"""
The weekly path must act on one league at a time.

Weekly models are trained on fantasy points under a league's own rules, so
they are namespaced exactly like the season models. Before this, three
leagues shared one flat `models/` directory and one flat `outputs/`: whichever
league trained last owned the models, and prediction CSVs — whose names carry
only position, season and week — silently overwrote each other.

Nothing here trains a real model; CatBoost is far too slow for a unit test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_predict import lineup, predict_week, train_model, weekly_api
from nfl_predict.leagues import (
    get_profile,
    league_keys,
    reset_active_league,
    set_active_league,
)

# ---------------------------------------------------------------------------
# Paths are per league
# ---------------------------------------------------------------------------


def test_every_league_has_its_own_predictions_dir():
    dirs = {k: get_profile(k).predictions_dir for k in league_keys()}
    # Two leagues share a directory only if they share a feature table.
    for key, d in dirs.items():
        assert d == Path("outputs") / get_profile(key).artifact_key


def test_predictions_dir_follows_the_artifact_key():
    """Leagues that score alike share models, so they share predictions."""
    for key in league_keys():
        profile = get_profile(key)
        assert profile.predictions_dir.name == profile.artifact_key
        assert profile.model_dir.name == profile.artifact_key


def test_distinct_scoring_never_shares_a_model_dir():
    """The bug this prevents: one league's models answering for another."""
    seen: dict[Path, str] = {}
    for key in league_keys():
        profile = get_profile(key)
        other = seen.setdefault(profile.model_dir, profile.artifact_key)
        assert other == profile.artifact_key


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def test_model_is_loaded_from_the_league_directory(tmp_path, monkeypatch):
    """A missing model must name the league's own path, not a flat one."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError) as exc:
        predict_week.load_model_and_meta(position="QB", league="ludopathy")
    assert "models/ludopathy/qb_catboost.cbm" in str(exc.value).replace("\\", "/")


def test_a_leagues_model_does_not_satisfy_another(tmp_path, monkeypatch):
    """Writing hoh's model must not make ludopathy's lookup succeed."""
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "models" / "hoh"
    d.mkdir(parents=True)
    (d / "qb_catboost.cbm").write_bytes(b"not a real model")
    (d / "qb_catboost_meta.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        predict_week.load_model_and_meta(position="QB", league="ludopathy")


# ---------------------------------------------------------------------------
# Prediction CSVs
# ---------------------------------------------------------------------------


def _write_preds(root: Path, key: str, pos: str, points: float) -> None:
    d = root / "outputs" / key
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "player_name": "A",
                "team": "KC",
                "position": pos,
                "expected_ppr_points": points,
            }
        ]
    ).to_csv(d / f"predictions_{pos.lower()}_2026_week2.csv", index=False)


def test_each_league_reads_its_own_predictions(tmp_path, monkeypatch):
    """The regression: identical file names under one flat directory."""
    monkeypatch.chdir(tmp_path)
    _write_preds(tmp_path, "ludopathy", "QB", 31.0)
    _write_preds(tmp_path, "hoh", "QB", 12.0)

    for key, expected in (("ludopathy", 31.0), ("hoh", 12.0)):
        token = set_active_league(key)
        try:
            files = weekly_api._latest_prediction_files()
            assert len(files) == 1
            assert pd.read_csv(files[0])["expected_ppr_points"].iloc[0] == expected
        finally:
            reset_active_league(token)


def test_lineup_cli_reads_the_named_league(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_preds(tmp_path, "ludopathy", "QB", 31.0)
    _write_preds(tmp_path, "hoh", "QB", 12.0)

    got = lineup._find_latest_prediction_files("hoh")
    assert len(got) == 1
    assert pd.read_csv(got[0])["expected_ppr_points"].iloc[0] == 12.0


def test_predictions_fall_back_to_the_flat_directory(tmp_path, monkeypatch):
    """CSVs written before namespacing must still render, not vanish."""
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "outputs"
    d.mkdir()
    pd.DataFrame(
        [
            {
                "player_name": "A",
                "team": "KC",
                "position": "QB",
                "expected_ppr_points": 20.0,
            }
        ]
    ).to_csv(d / "predictions_qb_2026_week2.csv", index=False)

    token = set_active_league("ludopathy")
    try:
        assert len(weekly_api._latest_prediction_files()) == 1
    finally:
        reset_active_league(token)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_training_only_covers_positions_the_league_starts(monkeypatch):
    """An IDP league needs LB/DL; a D/ST one does not."""
    captured: list[str] = []

    def fake_train(df, position, registry=None, league=None):
        captured.append(position)
        return None

    df = pd.DataFrame(
        {
            "position": ["QB", "RB", "WR", "TE", "K", "LB", "DL", "DB"] * 3,
            "season": [2023, 2024, 2025] * 8,
            "week": [1] * 24,
            "fantasy_points_custom": [10.0] * 24,
            "player_id": [f"p{i}" for i in range(24)],
        }
    )
    monkeypatch.setattr(train_model, "load_features", lambda league=None: df)
    monkeypatch.setattr(train_model, "add_target_next_week", lambda d: d)
    monkeypatch.setattr(train_model, "train_position_model", fake_train)

    train_model.main(use_registry=False, league="hoh")
    assert "LB" not in captured and "DL" not in captured
    assert {"QB", "RB", "WR", "TE"} <= set(captured)

    captured.clear()
    train_model.main(use_registry=False, league="ludopathy")
    assert {"LB", "DL", "DB"} <= set(captured)


def test_training_passes_the_league_down(monkeypatch):
    """train_position_model must be told which league it is fitting."""
    seen: list[str | None] = []

    def fake_train(df, position, registry=None, league=None):
        seen.append(league)
        return None

    df = pd.DataFrame(
        {
            "position": ["QB"] * 3,
            "season": [2023, 2024, 2025],
            "week": [1, 1, 1],
            "fantasy_points_custom": [10.0] * 3,
            "player_id": ["a", "b", "c"],
        }
    )
    monkeypatch.setattr(train_model, "load_features", lambda league=None: df)
    monkeypatch.setattr(train_model, "add_target_next_week", lambda d: d)
    monkeypatch.setattr(train_model, "train_position_model", fake_train)

    train_model.main(use_registry=False, league="rumble")
    assert seen == ["rumble"]


def test_training_loads_that_leagues_feature_table(monkeypatch):
    asked: list[str | None] = []

    def fake_load(league=None):
        asked.append(league)
        return pd.DataFrame(
            {
                "position": ["QB"],
                "season": [2025],
                "week": [1],
                "fantasy_points_custom": [10.0],
                "player_id": ["a"],
            }
        )

    monkeypatch.setattr(train_model, "load_features", fake_load)
    monkeypatch.setattr(train_model, "add_target_next_week", lambda d: d)
    monkeypatch.setattr(train_model, "train_position_model", lambda *a, **k: None)
    train_model.main(use_registry=False, league="hoh")
    assert asked == ["hoh"]
