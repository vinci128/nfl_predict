"""
Tests for the week-1 opponent adjustment and for the season-boundary fix
that lets a not-yet-started season be fetched at all.

Nothing here touches the network: the schedule is written to a tmp parquet
and the feature table is built inline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_predict import fetch_nfl_data, matchup

# ---------------------------------------------------------------------------
# Fetching a season that has not kicked off
# ---------------------------------------------------------------------------


def test_missing_season_detects_404():
    """A dataset published one parquet per season 404s until week 1."""
    exc = ConnectionError(
        "Failed to download .../player_week_2026.parquet: 404 Client Error"
    )
    assert fetch_nfl_data._is_missing_season(exc)


def test_missing_season_detects_range_validation():
    """Other loaders refuse the call before making it — same meaning."""
    exc = ValueError("Season must be between 2012 and 2025")
    assert fetch_nfl_data._is_missing_season(exc)


def test_missing_season_follows_the_cause_chain():
    """nflreadpy re-wraps requests' HTTPError, so the 404 is a cause."""
    try:
        try:
            raise OSError("404 Client Error: Not Found")
        except OSError as inner:
            raise ConnectionError("Failed to download") from inner
    except ConnectionError as exc:
        assert fetch_nfl_data._is_missing_season(exc)


def test_real_failures_still_raise():
    """A genuine error must not be mistaken for an unpublished season."""
    assert not fetch_nfl_data._is_missing_season(ValueError("schema mismatch"))
    assert not fetch_nfl_data._is_missing_season(ConnectionError("500 Server Error"))


def test_get_seasons_includes_the_fantasy_calendar_season(monkeypatch):
    """nflreadpy rolls over later than the fantasy calendar.

    On 2026-09-09 it still answered 2025 with week 1 kicking off, which left
    the pipeline with no 2026 schedule at all.
    """
    monkeypatch.setattr(fetch_nfl_data.nfl, "get_current_season", lambda **k: 2025)
    monkeypatch.setattr(
        fetch_nfl_data,
        "get_default_season_and_week",
        lambda: (2026, 1),
        raising=False,
    )
    import nfl_predict.predict_week as pw

    monkeypatch.setattr(pw, "get_default_season_and_week", lambda: (2026, 1))
    assert fetch_nfl_data.get_seasons()[-1] == 2026


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _features(season: int = 2025) -> pd.DataFrame:
    """A prior season where BAL smothers QBs and CAR concedes to them."""
    rows = []
    for week in range(1, 18):
        for team, qb_pts in (("BAL", 5.0), ("CAR", 30.0), ("DAL", 17.5)):
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "opponent_team": team,
                    "position": "QB",
                    "fantasy_points_custom": qb_pts,
                }
            )
            rows.append(
                {
                    "season": season,
                    "week": week,
                    "opponent_team": team,
                    "position": "WR",
                    "fantasy_points_custom": 10.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def schedule(tmp_path, monkeypatch):
    """Week 1 of 2026: PHI at CAR, KC at BAL. DET is on a bye."""
    df = pd.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "home_team": "CAR",
                "away_team": "PHI",
                "spread_line": -3.0,
                "total_line": 47.5,
            },
            {
                "season": 2026,
                "week": 1,
                "home_team": "BAL",
                "away_team": "KC",
                "spread_line": 2.5,
                "total_line": 44.0,
            },
        ]
    )
    (tmp_path / "data").mkdir()
    df.to_parquet(tmp_path / "data" / "schedules.parquet")
    monkeypatch.setattr(matchup, "DATA_DIR", tmp_path / "data")
    return df


# ---------------------------------------------------------------------------
# Opponent ratings
# ---------------------------------------------------------------------------


def test_dvp_ranks_defences_by_what_they_concede():
    dvp = matchup.prior_season_dvp(_features(), season=2026)
    qb = dvp[dvp.position == "QB"].set_index("team")
    assert qb.loc["CAR", "dvp_z"] > 0 > qb.loc["BAL", "dvp_z"]
    assert qb.loc["CAR", "dvp"] > qb.loc["DAL", "dvp"] > qb.loc["BAL", "dvp"]


def test_dvp_reads_the_prior_season_not_the_target():
    """Passing season=2026 must summarise 2025."""
    dvp = matchup.prior_season_dvp(_features(season=2025), season=2026)
    assert not dvp.empty
    assert matchup.prior_season_dvp(_features(season=2025), season=2030).empty


def test_dvp_z_is_clipped():
    """One lopsided prior season must not hand a QB unlimited points."""
    f = _features()
    f.loc[f.opponent_team == "CAR", "fantasy_points_custom"] = 500.0
    dvp = matchup.prior_season_dvp(f, season=2026)
    assert dvp.dvp_z.abs().max() <= matchup.Z_CLIP


# ---------------------------------------------------------------------------
# Applying the adjustment
# ---------------------------------------------------------------------------


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player_name": "QB v CAR",
                "position": "QB",
                "nfl_team": "PHI",
                "expected_ppr_points": 20.0,
            },
            {
                "player_name": "QB v BAL",
                "position": "QB",
                "nfl_team": "KC",
                "expected_ppr_points": 20.0,
            },
            {
                "player_name": "WR v CAR",
                "position": "WR",
                "nfl_team": "PHI",
                "expected_ppr_points": 12.0,
            },
            {
                "player_name": "bye",
                "position": "QB",
                "nfl_team": "DET",
                "expected_ppr_points": 18.0,
            },
        ]
    )


def test_soft_opponent_helps_and_tough_one_hurts(schedule):
    out = matchup.apply_matchup(_players(), _features(), season=2026, week=1)
    out = out.set_index("player_name")
    assert out.loc["QB v CAR", "expected_ppr_points"] > 20.0
    assert out.loc["QB v BAL", "expected_ppr_points"] < 20.0


def test_only_qb_and_rb_are_adjusted(schedule):
    """WR/TE/LB missed significance — applying them would add only noise."""
    out = matchup.apply_matchup(_players(), _features(), season=2026, week=1)
    wr = out.set_index("player_name").loc["WR v CAR"]
    assert wr["matchup_delta"] == 0.0
    assert wr["expected_ppr_points"] == 12.0
    assert wr["opponent"] == "CAR"  # still told who they face


def test_bye_week_keeps_the_base_projection(schedule):
    """A team absent from the schedule must not be silently dropped."""
    out = matchup.apply_matchup(_players(), _features(), season=2026, week=1)
    bye = out.set_index("player_name").loc["bye"]
    assert bye["expected_ppr_points"] == 18.0
    assert bye["matchup_delta"] == 0.0
    assert pd.isna(bye["opponent"])


def test_base_projection_is_preserved(schedule):
    out = matchup.apply_matchup(_players(), _features(), season=2026, week=1)
    assert list(out["expected_ppr_points_base"]) == [20.0, 20.0, 12.0, 18.0]


def test_delta_matches_the_fitted_coefficient(schedule):
    out = matchup.apply_matchup(_players(), _features(), season=2026, week=1)
    row = out.set_index("player_name").loc["QB v CAR"]
    expected = matchup.MATCHUP_COEF["QB"] * row["dvp_z"]
    assert row["matchup_delta"] == pytest.approx(expected, abs=0.01)


def test_missing_schedule_leaves_everything_untouched(tmp_path, monkeypatch):
    """No schedule for that week is a no-op, not a crash."""
    (tmp_path / "data").mkdir()
    pd.DataFrame(
        [{"season": 2026, "week": 1, "home_team": "CAR", "away_team": "PHI"}]
    ).to_parquet(tmp_path / "data" / "schedules.parquet")
    monkeypatch.setattr(matchup, "DATA_DIR", tmp_path / "data")
    out = matchup.apply_matchup(_players(), _features(), season=2026, week=9)
    assert list(out["expected_ppr_points"]) == [20.0, 20.0, 12.0, 18.0]
    assert (out["matchup_delta"] == 0).all()
