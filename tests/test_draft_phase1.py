"""
Tests for Phase 1 draft modules: season_features, season_model, draft_board.

All tests use synthetic DataFrames — no parquet files required.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nfl_predict.draft_board import (
    DraftSettings,
    assign_tiers,
    compute_vor,
)
from nfl_predict.season_features import build_all_inference_rows, build_season_snapshot
from nfl_predict.season_model import _get_season_feature_cols, build_training_data

# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic player-week data
# ---------------------------------------------------------------------------


def _make_weekly_df(
    n_players: int = 5,
    seasons: list[int] | None = None,
    n_weeks: int = 16,
    position: str = "WR",
) -> pd.DataFrame:
    """
    Build a minimal synthetic player-week DataFrame that mimics the shape
    produced by features.build_player_week_features().

    Contains the columns required by build_season_snapshot:
    - Identifiers: player_id, season, week, position, player_display_name, recent_team
    - Stats: fantasy_points_custom (non-null)
    - Feature suffixes: one _roll8, one _season_cum, one _season_mean column
    - games_played_roll8
    """
    if seasons is None:
        seasons = [2021, 2022, 2023]

    rows = []
    for pid in range(n_players):
        for season in seasons:
            cum = 0.0
            for week in range(1, n_weeks + 1):
                pts = float((pid + 1) * 10 + week)  # deterministic non-zero
                cum += pts
                rows.append(
                    {
                        "player_id": f"P{pid:02d}",
                        "player_display_name": f"Player {pid}",
                        "recent_team": "TM1",
                        "position": position,
                        "season": season,
                        "week": week,
                        "fantasy_points_custom": pts,
                        # Fake roll8 / season_cum / season_mean features
                        "fantasy_points_custom_roll8": pts * 0.9,
                        "receiving_yards_roll8": pts * 5.0,
                        "fantasy_points_custom_season_cum": cum,
                        "receiving_yards_season_cum": cum * 5.0,
                        "fantasy_points_custom_season_mean": cum / week,
                        # games played rolling
                        "games_played_roll8": min(week, 8),
                        "games_played_roll5": min(week, 5),
                        "games_played_roll3": min(week, 3),
                    }
                )
    return pd.DataFrame(rows)


def _make_rosters_df(
    n_players: int = 5, seasons: list[int] | None = None
) -> pd.DataFrame:
    """Build a minimal rosters DataFrame with gsis_id, birth_date, years_exp."""
    if seasons is None:
        seasons = [2021, 2022, 2023]
    rows = []
    for pid in range(n_players):
        for season in seasons:
            rows.append(
                {
                    "gsis_id": f"P{pid:02d}",
                    "season": season,
                    "birth_date": f"{1990 + pid}-03-15",
                    "years_exp": season - 2015 - pid,
                    "entry_year": 2015 + pid,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# season_features: build_season_snapshot
# ---------------------------------------------------------------------------


class TestBuildSeasonSnapshot:
    def test_one_row_per_player_season(self) -> None:
        df = _make_weekly_df(n_players=3, seasons=[2021, 2022, 2023])
        snap = build_season_snapshot(df)
        assert len(snap) == 3 * 3  # 3 players × 3 seasons

    def test_no_duplicate_player_seasons(self) -> None:
        df = _make_weekly_df(n_players=4, seasons=[2021, 2022])
        snap = build_season_snapshot(df)
        assert snap.duplicated(["player_id", "season"]).sum() == 0

    def test_season_total_current_matches_sum(self) -> None:
        df = _make_weekly_df(n_players=2, seasons=[2021, 2022], n_weeks=16)
        snap = build_season_snapshot(df)

        for _, row in snap.iterrows():
            pid, season = row["player_id"], row["season"]
            expected = df.loc[
                (df["player_id"] == pid) & (df["season"] == season),
                "fantasy_points_custom",
            ].sum()
            assert abs(row["season_total_pts_current"] - expected) < 1e-6

    def test_season_total_next_is_correct(self) -> None:
        df = _make_weekly_df(n_players=2, seasons=[2021, 2022, 2023])
        snap = build_season_snapshot(df)

        # For season 2021 the next-season total should equal the 2022 total
        p0_2021 = snap.loc[
            (snap["player_id"] == "P00") & (snap["season"] == 2021)
        ].iloc[0]
        p0_2022_total = df.loc[
            (df["player_id"] == "P00") & (df["season"] == 2022), "fantasy_points_custom"
        ].sum()
        assert abs(p0_2021["season_total_pts_next"] - p0_2022_total) < 1e-6

    def test_most_recent_season_has_nan_target(self) -> None:
        df = _make_weekly_df(n_players=3, seasons=[2021, 2022, 2023])
        snap = build_season_snapshot(df)
        last = snap[snap["season"] == 2023]
        assert last["season_total_pts_next"].isna().all()

    def test_earlier_seasons_have_non_nan_target(self) -> None:
        df = _make_weekly_df(n_players=3, seasons=[2021, 2022, 2023])
        snap = build_season_snapshot(df)
        earlier = snap[snap["season"] == 2021]
        assert earlier["season_total_pts_next"].notna().all()

    def test_keeps_roll8_columns(self) -> None:
        df = _make_weekly_df(n_players=2, seasons=[2022])
        snap = build_season_snapshot(df)
        assert "fantasy_points_custom_roll8" in snap.columns

    def test_keeps_season_cum_columns(self) -> None:
        df = _make_weekly_df(n_players=2, seasons=[2022])
        snap = build_season_snapshot(df)
        assert "fantasy_points_custom_season_cum" in snap.columns

    def test_games_played_season_is_positive(self) -> None:
        df = _make_weekly_df(n_players=2, seasons=[2022], n_weeks=10)
        snap = build_season_snapshot(df)
        assert (snap["games_played_season"] > 0).all()

    def test_roster_merge_adds_age(self) -> None:
        df = _make_weekly_df(n_players=3, seasons=[2021, 2022])
        rosters = _make_rosters_df(n_players=3, seasons=[2021, 2022])
        snap = build_season_snapshot(df, rosters=rosters)
        assert "age_at_season_start" in snap.columns
        assert snap["age_at_season_start"].notna().any()

    def test_roster_merge_age_reasonable(self) -> None:
        df = _make_weekly_df(n_players=1, seasons=[2022])
        rosters = _make_rosters_df(n_players=1, seasons=[2022])
        snap = build_season_snapshot(df, rosters=rosters)
        age = snap["age_at_season_start"].iloc[0]
        assert 20 <= age <= 40  # player born 1990, season 2022 → ~32 years old

    def test_no_raw_weekly_columns_in_snapshot(self) -> None:
        """Columns like 'week' or raw 'fantasy_points_custom' (non-suffixed) should not appear."""
        df = _make_weekly_df(n_players=2, seasons=[2022])
        snap = build_season_snapshot(df)
        # 'week' should not be a feature column (it's the groupby key consumed by .last())
        assert "week" not in snap.columns


# ---------------------------------------------------------------------------
# season_features: build_all_inference_rows
# ---------------------------------------------------------------------------


class TestBuildAllInferenceRows:
    def test_returns_only_requested_season_and_position(self) -> None:
        df = _make_weekly_df(n_players=4, seasons=[2021, 2022, 2023], position="WR")
        # Add some RB rows to ensure position filter works
        rb_df = _make_weekly_df(n_players=2, seasons=[2021, 2022, 2023], position="RB")
        rb_df["player_id"] = rb_df["player_id"].str.replace("P0", "RB")
        combined = pd.concat([df, rb_df], ignore_index=True)

        rows = build_all_inference_rows(combined, as_of_season=2023, position="WR")
        assert (rows["position"] == "WR").all()
        assert (rows["season"] == 2023).all()

    def test_no_target_columns(self) -> None:
        df = _make_weekly_df(n_players=3, seasons=[2021, 2022, 2023])
        rows = build_all_inference_rows(df, as_of_season=2023, position="WR")
        assert "season_total_pts_next" not in rows.columns
        assert "season_total_pts_current" not in rows.columns

    def test_empty_result_for_missing_season(self) -> None:
        df = _make_weekly_df(n_players=3, seasons=[2021, 2022])
        rows = build_all_inference_rows(df, as_of_season=2024, position="WR")
        assert rows.empty

    def test_correct_player_count(self) -> None:
        n = 6
        df = _make_weekly_df(n_players=n, seasons=[2021, 2022, 2023])
        rows = build_all_inference_rows(df, as_of_season=2023, position="WR")
        assert len(rows) == n


# ---------------------------------------------------------------------------
# season_model: _get_season_feature_cols
# ---------------------------------------------------------------------------


class TestGetSeasonFeatureCols:
    def _snapshot_df(self) -> pd.DataFrame:
        df = _make_weekly_df(n_players=5, seasons=[2021, 2022, 2023])
        return build_season_snapshot(df)

    def test_all_cols_are_numeric(self) -> None:
        snap = self._snapshot_df()
        cols = _get_season_feature_cols(snap, "WR")
        for col in cols:
            assert pd.api.types.is_numeric_dtype(snap[col]), f"{col} is not numeric"

    def test_drops_player_id(self) -> None:
        snap = self._snapshot_df()
        cols = _get_season_feature_cols(snap, "WR")
        assert "player_id" not in cols

    def test_drops_season(self) -> None:
        snap = self._snapshot_df()
        cols = _get_season_feature_cols(snap, "WR")
        assert "season" not in cols

    def test_drops_targets(self) -> None:
        snap = self._snapshot_df()
        cols = _get_season_feature_cols(snap, "WR")
        assert "season_total_pts_next" not in cols
        assert "season_total_pts_current" not in cols

    def test_includes_fantasy_points_features(self) -> None:
        snap = self._snapshot_df()
        cols = _get_season_feature_cols(snap, "WR")
        # At minimum, the roll8 and season_cum fantasy columns should be present
        fp_cols = [c for c in cols if "fantasy_points" in c]
        assert len(fp_cols) >= 1

    def test_non_empty_for_all_positions(self) -> None:
        snap = self._snapshot_df()
        for pos in ("QB", "RB", "WR", "TE", "K"):
            cols = _get_season_feature_cols(snap, pos)
            assert len(cols) > 0, f"No features for {pos}"


# ---------------------------------------------------------------------------
# season_model: build_training_data (uses synthetic data via monkeypatching)
# ---------------------------------------------------------------------------


class TestBuildTrainingData:
    def test_raises_on_insufficient_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With fewer than 50 rows the function should raise ValueError."""
        import nfl_predict.season_model as sm

        tiny_df = _make_weekly_df(n_players=2, seasons=[2021, 2022])
        monkeypatch.setattr(sm, "load_features", lambda: tiny_df)
        monkeypatch.setattr(sm, "load_rosters", lambda: None)

        with pytest.raises(ValueError, match="Not enough"):
            build_training_data("WR")

    def test_valid_split_is_last_season(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Validation set should be the most recent season in the snapshot."""
        import nfl_predict.season_model as sm

        # 14 players × 5 seasons → 14×4=56 rows with valid target (≥50 threshold)
        df = _make_weekly_df(
            n_players=14, seasons=[2019, 2020, 2021, 2022, 2023], n_weeks=16
        )
        monkeypatch.setattr(sm, "load_features", lambda: df)
        monkeypatch.setattr(sm, "load_rosters", lambda: None)

        df_train, df_valid, _ = build_training_data("WR")
        assert df_valid["season"].nunique() == 1
        assert df_valid["season"].iloc[0] == df_train["season"].max() + 1

    def test_feature_cols_are_subset_of_snapshot_cols(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nfl_predict.season_model as sm

        df = _make_weekly_df(
            n_players=14, seasons=[2019, 2020, 2021, 2022, 2023], n_weeks=16
        )
        monkeypatch.setattr(sm, "load_features", lambda: df)
        monkeypatch.setattr(sm, "load_rosters", lambda: None)

        df_train, _, feature_cols = build_training_data("WR")
        for col in feature_cols:
            assert col in df_train.columns


# ---------------------------------------------------------------------------
# draft_board: DraftSettings
# ---------------------------------------------------------------------------


class TestDraftSettings:
    def test_default_replacement_ranks_keys(self) -> None:
        settings = DraftSettings()
        ranks = settings.replacement_ranks()
        assert set(ranks.keys()) == {"QB", "RB", "WR", "TE", "K"}

    def test_scales_with_league_size(self) -> None:
        s12 = DraftSettings(league_size=12)
        s10 = DraftSettings(league_size=10)
        assert s12.replacement_ranks()["QB"] > s10.replacement_ranks()["QB"]

    def test_replacement_ranks_positive(self) -> None:
        ranks = DraftSettings().replacement_ranks()
        for pos, rank in ranks.items():
            assert rank > 0, f"Rank for {pos} should be positive"


# ---------------------------------------------------------------------------
# draft_board: compute_vor
# ---------------------------------------------------------------------------


def _make_projections(n_wr: int = 20, n_rb: int = 20, n_qb: int = 12) -> pd.DataFrame:
    """Build a minimal projections DataFrame for VOR / tier tests."""
    rows = []
    for i in range(n_qb):
        rows.append(
            {
                "player_id": f"QB{i}",
                "player_name": f"QB Player {i}",
                "position": "QB",
                "team": "TM",
                "proj_p10": 200.0 - i * 5,
                "proj_p50": 350.0 - i * 10,
                "proj_p90": 500.0 - i * 15,
                "projected_season": 2026,
            }
        )
    for i in range(n_rb):
        rows.append(
            {
                "player_id": f"RB{i}",
                "player_name": f"RB Player {i}",
                "position": "RB",
                "team": "TM",
                "proj_p10": 80.0 - i * 2,
                "proj_p50": 200.0 - i * 7,
                "proj_p90": 320.0 - i * 10,
                "projected_season": 2026,
            }
        )
    for i in range(n_wr):
        rows.append(
            {
                "player_id": f"WR{i}",
                "player_name": f"WR Player {i}",
                "position": "WR",
                "team": "TM",
                "proj_p10": 60.0 - i * 2,
                "proj_p50": 180.0 - i * 7,
                "proj_p90": 280.0 - i * 10,
                "projected_season": 2026,
            }
        )
    return pd.DataFrame(rows)


class TestComputeVor:
    def test_adds_vor_column(self) -> None:
        proj = _make_projections()
        result = compute_vor(proj)
        assert "vor" in result.columns

    def test_adds_replacement_baseline_column(self) -> None:
        proj = _make_projections()
        result = compute_vor(proj)
        assert "replacement_baseline" in result.columns

    def test_replacement_player_vor_is_zero(self) -> None:
        """The Nth ranked player at each position should have VOR ≈ 0."""
        settings = DraftSettings(
            league_size=12,
            qb_starters=1,
            rb_starters=2,
            wr_starters=2,
            te_starters=1,
            k_starters=1,
            flex_spots=0,
            replacement_buffer=0,
        )
        proj = _make_projections(n_qb=15, n_rb=30, n_wr=30)
        result = compute_vor(proj, settings=settings)

        qb_rank = settings.replacement_ranks()["QB"]  # 12
        qb_sorted = (
            result[result["position"] == "QB"]
            .sort_values("proj_p50", ascending=False)
            .reset_index(drop=True)
        )
        replacement_vor = float(qb_sorted.iloc[qb_rank - 1]["vor"])
        assert abs(replacement_vor) < 1e-6

    def test_top_players_have_positive_vor(self) -> None:
        proj = _make_projections()
        result = compute_vor(proj)
        top_wr = (
            result[result["position"] == "WR"]
            .sort_values("proj_p50", ascending=False)
            .head(5)
        )
        assert (top_wr["vor"] > 0).all()

    def test_bottom_players_have_negative_vor(self) -> None:
        proj = _make_projections(n_wr=40)
        result = compute_vor(proj)
        bottom_wr = (
            result[result["position"] == "WR"]
            .sort_values("proj_p50", ascending=True)
            .head(5)
        )
        assert (bottom_wr["vor"] < 0).all()

    def test_raises_without_proj_p50(self) -> None:
        proj = _make_projections().drop(columns=["proj_p50"])
        with pytest.raises(ValueError, match="proj_p50"):
            compute_vor(proj)

    def test_works_with_single_position(self) -> None:
        proj = _make_projections(n_rb=0, n_qb=0)
        result = compute_vor(proj)
        assert "vor" in result.columns
        assert len(result) == 20  # only WR rows


# ---------------------------------------------------------------------------
# draft_board: assign_tiers
# ---------------------------------------------------------------------------


class TestAssignTiers:
    def test_adds_tier_column(self) -> None:
        proj = compute_vor(_make_projections())
        result = assign_tiers(proj)
        assert "tier" in result.columns

    def test_adds_pos_tier_column(self) -> None:
        proj = compute_vor(_make_projections())
        result = assign_tiers(proj)
        assert "pos_tier" in result.columns

    def test_tiers_within_bounds(self) -> None:
        n_tiers = 6
        proj = compute_vor(_make_projections())
        result = assign_tiers(proj, n_tiers=n_tiers)
        assert result["tier"].min() >= 1
        assert result["tier"].max() <= n_tiers

    def test_top_player_is_tier_1(self) -> None:
        proj = compute_vor(_make_projections())
        result = assign_tiers(proj)
        top_player = result.sort_values("vor", ascending=False).iloc[0]
        assert top_player["tier"] == 1

    def test_raises_without_vor(self) -> None:
        proj = _make_projections()  # no VOR column
        with pytest.raises(ValueError, match="vor"):
            assign_tiers(proj)

    def test_pos_tier_starts_at_1_per_position(self) -> None:
        proj = compute_vor(_make_projections())
        result = assign_tiers(proj)
        for pos in result["position"].unique():
            pos_min = result.loc[result["position"] == pos, "pos_tier"].min()
            assert pos_min == 1


# ---------------------------------------------------------------------------
# draft_board: full build_draft_board integration (mocked predict_season)
# ---------------------------------------------------------------------------


class TestBuildDraftBoard:
    def test_sorted_by_vor_descending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nfl_predict.draft_board as db

        proj = _make_projections()

        # predict_season returns a slice of proj by position
        def mock_predict(position: str, as_of_season: int, **_: object) -> pd.DataFrame:
            return proj[proj["position"] == position].copy()

        monkeypatch.setattr(db, "predict_season", mock_predict)

        board = db.build_draft_board(as_of_season=2025, positions=["QB", "RB", "WR"])
        vors = board["vor"].tolist()
        assert vors == sorted(vors, reverse=True)

    def test_has_overall_rank_column(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nfl_predict.draft_board as db

        proj = _make_projections()

        def mock_predict(position: str, as_of_season: int, **_: object) -> pd.DataFrame:
            return proj[proj["position"] == position].copy()

        monkeypatch.setattr(db, "predict_season", mock_predict)
        board = db.build_draft_board(as_of_season=2025, positions=["QB", "RB", "WR"])
        assert "overall_rank" in board.columns
        assert board["overall_rank"].iloc[0] == 1

    def test_has_pos_rank_column(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nfl_predict.draft_board as db

        proj = _make_projections()

        def mock_predict(position: str, as_of_season: int, **_: object) -> pd.DataFrame:
            return proj[proj["position"] == position].copy()

        monkeypatch.setattr(db, "predict_season", mock_predict)
        board = db.build_draft_board(as_of_season=2025, positions=["QB", "RB", "WR"])
        assert "pos_rank" in board.columns
        # Each position's top player should have pos_rank == 1
        for pos in ["QB", "RB", "WR"]:
            top = board[board["position"] == pos]["pos_rank"].min()
            assert top == 1

    def test_raises_when_no_projections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import nfl_predict.draft_board as db

        monkeypatch.setattr(db, "predict_season", lambda *a, **kw: pd.DataFrame())
        with pytest.raises(ValueError, match="No projections"):
            db.build_draft_board(as_of_season=2025, positions=["WR"])


# ---------------------------------------------------------------------------
# draft_board: export_draft_board
# ---------------------------------------------------------------------------


class TestExportDraftBoard:
    def test_csv_export_creates_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import nfl_predict.draft_board as db

        proj = compute_vor(_make_projections())
        proj = assign_tiers(proj)
        proj["overall_rank"] = range(1, len(proj) + 1)
        proj["pos_rank"] = 1
        proj["projected_season"] = 2026

        out = tmp_path / "board.csv"
        result_path = db.export_draft_board(proj, out_path=str(out), fmt="csv")
        assert result_path.exists()
        loaded = pd.read_csv(result_path)
        assert len(loaded) == len(proj)

    def test_json_export_has_tiers_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import json

        import nfl_predict.draft_board as db

        proj = compute_vor(_make_projections())
        proj = assign_tiers(proj)
        proj["overall_rank"] = range(1, len(proj) + 1)
        proj["pos_rank"] = 1
        proj["projected_season"] = 2026
        if "player_name" not in proj.columns:
            proj["player_name"] = "Player"

        out = tmp_path / "board.json"
        db.export_draft_board(proj, out_path=str(out), fmt="json")
        data = json.loads(out.read_text())
        assert "tiers" in data
        assert "players" in data
        assert isinstance(data["tiers"], dict)

    def test_invalid_format_raises(self, tmp_path: Path) -> None:
        import nfl_predict.draft_board as db

        proj = compute_vor(_make_projections())
        proj = assign_tiers(proj)
        proj["overall_rank"] = range(1, len(proj) + 1)
        proj["pos_rank"] = 1
        proj["projected_season"] = 2026

        with pytest.raises(ValueError, match="Unknown format"):
            db.export_draft_board(proj, out_path=str(tmp_path / "x.txt"), fmt="xml")
