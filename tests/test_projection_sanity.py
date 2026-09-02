"""
Sanity checks that hold for every league's board.

These are the properties a reader would assume without checking: a floor below
a median below a ceiling, replacement level rising as the pool deepens, and
each league offering exactly the positions it starts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_predict.leagues import PROFILES, get_profile
from nfl_predict.season_model import _enforce_quantile_order

LEAGUES = sorted(PROFILES)


def _board(league: str) -> pd.DataFrame:
    profile = get_profile(league)
    path = profile.board_path(profile.season)
    if not path.exists():
        pytest.skip(f"no board for {league}; run `nfl-predict board --league {league}`")
    return pd.read_csv(path)


class TestQuantileOrder:
    """A ceiling below the median is nonsense on a board that shows both."""

    def test_repair_keeps_the_median_untouched(self) -> None:
        """p50 is what VOR ranks on and the only validated quantile."""
        df = pd.DataFrame(
            {"proj_p10": [73.2], "proj_p50": [207.1], "proj_p90": [180.7]}
        )
        out = _enforce_quantile_order(df.copy(), [0.1, 0.5, 0.9])
        assert out["proj_p50"].iloc[0] == 207.1

    def test_repair_pulls_a_low_ceiling_up_to_the_median(self) -> None:
        df = pd.DataFrame(
            {"proj_p10": [73.2], "proj_p50": [207.1], "proj_p90": [180.7]}
        )
        out = _enforce_quantile_order(df.copy(), [0.1, 0.5, 0.9])
        assert out["proj_p90"].iloc[0] == 207.1

    def test_repair_pushes_a_high_floor_down_to_the_median(self) -> None:
        df = pd.DataFrame(
            {"proj_p10": [140.2], "proj_p50": [133.2], "proj_p90": [153.8]}
        )
        out = _enforce_quantile_order(df.copy(), [0.1, 0.5, 0.9])
        assert out["proj_p10"].iloc[0] == 133.2

    def test_repair_leaves_a_well_ordered_row_alone(self) -> None:
        df = pd.DataFrame({"proj_p10": [80.0], "proj_p50": [120.0], "proj_p90": [160.0]})
        out = _enforce_quantile_order(df.copy(), [0.1, 0.5, 0.9])
        assert out.iloc[0].tolist() == [80.0, 120.0, 160.0]

    def test_repair_covers_rate_and_availability_too(self) -> None:
        df = pd.DataFrame(
            {
                "proj_ppg_p10": [20.0],
                "proj_ppg_p50": [15.0],
                "proj_ppg_p90": [10.0],
                "proj_games_p10": [17.0],
                "proj_games_p50": [12.0],
                "proj_games_p90": [9.0],
            }
        )
        out = _enforce_quantile_order(df.copy(), [0.1, 0.5, 0.9])
        assert out["proj_ppg_p10"].iloc[0] <= out["proj_ppg_p50"].iloc[0]
        assert out["proj_games_p90"].iloc[0] >= out["proj_games_p50"].iloc[0]

    @pytest.mark.parametrize("league", LEAGUES)
    def test_no_board_row_has_a_crossed_band(self, league: str) -> None:
        b = _board(league)
        bad = b[(b.proj_p10 > b.proj_p50) | (b.proj_p50 > b.proj_p90)]
        assert bad.empty, (
            f"{len(bad)} crossed bands in {league}, e.g. "
            f"{bad.head(3)[['player_name', 'proj_p10', 'proj_p50', 'proj_p90']].to_dict('records')}"
        )


class TestBoardShape:
    @pytest.mark.parametrize("league", LEAGUES)
    def test_board_offers_exactly_the_startable_positions(self, league: str) -> None:
        """An IDP league must not list a defence, and vice versa."""
        assert set(_board(league).position) == set(get_profile(league).roster.positions)

    @pytest.mark.parametrize("league", LEAGUES)
    def test_ranking_columns_are_populated(self, league: str) -> None:
        b = _board(league)
        assert b.proj_p50.notna().all()
        assert b.vor.notna().all()

    @pytest.mark.parametrize("league", LEAGUES)
    def test_overall_rank_follows_vor(self, league: str) -> None:
        b = _board(league).sort_values("overall_rank")
        assert b.vor.is_monotonic_decreasing

    @pytest.mark.parametrize("league", LEAGUES)
    def test_every_player_appears_once(self, league: str) -> None:
        b = _board(league)
        assert not b.player_id.duplicated().any()


class TestReplacementLevel:
    """The deeper the pool, the better the player you can still get for free."""

    def test_replacement_rises_as_leagues_shrink(self) -> None:
        base = {}
        for league in ("rumble", "hoh"):
            b = _board(league)
            base[league] = {
                p: b[b.position == p].replacement_baseline.iloc[0] for p in ("RB", "WR")
            }
        # 8 teams leaves better players undrafted than 14 does.
        assert base["rumble"]["RB"] > base["hoh"]["RB"]
        assert base["rumble"]["WR"] > base["hoh"]["WR"]

    def test_the_shallower_league_compresses_vor(self) -> None:
        """Same scoring, 8 teams vs 14: the top is worth much less."""
        h, r = _board("hoh"), _board("rumble")
        top_h = h.nsmallest(10, "overall_rank").vor.mean()
        top_r = r.nsmallest(10, "overall_rank").vor.mean()
        assert top_r < top_h


class TestIdenticalScoringAgrees:
    """hoh and rumble share one ScoringRules, so the inputs must match."""

    def test_the_feature_tables_are_the_same(self) -> None:
        col = "fantasy_points_custom"
        totals = {}
        for league in ("hoh", "rumble"):
            path = get_profile(league).features_path
            if not path.exists():
                pytest.skip("feature tables not built")
            totals[league] = pd.read_parquet(path, columns=[col])[col].sum()
        assert totals["hoh"] == pytest.approx(totals["rumble"])

    def test_ludopathy_scores_differently(self) -> None:
        """Sanity that the namespacing is doing anything at all."""
        col = "fantasy_points_custom"
        paths = {k: get_profile(k).features_path for k in ("hoh", "ludopathy")}
        if not all(p.exists() for p in paths.values()):
            pytest.skip("feature tables not built")
        a = pd.read_parquet(paths["hoh"], columns=[col])[col].sum()
        b = pd.read_parquet(paths["ludopathy"], columns=[col])[col].sum()
        assert b > a * 1.5
