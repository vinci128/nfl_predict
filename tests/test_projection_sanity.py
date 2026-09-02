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
        df = pd.DataFrame(
            {"proj_p10": [80.0], "proj_p50": [120.0], "proj_p90": [160.0]}
        )
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


class TestSharedArtifacts:
    """
    Leagues that score identically must share one fit.

    Training twice on the same target is waste, and the two runs disagreed:
    288 of 690 players got different projections, up to 28 points apart, so the
    same player was worth two different numbers depending on which league you
    were looking at.
    """

    def test_rumble_reuses_hell_or_highwaters_fit(self) -> None:
        assert get_profile("rumble").artifact_key == "hoh"
        assert get_profile("rumble").features_path == get_profile("hoh").features_path
        assert get_profile("rumble").model_dir == get_profile("hoh").model_dir

    def test_a_league_with_its_own_scoring_keeps_its_own(self) -> None:
        ludo = get_profile("ludopathy")
        assert ludo.artifact_key == "ludopathy"
        assert ludo.model_dir != get_profile("hoh").model_dir

    def test_boards_and_state_stay_per_league(self) -> None:
        """VOR depends on league size, and two drafts must not share a session."""
        hoh, rumble = get_profile("hoh"), get_profile("rumble")
        assert hoh.board_path(2026) != rumble.board_path(2026)
        assert hoh.state_path != rumble.state_path

    def test_building_every_league_builds_each_table_once(self) -> None:
        from nfl_predict.leagues import artifact_keys, league_keys

        assert len(artifact_keys()) < len(league_keys())
        assert set(artifact_keys()) == {"hoh", "ludopathy"}

    def test_sharing_is_rejected_when_scoring_differs(self) -> None:
        """The guard that stops a model being used on the wrong target."""
        from dataclasses import replace

        from nfl_predict.leagues import PROFILES, _validate_shared_artifacts

        bad = replace(get_profile("ludopathy"), shares_artifacts_with="hoh")
        original = dict(PROFILES)
        PROFILES["ludopathy"] = bad
        try:
            with pytest.raises(ValueError, match="score"):
                _validate_shared_artifacts()
        finally:
            PROFILES.clear()
            PROFILES.update(original)

    def test_sharing_with_an_unknown_league_is_rejected(self) -> None:
        from dataclasses import replace

        from nfl_predict.leagues import PROFILES, _validate_shared_artifacts

        bad = replace(get_profile("rumble"), shares_artifacts_with="nope")
        original = dict(PROFILES)
        PROFILES["rumble"] = bad
        try:
            with pytest.raises(ValueError, match="unknown league"):
                _validate_shared_artifacts()
        finally:
            PROFILES.clear()
            PROFILES.update(original)

    def test_the_shared_leagues_project_identically(self) -> None:
        """The whole point: one fit, one answer per player."""
        h, r = _board("hoh"), _board("rumble")
        m = h.merge(r, on=["player_name", "position"], suffixes=("_h", "_r"))
        assert (m.proj_p50_h == m.proj_p50_r).all()

    def test_but_their_vor_still_differs(self) -> None:
        """Same projections, different league size, so different value."""
        h, r = _board("hoh"), _board("rumble")
        m = h.merge(r, on=["player_name", "position"], suffixes=("_h", "_r"))
        assert (m.vor_h != m.vor_r).any()

    def test_ludopathy_scores_differently(self) -> None:
        """Sanity that the namespacing is doing anything at all."""
        col = "fantasy_points_custom"
        paths = {k: get_profile(k).features_path for k in ("hoh", "ludopathy")}
        if not all(p.exists() for p in paths.values()):
            pytest.skip("feature tables not built")
        a = pd.read_parquet(paths["hoh"], columns=[col])[col].sum()
        b = pd.read_parquet(paths["ludopathy"], columns=[col])[col].sum()
        assert b > a * 1.5
