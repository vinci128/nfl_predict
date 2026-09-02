"""
Tests for league profiles and league-specific scoring.

The scoring rules encode two real ESPN leagues. Where a value looks odd (a 50+
yard field goal worth nothing, whole-point yardage increments), the test says
so explicitly — these are the settings as configured, not oversights.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_predict.draft_board import compute_vor
from nfl_predict.leagues import (
    PROFILES,
    GameBonus,
    ScoringRules,
    fantasy_position,
    get_profile,
    league_keys,
    stat_series,
)


def frame(**cols) -> pd.DataFrame:
    """A one-row player-game frame from keyword stats."""
    return pd.DataFrame({k: [v] for k, v in cols.items()})


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


class TestProfileResolution:
    def test_every_league_is_configured(self) -> None:
        assert set(league_keys()) == {"ludopathy", "hoh", "rumble"}

    def test_default_is_ludopathy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NFL_PREDICT_LEAGUE", raising=False)
        assert get_profile().key == "ludopathy"

    def test_env_var_sets_the_league(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        assert get_profile().key == "hoh"

    def test_explicit_key_beats_the_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NFL_PREDICT_LEAGUE", "hoh")
        assert get_profile("ludopathy").key == "ludopathy"

    @pytest.mark.parametrize(
        "alias", ["HOH", "hell_or_highwater", "Hell Or Highwater", "hell-or-highwater"]
    )
    def test_aliases_resolve(self, alias: str) -> None:
        assert get_profile(alias).key == "hoh"

    def test_unknown_league_names_the_known_ones(self) -> None:
        with pytest.raises(KeyError, match="hoh"):
            get_profile("some-other-league")

    def test_passing_a_profile_through_is_idempotent(self) -> None:
        prof = get_profile("hoh")
        assert get_profile(prof) is prof

    def test_artifact_paths_are_namespaced_per_league(self) -> None:
        a, b = get_profile("ludopathy"), get_profile("hoh")
        assert a.features_path != b.features_path
        assert a.model_dir != b.model_dir
        assert a.board_path(2026) != b.board_path(2026)


# ---------------------------------------------------------------------------
# Stat aliases
# ---------------------------------------------------------------------------


class TestStatAliases:
    def test_sum_group_adds_its_members(self) -> None:
        df = frame(rushing_fumbles_lost=1, receiving_fumbles_lost=2)
        assert stat_series(df, "fumbles_lost").iloc[0] == 3

    def test_rename_group_takes_only_the_first_present(self) -> None:
        """Both names present must not double-count — they are one stat."""
        df = frame(passing_interceptions=2, interceptions=2)
        assert stat_series(df, "passing_interceptions").iloc[0] == 2

    def test_rename_group_falls_back_to_the_older_name(self) -> None:
        df = frame(interceptions=3)
        assert stat_series(df, "passing_interceptions").iloc[0] == 3

    def test_missing_stat_is_zero_not_an_error(self) -> None:
        assert stat_series(frame(passing_yards=100), "def_sacks").iloc[0] == 0.0

    def test_nulls_score_as_zero(self) -> None:
        df = pd.DataFrame({"passing_yards": [None, 50.0]})
        assert list(stat_series(df, "passing_yards")) == [0.0, 50.0]

    def test_unregistered_name_falls_through_to_the_raw_column(self) -> None:
        assert stat_series(frame(def_qb_hits=4), "def_qb_hits").iloc[0] == 4


# ---------------------------------------------------------------------------
# Scoring mechanics
# ---------------------------------------------------------------------------


class TestScoringMechanics:
    def test_increment_scoring_floors_the_remainder(self) -> None:
        """
        "Every 10 yards = 1 point" is not 0.1/yard: 347 yards is 34 points,
        because the trailing 7 yards never complete an increment.
        """
        rules = ScoringRules(per_increment={"passing_yards": (10.0, 1.0)})
        assert rules.score(frame(passing_yards=347)).iloc[0] == 34.0

    def test_per_unit_scoring_is_continuous(self) -> None:
        rules = ScoringRules(per_unit={"passing_yards": 0.04})
        assert rules.score(frame(passing_yards=347)).iloc[0] == pytest.approx(13.88)

    def test_scoring_a_stat_twice_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="passing_yards"):
            ScoringRules(
                per_unit={"passing_yards": 0.04},
                per_increment={"passing_yards": (10.0, 1.0)},
            )

    def test_negative_yardage_does_not_award_a_bonus(self) -> None:
        rules = ScoringRules(per_unit={"rushing_yards": 0.1})
        assert rules.score(frame(rushing_yards=-8)).iloc[0] == pytest.approx(-0.8)


class TestGameBonuses:
    rules = ScoringRules(
        bonuses=(
            GameBonus("rushing_yards", lo=100, hi=200, points=2.0),
            GameBonus("rushing_yards", lo=200, points=3.0),
        )
    )

    @pytest.mark.parametrize(
        "yards,expected",
        [(99, 0.0), (100, 2.0), (199, 2.0), (200, 3.0), (250, 3.0)],
    )
    def test_bands_are_exclusive_at_the_boundary(
        self, yards: int, expected: float
    ) -> None:
        """A 200-yard game earns the 200+ bonus only, never both bands."""
        assert self.rules.score(frame(rushing_yards=yards)).iloc[0] == expected

    def test_bonuses_are_per_game_not_per_season(self) -> None:
        """
        Two 120-yard games earn the bonus twice; the same 240 yards summed into
        one row earns the 200+ bonus once. Scoring must therefore run before
        any aggregation, which is why it lives at weekly grain.
        """
        weekly = pd.DataFrame({"rushing_yards": [120, 120]})
        assert self.rules.score(weekly).sum() == 4.0
        assert self.rules.score(frame(rushing_yards=240)).iloc[0] == 3.0


class TestFieldGoals:
    def test_distance_buckets_are_used_when_present(self) -> None:
        hoh = get_profile("hoh").scoring
        df = frame(fg_made_0_19=1, fg_made_40_49=1, fg_made_50_59=1)
        assert hoh.score(df).iloc[0] == pytest.approx(3 + 4 + 5)

    def test_falls_back_to_longest_made_when_buckets_are_absent(self) -> None:
        hoh = get_profile("hoh").scoring
        assert hoh.score(frame(fg_made=2, fg_long=45)).iloc[0] == pytest.approx(8.0)

    def test_buckets_win_over_the_fallback(self) -> None:
        hoh = get_profile("hoh").scoring
        df = frame(fg_made_30_39=1, fg_made=5, fg_long=55)
        assert hoh.score(df).iloc[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# The leagues themselves
# ---------------------------------------------------------------------------


class TestLudopathyScoring:
    scoring = PROFILES["ludopathy"].scoring

    def test_passing_is_a_point_per_ten_yards_floored(self) -> None:
        # 300 yards, 2 TD, 1 INT -> 30 + 8 - 4. INT is -4 here, not the usual -2.
        df = frame(passing_yards=300, passing_tds=2, passing_interceptions=1)
        assert self.scoring.score(df).iloc[0] == pytest.approx(34.0)

    def test_interception_costs_four(self) -> None:
        assert self.scoring.score(frame(passing_interceptions=1)).iloc[0] == -4.0

    def test_quarterback_is_charged_for_sacks_taken(self) -> None:
        assert self.scoring.score(frame(sacks_suffered=4)).iloc[0] == pytest.approx(
            -2.0
        )

    def test_long_field_goals_score(self) -> None:
        """Verified on ESPN's settings page: 50-59 is 5 and 60+ is 6."""
        assert self.scoring.score(frame(fg_made_50_59=1)).iloc[0] == 5.0
        assert self.scoring.score(frame(fg_made_60_=1)).iloc[0] == 6.0

    def test_forty_yard_field_goal_is_worth_the_same_as_a_short_one(self) -> None:
        assert self.scoring.score(frame(fg_made_40_49=1)).iloc[0] == 3.0
        assert self.scoring.score(frame(fg_made_30_39=1)).iloc[0] == 3.0

    def test_missed_field_goal_costs_a_point(self) -> None:
        assert self.scoring.score(frame(fg_missed=1)).iloc[0] == -1.0

    def test_two_point_conversions_score(self) -> None:
        assert self.scoring.score(frame(rushing_2pt_conversions=1)).iloc[0] == 2.0
        assert self.scoring.score(frame(passing_2pt_conversions=1)).iloc[0] == 2.0

    def test_four_hundred_yard_passing_game_bonus(self) -> None:
        plain = self.scoring.score(frame(passing_yards=399)).iloc[0]
        bonus = self.scoring.score(frame(passing_yards=400)).iloc[0]
        assert bonus - plain == pytest.approx(4.0 + 1.0)  # bonus + one increment

    def test_every_tackle_scores_twice(self) -> None:
        """ESPN pays Total Tackles (1) on top of Solo (1.5) or Assisted (0.5)."""
        df = frame(def_tackles_solo=8, def_tackle_assists=4, def_pass_defended=2)
        assert self.scoring.score(df).iloc[0] == pytest.approx(8 * 2.5 + 4 * 1.5 + 2)

    def test_idp_uses_the_defensive_players_table_not_the_dst_one(self) -> None:
        """A defender's sack is 4 and his interception 5; a defence's are 1 and 2."""
        assert self.scoring.per_unit["def_sacks"] == 4.0
        assert self.scoring.per_unit["def_interceptions"] == 5.0
        assert self.scoring.per_unit["def_fumbles_forced"] == 4.0
        assert self.scoring.per_unit["def_fumbles"] == 4.0

    def test_a_starting_linebacker_is_a_real_fantasy_asset(self) -> None:
        """10 solo, 6 assists, a sack: 25 + 9 + 4."""
        lb = frame(def_tackles_solo=10, def_tackle_assists=6, def_sacks=1)
        assert self.scoring.score(lb).iloc[0] == pytest.approx(38.0)

    def test_unmodelled_bonuses_are_declared(self) -> None:
        """The long-TD bonuses need play-level data; the gap must stay visible."""
        assert any("PTD40" in u for u in self.scoring.unmodelled)


class TestHellOrHighwaterScoring:
    scoring = PROFILES["hoh"].scoring

    def test_passing_uses_the_standard_rate(self) -> None:
        df = frame(passing_yards=300, passing_tds=2, passing_interceptions=1)
        assert self.scoring.score(df).iloc[0] == pytest.approx(12.0 + 8 - 2)

    def test_a_missed_field_goal_costs_a_point(self) -> None:
        assert self.scoring.score(frame(fg_missed=2)).iloc[0] == pytest.approx(-2.0)

    def test_two_point_conversions_count(self) -> None:
        assert self.scoring.score(frame(rushing_2pt_conversions=1)).iloc[0] == 2.0

    def test_defensive_players_score_nothing(self) -> None:
        """No IDP slots in this league, so defensive stats carry no value."""
        df = frame(def_tackles_solo=12, def_sacks=2, def_pass_defended=3)
        assert self.scoring.score(df).iloc[0] == 0.0


class TestTheTwoLeaguesDiffer:
    def test_the_same_quarterback_game_is_worth_far_more_in_ludopathy(self) -> None:
        """
        The single most consequential difference between the leagues: 0.1 vs
        0.04 per passing yard. A board built for one is wrong for the other.
        """
        game = frame(passing_yards=320, passing_tds=3, passing_interceptions=0)
        lud = PROFILES["ludopathy"].scoring.score(game).iloc[0]
        hoh = PROFILES["hoh"].scoring.score(game).iloc[0]
        assert lud == pytest.approx(44.0)
        assert hoh == pytest.approx(24.8)
        assert lud > hoh * 1.5

    def test_a_receiving_game_scores_almost_the_same_in_both(self) -> None:
        game = frame(receiving_yards=90, receptions=7, receiving_tds=1)
        lud = PROFILES["ludopathy"].scoring.score(game).iloc[0]
        hoh = PROFILES["hoh"].scoring.score(game).iloc[0]
        assert abs(lud - hoh) <= 1.0


# ---------------------------------------------------------------------------
# Roster configuration
# ---------------------------------------------------------------------------


class TestRosterConfig:
    @pytest.mark.parametrize("key,total", [("ludopathy", 12), ("hoh", 9)])
    def test_starters_sum_to_the_total_espn_reports(self, key: str, total: int) -> None:
        r = get_profile(key).roster
        assert sum(r.starters.values()) + r.flex_spots == total

    def test_roster_size_is_starters_plus_bench(self) -> None:
        """IR sits outside the roster limit in both leagues, so it is excluded."""
        for profile in PROFILES.values():
            r = profile.roster
            assert sum(r.starters.values()) + r.flex_spots + r.bench == r.roster_size

    def test_ludopathy_positions_include_idp(self) -> None:
        assert set(get_profile("ludopathy").roster.positions) == {
            "QB",
            "RB",
            "WR",
            "TE",
            "LB",
            "DL",
            "K",
        }

    def test_hoh_positions_include_dst_and_no_idp(self) -> None:
        positions = set(get_profile("hoh").roster.positions)
        assert "DST" in positions
        assert not positions & {"LB", "DL", "DB"}


class TestDraftSettingsFromProfile:
    def test_idp_slots_reach_the_replacement_ranks(self) -> None:
        ranks = get_profile("ludopathy").to_draft_settings().replacement_ranks()
        # 3 LB starters x 10 teams + buffer
        assert ranks["LB"] == 33
        assert ranks["DL"] == 13

    def test_league_size_drives_replacement_level(self) -> None:
        lud = get_profile("ludopathy").to_draft_settings().replacement_ranks()
        hoh = get_profile("hoh").to_draft_settings().replacement_ranks()
        assert hoh["WR"] > lud["WR"]  # 14 teams dig deeper into the pool

    def test_positions_with_no_starting_slot_are_dropped(self) -> None:
        ranks = get_profile("hoh").to_draft_settings().replacement_ranks()
        assert "LB" not in ranks

    def test_vor_ranks_idp_against_idp_only(self) -> None:
        settings = get_profile("ludopathy").to_draft_settings()
        proj = pd.DataFrame(
            {
                "position": ["LB"] * 40 + ["WR"] * 40,
                "proj_p50": list(range(140, 100, -1)) + list(range(400, 360, -1)),
            }
        )
        out = compute_vor(proj, settings)
        top_lb = out[out.position == "LB"].vor.max()
        # Replacement LB is the 33rd, so the best LB clears it by ~32 points —
        # it is not compared against wide receivers.
        assert 25 < top_lb < 40


class TestFantasyPositionMapping:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("DE", "DL"),
            ("DT", "DL"),
            ("NT", "DL"),
            ("ILB", "LB"),
            ("MLB", "LB"),
            ("OLB", "LB"),
            ("FB", "RB"),
            ("QB", "QB"),
            ("CB", "DB"),
        ],
    )
    def test_raw_positions_map_to_fantasy_slots(self, raw: str, expected: str) -> None:
        assert fantasy_position(raw) == expected

    @pytest.mark.parametrize("raw", ["OL", "G", "C", "OT", "LS", "P", None])
    def test_unstartable_positions_map_to_nothing(self, raw: str | None) -> None:
        assert fantasy_position(raw) is None

    def test_mapping_is_case_and_space_insensitive(self) -> None:
        assert fantasy_position("  de ") == "DL"


class TestRoyalRumbleScoring:
    """
    Verified against ESPN's settings page on 2026-09-02. Royal Rumble runs
    ESPN's default PPR, identical to Hell or Highwater, so the two share one
    rule set — these tests exist to catch that sharing being broken by a
    change meant for only one of them.
    """

    profile = PROFILES["rumble"]
    scoring = profile.scoring

    def test_scoring_is_identical_to_hell_or_highwater(self) -> None:
        assert self.scoring is PROFILES["hoh"].scoring
        assert self.profile.dst_scoring is PROFILES["hoh"].dst_scoring

    def test_passing_is_four_hundredths_a_yard(self) -> None:
        # 300 yards, 2 TD, 1 INT -> 12 + 8 - 2
        df = frame(passing_yards=300, passing_tds=2, passing_interceptions=1)
        assert self.scoring.score(df).iloc[0] == pytest.approx(18.0)

    def test_full_point_per_reception(self) -> None:
        assert self.scoring.score(frame(receptions=6)).iloc[0] == pytest.approx(6.0)

    def test_no_game_bonuses(self) -> None:
        """Unlike Ludopathy, a 400-yard game is worth no more than the yards."""
        assert self.scoring.bonuses == ()

    def test_eight_teams_is_the_shallowest_league(self) -> None:
        sizes = {
            k: PROFILES[k].roster.league_size for k in ("rumble", "hoh", "ludopathy")
        }
        assert sizes["rumble"] == 8
        assert sizes["rumble"] < sizes["ludopathy"] < sizes["hoh"]

    def test_starters_fill_nine_of_fourteen_spots(self) -> None:
        r = self.profile.roster
        assert sum(r.starters.values()) + r.flex_spots == 9
        assert sum(r.starters.values()) + r.flex_spots + r.bench == r.roster_size

    def test_roster_targets_fill_the_roster_exactly(self) -> None:
        """A draft that hits every target ends with no empty spot and no overflow."""
        assert (
            sum(self.profile.roster_targets.values()) == self.profile.roster.roster_size
        )

    def test_it_starts_a_defence_not_idps(self) -> None:
        assert "DST" in self.profile.roster.positions
        assert "LB" not in self.profile.roster.positions

    def test_board_and_session_are_its_own(self) -> None:
        """VOR depends on league size, and two drafts must not share a session."""
        assert self.profile.board_path(2026) != PROFILES["hoh"].board_path(2026)
        assert self.profile.state_path != PROFILES["hoh"].state_path
        assert "rumble" in str(self.profile.board_path(2026))

    def test_it_reuses_hell_or_highwaters_fit(self) -> None:
        """Identical scoring means an identical training target — fit it once."""
        assert self.profile.features_path == PROFILES["hoh"].features_path
        assert self.profile.model_dir == PROFILES["hoh"].model_dir
