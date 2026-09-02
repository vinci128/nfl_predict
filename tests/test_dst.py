"""
Tests for team defence / special teams projections.

The scoring here is almost entirely threshold ladders (points allowed, yards
allowed), and the two inputs those ladders read are *derived* — a team's points
allowed is its opponent's score, its yards allowed are its opponent's gains.
Getting either join backwards produces plausible-looking nonsense, so the joins
are tested directly rather than only through the final number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_predict.dst import (
    build_dst_weekly,
    dst_season_totals,
    fit_shrinkage,
    points_allowed,
    project_dst,
    yards_allowed,
)
from nfl_predict.leagues import get_profile

_HOH_DST = get_profile("hoh").dst_scoring
assert _HOH_DST is not None, "Hell or Highwater must define D/ST scoring"
RULES = _HOH_DST


@pytest.fixture
def schedules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 2],
            "game_type": ["REG", "REG"],
            "home_team": ["DEN", "KC"],
            "away_team": ["KC", "DEN"],
            "home_score": [10, 31],
            "away_score": [3, 0],
        }
    )


@pytest.fixture
def team_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2023] * 4,
            "week": [1, 1, 2, 2],
            "season_type": ["REG"] * 4,
            "team": ["DEN", "KC", "KC", "DEN"],
            "opponent_team": ["KC", "DEN", "DEN", "KC"],
            "passing_yards": [200, 150, 300, 80],
            "rushing_yards": [100, 40, 120, 15],
            "def_sacks": [3, 1, 2, 0],
            "def_interceptions": [1, 0, 0, 0],
            "def_fumbles": [0, 1, 0, 0],
            "def_safeties": [0, 0, 0, 0],
            "def_tds": [0, 0, 1, 0],
            "special_teams_tds": [0, 0, 0, 0],
            "def_punt_blocks": [0, 0, 0, 0],
            "def_pat_blocks": [0, 0, 0, 0],
            "def_fg_blocks": [0, 0, 0, 0],
        }
    )


class TestDerivedInputs:
    def test_points_allowed_is_the_opponents_score(self, schedules) -> None:
        pa = points_allowed(schedules)
        wk1 = pa[(pa.week == 1)].set_index("team")["points_allowed"]
        # DEN hosted and won 10-3, so DEN allowed 3 and KC allowed 10.
        assert wk1["DEN"] == 3
        assert wk1["KC"] == 10

    def test_every_game_yields_a_row_for_both_teams(self, schedules) -> None:
        assert len(points_allowed(schedules)) == 2 * len(schedules)

    def test_unplayed_games_are_dropped_not_scored_as_shutouts(self) -> None:
        sch = pd.DataFrame(
            {
                "season": [2026],
                "week": [1],
                "game_type": ["REG"],
                "home_team": ["DEN"],
                "away_team": ["KC"],
                "home_score": [None],
                "away_score": [None],
            }
        )
        assert points_allowed(sch).empty

    def test_postseason_is_excluded(self, schedules) -> None:
        sch = pd.concat(
            [
                schedules,
                schedules.assign(game_type="POST", week=19),
            ],
            ignore_index=True,
        )
        assert set(points_allowed(sch)["week"]) == {1, 2}

    def test_yards_allowed_rows_are_keyed_by_the_gaining_team(self, team_stats) -> None:
        """
        The frame is keyed on `opponent_team` so it joins onto whoever faced
        that offence — the row labelled DEN carries what DEN *gained*.
        """
        ya = yards_allowed(team_stats)
        wk1 = ya[ya.week == 1].set_index("opponent_team")["yards_allowed"]
        assert wk1["DEN"] == 300  # 200 passing + 100 rushing
        assert wk1["KC"] == 190  # 150 passing + 40 rushing

    def test_after_the_join_each_team_is_charged_its_opponents_yards(
        self, team_stats, schedules
    ) -> None:
        weekly = build_dst_weekly(team_stats, schedules, RULES)
        wk1 = weekly[weekly.week == 1].set_index("team")["yards_allowed"]
        assert wk1["KC"] == 300  # KC faced DEN's 300
        assert wk1["DEN"] == 190


class TestWeeklyScoring:
    def test_a_shutout_earns_the_top_points_allowed_band(
        self, team_stats, schedules
    ) -> None:
        weekly = build_dst_weekly(team_stats, schedules, RULES)
        kc_wk2 = weekly[(weekly.team == "KC") & (weekly.week == 2)].iloc[0]
        # KC won 31-0: 2 sacks (2) + def TD (6) + shutout (5) + DEN's 95 yards
        # allowed falling in the sub-100 band (5).
        assert kc_wk2["points_allowed"] == 0
        assert kc_wk2["yards_allowed"] == 95
        assert kc_wk2["dst_points"] == pytest.approx(2 + 6 + 5 + 5)

    def test_counting_stats_and_ladders_combine(self, team_stats, schedules) -> None:
        weekly = build_dst_weekly(team_stats, schedules, RULES)
        den_wk1 = weekly[(weekly.team == "DEN") & (weekly.week == 1)].iloc[0]
        # 3 sacks (3) + 1 INT (2) + allowed 3 pts (4) + allowed 190 yds (3).
        assert den_wk1["dst_points"] == pytest.approx(3 + 2 + 4 + 3)

    def test_points_allowed_bands_do_not_overlap(self) -> None:
        """Each score lands in exactly one band, so no game is paid twice."""
        df = pd.DataFrame({"points_allowed": list(range(0, 60)), "yards_allowed": 325})
        awarded = pd.DataFrame(
            {i: b.award(df) for i, b in enumerate(RULES.bonuses)}
        ).astype(bool)
        pa_bands = awarded.iloc[:, :7]
        assert pa_bands.sum(axis=1).max() <= 1

    def test_yards_allowed_bands_do_not_overlap(self) -> None:
        df = pd.DataFrame(
            {"yards_allowed": list(range(0, 700, 7)), "points_allowed": 21}
        )
        awarded = pd.DataFrame(
            {i: b.award(df) for i, b in enumerate(RULES.bonuses)}
        ).astype(bool)
        ya_bands = awarded.iloc[:, 7:]
        assert ya_bands.sum(axis=1).max() <= 1

    def test_a_mid_range_game_scores_no_ladder_points(self) -> None:
        """18-27 points and 300-349 yards allowed are both worth zero."""
        df = pd.DataFrame({"points_allowed": [21], "yards_allowed": [325]})
        assert RULES.score(df).iloc[0] == 0.0


class TestSeasonTotals:
    def test_totals_sum_the_weeks_and_count_games(self, team_stats, schedules) -> None:
        totals = dst_season_totals(team_stats, schedules, RULES)
        assert set(totals["team"]) == {"DEN", "KC"}
        assert totals["games"].tolist() == [2, 2]
        weekly = build_dst_weekly(team_stats, schedules, RULES)
        expected = weekly[weekly.team == "KC"]["dst_points"].sum()
        assert totals.set_index("team").loc["KC", "dst_points"] == expected


class TestShrinkage:
    def make_totals(self, slope: float, noise: float, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        teams = [f"T{i:02d}" for i in range(32)]
        rows = []
        prior = rng.normal(70, 35, size=32)
        for season in (2020, 2021, 2022):
            rows += [
                {"season": season, "team": t, "dst_points": v}
                for t, v in zip(teams, prior, strict=True)
            ]
            prior = slope * prior + 50 + rng.normal(0, noise, size=32)
        return pd.DataFrame(rows)

    def test_recovers_a_known_slope(self) -> None:
        totals = self.make_totals(slope=0.3, noise=1.0)
        slope, _, _, n = fit_shrinkage(totals)
        assert slope == pytest.approx(0.3, abs=0.05)
        assert n == 64  # two consecutive pairs x 32 teams

    def test_non_consecutive_seasons_are_not_paired(self) -> None:
        totals = self.make_totals(slope=0.3, noise=1.0)
        totals = totals[totals.season != 2021]
        with pytest.raises(ValueError, match="consecutive"):
            fit_shrinkage(totals)

    def test_needs_history(self) -> None:
        one = pd.DataFrame({"season": [2024], "team": ["DEN"], "dst_points": [100.0]})
        with pytest.raises(ValueError, match="consecutive"):
            fit_shrinkage(one)


class TestProjectDst:
    def test_returns_nothing_for_a_league_without_dst(
        self, team_stats, schedules
    ) -> None:
        """Ludopathy has no D/ST slot, so there is nothing to project."""
        out = project_dst(
            2023, league="ludopathy", team_stats=team_stats, schedules=schedules
        )
        assert out.empty

    def test_output_matches_the_board_schema(self) -> None:
        totals_source = _synthetic_league(seasons=(2021, 2022, 2023))
        out = project_dst(2023, league="hoh", **totals_source)
        assert not out.empty
        for col in (
            "player_id",
            "player_name",
            "team",
            "position",
            "proj_p10",
            "proj_p50",
            "proj_p90",
            "projected_season",
        ):
            assert col in out.columns
        assert set(out["position"]) == {"DST"}
        assert (out["projected_season"] == 2024).all()

    def test_quantiles_are_ordered(self) -> None:
        out = project_dst(2023, league="hoh", **_synthetic_league((2021, 2022, 2023)))
        assert (out["proj_p10"] <= out["proj_p50"]).all()
        assert (out["proj_p50"] <= out["proj_p90"]).all()

    def test_projections_are_shrunk_toward_the_mean(self) -> None:
        """
        The spread between the best and worst projected defence must be far
        narrower than the spread in the season that produced it — that is the
        whole point of shrinking a weak signal.
        """
        src = _synthetic_league((2021, 2022, 2023))
        out = project_dst(2023, league="hoh", **src)
        totals = dst_season_totals(src["team_stats"], src["schedules"], RULES)
        observed = totals[totals.season == 2023]["dst_points"]
        observed_spread = observed.max() - observed.min()
        projected_spread = out["proj_p50"].max() - out["proj_p50"].min()
        assert projected_spread < observed_spread

    def test_missing_parquet_returns_empty_rather_than_raising(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A board build must degrade to "no D/ST" rather than crash."""
        monkeypatch.setattr("nfl_predict.dst._load", lambda name: None)
        assert project_dst(2023, league="hoh").empty

    def test_empty_source_data_returns_empty_rather_than_raising(self) -> None:
        assert project_dst(
            2023, league="hoh", team_stats=pd.DataFrame(), schedules=pd.DataFrame()
        ).empty


def _synthetic_league(seasons: tuple[int, ...]) -> dict[str, pd.DataFrame]:
    """A small but structurally real league: 8 teams, 2 games each per season."""
    rng = np.random.default_rng(7)
    teams = ["DEN", "KC", "BUF", "MIA", "PHI", "DAL", "GB", "SF"]
    sched_rows, stat_rows = [], []
    for season in seasons:
        for week in (1, 2):
            shift = 0 if week == 1 else 1
            for i in range(0, len(teams), 2):
                home = teams[(i + shift) % len(teams)]
                away = teams[(i + 1 + shift) % len(teams)]
                hs, as_ = int(rng.integers(0, 40)), int(rng.integers(0, 40))
                sched_rows.append(
                    {
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "home_team": home,
                        "away_team": away,
                        "home_score": hs,
                        "away_score": as_,
                    }
                )
                for team, opp in ((home, away), (away, home)):
                    stat_rows.append(
                        {
                            "season": season,
                            "week": week,
                            "season_type": "REG",
                            "team": team,
                            "opponent_team": opp,
                            "passing_yards": int(rng.integers(80, 350)),
                            "rushing_yards": int(rng.integers(20, 180)),
                            "def_sacks": int(rng.integers(0, 6)),
                            "def_interceptions": int(rng.integers(0, 3)),
                            "def_fumbles": int(rng.integers(0, 2)),
                            "def_safeties": 0,
                            "def_tds": 0,
                            "special_teams_tds": 0,
                            "def_punt_blocks": 0,
                            "def_pat_blocks": 0,
                            "def_fg_blocks": 0,
                        }
                    )
    return {
        "team_stats": pd.DataFrame(stat_rows),
        "schedules": pd.DataFrame(sched_rows),
    }
