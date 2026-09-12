"""
Lineup choice in an elimination league, where only last place matters.

Royal Rumble puts all eight teams in one matchup and eliminates the lowest
scorer, so points above survival buy nothing. These tests pin the behaviour
that an expected-points objective cannot produce.

Nothing here touches the network or reads a parquet.
"""

from __future__ import annotations

import pytest

from nfl_predict import survival

MODEL = {"QB": (7.6, 0.03), "RB": (4.1, 0.29), "WR": (3.7, 0.33), "TE": (2.5, 0.43)}


def _team(settled=0.0, players=()):
    return {
        "settled": settled,
        "means": [m for m, _ in players],
        "sds": [s for _, s in players],
    }


# ---------------------------------------------------------------------------
# Spread model
# ---------------------------------------------------------------------------


def test_sd_grows_with_the_projection_where_the_fit_says_so():
    lo = survival.player_sd("WR", 5.0, MODEL)
    hi = survival.player_sd("WR", 20.0, MODEL)
    assert hi > lo


def test_qb_spread_is_essentially_flat_in_the_fit():
    """Measured slope is ~0.03: a big QB projection is not more volatile."""
    a, b = MODEL["QB"]
    assert a + b * 30 < 2 * (a + b * 5)


def test_sd_never_collapses_at_a_high_mean():
    """The floor stops a flat-slope position looking risk-free when large."""
    assert survival.player_sd("QB", 40.0, MODEL) >= 10.0


def test_unknown_position_falls_back_rather_than_raising():
    """D/ST has no weekly rows in any feature table."""
    assert survival.player_sd("DST", 8.0, {}) > 0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def test_probabilities_sum_to_one():
    teams = {
        "a": _team(0, [(10, 3), (10, 3)]),
        "b": _team(0, [(12, 4)]),
        "c": _team(5, [(9, 2)]),
    }
    p = survival.simulate_last_place(teams, sims=4000)
    assert sum(p.values()) == pytest.approx(1.0, abs=1e-6)


def test_the_weakest_team_is_likeliest_last():
    # Close enough that all three can plausibly finish last, or the two
    # better teams both sit at exactly zero and the ordering is untestable.
    teams = {
        "strong": _team(60, [(20, 6)]),
        "middle": _team(58, [(18, 6)]),
        "weak": _team(54, [(16, 6)]),
    }
    p = survival.simulate_last_place(teams, sims=20000)
    assert p["weak"] > p["middle"] > p["strong"]
    assert all(v > 0 for v in p.values())


def test_a_settled_lead_cannot_be_lost():
    """Nobody can catch a team already above every rival's ceiling."""
    teams = {
        "safe": _team(500, []),
        "other": _team(0, [(10, 2)]),
    }
    p = survival.simulate_last_place(teams, sims=2000)
    assert p["safe"] == 0.0


# ---------------------------------------------------------------------------
# The behaviour an expected-points objective cannot produce
# ---------------------------------------------------------------------------


def test_when_clear_of_the_cut_less_variance_is_better():
    """Ahead, the safe lineup wins even though it scores fewer points."""
    rivals = {"r1": _team(0, [(100, 8)]), "r2": _team(0, [(95, 8)])}

    risky = dict(rivals, me=_team(0, [(112, 40)]))  # higher mean, wild
    safe = dict(rivals, me=_team(0, [(108, 4)]))  # lower mean, steady

    p_risky = survival.simulate_last_place(risky, sims=20000)["me"]
    p_safe = survival.simulate_last_place(safe, sims=20000)["me"]
    assert p_safe < p_risky


def test_when_behind_more_variance_is_better():
    """Trailing badly, only the wild lineup can still get there."""
    rivals = {"r1": _team(0, [(100, 6)]), "r2": _team(0, [(98, 6)])}

    risky = dict(rivals, me=_team(0, [(70, 45)]))
    safe = dict(rivals, me=_team(0, [(72, 3)]))

    p_risky = survival.simulate_last_place(risky, sims=20000)["me"]
    p_safe = survival.simulate_last_place(safe, sims=20000)["me"]
    assert p_risky < p_safe


def test_the_simulation_is_reproducible():
    teams = {"a": _team(0, [(10, 5)]), "b": _team(0, [(11, 5)])}
    first = survival.simulate_last_place(teams, sims=3000, seed=7)
    again = survival.simulate_last_place(teams, sims=3000, seed=7)
    assert first == again
