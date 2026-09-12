"""
Waiver recommendations: what to claim, and who it costs.

A claim occupies a roster spot, so a suggestion is only useful if it names
the player it displaces. These tests pin that, and the guards that stop the
command proposing churn.

Nothing here touches the network; the client is stubbed.
"""

from __future__ import annotations

import pytest

from nfl_predict import waivers


class _FakeClient:
    def __init__(self, roster, free):
        self._roster, self._free = roster, free

    def fetch_rosters(self, week=None):
        return self._roster

    def fetch_free_agents(self, week, limit=200, min_owned=0.0):
        return self._free


def _rostered(name, position, proj, slot="BE"):
    return {
        "player_name": name,
        "position": position,
        "projected_points": proj,
        "lineup_slot": slot,
        "injury_status": "ACTIVE",
        "is_mine": True,
        "nfl_team": "KC",
    }


def _fa(name, position, proj, owned=50.0, injury="ACTIVE"):
    return {
        "player_name": name,
        "position": position,
        "projected_points": proj,
        "injury_status": injury,
        "percent_owned": owned,
        "nfl_team": "KC",
    }


def _install(monkeypatch, roster, free):
    from nfl_predict import espn_fantasy

    monkeypatch.setattr(
        espn_fantasy.EspnFantasyClient,
        "from_env",
        staticmethod(lambda league=None: _FakeClient(roster, free)),
    )


def test_an_upgrade_names_the_player_it_costs(monkeypatch):
    _install(
        monkeypatch,
        [_rostered("Weak QB", "QB", 10.0), _rostered("Good QB", "QB", 22.0, "QB")],
        [_fa("Better QB", "QB", 18.0)],
    )
    ups, _, _ = waivers.waiver_candidates(league="rumble", week=2)
    assert len(ups) == 1
    assert ups[0]["add"] == "Better QB"
    assert ups[0]["drop"] == "Weak QB"  # the worst, not the first
    assert ups[0]["gain"] == pytest.approx(8.0)


def test_a_marginal_gain_is_not_proposed(monkeypatch):
    """Projections are not accurate to tenths; a 0.5 swap is churn."""
    _install(
        monkeypatch,
        [_rostered("Mine", "RB", 12.0)],
        [_fa("Barely better", "RB", 12.5)],
    )
    ups, _, _ = waivers.waiver_candidates(league="rumble", week=2)
    assert ups == []


def test_a_position_the_league_cannot_start_is_ignored(monkeypatch):
    """Royal Rumble starts no IDP, so a linebacker is never an upgrade."""
    _install(
        monkeypatch,
        [_rostered("Mine", "RB", 4.0)],
        [_fa("Elite LB", "LB", 40.0)],
    )
    ups, _, pool = waivers.waiver_candidates(league="rumble", week=2)
    assert ups == []
    assert "Elite LB" not in {p["player_name"] for p in pool}


def test_idp_is_an_upgrade_in_the_idp_league(monkeypatch):
    """The same linebacker does count in Ludopathy."""
    _install(
        monkeypatch,
        [_rostered("Weak LB", "LB", 6.0)],
        [_fa("Elite LB", "LB", 20.0)],
    )
    ups, _, _ = waivers.waiver_candidates(league="ludopathy", week=2)
    assert ups and ups[0]["add"] == "Elite LB"


def test_one_player_is_not_dropped_twice(monkeypatch):
    """Two claims at one position must not both cost the same man."""
    _install(
        monkeypatch,
        [_rostered("Weak", "WR", 3.0), _rostered("OK", "WR", 8.0)],
        [_fa("Star", "WR", 18.0), _fa("Also good", "WR", 15.0)],
    )
    ups, _, _ = waivers.waiver_candidates(league="rumble", week=2)
    assert len({u["drop"] for u in ups}) == len(ups)


def test_an_ir_player_is_never_the_drop(monkeypatch):
    """An IR slot is not a roster spot you are competing for."""
    _install(
        monkeypatch,
        [_rostered("Injured", "RB", 0.0, "IR"), _rostered("Playing", "RB", 9.0)],
        [_fa("Upgrade", "RB", 14.0)],
    )
    ups, _, _ = waivers.waiver_candidates(league="rumble", week=2)
    assert ups and ups[0]["drop"] == "Playing"


def test_a_free_agent_without_a_projection_is_skipped(monkeypatch):
    _install(
        monkeypatch,
        [_rostered("Mine", "TE", 5.0)],
        [_fa("Unknown", "TE", None)],
    )
    ups, _, _ = waivers.waiver_candidates(league="rumble", week=2)
    assert ups == []


def test_no_team_of_mine_is_an_error(monkeypatch):
    _install(monkeypatch, [], [_fa("Anyone", "RB", 10.0)])
    with pytest.raises(RuntimeError, match="No team flagged as yours"):
        waivers.waiver_candidates(league="rumble", week=2)
