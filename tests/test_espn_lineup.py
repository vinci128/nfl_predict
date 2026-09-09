"""
Building a lineup from ESPN's own weekly projections.

In week 1 the season model has no in-season data at all — it cannot see an
offseason move, a depth-chart change, a same-day injury designation, or a
rookie it has never scored. ESPN's projection can, so it is what orders a
week-1 lineup.

Nothing here touches the network; the client is stubbed.
"""

from __future__ import annotations

import pytest

from nfl_predict import lineup


class _FakeClient:
    """Stands in for EspnFantasyClient, returning one team's roster."""

    def __init__(self, rows):
        self._rows = rows

    def fetch_rosters(self, week=None):
        return self._rows


def _player(name, position, slot, proj, injury="ACTIVE", mine=True):
    return {
        "player_name": name,
        "position": position,
        "nfl_team": "KC",
        "lineup_slot": slot,
        "injury_status": injury,
        "projected_points": proj,
        "is_mine": mine,
        "player_id": name.lower().replace(" ", "-"),
    }


def _install(monkeypatch, rows):
    from nfl_predict import espn_fantasy

    monkeypatch.setattr(
        espn_fantasy.EspnFantasyClient,
        "from_env",
        staticmethod(lambda league=None: _FakeClient(rows)),
    )


# ---------------------------------------------------------------------------
# The regression: slots come from the league being built, not the active one
# ---------------------------------------------------------------------------


def _hoh_roster():
    return [
        _player("QB1", "QB", "QB", 30.0),
        _player("RB1", "RB", "RB", 14.0),
        _player("RB2", "RB", "RB", 9.0),
        _player("WR1", "WR", "WR", 20.0),
        _player("WR2", "WR", "WR", 12.0),
        _player("TE1", "TE", "TE", 8.0),
        _player("Steelers D/ST", "DST", "DST", 7.5),
        _player("K1", "K", "K", 8.5),
        _player("WR3", "WR", "BE", 9.1),
    ]


def test_dst_is_started_in_a_dst_league(monkeypatch):
    """`_build_lineup` reads the *active* league for its slots.

    Unpinned, a Hell or Highwater lineup was built against Ludopathy's slots,
    which have no D/ST — so the defence was benched with nowhere to go.
    """
    _install(monkeypatch, _hoh_roster())
    starters, bench, out, total = lineup.espn_optimal(league="hoh")
    slots = {s["slot"] for s in starters}
    assert "DST" in slots
    assert "Steelers D/ST" in {s["player_name"] for s in starters}


def test_dst_league_is_unaffected_by_the_active_league(monkeypatch):
    """Pinning must hold even when another league is active."""
    from nfl_predict.leagues import reset_active_league, set_active_league

    _install(monkeypatch, _hoh_roster())
    token = set_active_league("ludopathy")
    try:
        starters, _, _, _ = lineup.espn_optimal(league="hoh")
    finally:
        reset_active_league(token)
    assert "Steelers D/ST" in {s["player_name"] for s in starters}


def test_idp_league_starts_its_defenders(monkeypatch):
    rows = [
        _player("QB1", "QB", "QB", 30.0),
        _player("RB1", "RB", "RB", 14.0),
        _player("RB2", "RB", "RB", 9.0),
        _player("WR1", "WR", "WR", 20.0),
        _player("WR2", "WR", "WR", 12.0),
        _player("TE1", "TE", "TE", 8.0),
        _player("LB1", "LB", "LB", 17.0),
        _player("LB2", "LB", "LB", 16.0),
        _player("DL1", "DE", "DL", 11.0),
        _player("DB1", "S", "DB", 14.0),
        _player("DB2", "CB", "DB", 12.0),
        _player("K1", "K", "K", 10.0),
    ]
    _install(monkeypatch, rows)
    starters, _, _, _ = lineup.espn_optimal(league="ludopathy")
    slots = {s["slot"] for s in starters}
    assert {"LB", "DL", "DB"} <= slots
    assert "DST" not in slots


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_ruled_out_players_never_start(monkeypatch):
    rows = _hoh_roster()
    rows[1] = _player("RB1", "RB", "RB", 0.0, injury="OUT")
    _install(monkeypatch, rows)
    starters, _, out, _ = lineup.espn_optimal(league="hoh")
    assert "RB1" not in {s["player_name"] for s in starters}
    assert "RB1" in {o["player_name"] for o in out}


def test_questionable_players_still_start(monkeypatch):
    """ESPN prices the risk into the projection; Q players mostly play."""
    rows = _hoh_roster()
    rows[3] = _player("WR1", "WR", "WR", 20.0, injury="QUESTIONABLE")
    _install(monkeypatch, rows)
    starters, _, _, _ = lineup.espn_optimal(league="hoh")
    assert "WR1" in {s["player_name"] for s in starters}


def test_a_player_without_a_projection_is_not_started(monkeypatch):
    rows = _hoh_roster()
    rows[5] = _player("TE1", "TE", "TE", None)
    _install(monkeypatch, rows)
    starters, _, out, _ = lineup.espn_optimal(league="hoh")
    assert "TE1" not in {s["player_name"] for s in starters}
    assert "TE1" in {o["player_name"] for o in out}


def test_the_better_projection_wins_the_flex(monkeypatch):
    _install(monkeypatch, _hoh_roster())
    starters, _, _, _ = lineup.espn_optimal(league="hoh")
    flex = [s for s in starters if s["slot"] == "FLEX"]
    assert flex and flex[0]["player_name"] == "WR3"  # 9.1 beats RB2's 9.0


def test_no_team_of_mine_is_an_error(monkeypatch):
    rows = [_player("QB1", "QB", "QB", 30.0, mine=False)]
    _install(monkeypatch, rows)
    with pytest.raises(RuntimeError, match="No team flagged as yours"):
        lineup.espn_optimal(league="hoh")


def test_total_counts_only_starters(monkeypatch):
    _install(monkeypatch, _hoh_roster())
    starters, _, _, total = lineup.espn_optimal(league="hoh")
    assert total == pytest.approx(
        sum(s["expected_ppr_points"] for s in starters), abs=0.05
    )
