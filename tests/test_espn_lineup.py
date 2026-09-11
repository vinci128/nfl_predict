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


def _player(name, position, slot, proj, injury="ACTIVE", mine=True, actual=None):
    return {
        "player_name": name,
        "position": position,
        "nfl_team": "KC",
        "lineup_slot": slot,
        "injury_status": injury,
        "projected_points": proj,
        "actual_points": actual,
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


# ---------------------------------------------------------------------------
# A change has to be worth making
# ---------------------------------------------------------------------------


def test_current_total_counts_a_ruled_out_starter_as_zero(monkeypatch):
    """An unnoticed OUT in a starting slot costs the whole slot."""
    rows = _hoh_roster()
    rows[1] = _player("RB1", "RB", "RB", 0.0, injury="OUT")
    _install(monkeypatch, rows)
    starters, bench, out, _ = lineup.espn_optimal(league="hoh")
    assert "RB1" not in {p["player_name"] for p in starters}
    # RB1 still occupies the RB slot on ESPN, and contributes nothing.
    assert lineup.current_total(starters, bench, out) == pytest.approx(
        sum(
            p["expected_ppr_points"]
            for p in starters + bench + out
            if p["current_slot"] not in ("BE", "IR")
            and p["injury_status"] not in lineup._CANNOT_PLAY
        ),
        abs=0.05,
    )


def test_a_sub_noise_swap_is_not_reported_as_needed(monkeypatch, capsys):
    """0.07 points is not a reason to bench the higher-ceiling player."""
    rows = _hoh_roster()
    rows[5] = _player("TE1", "TE", "TE", 8.3606, injury="QUESTIONABLE")
    rows[8] = _player("WR3", "WR", "FLEX", 9.1)  # otherwise FLEX is a real gain
    rows.append(_player("TE2", "TE", "BE", 8.4297))
    _install(monkeypatch, rows)
    changes = lineup.print_espn_lineup(league="hoh")
    assert changes == []
    assert "No change worth making" in capsys.readouterr().out


def test_a_real_gain_is_reported(monkeypatch, capsys):
    rows = _hoh_roster()
    rows[5] = _player("TE1", "TE", "TE", 3.0)
    rows.append(_player("TE2", "TE", "BE", 11.0))
    _install(monkeypatch, rows)
    changes = lineup.print_espn_lineup(league="hoh")
    assert changes
    assert "Changes needed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# A game that has kicked off is settled
# ---------------------------------------------------------------------------


def test_a_played_starter_counts_at_what_he_scored(monkeypatch):
    """Not at what he was projected to score — the week has happened."""
    rows = _hoh_roster()
    rows[0] = _player("QB1", "QB", "QB", 30.0, actual=11.4)
    _install(monkeypatch, rows)
    starters, _, _, total = lineup.espn_optimal(league="hoh")
    qb = next(s for s in starters if s["player_name"] == "QB1")
    assert qb["expected_ppr_points"] == 11.4
    assert qb["locked"] is True
    assert total == pytest.approx(
        sum(s["expected_ppr_points"] for s in starters), abs=0.05
    )


def test_a_played_starter_is_never_proposed_for_benching(monkeypatch, capsys):
    """His slot cannot be changed, so suggesting it wastes a trip to ESPN."""
    rows = _hoh_roster()
    rows[5] = _player("TE1", "TE", "TE", 8.4, actual=3.2)
    rows.append(_player("TE2", "TE", "BE", 9.9))
    _install(monkeypatch, rows)
    changes = lineup.print_espn_lineup(league="hoh")
    capsys.readouterr()
    # TE1 is settled: never proposed for benching, and still in the TE slot.
    assert not any("TE1" in c for c in changes)
    starters, _, _, _ = lineup.espn_optimal(league="hoh")
    te = next(s for s in starters if s["slot"] == "TE")
    assert te["player_name"] == "TE1"


def test_a_locked_slot_is_not_filled_twice(monkeypatch):
    """The locked starter spends his slot; nobody else may take it."""
    rows = _hoh_roster()
    rows[5] = _player("TE1", "TE", "TE", 8.4, actual=3.2)
    rows.append(_player("TE2", "TE", "BE", 20.0))
    _install(monkeypatch, rows)
    starters, bench, _, _ = lineup.espn_optimal(league="hoh")
    te_slots = [s for s in starters if s["slot"] == "TE"]
    assert len(te_slots) == 1
    assert te_slots[0]["player_name"] == "TE1"
    # TE2 may still take the open FLEX — a TE is flex-eligible here — but it
    # must never be handed the TE slot a played player is sitting in.
    te2 = next(p for p in starters + bench if p["player_name"] == "TE2")
    assert te2["slot"] != "TE"


def test_a_played_bench_player_stays_benched(monkeypatch):
    rows = _hoh_roster()
    rows[8] = _player("WR3", "WR", "BE", 9.1, actual=25.0)
    _install(monkeypatch, rows)
    starters, bench, _, _ = lineup.espn_optimal(league="hoh")
    assert "WR3" in {b["player_name"] for b in bench}
    assert "WR3" not in {s["player_name"] for s in starters}


def test_an_unplayed_slot_still_optimises_around_locked_ones(monkeypatch):
    """Locking the TE must not stop the open FLEX being filled."""
    rows = _hoh_roster()
    rows[5] = _player("TE1", "TE", "TE", 8.4, actual=3.2)
    _install(monkeypatch, rows)
    starters, _, _, _ = lineup.espn_optimal(league="hoh")
    flex = [s for s in starters if s["slot"] == "FLEX"]
    assert flex and flex[0]["player_name"] == "WR3"
