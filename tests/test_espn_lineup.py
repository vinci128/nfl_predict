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


_ESPN_IDS: dict[str, int] = {}


def _espn_id(name: str) -> int:
    """A stable, collision-free stand-in for ESPN's own player id."""
    if name not in _ESPN_IDS:
        _ESPN_IDS[name] = 1000 + len(_ESPN_IDS)
    return _ESPN_IDS[name]


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
        # ESPN's own id. The writer joins the optimiser's picks to the raw
        # roster on this, never on the name.
        "espn_player_id": _espn_id(name),
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


def test_a_player_in_an_ir_slot_is_never_started(monkeypatch):
    """IR is a roster state, not a deeper bench.

    Reaching a starting slot from IR takes an activation — a transaction, and
    one that needs a free roster spot. ESPN leaves a live designation on a
    player sitting there rather than overwriting it with INJURY_RESERVE, so
    the status does not rule him out and the slot has to.
    """
    rows = _hoh_roster()
    rows.append(_player("RB3", "RB", "IR", 25.0, injury="QUESTIONABLE"))
    _install(monkeypatch, rows)
    starters, _, out, _ = lineup.espn_optimal(league="hoh")
    assert "RB3" not in {s["player_name"] for s in starters}
    assert "RB3" in {o["player_name"] for o in out}


def test_an_ir_slot_player_does_not_inflate_the_best_total(monkeypatch):
    """The defect itself: a big projection in IR raised the lineup it proposed.

    Hell or Highwater's week 8 read 122.0 against a real best of 117.9,
    because Zach Charbonnet was counted from IR.
    """
    _install(monkeypatch, _hoh_roster())
    _, _, _, baseline = lineup.espn_optimal(league="hoh")
    rows = _hoh_roster()
    rows.append(_player("RB3", "RB", "IR", 25.0, injury="QUESTIONABLE"))
    _install(monkeypatch, rows)
    _, _, _, total = lineup.espn_optimal(league="hoh")
    assert total == pytest.approx(baseline, abs=0.05)


def test_an_ir_slot_player_is_never_proposed_as_a_start(monkeypatch, capsys):
    """Advice that needs a drop to follow is not a lineup change."""
    rows = _hoh_roster()
    rows[2] = _player("RB2", "RB", "RB", 6.0)
    rows.append(_player("RB3", "RB", "IR", 25.0, injury="QUESTIONABLE"))
    _install(monkeypatch, rows)
    changes = lineup.print_espn_lineup(league="hoh")
    capsys.readouterr()
    assert not any("RB3" in c for c in changes)


def test_an_ir_slot_player_is_not_counted_in_the_current_total(monkeypatch):
    _install(monkeypatch, _hoh_roster())
    starters, bench, out, _ = lineup.espn_optimal(league="hoh")
    before = lineup.current_total(starters, bench, out)
    rows = _hoh_roster()
    rows.append(_player("RB3", "RB", "IR", 25.0, injury="QUESTIONABLE"))
    _install(monkeypatch, rows)
    starters, bench, out, _ = lineup.espn_optimal(league="hoh")
    assert lineup.current_total(starters, bench, out) == pytest.approx(before, abs=0.05)


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


# ---------------------------------------------------------------------------
# Setting it on ESPN
#
# Everything below is about the one path that writes. The client is stubbed
# and the stub applies what it is sent, so the read-back verification is
# exercised rather than assumed.
# ---------------------------------------------------------------------------

_SLOT_ID = {
    "QB": 0,
    "RB": 2,
    "WR": 4,
    "TE": 6,
    "DST": 16,
    "K": 17,
    "FLEX": 23,
    "BE": 20,
    "IR": 21,
}

_ELIGIBLE = {
    "QB": [0, 7, 20, 21],
    "RB": [2, 3, 23, 7, 20, 21],
    "WR": [3, 4, 5, 23, 7, 20, 21],
    "TE": [5, 6, 23, 7, 20, 21],
    "DST": [16, 20, 21],
    "K": [17, 20, 21],
    "LB": [10, 15, 20, 21],
    "DL": [9, 11, 15, 20, 21],
    "DB": [13, 14, 15, 20, 21],
}


class _WritingClient:
    """A stub that serves both reads and the write, and remembers the write."""

    def __init__(self, rows, apply_on_submit=True):
        self.submitted = None
        self._apply = apply_on_submit
        self.rows = rows
        self._raw = [
            {
                "espn_player_id": r["espn_player_id"],
                "player_name": r["player_name"],
                "lineup_slot_id": _SLOT_ID[r["lineup_slot"]],
                "eligible_slots": list(_ELIGIBLE.get(r["position"], [20, 21])),
                "injury_status": r["injury_status"],
                "actual_points": r["actual_points"],
                "projected_points": r["projected_points"],
            }
            for r in rows
        ]

    def fetch_rosters(self, week=None):
        return self.rows

    def fetch_roster_slots(self, week=None):
        # Copies, as the real client builds fresh rows on every fetch. Serving
        # the live objects would let the writer's bookkeeping alias the plan.
        return {"team_id": "9", "week": 9, "entries": [dict(e) for e in self._raw]}

    def submit_lineup(self, items, week=None):
        self.submitted = {"items": items, "week": week}
        if self._apply:
            by_id = {e["espn_player_id"]: e for e in self._raw}
            for it in items:
                by_id[it["espn_player_id"]]["lineup_slot_id"] = it["to_slot_id"]
        return {"id": "txn-1", "type": "ROSTER", "status": "EXECUTED"}


def _install_writer(monkeypatch, rows, apply_on_submit=True):
    from nfl_predict import espn_fantasy

    client = _WritingClient(rows, apply_on_submit)
    monkeypatch.setattr(
        espn_fantasy.EspnFantasyClient,
        "from_env",
        staticmethod(lambda league=None: client),
    )
    return client


def _bench_upgrade():
    """TE1 starts at 3.0; TE2 sits on the bench at 11.0."""
    rows = _hoh_roster()
    rows[5] = _player("TE1", "TE", "TE", 3.0)
    rows.append(_player("TE2", "TE", "BE", 11.0))
    return rows


def test_a_bench_upgrade_is_paired_with_the_man_it_replaces(monkeypatch):
    _install_writer(monkeypatch, _bench_upgrade())
    plan = lineup.plan_lineup_moves(league="hoh")
    assert [(o["player_name"], i["player_name"]) for o, i in plan["pairs"]] == [
        ("TE1", "TE2")
    ]
    assert plan["gain"] == pytest.approx(8.0)


def test_a_surplus_starter_stays_where_he_is(monkeypatch):
    """Two starters should go, one player should come in.

    Benching both would leave the bench holding more players than it has
    slots — an illegal roster, and one ESPN rejects whole. So the better of
    the two stays: he is worth nothing less there than on the bench, since
    nothing replaces him either way.
    """
    rows = _hoh_roster()
    rows[1] = _player("RB1", "RB", "RB", 0.0, injury="OUT")
    rows[2] = _player("RB2", "RB", "RB", 4.0, injury="OUT")
    rows.append(_player("RB3", "RB", "BE", 9.0))
    _install_writer(monkeypatch, rows)
    plan = lineup.plan_lineup_moves(league="hoh")
    benched = [o["player_name"] for o, _ in plan["pairs"]]
    assert benched == ["RB1"]
    assert sum(1 for _o, i in plan["pairs"] if i["player_name"] == "RB3") == 1


def test_an_empty_starting_slot_is_left_empty(monkeypatch):
    """An unplaceable starter is named, not forced.

    Filling a slot nobody is leaving would mean naming a slot id outright —
    the guess that the swap form exists to avoid. Rare enough to prefer the
    safe failure, and it is visible in ``unplaced`` rather than silent.
    """
    rows = [r for r in _hoh_roster() if r["player_name"] != "WR3"]
    rows.append(_player("WR3", "WR", "BE", 25.0))
    _install_writer(monkeypatch, rows)
    plan = lineup.plan_lineup_moves(league="hoh")
    assert [e["player_name"] for e in plan["unplaced"]] == ["WR3"]
    assert plan["pairs"] == []


def test_an_unreachable_upgrade_is_not_called_optimal(monkeypatch):
    """ "Already the best lineup" is a claim about ESPN's roster, not ours."""
    rows = [r for r in _hoh_roster() if r["player_name"] != "WR3"]
    rows.append(_player("WR3", "WR", "BE", 25.0))
    _install_writer(monkeypatch, rows)
    result = lineup.apply_espn_lineup(league="hoh")
    assert result["status"] == "unreachable"
    assert "WR3 should start" in lineup.summarise_lineup_write(result)


def test_a_change_inside_the_noise_is_not_written(monkeypatch):
    rows = _hoh_roster()
    rows[5] = _player("TE1", "TE", "TE", 8.3606)
    rows[8] = _player("WR3", "WR", "FLEX", 9.1)
    rows.append(_player("TE2", "TE", "BE", 8.4297))
    client = _install_writer(monkeypatch, rows)
    result = lineup.apply_espn_lineup(league="hoh")
    assert result["status"] == "not_worth_it"
    assert client.submitted is None


def test_a_dry_run_writes_nothing(monkeypatch):
    client = _install_writer(monkeypatch, _bench_upgrade())
    result = lineup.apply_espn_lineup(league="hoh", dry_run=True)
    assert result["status"] == "dry_run"
    assert client.submitted is None


def test_the_write_is_one_transaction_and_is_verified(monkeypatch):
    """Both sides of the swap travel together, and the roster must agree."""
    client = _install_writer(monkeypatch, _bench_upgrade())
    result = lineup.apply_espn_lineup(league="hoh")

    assert result["status"] == "applied"
    assert result["wrong"] == []
    assert client.submitted is not None
    items = client.submitted["items"]
    assert len(items) == 2
    # The incoming player takes the exact slot the other is leaving; the
    # outgoing one goes to the bench.
    incoming, outgoing = items
    assert incoming["from_slot_id"] == 20
    assert outgoing["from_slot_id"] == incoming["to_slot_id"]
    assert outgoing["to_slot_id"] == 20


def test_a_write_espn_ignores_is_reported_as_unverified(monkeypatch):
    """A 200 means ESPN accepted a transaction, not that the lineup changed."""
    rows = _bench_upgrade()
    client = _install_writer(monkeypatch, rows, apply_on_submit=False)
    result = lineup.apply_espn_lineup(league="hoh")
    assert result["status"] == "unverified"
    assert "TE2" in result["wrong"][0]
    assert client.submitted is not None


def test_a_rejected_write_says_what_espn_said(monkeypatch):
    from nfl_predict.espn_fantasy import EspnFantasyError

    client = _install_writer(monkeypatch, _bench_upgrade())

    def _reject(items, week=None):
        raise EspnFantasyError("ESPN rejected the lineup change: Invalid Input.")

    monkeypatch.setattr(client, "submit_lineup", _reject)
    result = lineup.apply_espn_lineup(league="hoh")
    assert result["status"] == "rejected"
    assert "Invalid Input" in result["error"]


def test_a_league_espn_will_not_talk_about_is_not_called_optimal(monkeypatch):
    """A read that fails must not read as 'nothing to change'."""
    from nfl_predict import espn_fantasy

    def _boom(league=None):
        raise espn_fantasy.EspnFantasyError("ESPN returned HTTP 404 for league 1")

    monkeypatch.setattr(espn_fantasy.EspnFantasyClient, "from_env", staticmethod(_boom))
    result = lineup.apply_espn_lineup(league="hoh")
    assert result["status"] == "unavailable"
    assert "404" in result["error"]
    assert "could not read" in lineup.summarise_lineup_write(result)


def test_expected_points_is_zero_for_a_player_who_cannot_play():
    entry = {"injury_status": "OUT", "actual_points": None, "projected_points": 6.0}
    assert lineup._expected_points(entry) == 0.0


def test_expected_points_is_what_he_scored_once_he_has():
    entry = {"injury_status": "ACTIVE", "actual_points": 14.5, "projected_points": 9.0}
    assert lineup._expected_points(entry) == 14.5


def test_every_status_says_something_on_its_own(monkeypatch):
    """A silent no-op and a refused write must not look alike."""
    base = {
        "name": "Hell or Highwater",
        "week": 9,
        "pairs": [],
        "unplaced": [],
        "gain": 12.0,
        "error": "",
        "wrong": [],
    }
    for status in (
        "applied",
        "dry_run",
        "already_optimal",
        "not_worth_it",
        "rejected",
        "unverified",
        "unreachable",
        "unavailable",
    ):
        text = lineup.summarise_lineup_write({**base, "status": status})
        assert status not in text or status == "applied", text
        assert "Hell or Highwater" in text
        assert len(text) > 30


# ---------------------------------------------------------------------------
# The transaction ESPN actually receives
# ---------------------------------------------------------------------------


def _bare_client(**kwargs):
    from nfl_predict.espn_fantasy import EspnFantasyClient

    return EspnFantasyClient(league_id="581348581", season=2026, team_id="9", **kwargs)


def test_the_current_period_is_a_roster_transaction(monkeypatch):
    """Not "LINEUP" — that is the *item* type, and ESPN answers 400 with it.

    The type names come from ESPN's own bundle, and nothing about a 400 says
    which field was wrong; the first attempt at this sent LINEUP and read the
    rejection as a malformed body for an hour.
    """
    client = _bare_client()
    sent = {}

    monkeypatch.setattr(
        client, "get_league", lambda: {"status": {"latestScoringPeriod": 9}}
    )
    monkeypatch.setattr(client, "_post", lambda path, body: sent.update(body))
    client.submit_lineup(
        [{"espn_player_id": 123, "from_slot_id": 20, "to_slot_id": 6}], week=9
    )
    assert sent["type"] == "ROSTER"
    assert sent["scoringPeriodId"] == 9
    assert sent["executionType"] == "EXECUTE"
    assert sent["teamId"] == 9
    assert sent["items"][0] == {
        "playerId": 123,
        "type": "LINEUP",
        "fromLineupSlotId": 20,
        "toLineupSlotId": 6,
        "fromTeamId": 0,
        "toTeamId": 0,
        "isKeeper": False,
        "overallPickNumber": 0,
    }


def test_a_later_period_is_a_future_roster_transaction(monkeypatch):
    """Setting next week's lineup is legal — it just uses the other type."""
    client = _bare_client()
    sent = {}
    monkeypatch.setattr(
        client, "get_league", lambda: {"status": {"latestScoringPeriod": 8}}
    )
    monkeypatch.setattr(client, "_post", lambda path, body: sent.update(body))
    client.submit_lineup(
        [{"espn_player_id": 1, "from_slot_id": 20, "to_slot_id": 4}], week=10
    )
    assert sent["type"] == "FUTURE_ROSTER"
    assert sent["scoringPeriodId"] == 10


def test_an_earlier_period_is_booked_against_the_current_one(monkeypatch):
    """ESPN ignores the week you ask for; mirror that rather than fight it."""
    client = _bare_client()
    sent = {}
    monkeypatch.setattr(
        client, "get_league", lambda: {"status": {"latestScoringPeriod": 9}}
    )
    monkeypatch.setattr(client, "_post", lambda path, body: sent.update(body))
    client.submit_lineup(
        [{"espn_player_id": 1, "from_slot_id": 20, "to_slot_id": 4}], week=7
    )
    assert sent["type"] == "ROSTER"
    assert sent["scoringPeriodId"] == 9


def test_the_transaction_goes_to_the_write_host(monkeypatch):
    """The read host answers the same path with 405, so this is not cosmetic."""
    from nfl_predict import espn_fantasy

    client = _bare_client(espn_s2="s2", swid="{SWID}")
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"id": "txn"}'

    def _urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.headers)
        return _Response()

    monkeypatch.setattr(espn_fantasy.urllib.request, "urlopen", _urlopen)
    client._post("/transactions/", {"type": "ROSTER"})
    assert seen["url"].startswith("https://lm-api-writes.fantasy.espn.com/")
    assert seen["url"].endswith(
        "/seasons/2026/segments/0/leagues/581348581/transactions/"
    )
    assert seen["headers"]["X-fantasy-platform"] == "espn-fantasy-web"
    assert seen["headers"]["X-fantasy-source"] == "kona"
    assert "SWID" in seen["headers"]["Cookie"]
