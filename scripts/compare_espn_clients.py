"""
Compare our hand-rolled ESPN client against the `espn-api` library.

Both read the same undocumented ESPN v3 endpoint, so where they disagree one
of them is wrong. This runs them side by side and prints what differs.

    uv run python scripts/compare_espn_clients.py --league hoh
    uv run python scripts/compare_espn_clients.py --league hoh --week 3

Sections that have no data yet (rosters before a draft, box scores before
kickoff) are reported as skipped rather than as agreement — a comparison of
nothing is not a passing comparison.

A private league needs ESPN_S2 / ESPN_SWID in the environment, exactly as the
draft sync does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_predict.espn_fantasy import EspnFantasyClient, EspnFantasyError  # noqa: E402
from nfl_predict.leagues import get_profile  # noqa: E402

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Differences we chose on purpose: espn-api echoes ESPN's own labels, while we
# normalise onto the board's vocabulary (see leagues.fantasy_position). These
# are folded together before comparing so they do not drown out real
# disagreements — which are reported separately below.
_CANON = {
    "D/ST": "DST",
    "RB/WR/TE": "FLEX",
    "DT": "DL",
    "DE": "DL",
    "CB": "DB",
    "S": "DB",
    "ER": "EDR",
    # Two spellings of one franchise. ESPN writes LAR and WSH; nflverse -- and
    # so every table in this repo -- writes LA and WAS.
    "LAR": "LA",
    "WSH": "WAS",
}


def canon(label: Any) -> str:
    label = str(label or "").upper()
    return _CANON.get(label, label)


class Report:
    """Tallies agreements and divergences for one comparison section."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.checked = 0
        self.agreed = 0
        self.diffs: list[str] = []
        self.vocab: set[tuple[str, str]] = set()
        self.skipped: str | None = None

    def compare(self, field: str, subject: str, ours: Any, theirs: Any) -> None:
        """Compare one field, folding the deliberate label differences away."""
        self.checked += 1
        if canon(ours) == canon(theirs):
            self.agreed += 1
            if str(ours or "") != str(theirs or ""):
                self.vocab.add((str(theirs), str(ours)))
            return
        self.diffs.append(f"{subject}: {field} ours={ours!r} espn-api={theirs!r}")

    def note(self, message: str) -> None:
        self.diffs.append(message)
        self.checked += 1

    def skip(self, why: str) -> None:
        self.skipped = why

    def render(self) -> None:
        print(f"\n{'=' * 74}\n{self.name}\n{'=' * 74}")
        if self.skipped:
            print(f"  SKIPPED - {self.skipped}")
            return
        if not self.checked:
            print("  SKIPPED - nothing to compare")
            return

        print(f"  {self.agreed}/{self.checked} field comparisons agree")

        if self.vocab:
            print("\n  Deliberate label differences (counted as agreement):")
            for theirs, ours in sorted(self.vocab):
                print(f"    espn-api {theirs!r:<12} -> ours {ours!r}")

        if self.diffs:
            print(f"\n  {len(self.diffs)} divergence(s):")
            for line in self.diffs[:40]:
                print(f"    - {line}")
            if len(self.diffs) > 40:
                print(f"    ... and {len(self.diffs) - 40} more")
        else:
            print("\n  No divergences.")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def compare_league(ours: EspnFantasyClient, theirs: Any) -> Report:
    """Team count and team names."""
    r = Report("League metadata (?view=mTeam&mSettings)")

    payload = ours.get_league()
    our_teams = {
        str(t.get("id")): (t.get("name") or "").strip() for t in payload["teams"]
    }
    their_teams = {str(t.team_id): t.team_name.strip() for t in theirs.teams}

    r.compare("team count", "league", len(our_teams), len(their_teams))
    for team_id in sorted(set(our_teams) | set(their_teams)):
        r.compare(
            "team name",
            f"team {team_id}",
            our_teams.get(team_id),
            their_teams.get(team_id),
        )
    return r


def compare_player_identity(ours: EspnFantasyClient, season: int, limit: int) -> Report:
    """
    Name / position / pro team, derived from identical real player payloads.

    This is the one section that always has data: the player pool is populated
    long before any roster is. Both sides parse the same ESPN objects, so any
    divergence is purely a parsing difference.
    """
    from espn_api.football import Player

    r = Report(f"Player identity ({limit} real players from ?view=kona_player_info)")

    try:
        payload = ours._get(
            ["kona_player_info"],
            extra_headers={
                "x-fantasy-filter": json.dumps(
                    {
                        "players": {
                            "limit": limit,
                            "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
                        }
                    }
                )
            },
        )
    except EspnFantasyError as e:
        r.skip(f"could not read the player pool: {e}")
        return r

    entries = payload.get("players") or []
    if not entries:
        r.skip("ESPN returned an empty player pool")
        return r

    crosswalk = ours._players()

    for entry in entries:
        player = entry.get("player") or {}
        espn_id = player.get("id")

        # Wrap the real player object in a roster entry so both parsers see
        # exactly the same bytes.
        roster_entry = {"playerId": espn_id, "playerPoolEntry": {"player": player}}

        mine = ours._roster_entry(roster_entry, crosswalk)
        try:
            theirs = Player(roster_entry, season)
        except Exception as e:  # noqa: BLE001 - a parse failure is a finding
            r.note(f"player {espn_id}: espn-api raised {type(e).__name__}: {e}")
            continue

        subject = theirs.name or f"player {espn_id}"

        # We prefer the ffverse crosswalk for identity and fall back to ESPN;
        # espn-api only ever reads ESPN. Compare the fallback path directly
        # where the crosswalk resolved, or the names will differ merely
        # because one is "J.Allen" and the other "Josh Allen".
        matched = bool(mine["player_id"])
        if not matched:
            r.compare("name", subject, mine["player_name"], theirs.name)

        r.compare("position", subject, mine["position"], theirs.position)
        r.compare("injury status", subject, mine["injury_status"], theirs.injuryStatus)

        if matched:
            r.compare("pro team", subject, mine["nfl_team"], theirs.proTeam)

    return r


def compare_rosters(ours: EspnFantasyClient, theirs: Any, week: int | None) -> Report:
    """Every team's roster, player by player."""
    r = Report("Rosters (?view=mRoster)")

    our_rows = ours.fetch_rosters(week=week)
    if not our_rows:
        r.skip("no rosters yet - the draft has not happened")
        return r

    if week:
        theirs.load_roster_week(week)

    their_rows = {
        (str(team.team_id), p.playerId): p for team in theirs.teams for p in team.roster
    }
    our_index = {(row["team_id"], row["espn_player_id"]): row for row in our_rows}

    only_ours = set(our_index) - set(their_rows)
    only_theirs = set(their_rows) - set(our_index)
    for key in sorted(only_ours):
        r.note(f"{key}: on our roster, absent from espn-api's")
    for key in sorted(only_theirs):
        r.note(f"{key}: on espn-api's roster, absent from ours")

    for key in sorted(set(our_index) & set(their_rows), key=str):
        mine, theirs_p = our_index[key], their_rows[key]
        subject = theirs_p.name or str(key)
        r.compare("position", subject, mine["position"], theirs_p.position)
        r.compare("lineup slot", subject, mine["lineup_slot"], theirs_p.lineupSlot)
        r.compare(
            "acquisition", subject, mine["acquisition_type"], theirs_p.acquisitionType
        )
        if mine["player_id"]:
            r.compare("pro team", subject, mine["nfl_team"], theirs_p.proTeam)

    return r


def compare_boxscore(ours: EspnFantasyClient, theirs: Any, week: int) -> Report:
    """One week's actual and projected points."""
    r = Report(f"Box score, week {week} (?view=mBoxscore)")

    our_rows = ours.fetch_boxscore(week=week)
    try:
        their_boxes = theirs.box_scores(week=week)
    except Exception as e:  # noqa: BLE001 - a raise here is itself the finding
        if not our_rows:
            r.skip(f"neither client has data ({type(e).__name__}: {e})")
        else:
            r.note(
                f"espn-api raised {type(e).__name__}: {e}, ours returned {len(our_rows)} rows"
            )
        return r

    their_rows: dict[tuple[str, int], Any] = {}
    for box in their_boxes:
        for team, lineup in (
            (box.home_team, box.home_lineup),
            (box.away_team, box.away_lineup),
        ):
            if team is None:
                continue
            team_id = str(getattr(team, "team_id", team))
            for p in lineup:
                their_rows[(team_id, p.playerId)] = p

    if not our_rows and not their_rows:
        r.skip("no box score data yet - the week has not been played")
        return r

    our_index = {(row["team_id"], row["espn_player_id"]): row for row in our_rows}

    for key in sorted(set(our_index) - set(their_rows), key=str):
        r.note(f"{key}: in our box score, absent from espn-api's")
    for key in sorted(set(their_rows) - set(our_index), key=str):
        r.note(f"{key}: in espn-api's box score, absent from ours")

    for key in sorted(set(our_index) & set(their_rows), key=str):
        mine, theirs_p = our_index[key], their_rows[key]
        subject = theirs_p.name or str(key)
        r.compare("lineup slot", subject, mine["lineup_slot"], theirs_p.slot_position)
        # espn-api reports 0 where ESPN sent nothing; we report None. Only the
        # numbers are compared, so that difference is not treated as a clash.
        for field, mine_v, theirs_v in (
            ("actual points", mine["actual_points"], theirs_p.points),
            ("projected points", mine["projected_points"], theirs_p.projected_points),
        ):
            if mine_v is None and not theirs_v:
                continue
            r.compare(field, subject, round(mine_v or 0, 2), round(theirs_v or 0, 2))

    return r


def compare_draft(ours: EspnFantasyClient, theirs: Any) -> Report:
    """The draft pick log."""
    r = Report("Draft picks (?view=mDraftDetail)")

    our_picks = ours.fetch_all_picks()
    their_picks = list(getattr(theirs, "draft", []) or [])

    if not our_picks and not their_picks:
        r.skip("no draft picks recorded")
        return r

    r.compare("pick count", "draft", len(our_picks), len(their_picks))

    if not our_picks and their_picks:
        phantom = sum(
            1 for p in their_picks if int(getattr(p, "playerId", 0) or 0) <= 0
        )
        r.note(
            f"espn-api returned {len(their_picks)} picks of which {phantom} have "
            "playerId <= 0: ESPN pre-seeds the whole slate before the draft, and "
            "we filter those out"
        )
        return r

    for mine, theirs_p in zip(our_picks, their_picks, strict=False):
        subject = f"pick {mine['overall_pick']}"
        r.compare("player", subject, mine["player_name"], theirs_p.playerName)
        r.compare("round", subject, mine["round"], theirs_p.round_num)
        r.compare("pick in round", subject, mine["pick_in_round"], theirs_p.round_pick)

    return r


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="hoh", help="league profile key")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None, help="week to box score")
    parser.add_argument("--players", type=int, default=250, help="player pool sample")
    args = parser.parse_args()

    from espn_api.football import League

    profile = get_profile(args.league)
    season = args.season or profile.season

    ours = EspnFantasyClient.from_env(args.league)
    ours.season = season

    print(f"League : {profile.name} ({profile.espn_league_id})")
    print(f"Season : {season}")
    print(f"Auth   : {'cookies' if ours.espn_s2 else 'public (no cookies)'}")

    try:
        theirs = League(
            league_id=int(ours.league_id),
            year=season,
            espn_s2=ours.espn_s2,
            swid=ours.swid,
        )
    except Exception as e:  # noqa: BLE001
        print(f"\nespn-api could not open the league: {type(e).__name__}: {e}")
        return 1

    reports = [
        compare_league(ours, theirs),
        compare_player_identity(ours, season, args.players),
        compare_rosters(ours, theirs, args.week),
        compare_draft(ours, theirs),
    ]
    if args.week:
        reports.append(compare_boxscore(ours, theirs, args.week))

    for report in reports:
        report.render()

    total = sum(len(r.diffs) for r in reports)
    print(f"\n{'=' * 74}")
    print(f"{total} divergence(s) across {len(reports)} section(s)")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
