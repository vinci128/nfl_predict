"""
ESPN Fantasy API connector.

Polls the live draft picks endpoint so the draft UI can auto-record picks
without manual typing. This is the ESPN counterpart to ``nfl_fantasy`` and
exposes the same interface, so ``nfl-sync`` and the draft UI can use either.

Why this exists
---------------
NFL.com Fantasy is being folded into the ESPN app, which breaks the NFL.com
OAuth2 flow in ``nfl_fantasy``. Nothing about the *modelling* pipeline is
affected — weekly stats, rosters, injuries and schedules all come from
nflverse via ``nflreadpy`` and have no connection to a fantasy provider.
Only live pick sync changes.

Authentication
--------------
ESPN's fantasy API is undocumented but stable and read-only here.

  - **Public leagues** need no credentials at all.
  - **Private leagues** need the ``espn_s2`` and ``SWID`` cookies from a
    logged-in browser session (DevTools -> Application -> Cookies on
    fantasy.espn.com). There is no password grant; these cookies *are* the
    credential, and they expire, so expect to refresh them periodically.

The host moved from ``fantasy.espn.com`` to ``lm-api-reads.fantasy.espn.com``
in 2024; the old host now returns 403 for many leagues.

Player identity
---------------
ESPN returns its own numeric ``playerId``, which matches nothing on our draft
board. We map it to ``gsis_id`` through the ffverse crosswalk
(``nflreadpy.load_ff_playerids``) so picks are recorded by ID rather than by
name — this matters because abbreviated names collide (Bijan Robinson and
Brian Robinson are both ``B.Robinson``).

The crosswalk names teams in MFL's style (``SFO``, ``GBP``, ``JAC``) while
everything else here uses nflverse's (``SF``, ``GB``, ``JAX``), so team
abbreviations are normalised on the way out.

Configuration (environment variables)
--------------------------------------
    ESPN_LEAGUE_ID    numeric league ID from your league URL (required)
    ESPN_SEASON       season year (default: current year)
    ESPN_S2           espn_s2 cookie      (private leagues only)
    ESPN_SWID         SWID cookie         (private leagues only)
    ESPN_TEAM_ID      your team ID in the league (optional)
    ESPN_LEAGUE_SIZE  team count, used only as a round-math fallback

What this reads
---------------
The same league endpoint serves everything; only the ``view`` parameter
changes.

    ``mDraftDetail``   draft pick log      -> ``fetch_all_picks``
    ``mRoster``        current rosters     -> ``fetch_rosters``
    ``mBoxscore``      weekly scoring      -> ``fetch_boxscore``

Usage
-----
    from nfl_predict.espn_fantasy import EspnFantasyClient

    client = EspnFantasyClient.from_env()
    for pick in client.fetch_new_picks(already_recorded=5):
        print(pick["overall_pick"], pick["player_name"], pick["is_mine"])

    for row in client.fetch_rosters():
        print(row["team_name"], row["lineup_slot"], row["player_name"])

    for row in client.fetch_boxscore(week=1):
        print(row["player_name"], row["actual_points"], row["projected_points"])

CLI
---
    nfl-predict nfl-sync --provider espn --interval 30
"""

from __future__ import annotations

import datetime as _dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The pre-2024 host (fantasy.espn.com) now 403s for many leagues.
_API_BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl"

# ESPN sends a browser User-Agent check on some edges; a plain urllib agent
# is the most common cause of an unexplained 403.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

# ESPN's defensive/kicker slots use different labels from our board.
_POSITION_MAP = {
    "PK": "K",
    "K": "K",
    "DST": "DST",
    "D/ST": "DST",
    "DEF": "DST",
}

# ESPN identifies roster slots by a numeric ``lineupSlotId``. The labels here
# are *our* vocabulary, not ESPN's: its separate DT (8) and DE (9) slots both
# collapse to DL and CB/S to DB, matching ``leagues.fantasy_position``.
_LINEUP_SLOTS = {
    0: "QB",
    1: "TQB",
    2: "RB",
    3: "RB/WR",
    4: "WR",
    5: "WR/TE",
    6: "TE",
    7: "OP",
    8: "DL",
    9: "DL",
    10: "LB",
    11: "DL",
    12: "DB",
    13: "DB",
    14: "DB",
    15: "DP",
    16: "DST",
    17: "K",
    18: "P",
    19: "HC",
    20: "BE",
    21: "IR",
    23: "FLEX",
    24: "EDR",
}

# Slots that are not a starting lineup spot. Everything else counts as a
# start, including league-specific ones we do not use.
_BENCH_SLOTS = frozenset({20, 21})

# Slots naming exactly one position, so they can identify a player whom the
# crosswalk misses. The combination slots (RB/WR, FLEX, OP) and the roster
# housekeeping ones (BE, IR, HC) say nothing about what a player *is*.
_UNAMBIGUOUS_SLOTS = frozenset({0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 24})

# ESPN's stats[] entries carry both the real result and its own projection,
# distinguished only by this field.
_STAT_ACTUAL = 0
_STAT_PROJECTED = 1

# The ffverse crosswalk names teams in MFL's style; every other table in this
# repo (weekly_stats, the feature tables, the board) uses nflverse's. Left
# unmapped, an ESPN-sourced ``nfl_team`` silently matches nothing downstream.
# Codes absent here are already identical in both vocabularies.
_TEAM_MAP = {
    "GBP": "GB",
    "JAC": "JAX",
    "KCC": "KC",
    "LVR": "LV",
    "NEP": "NE",
    "NOS": "NO",
    "SFO": "SF",
    "TBB": "TB",
    # nflverse writes the Rams "LA" throughout, and the crosswalk keeps a
    # player's franchise history in this column, so the relocated franchises
    # resolve to whoever holds them now.
    "LAR": "LA",
    "RAM": "LA",
    "STL": "LA",
    "OAK": "LV",
    "SDC": "LAC",
    # Not a team. The crosswalk uses both spellings for a free agent.
    "FA": "",
    "FA*": "",
}


class EspnFantasyError(Exception):
    """Raised when ESPN Fantasy API interaction fails."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class EspnFantasyClient:
    """
    Thin read-only wrapper around ESPN's private fantasy API.

    Endpoints and payload shapes are undocumented and may change without
    notice. Everything here is a GET; nothing writes to ESPN.
    """

    league_id: str
    season: int
    espn_s2: str | None = None
    swid: str | None = None
    team_id: str | None = None
    # Only used to derive round numbers when ESPN omits them. Overridden by
    # the real team count as soon as a league payload is fetched.
    league_size: int = 12
    _player_map: dict[int, dict[str, str]] | None = field(
        default=None, repr=False, compare=False
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, league: str | None = None) -> EspnFantasyClient:
        """
        Create a client from a league profile, with environment overrides.

        The profile already knows each league's ESPN id, team id and size, so
        only the private-league cookies normally need to be in the environment.
        Every value can still be overridden:

        ESPN_LEAGUE_ID, ESPN_SEASON, ESPN_TEAM_ID, ESPN_LEAGUE_SIZE, and
        ESPN_S2 / ESPN_SWID (required for a private league, and never stored
        in a profile — they are credentials and they expire).
        """
        import os

        from nfl_predict.leagues import get_profile

        profile = get_profile(league)

        league_id = os.environ.get("ESPN_LEAGUE_ID") or (profile.espn_league_id or "")
        if not league_id:
            raise EspnFantasyError(
                "Set ESPN_LEAGUE_ID (the number in your league URL), or pick a "
                "league that has one configured. Private leagues also need "
                "ESPN_S2 and ESPN_SWID."
            )

        season = int(
            os.environ.get("ESPN_SEASON") or profile.season or _dt.date.today().year
        )
        size = os.environ.get("ESPN_LEAGUE_SIZE")

        return cls(
            league_id=league_id,
            season=season,
            espn_s2=os.environ.get("ESPN_S2") or None,
            swid=_normalise_swid(os.environ.get("ESPN_SWID")),
            team_id=os.environ.get("ESPN_TEAM_ID") or profile.espn_team_id or None,
            league_size=int(size) if size else profile.roster.league_size,
        )

    @staticmethod
    def credentials_available(league: str | None = None) -> bool:
        """
        True if the client can be constructed.

        Only the league ID is required — public leagues need no cookies, so
        demanding them would hide a working configuration. The ID may come
        from the environment or from the league profile.
        """
        import os

        from nfl_predict.leagues import get_profile

        if os.environ.get("ESPN_LEAGUE_ID"):
            return True
        try:
            return bool(get_profile(league).espn_league_id)
        except KeyError:
            return False

    # ------------------------------------------------------------------
    # Raw API helper
    # ------------------------------------------------------------------

    def _get(
        self,
        views: list[str],
        extra_headers: dict[str, str] | None = None,
        **params: Any,
    ) -> Any:
        """
        GET the league endpoint with one or more ``view`` parameters.

        Extra keyword arguments become query parameters. ``mBoxscore`` needs
        ``scoringPeriodId`` to say which week it should fill rosters for;
        without it ESPN answers for the current week whatever you ask.

        ``extra_headers`` carries ESPN's ``x-fantasy-filter``, the only way to
        page or filter the player pool — it takes a JSON document rather than
        query parameters.
        """
        query: list[tuple[str, Any]] = [("view", v) for v in views]
        query += [(k, v) for k, v in params.items() if v is not None]

        url = (
            f"{_API_BASE}/seasons/{self.season}"
            f"/segments/0/leagues/{self.league_id}"
            f"?{urllib.parse.urlencode(query)}"
        )

        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        if self.espn_s2 and self.swid:
            headers["Cookie"] = f"espn_s2={self.espn_s2}; SWID={self.swid}"
        headers.update(extra_headers or {})

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise EspnFantasyError(_explain_http_error(e, self)) from e
        except Exception as e:
            raise EspnFantasyError(f"ESPN API error ({views}): {e}") from e

        # leagueHistory-style responses come back as a single-element list.
        return payload[0] if isinstance(payload, list) and payload else payload

    # ------------------------------------------------------------------
    # League info
    # ------------------------------------------------------------------

    def get_league(self) -> dict:
        """Fetch league metadata (teams, members, settings)."""
        league = self._get(["mTeam", "mSettings"])

        # Prefer the real team count over the configured fallback.
        size = (league.get("settings") or {}).get("size") or len(
            league.get("teams") or []
        )
        if size:
            self.league_size = int(size)

        return league

    def get_my_team_id(self) -> str:
        """
        Resolve our team ID, from ESPN_TEAM_ID or by matching the SWID.

        ESPN identifies league members by SWID, so a private-league client
        can find its own team without being told which one it is.
        """
        if self.team_id:
            return self.team_id

        if not self.swid:
            raise EspnFantasyError(
                "Cannot determine your team: set ESPN_TEAM_ID, or provide "
                "ESPN_SWID so it can be matched against league members."
            )

        league = self.get_league()
        for team in league.get("teams") or []:
            owners = team.get("owners") or []
            if any(str(o).upper() == self.swid.upper() for o in owners):
                self.team_id = str(team.get("id"))
                return self.team_id

        raise EspnFantasyError(
            f"No team in league {self.league_id} is owned by SWID {self.swid}. "
            "Set ESPN_TEAM_ID explicitly."
        )

    # ------------------------------------------------------------------
    # Player identity
    # ------------------------------------------------------------------

    def _players(self) -> dict[int, dict[str, str]]:
        """
        Map ESPN ``playerId`` -> {player_id (gsis), name, position, team}.

        Built from the ffverse crosswalk, cached for the client's lifetime.
        Returns an empty map rather than raising if the crosswalk can't be
        loaded — picks still record by ordering, just without names.
        """
        if self._player_map is not None:
            return self._player_map

        mapping: dict[int, dict[str, str]] = {}
        try:
            import nflreadpy as nfl

            ids = nfl.load_ff_playerids()
            frame = ids.to_pandas() if hasattr(ids, "to_pandas") else ids
            frame = frame[frame["espn_id"].notna()]

            for row in frame.itertuples(index=False):
                espn_id = getattr(row, "espn_id", None)
                if espn_id is None or espn_id != espn_id:  # NaN check
                    continue
                gsis = getattr(row, "gsis_id", None)
                mapping[int(espn_id)] = {
                    "player_id": "" if gsis is None or gsis != gsis else str(gsis),
                    "name": str(getattr(row, "name", "") or ""),
                    "position": str(getattr(row, "position", "") or ""),
                    "team": str(getattr(row, "team", "") or ""),
                }
        except Exception as e:  # noqa: BLE001 - a missing crosswalk must not stop a draft
            print(f"  Warning: could not load the ESPN->gsis crosswalk: {e}")

        self._player_map = mapping
        return mapping

    # ------------------------------------------------------------------
    # Draft picks
    # ------------------------------------------------------------------

    def fetch_all_picks(self) -> list[dict]:
        """
        Fetch the full draft pick log.

        Returns a list of dicts with keys:
            overall_pick, round, pick_in_round, player_name, player_id,
            position, nfl_team, team_id, is_mine

        ``player_id`` is the gsis ID where the crosswalk resolves it, which
        lets ``mark_drafted`` match exactly instead of by name.
        """
        data = self._get(["mDraftDetail"])
        detail = data.get("draftDetail") or {}
        raw_picks: list[dict] = detail.get("picks") or []

        # ESPN pre-seeds the entire pick slate as soon as the draft is
        # scheduled: every unmade slot comes back with ``playerId: -1``. A
        # sync run before or during the draft would otherwise record the
        # whole remaining board as picks.
        raw_picks = [p for p in raw_picks if int(p.get("playerId") or -1) > 0]

        if not raw_picks:
            return []

        # ESPN usually returns picks in order, but sort defensively — the
        # caller slices by count, so a mis-ordered list would mis-record.
        raw_picks = sorted(
            raw_picks,
            key=lambda p: (
                p.get("roundId") or 0,
                p.get("roundPickNumber") or 0,
            ),
        )

        my_team_id = self._my_team_id_or_none()
        players = self._players()

        picks: list[dict] = []
        for i, p in enumerate(raw_picks):
            espn_player_id = p.get("playerId")
            info = players.get(int(espn_player_id)) if espn_player_id else None

            raw_pos = (info or {}).get("position", "")
            position = _normalise_position(raw_pos)

            name = (info or {}).get("name") or f"ESPN player {espn_player_id}"
            team_id = str(p.get("teamId") or "")

            picks.append(
                {
                    "overall_pick": i + 1,
                    "round": p.get("roundId") or (i // self.league_size) + 1,
                    "pick_in_round": p.get("roundPickNumber")
                    or (i % self.league_size) + 1,
                    "player_name": name,
                    "player_id": (info or {}).get("player_id", ""),
                    "position": position,
                    "nfl_team": _normalise_team((info or {}).get("team", "")),
                    "team_id": team_id,
                    "is_mine": bool(my_team_id and team_id == my_team_id),
                }
            )

        return picks

    def fetch_new_picks(self, already_recorded: int = 0) -> list[dict]:
        """
        Return only picks not yet recorded locally.

        Parameters
        ----------
        already_recorded : number of picks already in DraftState.picks
        """
        return self.fetch_all_picks()[already_recorded:]

    # ------------------------------------------------------------------
    # Rosters
    # ------------------------------------------------------------------

    def fetch_rosters(self, week: int | None = None) -> list[dict]:
        """
        Fetch every team's current roster.

        Returns one dict per rostered player, with the same identity keys as
        ``fetch_all_picks`` (``player_name``, ``player_id``, ``position``,
        ``nfl_team``, ``team_id``, ``is_mine``) plus:

            team_name        the fantasy team's name
            lineup_slot      QB / RB / ... / FLEX / BE / IR
            is_starter       False for bench and IR, True otherwise
            injury_status    ESPN's report string, "" when it says nothing
            acquisition_type DRAFT / ADD / TRADE, as ESPN labels it
            espn_player_id   ESPN's own id, for chasing a bad crosswalk hit

        ``week`` asks for the lineup as it stood in that scoring period;
        omitted, ESPN answers for the current one.
        """
        data = self._get(["mRoster", "mTeam"], scoringPeriodId=week)
        players = self._players()
        my_team_id = self._my_team_id_or_none()
        names = _team_names(data)

        rows: list[dict] = []
        for team in data.get("teams") or []:
            team_id = str(team.get("id") or "")
            for entry in (team.get("roster") or {}).get("entries") or []:
                row = self._roster_entry(entry, players)
                row["team_id"] = team_id
                row["team_name"] = names.get(team_id, "")
                row["is_mine"] = bool(my_team_id and team_id == my_team_id)
                rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Weekly scoring
    # ------------------------------------------------------------------

    def fetch_boxscore(self, week: int) -> list[dict]:
        """
        Fetch one week's per-player scoring for every team.

        Same keys as ``fetch_rosters``, plus:

            week              the scoring period asked for
            actual_points     what the player scored, None before he plays
            projected_points  ESPN's own projection, None when it has none
            team_points       the fantasy team's total for the week

        ``actual_points`` is ESPN's scoring, not ours — it reflects that
        league's ESPN settings, which is exactly what makes it useful as a
        check on ``ScoringRules``. It is not a model input.
        """
        data = self._get(["mBoxscore", "mMatchupScore", "mTeam"], scoringPeriodId=week)
        players = self._players()
        my_team_id = self._my_team_id_or_none()
        names = _team_names(data)

        games = data.get("schedule") or []
        # Matchup periods and scoring periods differ in a league with
        # multi-week matchups, so fall back to the whole schedule rather than
        # returning nothing. The dedupe below keeps that fallback honest.
        in_week = [g for g in games if g.get("matchupPeriodId") == week] or games

        rows: list[dict] = []
        seen: set[tuple[str, Any]] = set()

        for game in in_week:
            for which in ("home", "away"):
                side = game.get(which) or {}
                team_id = str(side.get("teamId") or "")

                # ESPN fills rosterForCurrentScoringPeriod only for the week
                # named by scoringPeriodId; a completed week comes back under
                # rosterForMatchupPeriod instead.
                roster = (
                    side.get("rosterForCurrentScoringPeriod")
                    or side.get("rosterForMatchupPeriod")
                    or {}
                )

                for entry in roster.get("entries") or []:
                    row = self._roster_entry(entry, players)

                    key = (team_id, row["espn_player_id"])
                    if key in seen:
                        continue
                    seen.add(key)

                    player = (entry.get("playerPoolEntry") or {}).get("player") or {}
                    actual, projected = _week_points(player, week)

                    row.update(
                        {
                            "week": week,
                            "team_id": team_id,
                            "team_name": names.get(team_id, ""),
                            "is_mine": bool(my_team_id and team_id == my_team_id),
                            "actual_points": actual,
                            "projected_points": projected,
                            "team_points": side.get("totalPoints"),
                        }
                    )
                    rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Shared roster-entry parsing
    # ------------------------------------------------------------------

    def _roster_entry(self, entry: dict, players: dict[int, dict[str, str]]) -> dict:
        """
        Turn one ESPN roster entry into our vocabulary.

        Identity comes from the ffverse crosswalk wherever it resolves, so
        rosters join to the board on ``player_id`` exactly as picks do. Where
        it misses — most reliably for D/ST, which has no gsis id at all —
        ESPN's own name and eligible slots stand in.
        """
        espn_player_id = entry.get("playerId")
        player = (entry.get("playerPoolEntry") or {}).get("player") or {}
        info = players.get(int(espn_player_id)) if espn_player_id is not None else None

        slot_id = entry.get("lineupSlotId")
        position = _normalise_position((info or {}).get("position", ""))
        if not position:
            position = _position_from_slots(player.get("eligibleSlots"))

        name = (
            (info or {}).get("name")
            or player.get("fullName")
            or f"ESPN player {espn_player_id}"
        )

        return {
            "espn_player_id": espn_player_id,
            "player_name": name,
            "player_id": (info or {}).get("player_id", ""),
            "position": position,
            "nfl_team": _normalise_team((info or {}).get("team", "")),
            "lineup_slot": _LINEUP_SLOTS.get(slot_id, ""),
            "is_starter": slot_id is not None and slot_id not in _BENCH_SLOTS,
            "injury_status": player.get("injuryStatus") or "",
            "acquisition_type": entry.get("acquisitionType") or "",
        }

    def _my_team_id_or_none(self) -> str | None:
        """Our team id where it can be resolved; None is not an error here."""
        import contextlib

        with contextlib.suppress(EspnFantasyError):
            return self.get_my_team_id()
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_position(raw: str) -> str:
    """Map ESPN's position label onto our board's vocabulary."""
    raw = (raw or "").upper()
    return _POSITION_MAP.get(raw, raw)


def _normalise_team(raw: str) -> str:
    """Map the crosswalk's team abbreviation onto nflverse's."""
    raw = (raw or "").upper()
    return _TEAM_MAP.get(raw, raw)


def _position_from_slots(eligible: Any) -> str:
    """
    Infer a position from ESPN's ``eligibleSlots``.

    ESPN lists a player's primary slot first, so the first entry naming
    exactly one position is his. This deliberately avoids
    ``defaultPositionId``, whose numbering is offset from the slot ids and is
    a well-known source of silently wrong positions.
    """
    for slot_id in eligible or []:
        if slot_id in _UNAMBIGUOUS_SLOTS:
            return _LINEUP_SLOTS.get(slot_id, "")
    return ""


def _team_names(payload: dict) -> dict[str, str]:
    """
    Map team id -> display name from an ``mTeam`` payload.

    ESPN moved to a single ``name`` field in 2023; older seasons split it
    across ``location`` and ``nickname``.
    """
    names: dict[str, str] = {}
    for team in payload.get("teams") or []:
        name = team.get("name") or " ".join(
            str(team.get(k) or "").strip() for k in ("location", "nickname")
        )
        names[str(team.get("id") or "")] = name.strip()
    return names


def _week_points(player: dict, week: int) -> tuple[float | None, float | None]:
    """
    Pull (actual, projected) fantasy points for one week out of ``stats[]``.

    ESPN returns season totals and every week in the same list, separated
    only by ``scoringPeriodId``; ``statSourceId`` then distinguishes the real
    result from ESPN's projection.
    """
    actual: float | None = None
    projected: float | None = None

    for stat in player.get("stats") or []:
        if stat.get("scoringPeriodId") != week:
            continue
        total = stat.get("appliedTotal")
        if total is None:
            continue
        if stat.get("statSourceId") == _STAT_ACTUAL:
            actual = float(total)
        elif stat.get("statSourceId") == _STAT_PROJECTED:
            projected = float(total)

    return actual, projected


def _normalise_swid(swid: str | None) -> str | None:
    """ESPN's SWID cookie is brace-wrapped; accept it with or without."""
    if not swid:
        return None
    swid = swid.strip()
    if not swid.startswith("{"):
        swid = "{" + swid
    if not swid.endswith("}"):
        swid = swid + "}"
    return swid


def _explain_http_error(e: urllib.error.HTTPError, client: EspnFantasyClient) -> str:
    """Turn ESPN's bare status codes into something actionable."""
    if e.code in (401, 403):
        if not (client.espn_s2 and client.swid):
            return (
                f"ESPN returned HTTP {e.code} for league {client.league_id}. "
                "This is a private league — set ESPN_S2 and ESPN_SWID from a "
                "logged-in browser session."
            )
        return (
            f"ESPN returned HTTP {e.code} for league {client.league_id} despite "
            "credentials. The cookies have most likely expired — grab fresh "
            "espn_s2 / SWID values from your browser."
        )
    if e.code == 404:
        return (
            f"League {client.league_id} not found for season {client.season}. "
            "Check ESPN_LEAGUE_ID, and ESPN_SEASON if drafting for next year."
        )
    return f"ESPN returned HTTP {e.code} for league {client.league_id}."


# ---------------------------------------------------------------------------
# Polling helper (for CLI use)
# ---------------------------------------------------------------------------


def poll_draft(
    client: EspnFantasyClient,
    on_pick: Any,  # callable(pick_dict) -> None
    interval: int = 30,
    max_rounds: int = 20,
    initial_recorded: int = 0,
) -> None:
    """
    Poll for new draft picks every ``interval`` seconds.

    Mirrors ``nfl_fantasy.poll_draft`` so the CLI can use either provider.
    """
    recorded = initial_recorded
    print(f"Polling ESPN draft (league {client.league_id}) every {interval}s…")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            try:
                for pick in client.fetch_new_picks(already_recorded=recorded):
                    on_pick(pick)
                    recorded += 1
                    if pick["round"] >= max_rounds:
                        print("Max rounds reached. Stopping poll.")
                        return
            except EspnFantasyError as e:
                print(f"  Warning: {e}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")
