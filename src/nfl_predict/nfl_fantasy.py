"""
NFL Fantasy API connector.

Authenticates to NFL.com and polls the live draft picks endpoint so the
draft UI can auto-record picks without manual typing.

Authentication notes
--------------------
This module uses the NFL Fantasy **mobile app's internal OAuth2 endpoint**
(``https://id.nfl.com/account/login``) with the hardcoded client_id
``"fantasy-football"``.  This is reverse-engineered from the NFL Fantasy
iOS app (v25.x) and is **not officially documented**.

Known risks:
  - NFL rotated its identity infrastructure in late 2024; the endpoint has
    been intermittently unreliable.  If auth returns HTTP 4xx or a body
    with no ``access_token``, the most likely cause is a backend change.
  - There is no public Gigya Site ID available, so the officially
    documented Gigya-based flow is not available for personal use.
  - ``client_id`` may be rotated without notice.
  - Token refresh (``/account/token``) is unverified — this module always
    re-authenticates from scratch when the cached token expires, which is
    the safe fallback.

Configuration (environment variables)
--------------------------------------
    NFL_FANTASY_USERNAME   your NFL.com email
    NFL_FANTASY_PASSWORD   your NFL.com password
    NFL_FANTASY_LEAGUE_ID  the numeric league ID from your league URL
    NFL_FANTASY_TEAM_ID    your team ID in the league (1-based, optional)

Usage
-----
    from nfl_predict.nfl_fantasy import NflFantasyClient

    client = NflFantasyClient.from_env()
    new_picks = client.fetch_new_picks(already_recorded=5)
    for pick in new_picks:
        print(pick["overall_pick"], pick["player_name"], pick["is_mine"])

CLI
---
    nfl-predict nfl-sync --interval 30      # poll every 30 seconds
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mobile app reverse-engineered endpoint.  No refresh URL is used — we always
# re-auth from scratch because the token endpoint is unverified.
_AUTH_URL = "https://id.nfl.com/account/login"
_API_BASE = "https://api.fantasy.nfl.com/v2"

# Maps NFL.com position strings to our internal position codes
_POSITION_MAP = {
    "QB": "QB",
    "RB": "RB",
    "WR": "WR",
    "TE": "TE",
    "K": "K",
    "DEF": "DST",
    "D/ST": "DST",
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NflFantasyError(Exception):
    """Raised when NFL Fantasy API interaction fails."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class NflFantasyClient:
    """
    Thin wrapper around the NFL Fantasy private API.

    Authentication uses NFL.com's OAuth2 password-grant flow.  The session
    token is cached in-memory for the lifetime of the client object.

    Note: NFL.com's API is not publicly documented.  Endpoints and payload
    shapes were reverse-engineered from network traffic in the NFL Fantasy
    mobile app (v25.x) and may break without notice on NFL.com updates.
    """

    username: str
    password: str
    league_id: str
    team_id: str | None = None
    _token: str | None = None
    _token_expiry: float = 0.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> NflFantasyClient:
        """
        Create a client from environment variables.

        Required: NFL_FANTASY_USERNAME, NFL_FANTASY_PASSWORD,
                  NFL_FANTASY_LEAGUE_ID
        Optional: NFL_FANTASY_TEAM_ID
        """
        username = os.environ.get("NFL_FANTASY_USERNAME", "")
        password = os.environ.get("NFL_FANTASY_PASSWORD", "")
        league_id = os.environ.get("NFL_FANTASY_LEAGUE_ID", "")
        team_id = os.environ.get("NFL_FANTASY_TEAM_ID")

        if not username or not password or not league_id:
            raise NflFantasyError(
                "Set NFL_FANTASY_USERNAME, NFL_FANTASY_PASSWORD, and "
                "NFL_FANTASY_LEAGUE_ID environment variables."
            )

        return cls(
            username=username,
            password=password,
            league_id=league_id,
            team_id=team_id,
        )

    @staticmethod
    def credentials_available() -> bool:
        """Return True if the required env vars are set."""
        return all(
            os.environ.get(k)
            for k in (
                "NFL_FANTASY_USERNAME",
                "NFL_FANTASY_PASSWORD",
                "NFL_FANTASY_LEAGUE_ID",
            )
        )

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _ensure_token(self) -> str:
        """
        Return a valid bearer token, re-authenticating from scratch if expired.

        Always re-auths (no separate refresh endpoint) — the token refresh
        URL is unverified so we take the safe path.
        """
        if self._token and time.time() < self._token_expiry - 60:
            return self._token  # type: ignore[return-value]

        import json
        import urllib.request

        # The NFL Fantasy mobile app uses client_id="fantasy-football".
        # If that stops working, "nfl-fantasy-football" is a known alternate.
        payload = json.dumps(
            {
                "username": self.username,
                "password": self.password,
                "grant_type": "password",
                "client_id": "fantasy-football",
            }
        ).encode()

        req = urllib.request.Request(
            _AUTH_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "NFLFantasy/25.0 (iOS)",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data: dict = json.loads(resp.read())
        except Exception as e:
            raise NflFantasyError(
                f"NFL.com auth failed: {e}\n"
                "  Check that NFL_FANTASY_USERNAME / PASSWORD are correct and\n"
                "  that https://id.nfl.com is reachable from this machine."
            ) from e

        token = data.get("access_token") or data.get("token")
        if not token:
            raise NflFantasyError(
                f"No access_token in NFL.com auth response (endpoint may have changed).\n"
                f"  Keys returned: {list(data.keys())}\n"
                "  If NFL rotated the client_id, try setting client_id to\n"
                "  'nfl-fantasy-football' in nfl_fantasy._ensure_token."
            )

        self._token = token
        expires_in = int(data.get("expires_in", 3600))
        self._token_expiry = time.time() + expires_in
        return self._token

    # ------------------------------------------------------------------
    # Raw API helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """Authenticated GET to the NFL Fantasy API."""
        import json
        import urllib.parse
        import urllib.request

        token = self._ensure_token()
        url = f"{_API_BASE}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "NFLFantasy/25.0 (iOS)",
                "Accept": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except Exception as e:
            raise NflFantasyError(f"NFL Fantasy API error ({path}): {e}") from e

    # ------------------------------------------------------------------
    # League info
    # ------------------------------------------------------------------

    def get_league(self) -> dict:
        """Fetch league metadata (teams, settings, season)."""
        return self._get(f"/leagues/{self.league_id}")

    def get_my_team_id(self) -> str:
        """Resolve team_id from username if not set explicitly."""
        if self.team_id:
            return self.team_id

        league = self.get_league()
        teams = league.get("teams") or league.get("data", {}).get("teams", [])
        for team in teams:
            owner = team.get("owners", [{}])[0]
            if owner.get("userName", "").lower() == self.username.lower():
                self.team_id = str(team.get("teamId") or team.get("id"))
                return self.team_id  # type: ignore[return-value]

        raise NflFantasyError(
            f"Could not find team for '{self.username}' in league {self.league_id}. "
            "Set NFL_FANTASY_TEAM_ID explicitly."
        )

    # ------------------------------------------------------------------
    # Draft picks
    # ------------------------------------------------------------------

    def fetch_all_picks(self) -> list[dict]:
        """
        Fetch the full draft pick log from the NFL Fantasy API.

        Returns a list of dicts with keys:
            overall_pick, round, pick_in_round, player_name, position,
            team, nfl_team, team_id, is_mine
        """
        data = self._get(f"/leagues/{self.league_id}/draft/picks")

        # The picks are usually under data.picks or data.draftPicks
        raw_picks: list[dict] = (
            data.get("picks")
            or data.get("draftPicks")
            or data.get("data", {}).get("picks", [])
            or []
        )

        import contextlib

        my_team_id: str | None = None
        with contextlib.suppress(NflFantasyError):
            my_team_id = self.get_my_team_id()

        picks = []
        for i, p in enumerate(raw_picks):
            player = p.get("player") or p.get("playerInfo") or {}
            raw_pos = player.get("position", "") or p.get("position", "")
            pos = _POSITION_MAP.get(raw_pos.upper(), raw_pos.upper())

            name = (
                player.get("name")
                or player.get("displayName")
                or player.get("firstName", "") + " " + player.get("lastName", "")
            ).strip()

            team_id = str(p.get("teamId") or p.get("rosterId") or "")

            picks.append(
                {
                    "overall_pick": i + 1,
                    "round": p.get("round", (i // 10) + 1),
                    "pick_in_round": p.get("pick")
                    or p.get("pickInRound")
                    or (i % 10) + 1,
                    "player_name": name,
                    "position": pos,
                    "nfl_team": player.get("nflTeam") or player.get("teamAbbr") or "",
                    "team_id": team_id,
                    "is_mine": bool(my_team_id and team_id == my_team_id),
                }
            )

        return picks

    def fetch_new_picks(self, already_recorded: int = 0) -> list[dict]:
        """
        Return only picks that haven't been recorded in the local state yet.

        Parameters
        ----------
        already_recorded : number of picks already in DraftState.picks

        Returns
        -------
        List of new pick dicts (empty list if no new picks).
        """
        all_picks = self.fetch_all_picks()
        return all_picks[already_recorded:]


# ---------------------------------------------------------------------------
# Polling helper (for CLI use)
# ---------------------------------------------------------------------------


def poll_draft(
    client: NflFantasyClient,
    on_pick: Any,  # callable(pick_dict) -> None
    interval: int = 30,
    max_rounds: int = 20,
) -> None:
    """
    Poll for new draft picks every ``interval`` seconds.

    Parameters
    ----------
    client    : authenticated NflFantasyClient
    on_pick   : callback called for each new pick dict
    interval  : seconds between polls
    max_rounds: stop after this many rounds (safety limit)
    """
    recorded = 0
    print(f"Polling NFL Fantasy draft (league {client.league_id}) every {interval}s…")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            try:
                new_picks = client.fetch_new_picks(already_recorded=recorded)
                for pick in new_picks:
                    on_pick(pick)
                    recorded += 1
                    if pick["round"] >= max_rounds:
                        print("Max rounds reached. Stopping poll.")
                        return
            except NflFantasyError as e:
                print(f"  Warning: {e}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")
