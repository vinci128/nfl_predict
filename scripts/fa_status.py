#!/usr/bin/env python3
"""Is a free agent still add-now, or has he gone to waivers? WhatsApp the answer.

For the morning after the last game, when ESPN rolls the scoring period and
may sweep every unowned player onto waivers. The difference decides whether
speed matters: a FREEAGENT is first come first served this minute, a WAIVERS
player is resolved by priority on the next processing day, and no amount of
speed changes that.

Usage:  fa_status.py --league rumble --player "Brock Purdy"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from waiver_notify import TOKEN_PATH, _call, own_jid  # noqa: E402


def _when(ms) -> str:
    """ESPN's waiverProcessDate is unix milliseconds; say it in local time."""
    import datetime as dt
    import zoneinfo

    t = dt.datetime.fromtimestamp(int(ms) / 1000, dt.UTC)
    return t.astimezone(zoneinfo.ZoneInfo("Europe/Rome")).strftime("%a %H:%M")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="rumble")
    ap.add_argument("--player", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from nfl_predict.espn_fantasy import EspnFantasyClient
    from nfl_predict.leagues import get_profile

    profile = get_profile(args.league)
    c = EspnFantasyClient.from_env(args.league)
    filt = {
        "players": {
            "filterStatus": {"value": ["FREEAGENT", "WAIVERS", "ONTEAM"]},
            "limit": 400,
            "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
        }
    }
    d = c._get(
        ["kona_player_info"], extra_headers={"x-fantasy-filter": json.dumps(filt)}
    )
    hit = next(
        (
            p
            for p in d.get("players") or []
            if ((p.get("player") or {}).get("fullName") or "").lower()
            == args.player.lower()
        ),
        None,
    )
    if hit is None:
        text = f"*{profile.name}*: could not find {args.player} in the pool."
    else:
        status = hit.get("status")
        team = hit.get("onTeamId") or 0
        wpd = hit.get("waiverProcessDate")
        if team:
            text = f"*{profile.name}*: ❌ {args.player} is already rostered (team {team}). Gone."
        elif status == "FREEAGENT":
            text = (
                f"*{profile.name}*: 🟢 *{args.player} is a FREE AGENT — add him NOW.* "
                f"First come first served; every minute counts."
            )
        elif status == "WAIVERS":
            text = (
                f"*{profile.name}*: 🟡 {args.player} is ON WAIVERS. Submit the claim "
                f"(drop Lawrence); it resolves by priority"
                + (f" on {_when(wpd)}" if wpd else " at the next processing")
                + ". You are rank 6 of 8 — speed does not help here."
            )
        else:
            text = f"*{profile.name}*: {args.player} status={status}, onTeam={team}."

    print(text)
    if args.dry_run:
        return 0
    token = TOKEN_PATH.read_text().strip()
    try:
        _call("/send", token, {"jid": own_jid(token), "text": text})
    except Exception as exc:  # noqa: BLE001
        print(f"WhatsApp send failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
