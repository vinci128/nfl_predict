#!/usr/bin/env python3
"""Check one player's game-day status and WhatsApp you the verdict.

For the case where a Questionable starter's inactive designation lands
after you have stepped away — 90 minutes before his kickoff, which for a
late-window game is mid-evening here. The message says whether he is
playing and, if not, who to swap in, so the reply from a phone is the swap
itself rather than a lookup.

Usage:
    player_status.py --league rumble --player "Jeremiyah Love" [--fallback "Cam Skattebo"]

Schedule a one-off check with systemd (local time):
    systemd-run --user --on-calendar="2026-09-13 20:55:00" \\
        --working-directory=/home/vinci/nfl_predict \\
        uv run python scripts/player_status.py --league rumble --player "Jeremiyah Love"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from waiver_notify import TOKEN_PATH, _call, own_jid  # noqa: E402

_CANNOT_PLAY = {"OUT", "DOUBTFUL", "INJURY_RESERVE", "SUSPENSION", "NOT_ACTIVE"}


def check(league: str, player: str, fallback: str | None) -> tuple[str, bool]:
    """Return (message, needs_action)."""
    from nfl_predict.espn_fantasy import EspnFantasyClient
    from nfl_predict.leagues import get_profile

    profile = get_profile(league)
    rows = [
        r for r in EspnFantasyClient.from_env(league).fetch_rosters() if r["is_mine"]
    ]
    me = next((r for r in rows if r["player_name"].lower() == player.lower()), None)
    if me is None:
        return f"*{profile.name}*: {player} is not on your roster.", False

    status = me["injury_status"] or "ACTIVE"
    slot = me["lineup_slot"]
    proj = me["projected_points"] or 0.0
    played = me["actual_points"] is not None

    if played:
        return (
            f"*{profile.name}*: {player} has already played "
            f"({me['actual_points']:.1f} pts). Nothing to do.",
            False,
        )

    if status in _CANNOT_PLAY:
        lines = [f"*{profile.name}*: ⚠️ *{player} is {status}* (slot {slot})."]
        fb = None
        if fallback:
            fb = next(
                (r for r in rows if r["player_name"].lower() == fallback.lower()), None
            )
        if fb is None:
            # Best bench player who can take the slot, by projection.
            pos = me["position"]
            bench = [
                r
                for r in rows
                if r["lineup_slot"] in ("BE",)
                and r["actual_points"] is None
                and (r["injury_status"] or "ACTIVE") not in _CANNOT_PLAY
                and r["position"] == pos
            ]
            fb = max(bench, key=lambda r: r["projected_points"] or 0.0, default=None)
        if fb:
            lines.append(
                f"Swap in *{fb['player_name']}* ({fb['projected_points'] or 0:.1f} proj, "
                f"{fb['injury_status'] or 'ACTIVE'}) before his game locks."
            )
        else:
            lines.append("No healthy bench replacement at that position.")
        return "\n".join(lines), True

    if status in ("QUESTIONABLE", "DAY_TO_DAY"):
        return (
            f"*{profile.name}*: {player} still *{status}* ({proj:.1f} proj, slot {slot}). "
            f"ESPN has not updated yet — check again in 15 min.",
            False,
        )

    return (
        f"*{profile.name}*: ✅ {player} is *{status}* ({proj:.1f} proj). Leave him in.",
        False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="rumble")
    ap.add_argument("--player", required=True)
    ap.add_argument("--fallback", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text, _ = check(args.league, args.player, args.fallback)
    print(text)
    if args.dry_run:
        return 0
    if not TOKEN_PATH.exists():
        print("no bridge token; not sending", file=sys.stderr)
        return 1
    token = TOKEN_PATH.read_text().strip()
    try:
        _call("/send", token, {"jid": own_jid(token), "text": text})
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        print(f"WhatsApp send failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
