#!/usr/bin/env python3
"""Send the week's waiver tips to WhatsApp, as a message to yourself.

Called by `waiver_check.sh` after the full report is written. The report has
the detail; this is the part worth reading on a phone.

It talks to the local Baileys bridge over its loopback HTTP API rather than
through the MCP server, because cron has no MCP client — the bridge is the
same daemon either way, so nothing is bypassed.

Default recipient is your own chat. That is deliberate: the league groups are
one keystroke away and posting your own waiver targets into one hands nine
rivals your homework.

Usage:
    waiver_notify.py [--league rumble] [--to <jid|self>] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BRIDGE = "http://127.0.0.1:8787"
TOKEN_PATH = Path.home() / "whatsapp-connection" / ".token"
TIMEOUT = 15


def _call(path: str, token: str, payload: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"{BRIDGE}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def own_jid(token: str) -> str:
    """This account's own chat, for a message to yourself.

    Baileys reports `me.id` with a device suffix (`3934…:12@s.whatsapp.net`).
    Sending to that exact string addresses one linked device rather than the
    conversation, so the suffix has to go.
    """
    me = (_call("/health", token) or {}).get("me") or ""
    if "@" not in me:
        raise RuntimeError("bridge is not linked to an account")
    user, server = me.split("@", 1)
    return f"{user.split(':')[0]}@{server}"


def build_message(league: str, week: int | None = None) -> str:
    """The short version: what changed, what to claim, how safe you are."""
    from nfl_predict.leagues import get_profile
    from nfl_predict.survival import (
        elimination_state,
        league_teams,
        simulate_last_place,
    )
    from nfl_predict.waivers import next_open_week, waiver_candidates

    profile = get_profile(league)
    from nfl_predict.predict_week import get_default_season_and_week

    season, fallback = get_default_season_and_week()
    week = week or next_open_week(season) or fallback

    lines = [f"*{profile.name}* — waivers for week {week}", ""]

    try:
        st = elimination_state(league)
        if st["n_eliminated"]:
            gone = ", ".join(
                st["names"][t]
                for t in sorted(st["eliminated"], key=lambda x: st["eliminated"][x])
            )
            lines.append(f"Eliminated so far: {gone}")
        danger = [st["names"][t] for t, v in st["status"].items() if v == "DANGER"]
        if danger:
            lines.append(f"In danger: {', '.join(danger)}")
        lines.append("")
    except Exception:
        # An elimination read failing must not cost you the waiver tips.
        pass

    ups, _, pool = waiver_candidates(league=league, week=week)
    if ups:
        lines.append(f"Worth claiming ({len(pool)} free agents scanned):")
        for u in ups[:6]:
            lines.append(
                f"• {u['position']} *{u['add']}* {u['add_proj']:.1f} "
                f"— drop {u['drop']} {u['drop_proj']:.1f}  ({u['gain']:+.1f})"
            )
    else:
        lines.append("Nothing on the wire beats what you have.")
    lines.append("")

    try:
        board = league_teams(league)
        probs = simulate_last_place(board)
        mine = next((t for t in board if board[t]["is_mine"]), None)
        if mine:
            rank = sorted(probs.values(), reverse=True).index(probs[mine]) + 1
            lines.append(
                f"Risk of going out: *{probs[mine]:.1%}* "
                f"({rank}th most exposed of {len(probs)})"
            )
            worst = max(probs, key=lambda t: probs[t])
            lines.append(f"Likeliest out: {board[worst]['name']} {probs[worst]:.1%}")
    except Exception:
        pass

    lines.append("")
    lines.append("_Claims are not automatic — make them in the ESPN app._")
    return "\n".join(lines).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="rumble")
    ap.add_argument("--to", default="self", help='jid, or "self" for your own chat')
    ap.add_argument("--week", type=int, default=None, help="override the week")
    ap.add_argument("--dry-run", action="store_true", help="print, do not send")
    args = ap.parse_args()

    text = build_message(args.league, args.week)
    if args.dry_run:
        print(text)
        return 0

    if not TOKEN_PATH.exists():
        print(f"no bridge token at {TOKEN_PATH}; not sending", file=sys.stderr)
        return 1
    token = TOKEN_PATH.read_text().strip()

    try:
        jid = own_jid(token) if args.to == "self" else args.to
        _call("/send", token, {"jid": jid, "text": text})
    except (urllib.error.URLError, RuntimeError, OSError) as exc:
        # The report is already on disk; a dead bridge is not a failed check.
        print(f"WhatsApp send failed: {exc}", file=sys.stderr)
        return 1
    print("sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
