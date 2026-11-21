#!/usr/bin/env python3
"""Send the Sunday lineup result to WhatsApp, as a message to yourself.

Called by `lineup_check.sh` after the lineup has been set (or has refused to
be). The report on disk has the detail; this is the part worth reading on a
phone — and the only part, most Sundays, because the point of the job is that
nobody has to open anything.

It reads the newest `outputs/lineups/lineup-<league>-*.json`, written by
`nfl-predict espn-lineup --apply`. Reading that file rather than recomputing
the plan is deliberate: the message has to describe what ESPN was actually
asked to do and what it answered, and recomputing would describe a plan that
was never sent — a second, possibly different, answer.

It talks to the local Baileys bridge over its loopback HTTP API rather than
through the MCP server, because a timer has no MCP client — the bridge is the
same daemon either way, so nothing is bypassed.

Default recipient is your own chat. That is deliberate: the league groups are
one keystroke away, and announcing your lineup there hands nine rivals your
waiver plans.

Usage:
    lineup_notify.py [--league hoh] [--to <jid|self>] [--dry-run]
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
ARTEFACT_DIR = Path("outputs") / "lineups"
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

    Baileys reports `me.id` with a device suffix (`3934...:12@s.whatsapp.net`).
    Sending to that exact string addresses one linked device rather than the
    conversation, so the suffix has to go.
    """
    me = (_call("/health", token) or {}).get("me") or ""
    if "@" not in me:
        raise RuntimeError("bridge is not linked to an account")
    user, server = me.split("@", 1)
    return f"{user.split(':')[0]}@{server}"


def latest_artefact(league: str, directory: Path = ARTEFACT_DIR) -> dict:
    """The newest result for this league, or {} when the job has never run.

    Names carry a timestamp, so the sort is the ordering — no mtime, which
    a copied or restored file would get wrong.
    """
    files = sorted(directory.glob(f"lineup-{league}-*.json"))
    if not files:
        return {}
    try:
        return json.loads(files[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def build_message(league: str, result: dict | None = None) -> str:
    """The short version: what was set, or why nothing was."""
    from nfl_predict.leagues import get_profile

    result = latest_artefact(league) if result is None else result
    name = result.get("name") or get_profile(league).name

    if not result:
        return (
            f"*{name}* — no lineup result on disk.\n"
            "_The Sunday check has not run, or it failed before writing one._"
        )

    head = f"*{name}* — week {result['week']}"
    status = result["status"]
    moves = "\n".join(
        f"• {m['in']} in for {m['out']}" for m in result.get("moves") or []
    )

    if status == "applied":
        body = f"*Lineup set on ESPN* (+{result['gain']:.1f} proj)"
        if moves:
            body += "\n" + moves
    elif status == "dry_run":
        body = f"Would set the lineup (+{result['gain']:.1f} proj), nothing written"
        if moves:
            body += "\n" + moves
    elif status == "already_optimal":
        body = "Already the best lineup — nothing to change."
    elif status == "unreachable":
        body = (
            "No legal move reaches the best lineup; wanted "
            + ", ".join(result.get("unplaced") or [])
            + " starting, but no starting slot is free."
        )
    elif status == "not_worth_it":
        body = (
            f"Best change is worth +{result['gain']:.2f}, under the threshold. "
            "Left alone."
        )
    elif status == "rejected":
        body = f"⚠️ *ESPN refused the change* — {result['error']}"
        if moves:
            body += "\n" + moves
    elif status == "unavailable":
        body = f"⚠️ Could not read the roster from ESPN — {result['error']}"
    elif status == "unverified":
        body = "⚠️ ESPN accepted the change but the roster does not match it — " + (
            "; ".join(result.get("wrong") or []) or result["error"]
        )
    else:
        body = f"unexpected result: {status}"

    tail = f"\n_checked {result.get('written_at', '?')}_"
    if result.get("unplaced"):
        tail = f"\n_wanted but could not start: {', '.join(result['unplaced'])}_" + tail
    return f"{head}: {body}{tail}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="ludopathy")
    ap.add_argument("--to", default="self", help='jid, or "self" for your own chat')
    ap.add_argument("--dry-run", action="store_true", help="print, do not send")
    args = ap.parse_args()

    text = build_message(args.league)
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
        # The lineup is set either way; a dead bridge is not a failed check.
        print(f"WhatsApp send failed: {exc}", file=sys.stderr)
        return 1
    print("sent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
