#!/usr/bin/env python3
"""Did a waiver claim go through? WhatsApp the answer.

For the morning after ESPN processes waivers. A claim either landed (the
player is on your roster), lost to a higher priority (he is on someone
else's), or never resolved (still FREEAGENT/WAIVERS). Those three need
different follow-ups, so the message says which one happened rather than
just "check ESPN".

Pairs with `fa_status.py`, which answers the question *before* processing.

Usage:  claim_result.py --league ludopathy --player "Kyle Monangai" \
            [--dropped "TreVeyon Henderson"] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from waiver_notify import TOKEN_PATH, _call, own_jid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="ludopathy")
    ap.add_argument("--player", required=True)
    ap.add_argument("--dropped", help="player the claim was paired to drop")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from nfl_predict.espn_fantasy import EspnFantasyClient
    from nfl_predict.leagues import get_profile

    profile = get_profile(args.league)
    c = EspnFantasyClient.from_env(args.league)

    rosters = c.fetch_rosters()
    mine = {r["player_name"] for r in rosters if r["is_mine"]}
    # Whoever ended up with him, if not us.
    owner = next(
        (r for r in rosters if r["player_name"].lower() == args.player.lower()),
        None,
    )

    if args.player in mine:
        text = f"*{profile.name}*: ✅ *Claim landed — {args.player} is on your roster.*"
        if args.dropped:
            gone = args.dropped not in mine
            text += (
                f"\n{args.dropped} was dropped as paired."
                if gone
                else f"\n⚠️ But {args.dropped} is still rostered — check your roster size."
            )
    elif owner is not None:
        text = (
            f"*{profile.name}*: ❌ Claim lost — {args.player} went to "
            f"{owner['team_name']}. Waiver priority unspent."
        )
        if args.dropped:
            text += f"\n{args.dropped} was kept (the drop was conditional)."
    else:
        # Not on any roster: the claim did not resolve into an add.
        filt = {
            "players": {
                "filterStatus": {"value": ["FREEAGENT", "WAIVERS"]},
                "limit": 700,
                "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
            }
        }
        d = c._get(
            ["kona_player_info"],
            extra_headers={"x-fantasy-filter": json.dumps(filt)},
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
        status = (hit or {}).get("status", "unknown")
        if status == "WAIVERS":
            # Still on waivers after the processing window means the run has
            # not happened yet, not that the claim was rejected.
            text = (
                f"*{profile.name}*: ⏳ {args.player} is still on WAIVERS — "
                f"processing has not run yet. Check again later."
            )
        else:
            text = (
                f"*{profile.name}*: ⚠️ Claim did not land — {args.player} is "
                f"unrostered (status={status}) and addable right now."
            )

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
