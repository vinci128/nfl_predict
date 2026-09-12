# Tuesday waiver check

Runs `scripts/waiver_check.sh` every Tuesday at 09:00 local time: elimination
state, the waiver wire, and the survival-ranked lineup, written to
`outputs/waivers/` with a desktop notification.

## Why local rather than a cloud routine

Royal Rumble — the league with eliminations, and so the only one where the
waiver wire is time-critical — is a **private** ESPN league. Reading it needs
the `espn_s2` and `SWID` cookies, which live in `.env` and are deliberately
never committed. A cloud routine has neither the file nor the machine, and
was measured failing on exactly that: Hell or Highwater (public) reads fine
without cookies, Royal Rumble raises `EspnFantasyError`.

## Install

```bash
mkdir -p ~/.config/systemd/user
cp scripts/systemd/fantasy-waivers.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fantasy-waivers.timer
```

`crontab` is not installed on this machine (systemd-only), which is why this
is a timer rather than a cron entry.

## Check it

```bash
systemctl --user list-timers fantasy-waivers.timer   # when it next fires
systemctl --user start fantasy-waivers.service       # run it now
journalctl --user -u fantasy-waivers.service -n 50   # what happened
```

`Persistent=true` means a machine asleep at 09:00 runs the check when it
wakes. `loginctl show-user "$USER" -p Linger` should say `Linger=yes`, or the
timer only runs while you are logged in.

## WhatsApp

The check also sends the short version — elimination state, what to claim and
who to drop, and your risk of going out — as a WhatsApp message **to
yourself**, via `scripts/waiver_notify.py`.

It posts to the local Baileys bridge on `127.0.0.1:8787` rather than through
the MCP server, because a timer has no MCP client. Same daemon either way, so
nothing is bypassed; it needs `whatsapp-bridge.service` running and
`~/whatsapp-connection/.token` present. If the bridge is down the send fails
and the check still succeeds — the report is already on disk.

Recipient defaults to your own chat, deliberately. `--to <jid>` can target a
group, but posting your waiver targets into a league group hands your rivals
your homework.

```bash
uv run python scripts/waiver_notify.py --week 2 --dry-run   # preview, no send
```

## What it will not do

Make the claim. The ESPN client here is read-only — every endpoint is a GET,
and there is no add/drop path. The report names the add and the drop; the
transaction is yours to make.
