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

## What it will not do

Make the claim. The ESPN client here is read-only — every endpoint is a GET,
and there is no add/drop path. The report names the add and the drop; the
transaction is yours to make.
