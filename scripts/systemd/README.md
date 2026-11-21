# Scheduled fantasy jobs

Two systemd user timers, both on this machine:

| when | script | what it does |
|---|---|---|
| Tue 09:00 | `waiver_check.sh` | waiver wire + best lineup, **reported** |
| Sun 11:00 and 18:00 | `lineup_check.sh` | best lineup, **set on ESPN** |

The first is described in full below; the Sunday one follows it.

---

# Tuesday waiver check

Runs `scripts/waiver_check.sh` every Tuesday at 09:00 local time: the waiver
wire and the best lineup for each live league, written to `outputs/waivers/`
with a desktop notification and a WhatsApp message per league.

## Which leagues

`$LEAGUES`, default **`ludopathy hoh`**. One report and one message each;
a league that cannot be read does not stop the others, and the exit status
is the number that failed.

Royal Rumble is deliberately not in that default. The team was eliminated in
week 2 of 2026 and its roster dissolved, so the check spent two minutes every
Tuesday reporting on a team that no longer exists. For a new season:

```bash
LEAGUES="rumble" scripts/waiver_check.sh
```

Elimination leagues get two extra sections — the elimination state and
`espn-lineup --survive` — and the others do not. That is keyed off
`LeagueProfile.elimination`, not off the league name, because
`survival.elimination_state()` does not fail in a normal league: it cheerfully
reports "nobody out", which reads as reassurance rather than as a question
that does not apply.

## Why local rather than a cloud routine

Ludopathy Bowl is a **private** ESPN league. Reading it needs the `espn_s2`
and `SWID` cookies, which live in `.env` and are deliberately never committed.
A cloud routine has neither the file nor the machine, and was measured failing
on exactly that: Hell or Highwater (public) reads fine without cookies, the
private leagues raise `EspnFantasyError`.

Note this applies to anything scheduled, not just this check — and use an
**installed** unit, never `systemd-run --user`. A transient unit lives only in
the running user manager, so a reboot discards it; `Persistent=true` cannot
resurrect a unit that was never written to disk. Measured 2026-10-17, when a
one-shot reminder armed that way simply never fired and left no journal entry.

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

The check also sends the short version — what to claim and who to drop, plus
the elimination state and your risk of going out in a league that has them —
as a WhatsApp message **to yourself**, via `scripts/waiver_notify.py`.

`NOTIFY=0 scripts/waiver_check.sh` writes that message into the report instead
of sending it, which is how to exercise the whole check without putting test
messages in your own chat.

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

Make the claim. There is no add/drop code path — every roster-changing
endpoint is absent.

The lineup is the exception, as of the Sunday check below: `espn-lineup
--apply` does set the lineup. Claims and trades are still yours.

---

# Sunday lineup check

Runs `scripts/lineup_check.sh` twice on Sunday — 11:00 and 18:00 local — and
**sets** each live league's lineup on ESPN: one report per league in
`outputs/lineups/`, one JSON result per league the notifier reads, a desktop
notification, and a WhatsApp message to yourself.

```bash
cp scripts/systemd/fantasy-lineup.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fantasy-lineup.timer
systemctl --user list-timers fantasy-lineup.timer
```

## Why two runs, and why those hours

The NFL's 13:00 ET slate is 19:00 CET. Both runs sit before it:

- **11:00** catches Saturday's news and sets a lineup even on a morning you
  slept through.
- **18:00** is an hour before kickoff, which is when the inactive lists land.
  That is the run that matters for a player whose status changed overnight.

The occasional 9:30 ET London game is 15:30 CET and falls *between* them — it
is not covered. Set that week's lineup the night before.

`Persistent=true` means a machine asleep at 11:00 runs it on waking. Lateness
is survivable by design: a player whose game has kicked off is a locked
starter and the check only ever moves players who are still free to move. A
machine that wakes at 19:30 still fixes anyone playing Monday.

## Why it is local, and why the run cost matters

Same reason as the waiver check, only more so: this one **writes**. Ludopathy
Bowl and Royal Rumble are private ESPN leagues, so the `espn_s2` / `SWID`
cookies in `.env` are the credential, and a cloud routine has neither the file
nor the machine. Hell or Highwater is public and would work anywhere; the
other two would silently not.

Because it writes, the preflight matters more than the waiver check's. Before
each league it reads one roster and distinguishes a dead network from an
expired cookie: a machine that has just woken routinely has no route yet, and
five retries over ~3 minutes absorb that. A rejected cookie is reported as an
expired cookie, with the fix (`nfl-predict espn-login`) in the report, and
never as "no change needed" — the failure mode that looks like success.

## What it will not do

- **Make a claim, add or drop.** There is no code path for any of them; the
  lineup is the only write this client makes.
- **Fill a starting slot nobody is leaving.** The bench holds a fixed number
  of players, so vacating a slot without filling it is an illegal roster ESPN
  rejects whole — and naming the slot outright would mean guessing which of a
  league's interchangeable slots a player is meant to occupy. If the optimiser
  wants someone who cannot be reached that way, the result says
  `unreachable` and names him rather than quietly calling the lineup optimal.
- **Benching a starter nobody replaces.** An OUT starter and an empty slot
  score the same zero.
- **Touch a player whose game has started**, or one sitting in IR (`--survive`
  in an elimination league is also refused: that lineup minimises a
  probability over the whole league, not a projected-point total).
- **Churn for nothing.** Under the same `MIN_GAIN` the report uses, it does
  nothing at all. Two runs that each reshuffle a lineup for 0.07 points are
  worse than none, because each pass is a transaction you have to read.

## The write API, which is not documented anywhere

Worth knowing if this breaks:

- **A different host.** Reads go to `lm-api-reads.fantasy.espn.com`; writes go
  to `lm-api-writes.fantasy.espn.com`, which serves the same paths. The read
  host answers a POST to `/transactions/` with 405.
- **The transaction type is not `LINEUP`.** It is `ROSTER` for the current
  scoring period and `FUTURE_ROSTER`, carrying its own `scoringPeriodId`, for
  a later one. `LINEUP` is the *item* type. Sending the item's name as the
  transaction's is answered with a bare `{"messages":["Invalid Input."]}` and
  no field name, which reads like a malformed body for as long as you believe
  it — so the type is read from ESPN's own bundle, not guessed. These values
  live in `EspnFantasyClient.submit_lineup`; the reasoning is in its
  docstring.
- **Two headers are load-bearing.** `x-fantasy-platform: espn-fantasy-web`
  (not the `kona-PROD` the read side accepts) and `x-fantasy-source: kona`.
- **`memberId` is not needed.** The server fills it from the cookie.
- **A transaction is atomic.** A rejected one leaves the roster exactly as it
  found it, which is what makes a guess cheap: a bad plan costs a 400, never
  half a lineup.
- **ESPN refuses a lineup recommendation for a future period** (409,
  "not available for future scoring periods") — but it does accept a
  `FUTURE_ROSTER` change. The recommendation endpoint and the write path
  disagree about this, so do not use the former to test the latter.

## Reading a result

`outputs/lineups/lineup-<league>-<stamp>.json` holds what ESPN was asked and
what it answered. `status` is one of:

| status | meaning |
|---|---|
| `applied` | written, and the roster read back matches |
| `dry_run` | planned, nothing written (`APPLY=0`) |
| `already_optimal` | nothing worth changing |
| `not_worth_it` | the best change is under `MIN_GAIN` |
| `unreachable` | a starter should change but no legal move gets him there |
| `rejected` | ESPN refused, with its own words in `error` |
| `unverified` | accepted, but the roster does not match — the one to look at |

`unverified` is the only one that means something is wrong with the client
rather than with the lineup: a 200 from ESPN means it accepted a transaction,
not that the lineup changed, so every write is read back before it is called
done.

## Check it without touching anything

```bash
APPLY=0 NOTIFY=0 scripts/lineup_check.sh        # report only, no write, no send
NOTIFY=0 scripts/lineup_check.sh                # the real thing, message on disk
uv run nfl-predict espn-lineup --league hoh     # one league, report only
uv run nfl-predict espn-lineup --league hoh --apply   # one league, sets it
uv run python scripts/lineup_notify.py --league hoh --dry-run
```
