#!/usr/bin/env bash
# Tuesday waiver check for the elimination league.
#
# Runs LOCALLY, from cron, and that is deliberate rather than a fallback.
# Royal Rumble is a private ESPN league: reading it needs the espn_s2 and
# SWID cookies, which live in .env and are never committed. A cloud routine
# has neither the file nor the machine, so it can read Hell or Highwater
# (public) and nothing else — and Hell or Highwater is not the league with
# eliminations in it.
#
# This does everything up to the claim and then stops. Making the claim is
# a write, and the ESPN client here is read-only by design; see the report
# footer.
#
# Install:  crontab -e   ->   0 9 * * 2 /home/vinci/nfl_predict/scripts/waiver_check.sh
# (09:00 Europe/Rome each Tuesday; cron uses the machine's local time.)

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

LEAGUE="${LEAGUE:-rumble}"
STAMP="$(date +%Y%m%d-%H%M)"
REPORT_DIR="$PROJECT_DIR/outputs/waivers"
REPORT="$REPORT_DIR/waivers-${LEAGUE}-${STAMP}.txt"
mkdir -p "$REPORT_DIR"

# cron gets a minimal PATH; uv is usually not on it.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

{
  echo "waiver check — $LEAGUE — $(date '+%Y-%m-%d %H:%M %Z')"
  echo "======================================================================"
  echo

  # Cookies expire. Say so loudly rather than reporting an empty wire, which
  # looks like "nothing to do" and is the one failure that costs a claim.
  if ! uv run python -c "
from nfl_predict.espn_fantasy import EspnFantasyClient
EspnFantasyClient.from_env('$LEAGUE').fetch_rosters(week=1)
" >/dev/null 2>&1; then
    echo "!! Cannot read $LEAGUE from ESPN."
    echo "!! Most likely the espn_s2 / SWID cookies in .env have expired."
    echo "!! Fix with:  uv run nfl-predict espn-login"
    exit 1
  fi

  echo "-- elimination state"
  uv run python -c "
from nfl_predict.survival import elimination_state
s = elimination_state('$LEAGUE')
print(f\"   {len(s['alive'])} of {len(s['names'])} teams alive, {s['n_eliminated']} eliminated\")
for tid, period in sorted(s['eliminated'].items(), key=lambda kv: kv[1]):
    print(f\"   OUT (week {period}): {s['names'][tid]}\")
danger = [s['names'][t] for t, v in s['status'].items() if v == 'DANGER']
if danger:
    print('   in DANGER now: ' + ', '.join(danger))
" 2>&1
  echo

  echo "-- waiver wire"
  uv run nfl-predict waivers --league "$LEAGUE" 2>&1
  echo

  echo "-- lineup, ranked to survive"
  uv run nfl-predict espn-lineup --league "$LEAGUE" --survive 2>&1
  echo

  echo "======================================================================"
  echo "Claims are NOT made automatically: this client only reads from ESPN."
  echo "Act on anything above in the ESPN app, or ask Claude to drive it."
} >"$REPORT" 2>&1

# Surface it. A report nobody opens is the same as no report.
if command -v notify-send >/dev/null 2>&1; then
  SUMMARY="$(grep -cE '^  (QB|RB|WR|TE|K|DST|LB|DL|DB) ' "$REPORT" 2>/dev/null || echo 0)"
  notify-send -u normal "Fantasy waivers — $LEAGUE" \
    "$SUMMARY upgrade(s) on the wire. Report: $REPORT"
fi

echo "$REPORT"
