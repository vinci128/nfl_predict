#!/usr/bin/env bash
# Tuesday waiver check for the leagues that are still live.
#
# Runs LOCALLY, from a systemd user timer, and that is deliberate rather than
# a fallback. Ludopathy Bowl is a private ESPN league: reading it needs the
# espn_s2 and SWID cookies, which live in .env and are never committed. A
# cloud routine has neither the file nor the machine, so it can read Hell or
# Highwater (public) and nothing else.
#
# This does everything up to the claim and then stops. Making the claim is a
# write, and there is no code path for one; see the report footer. Setting the
# LINEUP is the one write this project makes, and it is the Sunday job's
# (scripts/lineup_check.sh), not this one.
#
# Leagues come from $LEAGUES, default "ludopathy hoh". Royal Rumble is not in
# that list: the team was eliminated in week 2 of 2026 and its roster
# dissolved, so the check spent two minutes a week reporting on nothing. Put
# it back with  LEAGUES="rumble" scripts/waiver_check.sh  for a new season.
#
# Each league gets its own report and its own WhatsApp message. One league
# failing — expired cookies, a league that no longer exists — must not cost
# you the other's tips, so the per-league work returns rather than exits and
# the exit status is the count of leagues that failed.
#
# Install: see scripts/systemd/README.md (crontab is not installed here).

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

# $LEAGUE stays supported: it used to name the single league this checked.
LEAGUES="${LEAGUES:-${LEAGUE:-ludopathy hoh}}"
NOTIFY="${NOTIFY:-1}"   # 0 = write the WhatsApp text to the report, send nothing
STAMP="$(date +%Y%m%d-%H%M)"
REPORT_DIR="$PROJECT_DIR/outputs/waivers"
mkdir -p "$REPORT_DIR"

# A timer gets a minimal PATH; uv is usually not on it.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

# Is this a league where last place is knocked out each week? Only there do
# the elimination state and the survival-ranked lineup mean anything — in a
# normal league elimination_state() happily reports "nobody out", which reads
# as reassurance rather than as a question that does not apply.
is_elimination_league() {
  uv run python -c "
from nfl_predict.leagues import get_profile
raise SystemExit(0 if get_profile('$1').elimination else 1)
" 2>/dev/null
}

# Everything for one league, written to its own report. Returns non-zero if
# the league could not be read at all.
check_league() {
  local LEAGUE="$1"
  local REPORT="$REPORT_DIR/waivers-${LEAGUE}-${STAMP}.txt"

  # A subshell, not a brace group: the credential and network branches below
  # end the report with `exit`, and in a brace group that would end the whole
  # run and silently skip every league after this one.
  (
    echo "waiver check — $LEAGUE — $(date '+%Y-%m-%d %H:%M %Z')"
    echo "======================================================================"
    echo

    # Cookies expire. Say so loudly rather than reporting an empty wire, which
    # looks like "nothing to do" and is the one failure that costs a claim.
    # Distinguish "your cookies died" from "the network was not up yet". With
    # Persistent=true a machine asleep at 09:00 runs this the moment it wakes,
    # which on 2026-09-15 was 09:40 with no route to ESPN — and the check
    # blamed the cookies, which sends you to re-login for nothing. Retry first,
    # then say which of the two it actually is.
    local PREFLIGHT=""
    for attempt in 1 2 3 4 5; do
      PREFLIGHT="$(uv run python -c "
from nfl_predict.espn_fantasy import EspnFantasyClient
try:
    EspnFantasyClient.from_env('$LEAGUE').fetch_rosters(week=1)
    print('OK')
except Exception as exc:
    msg = str(exc)
    auth = any(k in msg for k in ('401', '403', 'cookie', 'private'))
    print('AUTH' if auth else 'NET', msg[:200])
" 2>&1 | tail -1)"
      case "$PREFLIGHT" in
        OK*)   break ;;
        AUTH*) break ;;   # retrying a rejected cookie just wastes minutes
      esac
      [ "$attempt" -lt 5 ] && sleep $((attempt * 20))
    done

    case "$PREFLIGHT" in
      OK*)
        ;;
      AUTH*)
        echo "!! ESPN rejected the credentials for $LEAGUE."
        echo "!! The espn_s2 / SWID cookies in .env have expired."
        echo "!! Fix with:  uv run nfl-predict espn-login"
        echo "!! ($PREFLIGHT)"
        exit 1
        ;;
      *)
        echo "!! Could not reach ESPN after 5 attempts over ~3 minutes."
        echo "!! This is a network problem, NOT your cookies — do not re-login."
        echo "!! Most likely the machine had just woken and had no route yet."
        echo "!! ($PREFLIGHT)"
        exit 1
        ;;
    esac

    if is_elimination_league "$LEAGUE"; then
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
    fi

    echo "-- waiver wire"
    uv run nfl-predict waivers --league "$LEAGUE" 2>&1
    echo

    if is_elimination_league "$LEAGUE"; then
      echo "-- lineup, ranked to survive"
      uv run nfl-predict espn-lineup --league "$LEAGUE" --survive 2>&1
    else
      echo "-- lineup, by ESPN projection"
      uv run nfl-predict espn-lineup --league "$LEAGUE" 2>&1
    fi
    echo

    echo "======================================================================"
    echo "Claims are NOT made automatically: this client only reads from ESPN."
    echo "Act on anything above in the ESPN app, or ask Claude to drive it."
  ) >"$REPORT" 2>&1
  local RC
  RC=$?

  # A league that could not be read has nothing worth sending on. Its report
  # is on disk and says why.
  if [ "$RC" -ne 0 ]; then
    echo "$REPORT"
    return "$RC"
  fi

  # WhatsApp the short version to yourself. The report on disk has the detail;
  # this is the part that reaches you when you are not at the machine. A dead
  # bridge must not fail the check — the report is already written.
  # NOTIFY=0 writes the message into the report instead of sending it, so the
  # whole check can be exercised without putting messages in your own chat.
  if [ "$NOTIFY" = "0" ]; then
    echo "-- WhatsApp (not sent, NOTIFY=0)" >>"$REPORT"
    uv run python "$PROJECT_DIR/scripts/waiver_notify.py" --league "$LEAGUE" \
      --dry-run >>"$REPORT" 2>&1
  else
    uv run python "$PROJECT_DIR/scripts/waiver_notify.py" --league "$LEAGUE" \
      >>"$REPORT" 2>&1 || echo "(WhatsApp notify failed — see report)" >>"$REPORT"
  fi

  # Surface it locally too. A report nobody opens is the same as no report.
  if command -v notify-send >/dev/null 2>&1; then
    local SUMMARY
    SUMMARY="$(grep -cE '^  (QB|RB|WR|TE|K|DST|LB|DL|DB) ' "$REPORT" 2>/dev/null || echo 0)"
    notify-send -u normal "Fantasy waivers — $LEAGUE" \
      "$SUMMARY upgrade(s) on the wire. Report: $REPORT" || true
  fi

  echo "$REPORT"
  return 0
}

FAILED=0
for league in $LEAGUES; do
  check_league "$league" || FAILED=$((FAILED + 1))
done
exit "$FAILED"
