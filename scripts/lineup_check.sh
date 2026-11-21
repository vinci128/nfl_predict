#!/usr/bin/env bash
# Sunday lineup check: set the best lineup on ESPN before the games kick off.
#
# Runs LOCALLY, from a systemd user timer, and that is deliberate rather than
# a fallback. Ludopathy Bowl is a private ESPN league: reading it — and
# reading is the least of it, since this one writes — needs the espn_s2 and
# SWID cookies, which live in .env and are never committed. A cloud routine
# has neither the file nor the machine, so it could set Hell or Highwater
# (public) and nothing else.
#
# This is the one scheduled job that CHANGES something on ESPN. It is the
# same command the Tuesday waiver check prints, with --apply: the lineup the
# `espn-lineup` report names is the lineup that gets set. What it will not do
# is make a claim, a trade or an add — those have no code path.
#
# Fires TWICE, and the second run is the one that matters:
#
#   11:00  catches Saturday's news, and leaves you a lineup rather than a
#          blank on a morning you slept in.
#   18:00  an hour before the 19:00 CET slate, which is when the inactive
#          lists land. Injured players are the ones this is for.
#
# Kickoff is 19:00 CET for the 13:00 ET slate. The occasional 9:30 ET London
# game is 15:30 CET and is NOT covered by either run — set that one the night
# before, or move the first run earlier.
#
# Leagues come from $LEAGUES, default "ludopathy hoh". Royal Rumble is not in
# that list: the team was eliminated in week 2 of 2026 and its roster
# dissolved. Put it back with  LEAGUES="rumble" scripts/lineup_check.sh  for
# a new season.
#
# Each league gets its own report, its own ESPN transaction and its own
# WhatsApp message. One league failing — expired cookies, a league that no
# longer exists — must not cost you the other's lineup, so the per-league
# work returns rather than exits and the exit status is the count that failed.
#
# Install: see scripts/systemd/README.md (crontab is not installed here).

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

# $LEAGUE stays supported: it used to name the single league a check covered.
LEAGUES="${LEAGUES:-${LEAGUE:-ludopathy hoh}}"
NOTIFY="${NOTIFY:-1}"   # 0 = write the WhatsApp text to the report, send nothing
APPLY="${APPLY:-1}"     # 0 = report the lineup without setting it
STAMP="$(date +%Y%m%d-%H%M)"
REPORT_DIR="$PROJECT_DIR/outputs/lineups"
mkdir -p "$REPORT_DIR"

# A timer gets a minimal PATH; uv is usually not on it.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:$PATH"

# Everything for one league, written to its own report. Returns non-zero if
# the league could not be read at all.
check_league() {
  local LEAGUE="$1"
  local REPORT="$REPORT_DIR/lineup-${LEAGUE}-${STAMP}.txt"

  # A league with no ESPN id has no lineup to set. Skip it rather than
  # reporting a failure: `rumble`-style entries are configured but not
  # necessarily linked.
  if ! uv run python -c "
from nfl_predict.leagues import get_profile
raise SystemExit(0 if get_profile('$LEAGUE').espn_league_id else 1)
" 2>/dev/null; then
    echo "no ESPN league id for $LEAGUE — skipped"
    return 0
  fi

  # A subshell, not a brace group: the credential and network branches below
  # end the report with `exit`, and in a brace group that would end the whole
  # run and silently skip every league after this one.
  (
    echo "lineup check — $LEAGUE — $(date '+%Y-%m-%d %H:%M %Z')"
    echo "======================================================================"
    echo

    # Cookies expire, and this job writes: an expired cookie is a lineup
    # nobody set. Say which failure it is rather than reporting "no change",
    # which looks like success. With Persistent=true a machine asleep at
    # 11:00 runs this the moment it wakes, routinely before the network is up
    # — so retry, then distinguish the two.
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

    if [ "$APPLY" = "0" ]; then
      echo "-- lineup, by ESPN projection (NOT APPLIED, APPLY=0)"
      uv run nfl-predict espn-lineup --league "$LEAGUE" 2>&1
    else
      echo "-- lineup, by ESPN projection, and setting it"
      uv run nfl-predict espn-lineup --league "$LEAGUE" --apply 2>&1
    fi
    echo

    echo "======================================================================"
    echo "Claims and adds are still NOT automatic: setting the lineup is the"
    echo "only write this client makes. Everything else is in the ESPN app."
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
  # bridge must not fail the check — the lineup is already set either way.
  # NOTIFY=0 writes the message into the report instead of sending it, so the
  # whole check can be exercised without putting messages in your own chat.
  if [ "$NOTIFY" = "0" ]; then
    echo "-- WhatsApp (not sent, NOTIFY=0)" >>"$REPORT"
    uv run python "$PROJECT_DIR/scripts/lineup_notify.py" --league "$LEAGUE" \
      --dry-run >>"$REPORT" 2>&1
  else
    uv run python "$PROJECT_DIR/scripts/lineup_notify.py" --league "$LEAGUE" \
      >>"$REPORT" 2>&1 || echo "(WhatsApp notify failed — see report)" >>"$REPORT"
  fi

  # Surface it locally too. A report nobody opens is the same as no report.
  if command -v notify-send >/dev/null 2>&1; then
    local STATE
    STATE="$(grep -oE 'lineup set|ESPN refused|already the best|left alone|not match' \
      "$REPORT" | head -1)"
    notify-send -u normal "Fantasy lineup — $LEAGUE" \
      "${STATE:-see report}. Report: $REPORT" || true
  fi

  echo "$REPORT"
  return 0
}

FAILED=0
for league in $LEAGUES; do
  check_league "$league" || FAILED=$((FAILED + 1))
done
exit "$FAILED"
