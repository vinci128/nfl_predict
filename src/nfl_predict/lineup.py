import re
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer()
# CWD-relative, consistent with the rest of the pipeline — run from the
# project root (the __file__-relative path breaks when pip-installed).
OUT_DIR = Path("outputs")


def _predictions_dir(league: str | None = None) -> Path:
    """Where this league's prediction CSVs live.

    Falls back to the flat `outputs/` when the league has no directory yet,
    so CSVs written before predictions were namespaced still resolve.
    """
    from nfl_predict.leagues import get_profile

    d = get_profile(league).predictions_dir
    return d if d.exists() else OUT_DIR


def _find_latest_prediction_files(league: str | None = None) -> list[Path]:
    """
    Scans this league's prediction directory for all predictions_*.csv files
    and for each position picks the file with the most recent season/week.

    Expected name: predictions_<pos>_<season>_week<week>.csv
    e.g.: predictions_wr_2025_week12.csv
    """
    out_dir = _predictions_dir(league)
    pred_files = list(out_dir.glob("predictions_*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions_*.csv files found in {out_dir}")

    pattern = re.compile(
        r"predictions_(?P<position>[A-Za-z]+)_(?P<season>\d{4})_week(?P<week>\d+)\.csv"
    )

    records = []
    for path in pred_files:
        m = pattern.fullmatch(path.name)
        if not m:
            # skip files that don't match the pattern
            continue
        records.append(
            {
                "position": m.group("position").upper(),
                "season": int(m.group("season")),
                "week": int(m.group("week")),
                "path": path,
            }
        )

    if not records:
        raise RuntimeError(
            f"Files found in {out_dir}, but none matching the pattern "
            "'predictions_<pos>_<season>_week<week>.csv'"
        )

    df_files = pd.DataFrame(records)
    # sort by position, season, week
    df_files = df_files.sort_values(["position", "season", "week"])
    # take the last row per position (max season/week)
    latest = df_files.groupby("position", as_index=False).tail(1)

    return list(latest["path"])


@app.command()
def suggest(
    roster_path: str = typer.Argument("my_roster.csv", help="CSV with your roster"),
    top_n: int = typer.Option(20, help="How many players to show"),
    league: str | None = typer.Option(
        None, help="League profile key (see `nfl-predict leagues`)."
    ),
):
    """
    Reads the most recent predictions_*.csv for each position in this
    league's output directory and shows your players ranked by expected PPR.
    """
    # --- find the most up-to-date prediction files ---
    latest_files = _find_latest_prediction_files(league)
    typer.echo("Using the following most recent prediction files:")
    for p in latest_files:
        typer.echo(f"  - {p.name}")

    # --- load and concatenate all predictions ---
    preds_list = []
    for path in latest_files:
        df = pd.read_csv(path)
        # normalize position
        if "position" in df.columns:
            df["position"] = df["position"].str.upper()
        preds_list.append(df)

    preds = pd.concat(preds_list, ignore_index=True)

    # --- load roster ---
    roster = pd.read_csv(roster_path)

    if "player_name" not in roster.columns or "position" not in roster.columns:
        raise ValueError(
            "Roster must contain at least the columns: 'player_name' and 'position'. "
            "Optional but recommended: 'team'."
        )

    roster["position"] = roster["position"].str.upper()

    # --- dynamic join keys ---
    join_keys = ["player_name", "position"]
    if "team" in preds.columns and "team" in roster.columns:
        join_keys.append("team")

    merged = preds.merge(
        roster,
        on=join_keys,
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "No matches between predictions and roster. "
            "Check that player names (and teams, if used) align."
        )

    merged = merged.sort_values("expected_ppr_points", ascending=False)

    print("\nYOUR players (ranked by expected PPR):\n")
    cols = [
        c
        for c in ["player_name", "team", "position", "expected_ppr_points"]
        if c in merged.columns
    ]
    print(merged[cols].head(top_n).to_string(index=False))


# ---------------------------------------------------------------------------
# Gameday lineup, from ESPN's own projections
# ---------------------------------------------------------------------------

# ESPN designations that mean the player cannot score this week. DAY_TO_DAY
# and QUESTIONABLE are deliberately absent: they still play more often than
# not, and ESPN already prices the risk into its projection.
_CANNOT_PLAY = {"OUT", "DOUBTFUL", "INJURY_RESERVE", "SUSPENSION", "NOT_ACTIVE"}


def espn_optimal(league: str | None = None, week: int | None = None):
    """Best startable lineup for one league, ranked on ESPN's projections.

    Returns ``(starters, bench, benched_out, total)``.

    ESPN's weekly number is used rather than this project's own because in
    week 1 the season model has no in-season data at all: it cannot see an
    offseason move, a depth-chart change, a same-day injury, or a rookie it
    has never scored. Measured on 2026 week 1, agreement between the two
    among *startable* players was 0.67 at RB and 0.64 at WR but
    indistinguishable from zero at QB, DB and K — so for this week ESPN
    orders a lineup better than we do. From week 2 the weekly model has real
    signal and this stops being true.
    """
    from nfl_predict.espn_fantasy import EspnFantasyClient
    from nfl_predict.leagues import (
        fantasy_position,
        get_profile,
        reset_active_league,
        set_active_league,
    )
    from nfl_predict.weekly_api import _lineup_slots

    profile = get_profile(league)
    rows = EspnFantasyClient.from_env(profile.key).fetch_rosters(week=week)
    mine = [r for r in rows if r["is_mine"]]
    if not mine:
        raise RuntimeError(
            f"No team flagged as yours in {profile.name} — check ESPN_TEAM_ID."
        )

    pool, out, locked = [], [], []
    for r in mine:
        # Once a player's game has kicked off ESPN reports a real score, and
        # the slot he is in cannot be changed. He is part of the week now,
        # at what he actually scored — not at what he was projected to.
        played = r["actual_points"] is not None
        rec = {
            "player_name": r["player_name"],
            "position": fantasy_position(r["position"]) or r["position"],
            "nfl_team": r["nfl_team"],
            "injury_status": r["injury_status"] or "",
            "current_slot": r["lineup_slot"],
            "expected_ppr_points": (
                r["actual_points"] if played else r["projected_points"]
            ),
            "locked": played,
            "actual_points": r["actual_points"],
        }
        if played:
            locked.append(rec)
        elif rec["injury_status"] in _CANNOT_PLAY or rec["expected_ppr_points"] is None:
            out.append(rec)
        else:
            pool.append(rec)

    # `_build_lineup` reads the *active* league for its starting slots, not an
    # argument. Without pinning it, a Hell or Highwater lineup gets built
    # against Ludopathy's slots — which have no D/ST — and the defence is
    # silently benched with no slot to put it back in.
    token = set_active_league(profile.key)
    try:
        slots = _lineup_slots()
        flex_positions = list(profile.roster.flex_positions)
    finally:
        reset_active_league(token)

    locked_starters = [x for x in locked if x["current_slot"] not in ("BE", "IR")]
    for x in locked_starters:
        x["slot"] = x["current_slot"]

    starters, bench = _fill_open_slots(slots, flex_positions, locked_starters, pool)
    bench += [x for x in locked if x["current_slot"] in ("BE", "IR")]
    total = round(sum(s["expected_ppr_points"] or 0.0 for s in starters), 1)
    return starters, bench, out, total


def _fill_open_slots(slots, flex_positions, locked_starters, movable):
    """Fill whatever starting slots are still open, best projection first.

    Slots already held by a player whose game has kicked off are spent: he
    cannot be moved out and nobody else can be moved in. Returns
    ``(starters, bench)`` with ``starters`` including the locked ones.
    """
    remaining = dict(slots)
    flex_left = remaining.pop("FLEX", 0)
    for x in locked_starters:
        slot = x["slot"]
        if slot == "FLEX":
            flex_left = max(0, flex_left - 1)
        elif slot in remaining:
            remaining[slot] = max(0, remaining[slot] - 1)

    left = sorted(movable, key=lambda z: -(z["expected_ppr_points"] or 0.0))
    starters = list(locked_starters)
    used: set[int] = set()

    for pos, n in remaining.items():
        for x in left:
            if n <= 0:
                break
            if id(x) in used or x["position"] != pos:
                continue
            starters.append({**x, "slot": pos})
            used.add(id(x))
            n -= 1

    for x in left:
        if flex_left <= 0:
            break
        if id(x) in used or x["position"] not in flex_positions:
            continue
        starters.append({**x, "slot": "FLEX"})
        used.add(id(x))
        flex_left -= 1

    bench = [{**x, "slot": "BN"} for x in left if id(x) not in used]
    return starters, bench


# A lineup change has to be worth making. ESPN quotes projections to four
# decimals, so the optimiser will happily tell you to bench George Kittle for
# Pat Freiermuth on 0.069 points — a real roster move, giving up the higher
# ceiling, for a gain indistinguishable from rounding.
MIN_GAIN = 0.5


def current_total(starters, bench, out) -> float:
    """What the lineup as currently set on ESPN projects.

    Players who cannot play count zero, which is the point: that is what an
    unnoticed OUT in a starting slot actually costs.
    """
    total = 0.0
    for p in list(starters) + list(bench) + list(out):
        if p["current_slot"] in ("BE", "IR"):
            continue
        if p["injury_status"] in _CANNOT_PLAY:
            continue
        total += p["expected_ppr_points"] or 0.0
    return round(total, 1)


def _slot_order(profile):
    """Starting slots in the order a lineup page lists them, FLEX last."""
    slots = [p for p, n in profile.roster.starters.items() if n > 0]
    if profile.roster.flex_spots:
        slots.append("FLEX")
    return slots


def print_espn_lineup(league: str | None = None, week: int | None = None) -> list[str]:
    """Print the best lineup by ESPN's projections; return the changes needed."""
    from nfl_predict.leagues import get_profile

    profile = get_profile(league)
    starters, bench, out, total = espn_optimal(league, week)

    order = list(_slot_order(profile))
    starters = sorted(
        starters,
        key=lambda z: order.index(z["slot"]) if z["slot"] in order else 99,
    )

    typer.echo(f"\n{profile.name} — best lineup by ESPN projection\n")
    settled = 0.0
    for s in starters:
        if s.get("locked"):
            # Already scored: this is what he got, not what he might get.
            note = "  final"
            settled += s["expected_ppr_points"] or 0.0
        elif s["injury_status"] not in ("", "ACTIVE"):
            note = f"  [{s['injury_status']}]"
        else:
            note = ""
        typer.echo(
            f"  {s['slot']:<5} {s['position']:<3} {s['player_name']:<24} "
            f"{s['expected_ppr_points']:>6.1f}{note}"
        )
    n_locked = sum(1 for s in starters if s.get("locked"))
    typer.echo(f"  {'':<5} {'':<3} {'TOTAL':<24} {total:>6.1f}")
    if n_locked:
        typer.echo(
            f"  {'':<5} {'':<3} {'':<24} "
            f"({settled:.1f} already scored across {n_locked} "
            f"player{'s' if n_locked != 1 else ''}, "
            f"{total - settled:.1f} still to play)"
        )
    typer.echo("")

    changes: list[str] = []
    for s in starters:
        if s["current_slot"] in ("BE", "IR"):
            changes.append(f"  START  {s['player_name']:<24} -> {s['slot']}")
    for b in bench:
        if b["current_slot"] not in ("BE", "IR"):
            changes.append(
                f"  BENCH  {b['player_name']:<24} (was {b['current_slot']}, "
                f"{b['expected_ppr_points']:.1f} proj)"
            )
    for o in out:
        if o["current_slot"] not in ("BE", "IR"):
            why = o["injury_status"] or "no projection"
            changes.append(
                f"  BENCH  {o['player_name']:<24} (was {o['current_slot']}, {why})"
            )
    # A player moving between two *starting* slots is cosmetic: the same
    # people start and the total is identical, so it is not a change to make
    # and reporting it sends you to ESPN for nothing.

    now = current_total(starters, bench, out)
    gain = round(total - now, 1)

    if not changes:
        typer.echo(f"Lineup is already optimal ({now:.1f} projected).")
    elif gain < MIN_GAIN:
        # Worth showing, not worth doing: naming the players keeps the call
        # yours rather than hiding it.
        typer.echo(
            f"No change worth making — the best alternative lineup gains "
            f"{gain:+.1f} pts, inside the noise:"
        )
        for c in changes:
            typer.echo(f"  (optional){c[9:]}")
    else:
        typer.echo(f"Changes needed ({now:.1f} -> {total:.1f}, {gain:+.1f}):")
        for c in changes:
            typer.echo(c)
    typer.echo("")
    return changes if gain >= MIN_GAIN else []


if __name__ == "__main__":
    app()


def print_survival_lineup(
    league: str | None = None, week: int | None = None
) -> list[str]:
    """Print the lineup that minimises P(finishing last), and what to change.

    For an elimination league only. Points above survival buy nothing there,
    so the highest-mean lineup is not automatically the right one.
    """
    from nfl_predict.leagues import get_profile
    from nfl_predict.survival import league_teams, simulate_last_place, survival_optimal

    profile = get_profile(league)
    starters, bench, before, after = survival_optimal(profile.key, week)

    order = list(_slot_order(profile))
    starters = sorted(
        starters, key=lambda z: order.index(z["slot"]) if z["slot"] in order else 99
    )

    typer.echo(f"\n{profile.name} — lineup to survive (minimise P(last))\n")
    for s in starters:
        note = (
            "  final"
            if s.get("locked")
            else (
                f"  [{s['injury_status']}]"
                if s["injury_status"] not in ("", "ACTIVE")
                else ""
            )
        )
        typer.echo(
            f"  {s['slot']:<5} {s['position']:<3} {s['player_name']:<24} "
            f"{s['expected_ppr_points']:>6.1f}{note}"
        )
    total = round(sum(s["expected_ppr_points"] or 0.0 for s in starters), 1)
    typer.echo(f"  {'':<5} {'':<3} {'TOTAL':<24} {total:>6.1f}")
    typer.echo(f"\n  P(last) — points-max lineup: {before:6.2%}")
    typer.echo(f"  P(last) — survival lineup  : {after:6.2%}")

    changes = [
        f"  START  {s['player_name']:<24} -> {s['slot']}"
        for s in starters
        if s["current_slot"] in ("BE", "IR")
    ] + [
        f"  BENCH  {b['player_name']:<24} (was {b['current_slot']})"
        for b in bench
        if b["current_slot"] not in ("BE", "IR")
    ]
    if changes:
        typer.echo(f"\nChanges ({before:.2%} -> {after:.2%} risk of going out):")
        for c in changes:
            typer.echo(c)
    else:
        typer.echo("\nAlready the safest lineup available.")

    # One fetch, not one per printed row.
    board = league_teams(profile.key, week)
    typer.echo("\n  league-wide P(last), on lineups as they stand:")
    for tid, p in sorted(simulate_last_place(board).items(), key=lambda kv: -kv[1]):
        t = board[tid]
        mark = "  <-- you" if t["is_mine"] else ""
        if p > 0.001:
            typer.echo(f"    {t['name']:<28} {p:6.2%}{mark}")
    typer.echo("")
    return changes
