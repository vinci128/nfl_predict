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
    from nfl_predict.weekly_api import _build_lineup

    profile = get_profile(league)
    rows = EspnFantasyClient.from_env(profile.key).fetch_rosters(week=week)
    mine = [r for r in rows if r["is_mine"]]
    if not mine:
        raise RuntimeError(
            f"No team flagged as yours in {profile.name} — check ESPN_TEAM_ID."
        )

    pool, out = [], []
    for r in mine:
        rec = {
            "player_name": r["player_name"],
            "position": fantasy_position(r["position"]) or r["position"],
            "nfl_team": r["nfl_team"],
            "injury_status": r["injury_status"] or "",
            "current_slot": r["lineup_slot"],
            "expected_ppr_points": r["projected_points"],
        }
        if rec["injury_status"] in _CANNOT_PLAY or rec["expected_ppr_points"] is None:
            out.append(rec)
        else:
            pool.append(rec)

    df = pd.DataFrame(pool)
    if df.empty:
        return [], [], out, 0.0

    # `_build_lineup` reads the *active* league for its starting slots, not an
    # argument. Without pinning it, a Hell or Highwater lineup gets built
    # against Ludopathy's slots — which have no D/ST — and the defence is
    # silently benched with no slot to put it back in.
    token = set_active_league(profile.key)
    try:
        starters, bench, total = _build_lineup(df)
    finally:
        reset_active_league(token)
    return starters, bench, out, total


def print_espn_lineup(league: str | None = None, week: int | None = None) -> list[str]:
    """Print the best lineup by ESPN's projections; return the changes needed."""
    from nfl_predict.leagues import get_profile

    profile = get_profile(league)
    starters, bench, out, total = espn_optimal(league, week)

    typer.echo(f"\n{profile.name} — best lineup by ESPN projection\n")
    for s in starters:
        flag = (
            "" if s["injury_status"] in ("", "ACTIVE") else f"  [{s['injury_status']}]"
        )
        typer.echo(
            f"  {s['slot']:<5} {s['position']:<3} {s['player_name']:<24} "
            f"{s['expected_ppr_points']:>6.1f}{flag}"
        )
    typer.echo(f"  {'':<5} {'':<3} {'TOTAL':<24} {total:>6.1f}\n")

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

    if changes:
        typer.echo("Changes needed:")
        for c in changes:
            typer.echo(c)
    else:
        typer.echo("Lineup is already optimal.")
    typer.echo("")
    return changes


if __name__ == "__main__":
    app()
