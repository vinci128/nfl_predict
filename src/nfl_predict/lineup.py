"""
Lineup optimizer — reads weekly predictions and selects the optimal
starting lineup from your roster given positional slot constraints.

Commands
--------
suggest   Show all your rostered players sorted by projected points.
optimize  Pick the best legal starting lineup from your roster.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(help="Lineup suggestion and optimization tools.")

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "outputs"

# ---------------------------------------------------------------------------
# Default slot configuration (standard 1-QB league)
# Change counts here or override with --slots on the CLI.
# ---------------------------------------------------------------------------
DEFAULT_SLOTS: dict[str, int] = {
    "QB": 1,
    "RB": 2,
    "WR": 3,
    "TE": 1,
    "FLEX": 1,  # eligible: RB / WR / TE
    "K": 1,
}

FLEX_ELIGIBLE: set[str] = {"RB", "WR", "TE"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _find_latest_prediction_files() -> list[Path]:
    """
    Scan outputs/ for predictions_*.csv and return the most-recent
    file per position (highest season + week).

    Expected filename: predictions_<pos>_<season>_week<week>.csv
    Example:          predictions_wr_2025_week12.csv
    """
    pred_files = list(OUT_DIR.glob("predictions_*.csv"))
    if not pred_files:
        raise FileNotFoundError(
            f"No predictions_*.csv files found in {OUT_DIR}. "
            "Run `nfl-predict predict --position <POS>` first."
        )

    pattern = re.compile(
        r"predictions_(?P<position>[A-Za-z]+)_(?P<season>\d{4})_week(?P<week>\d+)\.csv"
    )

    records = []
    for path in pred_files:
        m = pattern.fullmatch(path.name)
        if not m:
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
            f"Files found in {OUT_DIR} but none match the pattern "
            "'predictions_<pos>_<season>_week<week>.csv'."
        )

    df_files = pd.DataFrame(records).sort_values(["position", "season", "week"])
    latest = df_files.groupby("position", as_index=False).tail(1)
    return list(latest["path"])


def _load_predictions() -> pd.DataFrame:
    files = _find_latest_prediction_files()
    typer.echo("Using prediction files:")
    for p in files:
        typer.echo(f"  {p.name}")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        if "position" in df.columns:
            df["position"] = df["position"].str.upper()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_roster(roster_path: str) -> pd.DataFrame:
    roster = pd.read_csv(roster_path)
    if "player_name" not in roster.columns or "position" not in roster.columns:
        raise ValueError(
            "Roster CSV must have at least 'player_name' and 'position' columns. "
            "'team' is optional but improves matching accuracy."
        )
    roster["position"] = roster["position"].str.upper()
    return roster


def _merge_roster_with_preds(preds: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    join_keys = ["player_name", "position"]
    if "team" in preds.columns and "team" in roster.columns:
        join_keys.append("team")

    merged = preds.merge(roster, on=join_keys, how="inner")
    if merged.empty:
        raise ValueError(
            "No overlap between predictions and roster. "
            "Check that player names (and teams if used) match exactly."
        )
    return merged.sort_values("expected_ppr_points", ascending=False).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


def _optimize_lineup(
    players: pd.DataFrame,
    slots: dict[str, int],
    flex_eligible: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Greedy lineup optimizer.

    Strategy:
    1. Fill each named positional slot (QB, RB, WR, TE, K) greedily —
       highest projected points first.
    2. Fill FLEX from remaining eligible players (RB/WR/TE) by best projection.

    Returns (starters_df, bench_df).
    """
    available = players.copy().reset_index(drop=True)
    starters_rows: list[dict] = []

    def _take_best(pos: str, n: int, label: str) -> None:
        nonlocal available
        pool = available[available["position"] == pos].head(n)
        if len(pool) < n:
            typer.echo(
                f"  Warning: needed {n} {pos} but only {len(pool)} available on roster."
            )
        for _, row in pool.iterrows():
            starters_rows.append({**row.to_dict(), "slot": label})
        available = available.drop(pool.index)

    # Fill named slots
    for pos, count in slots.items():
        if pos == "FLEX":
            continue
        _take_best(pos, count, pos)

    # Fill FLEX
    flex_count = slots.get("FLEX", 0)
    if flex_count:
        pool = available[available["position"].isin(flex_eligible)].head(flex_count)
        if len(pool) < flex_count:
            typer.echo(
                f"  Warning: needed {flex_count} FLEX but only {len(pool)} eligible players left."
            )
        for _, row in pool.iterrows():
            starters_rows.append({**row.to_dict(), "slot": "FLEX"})
        available = available.drop(pool.index)

    starters = pd.DataFrame(starters_rows)
    bench = available.copy()
    bench["slot"] = "BENCH"

    return starters, bench


def _print_lineup(starters: pd.DataFrame, bench: pd.DataFrame) -> None:
    slot_order = ["QB", "RB", "WR", "TE", "FLEX", "K", "BENCH"]
    starters["_order"] = (
        starters["slot"].map({s: i for i, s in enumerate(slot_order)}).fillna(99)
    )
    starters = starters.sort_values("_order").drop(columns="_order")

    display_cols = [
        c
        for c in ["slot", "player_name", "team", "position", "expected_ppr_points"]
        if c in starters.columns
    ]

    total = starters["expected_ppr_points"].sum()
    bench_total = bench["expected_ppr_points"].sum() if not bench.empty else 0.0

    typer.echo("\n── OPTIMAL STARTING LINEUP ─────────────────────────────")
    typer.echo(starters[display_cols].to_string(index=False))
    typer.echo(f"\n  Projected starter total:  {total:.1f} pts")

    if not bench.empty:
        bench_display = [c for c in display_cols if c in bench.columns]
        typer.echo("\n── BENCH ────────────────────────────────────────────────")
        typer.echo(bench[bench_display].to_string(index=False))
        typer.echo(f"\n  Projected bench total:    {bench_total:.1f} pts")

    typer.echo("─────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def suggest(
    roster_path: str = typer.Argument("my_roster.csv", help="CSV with your roster"),
    top_n: int = typer.Option(20, help="Number of players to show"),
) -> None:
    """Show your rostered players sorted by projected PPR points."""
    preds = _load_predictions()
    roster = _load_roster(roster_path)
    merged = _merge_roster_with_preds(preds, roster)

    display_cols = [
        c
        for c in ["player_name", "team", "position", "expected_ppr_points"]
        if c in merged.columns
    ]
    typer.echo("\nYour players (sorted by projected PPR):\n")
    typer.echo(merged[display_cols].head(top_n).to_string(index=False))


@app.command()
def optimize(
    roster_path: str = typer.Argument("my_roster.csv", help="CSV with your roster"),
    qb: int = typer.Option(DEFAULT_SLOTS["QB"], help="QB starter slots"),
    rb: int = typer.Option(DEFAULT_SLOTS["RB"], help="RB starter slots"),
    wr: int = typer.Option(DEFAULT_SLOTS["WR"], help="WR starter slots"),
    te: int = typer.Option(DEFAULT_SLOTS["TE"], help="TE starter slots"),
    flex: int = typer.Option(
        DEFAULT_SLOTS["FLEX"], help="FLEX starter slots (RB/WR/TE eligible)"
    ),
    k: int = typer.Option(DEFAULT_SLOTS["K"], help="K starter slots"),
) -> None:
    """
    Pick the optimal starting lineup from your roster.

    Uses a greedy algorithm: fills each positional slot with the
    highest-projected available player, then fills FLEX from the
    remaining RB/WR/TE pool.

    Example:
        optimize my_roster.csv --flex 2 --qb 2  # superflex / 2-QB league
    """
    slots: dict[str, int] = {
        "QB": qb,
        "RB": rb,
        "WR": wr,
        "TE": te,
        "FLEX": flex,
        "K": k,
    }

    preds = _load_predictions()
    roster = _load_roster(roster_path)
    merged = _merge_roster_with_preds(preds, roster)

    starters, bench = _optimize_lineup(merged, slots, FLEX_ELIGIBLE)
    _print_lineup(starters, bench)


if __name__ == "__main__":
    app()
