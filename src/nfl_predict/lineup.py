import re
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer()
BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "outputs"


def _find_latest_prediction_files() -> list[Path]:
    """
    Scans OUT_DIR for all predictions_*.csv files and for each position
    picks the file with the most recent season/week.

    Expected name: predictions_<pos>_<season>_week<week>.csv
    e.g.: predictions_wr_2025_week12.csv
    """
    pred_files = list(OUT_DIR.glob("predictions_*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions_*.csv files found in {OUT_DIR}")

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
            f"Files found in {OUT_DIR}, but none matching the pattern "
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
):
    """
    Reads the most recent predictions_*.csv for each position in outputs/
    and shows your players ranked by expected PPR.
    """
    # --- find the most up-to-date prediction files ---
    latest_files = _find_latest_prediction_files()
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


if __name__ == "__main__":
    app()
