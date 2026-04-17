"""
Build player-season snapshots for the draft / season-projection model.

Each row represents a player's end-of-season feature vector (season S)
paired with the target: total fantasy points in season S+1.

The "snapshot" for a given season is the *last* week's row, which already
contains season-cumulative and rolling-mean features that encode full-season
form.  Birth-date / experience data are merged in from the rosters file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"

# Feature suffixes we keep from the end-of-season row
_SNAP_SUFFIXES = ("_roll8", "_season_cum", "_season_mean")

# Rolling games-played columns
_GAMES_PLAYED_COLS = ["games_played_roll3", "games_played_roll5", "games_played_roll8"]

# Identifier / metadata columns (not features but needed for joins / display)
_ID_COLS = ["player_id", "position", "season"]
_DISPLAY_COLS = ["player_display_name", "player_name", "recent_team"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_features() -> pd.DataFrame:
    """Load the player-week feature parquet produced by features.py."""
    path = PROCESSED_DIR / "player_week_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `nfl-predict update-all --no-train`"
        )
    return pd.read_parquet(path)


def load_rosters() -> pd.DataFrame | None:
    """Load rosters parquet (for birth_date / years_exp). Returns None if missing."""
    path = DATA_DIR / "rosters.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def build_season_snapshot(
    df: pd.DataFrame,
    rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Convert player-week features into player-season snapshots.

    For each (player_id, season) takes the LAST week's feature row (which
    carries season-cumulative and roll-8 stats) and attaches two targets:

    - season_total_pts_current : fantasy_points_custom summed over this season
    - season_total_pts_next    : same sum for season + 1  ← **training target**

    Rows with no next-season target (the most recent season in the data) are
    kept — they are used for draft-time inference.

    Parameters
    ----------
    df      : player_week_features DataFrame
    rosters : optional rosters DataFrame containing birth_date / years_exp

    Returns
    -------
    One row per (player_id, season) with snapshot features + targets.
    """
    df = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Season totals (used as targets)                                   #
    # ------------------------------------------------------------------ #
    season_totals = (
        df.groupby(["player_id", "season"])["fantasy_points_custom"]
        .sum()
        .rename("season_total_pts")
        .reset_index()
    )

    # Games with any fantasy scoring
    games_played = (
        df[df["fantasy_points_custom"] > 0]
        .groupby(["player_id", "season"])
        .size()
        .rename("games_played_season")
        .reset_index()
    )

    # ------------------------------------------------------------------ #
    # 2. End-of-season snapshot (last week per player-season)              #
    # ------------------------------------------------------------------ #
    snapshot = (
        df.sort_values(["player_id", "season", "week"])
        .groupby(["player_id", "season"])
        .last()
        .reset_index(drop=False)
    )

    # Select columns: IDs + display + games-played + roll8/season_cum/mean
    keep: list[str] = []
    for col in snapshot.columns:
        if col in _ID_COLS or col in _DISPLAY_COLS:
            keep.append(col)
            continue
        if col in _GAMES_PLAYED_COLS:
            keep.append(col)
            continue
        if any(col.endswith(suf) for suf in _SNAP_SUFFIXES):
            keep.append(col)

    snapshot = snapshot[[c for c in keep if c in snapshot.columns]].copy()

    # Merge actual games played this season
    snapshot = snapshot.merge(games_played, on=["player_id", "season"], how="left")

    # ------------------------------------------------------------------ #
    # 3. Merge roster metadata (age, experience)                           #
    # ------------------------------------------------------------------ #
    if rosters is not None:
        roster_keep = ["season"]
        # Rosters use gsis_id; rename to player_id for the join
        if "gsis_id" in rosters.columns:
            roster_keep.append("gsis_id")
        for c in ("birth_date", "years_exp", "entry_year"):
            if c in rosters.columns:
                roster_keep.append(c)

        if len(roster_keep) > 2:  # more than just season + id
            rosters_min = (
                rosters[roster_keep]
                .rename(columns={"gsis_id": "player_id"})
                .drop_duplicates(["player_id", "season"], keep="last")
            )
            snapshot = snapshot.merge(
                rosters_min, on=["player_id", "season"], how="left"
            )

    # Derive age from birth_date
    if "birth_date" in snapshot.columns:
        snapshot["birth_date"] = pd.to_datetime(snapshot["birth_date"], errors="coerce")
        season_start = pd.to_datetime(
            snapshot["season"].astype(str) + "-09-01", format="%Y-%m-%d"
        )
        snapshot["age_at_season_start"] = (
            (season_start - snapshot["birth_date"]).dt.days / 365.25
        ).round(1)
        snapshot.drop(columns=["birth_date"], inplace=True)

    # ------------------------------------------------------------------ #
    # 4. Attach targets                                                    #
    # ------------------------------------------------------------------ #
    # Current season total (useful for diagnostics)
    snapshot = snapshot.merge(
        season_totals.rename(columns={"season_total_pts": "season_total_pts_current"}),
        on=["player_id", "season"],
        how="left",
    )

    # Next season total (the training target)
    next_totals = season_totals.copy()
    next_totals["season"] = next_totals["season"] - 1  # shift: features@S → target@S+1
    next_totals = next_totals.rename(
        columns={"season_total_pts": "season_total_pts_next"}
    )
    snapshot = snapshot.merge(next_totals, on=["player_id", "season"], how="left")

    return snapshot


def build_all_inference_rows(
    df: pd.DataFrame,
    as_of_season: int,
    position: str,
    rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build inference feature rows for all active players of a position.

    Uses their stats through ``as_of_season`` to project ``as_of_season + 1``.

    Parameters
    ----------
    df          : player_week_features DataFrame
    as_of_season: the most recently completed season (e.g. 2024)
    position    : "WR", "RB", "QB", "TE", or "K"
    rosters     : optional rosters DataFrame

    Returns
    -------
    One row per player; targets are absent (they don't exist yet).
    """
    snapshot = build_season_snapshot(df, rosters=rosters)
    mask = (snapshot["season"] == as_of_season) & (
        snapshot["position"] == position.upper()
    )
    inference = snapshot[mask].copy()
    inference.drop(
        columns=[
            c
            for c in ("season_total_pts_next", "season_total_pts_current")
            if c in inference.columns
        ],
        inplace=True,
    )
    return inference
