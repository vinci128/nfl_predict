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

# Outcome columns. Never features — `games_played_next` in particular would
# otherwise be picked up by the season model's "games_played" pattern match
# and leak the availability target straight into the feature set.
TARGET_COLS = (
    "season_total_pts_current",
    "season_total_pts_next",
    "games_played_next",
    "season_ppg_next",
)


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
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """
    Convert player-week features into player-season snapshots.

    For each (player_id, season) takes the LAST week's feature row (which
    carries season-cumulative and roll-8 stats) and attaches the targets:

    - season_total_pts_current : fantasy_points_custom summed over this season
    - season_total_pts_next    : same sum for season + 1
    - games_played_next        : games played in season + 1
    - season_ppg_next          : points per game in season + 1

    The model targets ``season_ppg_next`` (talent) and ``games_played_next``
    (availability) separately; their product reconstructs the season total.
    Predicting the total directly conflates the two, which systematically
    under-projects players who missed time — see season_model.py.

    Rows with no next-season target (the most recent season in the data) are
    kept — they are used for draft-time inference.

    Parameters
    ----------
    df                  : player_week_features DataFrame
    rosters             : optional rosters DataFrame with birth_date / years_exp
    regular_season_only : drop postseason rows before aggregating. Fantasy
                          seasons are regular-season only, and playoff games
                          otherwise inflate totals for players on deep runs —
                          a team-quality artifact that has nothing to do with
                          the player's own scoring rate.

    Returns
    -------
    One row per (player_id, season) with snapshot features + targets.
    """
    df = df.copy()

    if regular_season_only and "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]

    # ------------------------------------------------------------------ #
    # 1. Season totals (used as targets)                                   #
    # ------------------------------------------------------------------ #
    season_totals = (
        df.groupby(["player_id", "season"])["fantasy_points_custom"]
        .sum()
        .rename("season_total_pts")
        .reset_index()
    )

    # Games played = games the player appeared in. Counting only games with
    # positive fantasy points undercounts availability: a QB with two picks
    # and no TDs scores <= 0, and a kicker with no attempts scores 0, yet both
    # played. That undercount inflates any per-game rate derived from it.
    games_played = (
        df.groupby(["player_id", "season"])
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

    # Next-season targets: total points, games played, and their ratio.
    # Shift season by -1 so features@S line up with outcomes@S+1.
    next_targets = season_totals.merge(
        games_played, on=["player_id", "season"], how="left"
    )
    next_targets["season"] = next_targets["season"] - 1
    next_targets = next_targets.rename(
        columns={
            "season_total_pts": "season_total_pts_next",
            "games_played_season": "games_played_next",
        }
    )
    snapshot = snapshot.merge(next_targets, on=["player_id", "season"], how="left")

    # Per-game scoring rate next season. Undefined with no games played.
    snapshot["season_ppg_next"] = snapshot["season_total_pts_next"] / snapshot[
        "games_played_next"
    ].where(snapshot["games_played_next"] > 0)

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
        columns=[c for c in TARGET_COLS if c in inference.columns],
        inplace=True,
    )
    return inference
