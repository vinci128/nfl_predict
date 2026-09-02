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
from typing import cast

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

# Injury-report summary columns. Reporting only — see
# build_injury_season_features for why they are not model features.
_INJURY_COLS = (
    "inj_weeks_out",
    "inj_weeks_on_report",
    "inj_weeks_dnp",
    "inj_primary",
)

# Injury-report status -> severity. Higher means less likely to play.
_REPORT_STATUS_SEVERITY = {
    "Out": 3.0,
    "Doubtful": 2.0,
    "Questionable": 1.0,
    "Probable": 0.5,
}
_PRACTICE_SEVERITY = {
    "Did Not Participate In Practice": 3.0,
    "Out (Definitely Will Not Play)": 3.0,
    "Limited Participation in Practice": 2.0,
    "Full Participation in Practice": 1.0,
}
_OUT_SEVERITY = 3.0
_DNP_SEVERITY = 3.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_features(league: str | None = None) -> pd.DataFrame:
    """Load the player-week feature table scored for `league`."""
    from nfl_predict.leagues import resolve_features_path

    path = resolve_features_path(league)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `nfl-predict features --league "
            f"{league or 'ludopathy'}` (or `update-all`)."
        )
    return pd.read_parquet(path)


def load_rosters() -> pd.DataFrame | None:
    """Load rosters parquet (for birth_date / years_exp). Returns None if missing."""
    path = DATA_DIR / "rosters.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_injuries() -> pd.DataFrame | None:
    """Load the weekly injury report. Returns None if missing."""
    path = DATA_DIR / "injuries.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def build_injury_season_features(injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the weekly injury report into one row per (player_id, season).

    These are **reporting columns, not model features**. Walk-forward testing
    (2019-2024) found no incremental predictive value from injury history on
    any of the three season targets — deltas within +/-1.3% MAE, mostly
    slightly worse, and unchanged when restricted to established starters.
    `games_played_season` is already in the feature set and is itself the
    strong injury proxy; the report detail adds nothing on top of it. They are
    surfaced on the draft board so a human can apply the judgement the model
    demonstrably cannot — an Achilles at 30 reads differently from a hamstring
    at 24, and neither is legible to the model.

    Aggregating the report directly (rather than reading the weekly feature
    table) is essential: the weekly merge attaches injury status to games the
    player *played*, so it never sees a week he was Out — 0% of `Out` rows
    have a weekly stat line. Anything derived from the weekly path measures
    "played while listed", not "missed time".
    """
    inj = injuries.copy()

    if "game_type" in inj.columns:
        inj = inj[inj["game_type"] == "REG"]
    if inj.empty:
        return pd.DataFrame(columns=pd.Index(["player_id", "season", *_INJURY_COLS]))

    inj = inj.dropna(subset=["gsis_id", "season", "week"])
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)
    inj["_severity"] = inj["report_status"].map(_REPORT_STATUS_SEVERITY).fillna(0.0)
    inj["_dnp"] = (
        inj["practice_status"].map(_PRACTICE_SEVERITY).fillna(0.0) >= _DNP_SEVERITY
    )

    # One row per player-week, keeping the worst status reported that week.
    inj = (
        inj.sort_values("_severity")
        .groupby(["gsis_id", "season", "week"], as_index=False)
        .last()
    )

    agg = (
        inj.groupby(["gsis_id", "season"])
        .agg(
            inj_weeks_out=("_severity", lambda s: int((s >= _OUT_SEVERITY).sum())),
            inj_weeks_on_report=("week", "nunique"),
            inj_weeks_dnp=("_dnp", "sum"),
        )
        .reset_index()
    )
    agg["inj_weeks_dnp"] = agg["inj_weeks_dnp"].astype(int)

    # Most frequent body part among weeks the player was actually held out;
    # fall back to the overall most frequent when he never missed a week.
    def _primary(group: pd.DataFrame) -> str | None:
        held_out = group[group["_severity"] >= _OUT_SEVERITY]
        source = held_out if not held_out.empty else group
        parts = source["report_primary_injury"].dropna()
        if parts.empty:
            parts = source["practice_primary_injury"].dropna()
        return parts.mode().iloc[0] if not parts.empty else None

    primary_per_season = cast(
        "pd.Series",
        inj.groupby(["gsis_id", "season"])[
            ["_severity", "report_primary_injury", "practice_primary_injury"]
        ].apply(_primary),
    )
    primary = primary_per_season.rename("inj_primary").reset_index()

    out = agg.merge(primary, on=["gsis_id", "season"], how="left")
    return out.rename(columns={"gsis_id": "player_id"})


def build_season_snapshot(
    df: pd.DataFrame,
    rosters: pd.DataFrame | None = None,
    regular_season_only: bool = True,
    injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Convert player-week features into player-season snapshots.

    For each (player_id, season) takes the LAST week's feature row (which
    carries season-cumulative and roll-8 stats) and attaches the targets:

    - season_total_pts_current : fantasy_points_custom summed over this season
    - season_total_pts_next    : same sum for season + 1
    - games_played_next        : games played in season + 1
    - season_ppg_next          : points per game in season + 1

    All three are modelled independently. The total is *not* derived as
    ppg x games — the median of a product is not the product of the medians,
    and multiplying component medians measurably worsened accuracy. The rate
    and availability targets exist to explain a projection, not to build it:
    a player who missed time reads as "elite rate, low games" rather than one
    deflated total. See season_model.py.

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
    injuries            : optional weekly injury report. Adds the reporting
                          columns in ``_INJURY_COLS`` — displayed on the draft
                          board, never used as model features.

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
    games_counts = cast("pd.Series", df.groupby(["player_id", "season"]).size())
    games_played = games_counts.rename("games_played_season").reset_index()

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

    # ------------------------------------------------------------------ #
    # 5. Injury-report summary (display only)                              #
    # ------------------------------------------------------------------ #
    if injuries is not None:
        inj_season = build_injury_season_features(injuries)
        if not inj_season.empty:
            snapshot = snapshot.merge(
                inj_season, on=["player_id", "season"], how="left"
            )
            for col in ("inj_weeks_out", "inj_weeks_on_report", "inj_weeks_dnp"):
                # No injury rows means the player was never on the report.
                snapshot[col] = snapshot[col].fillna(0).astype(int)

    return snapshot


def build_all_inference_rows(
    df: pd.DataFrame,
    as_of_season: int,
    position: str,
    rosters: pd.DataFrame | None = None,
    injuries: pd.DataFrame | None = None,
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
    snapshot = build_season_snapshot(df, rosters=rosters, injuries=injuries)
    mask = (snapshot["season"] == as_of_season) & (
        snapshot["position"] == position.upper()
    )
    inference = snapshot[mask].copy()
    inference.drop(
        columns=[c for c in TARGET_COLS if c in inference.columns],
        inplace=True,
    )
    return inference
