from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Rolling windows (in games)
ROLLING_WINDOWS = [3, 5, 8]

# Injury status encoding (higher = more injured)
_INJURY_STATUS_MAP = {
    "Questionable": 1,
    "Doubtful": 2,
    "Out": 3,
    "Injured Reserve": 4,
    "Reserve/COVID-19": 4,
    "Physically Unable to Perform": 4,
}

# Practice status encoding (higher = less participation)
_PRACTICE_STATUS_MAP = {
    "Full Participation in Practice": 1,
    "Limited Participation in Practice": 2,
    "Did Not Participate in Practice": 3,
}


def _load_parquet(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fetch_nfl_data.py first.")
    return pd.read_parquet(path)


def load_raw_data():
    """Load the main parquets created by fetch_nfl_data.py."""
    weekly = _load_parquet("weekly_stats")
    rosters = _load_parquet("rosters")
    try:
        snaps = _load_parquet("snap_counts")
    except FileNotFoundError:
        snaps = None
    try:
        schedules = _load_parquet("schedules")
    except FileNotFoundError:
        schedules = None
    try:
        injuries = _load_parquet("injuries")
    except FileNotFoundError:
        injuries = None
    return weekly, rosters, snaps, schedules, injuries


def prepare_base_weekly(
    weekly: pd.DataFrame,
    rosters: pd.DataFrame,
    snaps: pd.DataFrame | None = None,
    offensive_only: bool = False,
) -> pd.DataFrame:
    # Local copies for safety
    weekly = weekly.copy()
    rosters = rosters.copy()

    # ------------------------------------------------------------------
    # 1) Harmonize player ID between weekly and rosters
    #    - nflreadpy uses "gsis_id" in rosters
    #    - weekly already has "player_id" like "00-0019596"
    # ------------------------------------------------------------------
    if "player_id" not in rosters.columns:
        if "gsis_id" in rosters.columns:
            rosters["player_id"] = rosters["gsis_id"]
        else:
            raise KeyError(
                "Rosters has neither 'player_id' nor 'gsis_id'. "
                "Check rosters.parquet columns."
            )

    # ------------------------------------------------------------------
    # 2) Harmonize team/recent_team columns
    # ------------------------------------------------------------------
    if "recent_team" not in weekly.columns and "team" in weekly.columns:
        weekly.rename(columns={"team": "recent_team"}, inplace=True)

    if "team" not in rosters.columns and "recent_team" in rosters.columns:
        rosters.rename(columns={"recent_team": "team"}, inplace=True)

    # ------------------------------------------------------------------
    # 3) Select minimal roster columns
    # ------------------------------------------------------------------
    rosters_min = rosters[
        [
            c
            for c in ["player_id", "position", "team", "season", "week"]
            if c in rosters.columns
        ]
    ].copy()

    if "week" in rosters_min.columns:
        rosters_min = rosters_min.sort_values(["season", "week"]).drop_duplicates(
            ["season", "player_id"], keep="last"
        )
    else:
        rosters_min = rosters_min.sort_values(["season"]).drop_duplicates(
            ["season", "player_id"], keep="last"
        )

    # ------------------------------------------------------------------
    # 4) Merge with weekly on common fields (player_id + season)
    # ------------------------------------------------------------------
    merge_keys = ["player_id"]
    if "season" in weekly.columns and "season" in rosters_min.columns:
        merge_keys.append("season")

    df = weekly.merge(rosters_min, on=merge_keys, how="left", suffixes=("", "_roster"))

    if "position" in weekly.columns:
        df["position"] = df["position"].fillna(df["position_roster"])
    else:
        df["position"] = df["position_roster"]

    df.drop(columns=[c for c in ["position_roster"] if c in df.columns], inplace=True)

    # Filter only offensive roles for fantasy (QB, RB, WR, TE, K)
    if offensive_only:
        df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()
    else:
        print(
            "Keeping all positions (filtering positions that have no fantasy points)."
        )
        df = df[~df["position"].isin(["LS", "NT", "DL", "OL"])].copy()

    # --- Merge with snap counts (if available) ---
    if snaps is not None:
        if "player_id" in snaps.columns:
            snap_cols = ["season", "week", "player_id"]
            for c in ["offense_snaps", "offense_pct"]:
                if c in snaps.columns:
                    snap_cols.append(c)
            snaps_min = snaps[snap_cols].copy()
            df = df.merge(snaps_min, on=["season", "week", "player_id"], how="left")
            if "offense_pct" in df.columns:
                df.rename(columns={"offense_pct": "snap_pct_offense"}, inplace=True)
            if "offense_snaps" in df.columns:
                df.rename(columns={"offense_snaps": "snaps_offense"}, inplace=True)
    else:
        print("snap_counts not available — skipping snap features")

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    return df


def _select_stat_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numeric stat columns for lag/rolling features.
    Includes performance stats, usage, kicking, and contextual signals.
    """
    candidate_cols = [
        # passing
        "passing_yards",
        "passing_tds",
        "passing_attempts",
        "passing_completions",
        "interceptions",
        # rushing
        "rushing_yards",
        "rushing_tds",
        "rushing_attempts",
        # receiving
        "receiving_yards",
        "receiving_tds",
        "receptions",
        "targets",
        # ball security
        "fumbles_lost",
        # fantasy (both PPR and custom — lag of custom is the ideal baseline)
        "fantasy_points_ppr",
        "fantasy_points_custom",
        # usage from snaps
        "snaps_offense",
        "snap_pct_offense",
        # kicking
        "fg_made",
        "fg_att",
        "fg_missed",
        "fg_long",
        "fg_pct",
        "fg_made_0_19",
        "fg_made_20_29",
        "fg_made_30_39",
        "fg_made_40_49",
        "fg_made_50_59",
        "fg_made_60_",
        "fg_missed_0_19",
        "fg_missed_20_29",
        "fg_missed_30_39",
        "fg_missed_40_49",
        "fg_missed_50_59",
        "fg_missed_60_",
        "fg_made_distance",
        "fg_missed_distance",
        "pat_made",
        "pat_att",
        "pat_missed",
        "pat_blocked",
        "pat_pct",
        # contextual signals (injury trends)
        "injury_status",
        "practice_status_enc",
    ]

    present = [c for c in candidate_cols if c in df.columns]
    return [c for c in present if pd.api.types.is_numeric_dtype(df[c])]


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag1 and rolling mean features (windows 3/5/8) for all stat columns.
    Uses only past values (no data leakage).
    Batches all new columns into two pd.concat calls to avoid DataFrame fragmentation.
    """
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    stat_cols = _select_stat_columns(df)
    grouped = df.groupby("player_id", group_keys=False)

    # Pass 1: all lag1 columns at once
    lag_cols = {f"{col}_lag1": grouped[col].shift(1) for col in stat_cols}
    df = pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1)

    # Pass 2: all rolling columns at once (re-group since df grew)
    grouped = df.groupby("player_id", group_keys=False)
    roll_cols: dict[str, pd.Series] = {}
    for col in stat_cols:
        lag_col = f"{col}_lag1"
        for w in ROLLING_WINDOWS:
            roll_cols[f"{col}_roll{w}"] = (
                grouped[lag_col]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
    if roll_cols:
        df = pd.concat([df, pd.DataFrame(roll_cols, index=df.index)], axis=1)

    # Games played rolling count
    if "fantasy_points_ppr_lag1" in df.columns:
        grouped = df.groupby("player_id", group_keys=False)
        gp_cols = {
            f"games_played_roll{w}": grouped["fantasy_points_ppr_lag1"]
            .rolling(window=w, min_periods=1)
            .count()
            .reset_index(level=0, drop=True)
            for w in ROLLING_WINDOWS
        }
        df = pd.concat([df, pd.DataFrame(gp_cols, index=df.index)], axis=1)

    return df


def add_simple_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cumulative season stats (expanding) up to the previous week.
    Batches all new columns into a single pd.concat call to avoid fragmentation.
    """
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    stat_cols = _select_stat_columns(df)
    grouped = df.groupby(["player_id", "season"], group_keys=False)

    season_cols: dict[str, pd.Series | pd.DataFrame] = {}
    for col in stat_cols:
        src_col = f"{col}_lag1" if f"{col}_lag1" in df.columns else col
        season_cols[f"{col}_season_cum"] = grouped[src_col].cumsum()
        season_cols[f"{col}_season_mean"] = (
            grouped[src_col]
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
    if season_cols:
        df = pd.concat([df, pd.DataFrame(season_cols, index=df.index)], axis=1)

    return df


def add_custom_league_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate fantasy points per custom NFL.com league scoring:

    Passing: 0.1 pts/yd, 4 pts/TD, -2 pts/INT
    Rushing: 0.1 pts/yd, 6 pts/TD
    Receiving: 1 pt/rec, 0.1 pts/yd, 6 pts/TD
    Return/Fumble TDs: 6 pts
    Fumbles lost: -2 pts
    2-pt conversions: 2 pts
    PAT made: 1 pt
    FG 0-39: 3 pts, FG 40-49: 4 pts, FG 50+: 5 pts
    """
    pts = pd.Series(0.0, index=df.index, dtype="float64")

    def col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].fillna(0)
        return pd.Series(0, index=df.index, dtype="float64")

    # Passing
    pts += 0.1 * col("passing_yards")
    pts += 4.0 * col("passing_tds")
    pts += -2.0 * (col("interceptions") + col("passing_interceptions"))

    # Rushing
    pts += 0.1 * col("rushing_yards")
    pts += 6.0 * col("rushing_tds")

    # Receiving
    pts += 1.0 * col("receptions")
    pts += 0.1 * col("receiving_yards")
    pts += 6.0 * col("receiving_tds")

    # Return TDs
    pts += 6.0 * (
        col("kick_return_tds") + col("punt_return_tds") + col("special_teams_tds")
    )

    # Fumble recovery TDs
    pts += 6.0 * (col("fumble_recovery_tds") + col("defense_tds"))

    # Fumbles lost
    fumbles_lost = (
        col("fumbles_lost")
        + col("rushing_fumbles_lost")
        + col("receiving_fumbles_lost")
        + col("sack_fumbles_lost")
    )
    pts += -2.0 * fumbles_lost

    # 2-point conversions
    two_pt = (
        col("passing_2pt_conversions")
        + col("rushing_2pt_conversions")
        + col("receiving_2pt_conversions")
        + col("two_point_conversions")
    )
    pts += 2.0 * two_pt

    # Kicking — prefer distance buckets, fall back to fg_long
    fg_made = col("field_goals_made") + col("fg_made") + col("fgm")
    fg_long = col("field_goals_longest") + col("fg_long") + col("fg_longest")

    fg0 = col("fg_made_0_19")
    fg20 = col("fg_made_20_29")
    fg30 = col("fg_made_30_39")
    fg40 = col("fg_made_40_49")
    fg50 = col("fg_made_50_59")
    fg60 = col("fg_made_60_")

    if (fg0.sum() + fg20.sum() + fg30.sum() + fg40.sum() + fg50.sum() + fg60.sum()) > 0:
        fg_points = 3 * (fg0 + fg20 + fg30) + 4 * fg40 + 5 * (fg50 + fg60)
    else:
        fg_points = pd.Series(0.0, index=df.index)
        fg_points[fg_long >= 50] = 5 * fg_made[fg_long >= 50]
        fg_points[(fg_long >= 40) & (fg_long < 50)] = (
            4 * fg_made[(fg_long >= 40) & (fg_long < 50)]
        )
        fg_points[fg_long < 40] = 3 * fg_made[fg_long < 40]

    pts += fg_points
    pts += (col("extra_points_made") + col("xpmade") + col("pat_made")) * 1

    df["fantasy_points_custom"] = pts
    return df


def add_opponent_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add opponent defense strength features: points allowed to each position
    (and total) in prior weeks (lag1 + rolling 3/5/8).
    """
    df = df.copy()

    if "opponent_team" not in df.columns:
        print("No 'opponent_team' column — skipping opponent defense features.")
        return df

    if "fantasy_points_custom" not in df.columns:
        print("No 'fantasy_points_custom' — call add_custom_league_points first.")
        return df

    # Points allowed by team per position per week
    team_allowed = (
        df.groupby(["season", "week", "opponent_team", "position"], dropna=False)[
            "fantasy_points_custom"
        ]
        .sum()
        .reset_index(name="points_allowed")
        .rename(columns={"opponent_team": "team"})
    )

    team_allowed = team_allowed.sort_values(
        ["team", "position", "season", "week"]
    ).reset_index(drop=True)
    grouped = team_allowed.groupby(["team", "position"], group_keys=False)
    team_allowed["points_allowed_lag1"] = grouped["points_allowed"].shift(1)
    for w in ROLLING_WINDOWS:
        team_allowed[f"points_allowed_roll{w}"] = (
            grouped["points_allowed_lag1"]
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )

    # Total points allowed (all positions combined)
    total_allowed = (
        df.groupby(["season", "week", "opponent_team"], dropna=False)[
            "fantasy_points_custom"
        ]
        .sum()
        .reset_index(name="points_allowed_total")
        .rename(columns={"opponent_team": "team"})
    )

    total_allowed = total_allowed.sort_values(["team", "season", "week"]).reset_index(
        drop=True
    )
    gtot = total_allowed.groupby(["team"], group_keys=False)
    total_allowed["points_allowed_total_lag1"] = gtot["points_allowed_total"].shift(1)
    for w in ROLLING_WINDOWS:
        total_allowed[f"points_allowed_total_roll{w}"] = (
            gtot["points_allowed_total_lag1"]
            .rolling(window=w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # Rename to avoid collision
    rename_map = {"points_allowed_lag1": "opp_points_allowed_lag1"}
    for w in ROLLING_WINDOWS:
        rename_map[f"points_allowed_roll{w}"] = f"opp_points_allowed_roll{w}"
        rename_map[f"points_allowed_total_roll{w}"] = (
            f"opp_points_allowed_total_roll{w}"
        )
    rename_map["points_allowed_total_lag1"] = "opp_points_allowed_total_lag1"

    team_allowed = team_allowed.rename(
        columns={k: v for k, v in rename_map.items() if k in team_allowed.columns}
    )
    total_allowed = total_allowed.rename(
        columns={k: v for k, v in rename_map.items() if k in total_allowed.columns}
    )

    df = df.merge(
        team_allowed[
            ["season", "week", "team", "position"]
            + [c for c in team_allowed.columns if c.startswith("opp_")]
        ],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "team", "position"],
        how="left",
    )

    df = df.merge(
        total_allowed[
            ["season", "week", "team"]
            + [c for c in total_allowed.columns if c.startswith("opp_")]
        ],
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "team"],
        how="left",
        suffixes=(None, "_tot"),
    )

    for c in ["team", "team_tot"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    return df


def add_game_context_features(
    df: pd.DataFrame, schedules: pd.DataFrame
) -> pd.DataFrame:
    """
    Add contextual game signals from the schedule:
    - Vegas lines: team_spread, total_line
    - Weather: temp, wind, is_dome
    - Rest days
    - is_home flag
    - bye_next_week indicator
    """
    df = df.copy()

    want = [
        "season",
        "week",
        "home_team",
        "away_team",
        "spread_line",
        "total_line",
        "temp",
        "wind",
        "roof",
        "home_rest",
        "away_rest",
    ]
    sched = schedules[[c for c in want if c in schedules.columns]].copy()

    # Build a per-team view of each game
    def _team_view(
        team_col: str, is_home: int, spread_sign: int, rest_col: str
    ) -> pd.DataFrame:
        keep = ["season", "week", team_col]
        for c in ["spread_line", "total_line", "temp", "wind", "roof", rest_col]:
            if c in sched.columns:
                keep.append(c)
        v = sched[keep].copy().rename(columns={team_col: "team"})
        v["is_home"] = is_home
        if "spread_line" in v.columns:
            v["team_spread"] = spread_sign * v["spread_line"]
            v.drop(columns=["spread_line"], inplace=True)
        if rest_col in v.columns:
            v.rename(columns={rest_col: "rest_days"}, inplace=True)
        return v

    home_view = _team_view("home_team", 1, 1, "home_rest")
    away_view = _team_view("away_team", 0, -1, "away_rest")
    game_ctx = pd.concat([home_view, away_view], ignore_index=True)

    if "roof" in game_ctx.columns:
        game_ctx["is_dome"] = game_ctx["roof"].isin(["dome", "closed"]).astype("int8")
        game_ctx.drop(columns=["roof"], inplace=True)

    # Determine team column in player data
    team_col = "recent_team" if "recent_team" in df.columns else "team"
    pre_cols = set(df.columns)

    df = df.merge(
        game_ctx,
        left_on=["season", "week", team_col],
        right_on=["season", "week", "team"],
        how="left",
    )
    # Drop extra "team" column introduced by merge if team_col != "team"
    if "team" not in pre_cols and "team" in df.columns:
        df.drop(columns=["team"], inplace=True)

    # --- Bye next week ---
    # For each (season, week, team): does the team play in week+1?
    teams_playing = pd.concat(
        [
            sched[["season", "week", "home_team"]].rename(
                columns={"home_team": "team"}
            ),
            sched[["season", "week", "away_team"]].rename(
                columns={"away_team": "team"}
            ),
        ]
    ).drop_duplicates()

    # Shift perspective: row (s, w, t) will have plays_next_week=1
    # if (s, w+1, t) exists — equivalent to checking (s, w, t) in next_week frame
    # with week decremented by 1.
    next_week = teams_playing.copy()
    next_week["week"] = next_week["week"] - 1  # now "week" = the week before
    next_week["plays_next_week"] = 1

    df = df.merge(
        next_week,
        left_on=["season", "week", team_col],
        right_on=["season", "week", "team"],
        how="left",
        suffixes=("", "_nw"),
    )
    df["bye_next_week"] = df["plays_next_week"].isna().astype("int8")
    drop_extra = [c for c in ["plays_next_week", "team_nw"] if c in df.columns]
    if drop_extra:
        df.drop(columns=drop_extra, inplace=True)
    # Also drop the "team" brought in by this merge if it's a new column
    if "team" not in pre_cols and "team" in df.columns:
        df.drop(columns=["team"], inplace=True)

    return df


def add_injury_features(df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """
    Add injury-report features from weekly injury data:
    - injury_status: 0=healthy, 1=Q, 2=D, 3=Out, 4=IR/PUP
    - practice_status_enc: 0=not reported, 1=full, 2=limited, 3=DNP
    """
    df = df.copy()
    inj = injuries.copy()

    if "player_id" not in inj.columns:
        if "gsis_id" in inj.columns:
            inj["player_id"] = inj["gsis_id"]
        else:
            print("injuries: no player_id/gsis_id — skipping injury features")
            return df

    inj["injury_status"] = (
        inj["report_status"].map(_INJURY_STATUS_MAP).fillna(0).astype("int8")
    )
    inj["practice_status_enc"] = (
        inj["practice_status"].map(_PRACTICE_STATUS_MAP).fillna(0).astype("int8")
    )

    inj_agg = inj.groupby(["season", "week", "player_id"], as_index=False).agg(
        injury_status=("injury_status", "max"),
        practice_status_enc=("practice_status_enc", "max"),
    )

    df = df.merge(inj_agg, on=["season", "week", "player_id"], how="left")
    df["injury_status"] = df["injury_status"].fillna(0).astype("int8")
    df["practice_status_enc"] = df["practice_status_enc"].fillna(0).astype("int8")

    return df


def validate_features(df: pd.DataFrame) -> None:
    """Print a data quality report on the final feature DataFrame."""
    print("\n=== Feature Quality Report ===")
    print(f"Shape: {df.shape}")
    if "season" in df.columns:
        print(f"Seasons: {sorted(df['season'].unique())}")
    if "position" in df.columns:
        print(f"Positions: {sorted(df['position'].dropna().unique())}")

    key_cols = [
        "fantasy_points_ppr",
        "fantasy_points_custom",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "total_line",
        "team_spread",
        "temp",
        "wind",
        "is_home",
        "rest_days",
        "bye_next_week",
        "injury_status",
        "snap_pct_offense",
    ]
    print("\nNull rates for key columns:")
    for c in key_cols:
        if c in df.columns:
            null_pct = df[c].isna().mean() * 100
            flag = "  ⚠ high null rate" if null_pct > 50 else ""
            print(f"  {c}: {null_pct:.1f}%{flag}")

    # Warn if any context column is entirely zero (likely merge failure)
    zero_cols = [
        c
        for c in ["total_line", "team_spread", "bye_next_week", "injury_status"]
        if c in df.columns and df[c].fillna(0).eq(0).all()
    ]
    if zero_cols:
        print(
            f"\n  WARNING: these columns are all-zero (possible merge failure): {zero_cols}"
        )
    else:
        print("\n  Context columns look OK (non-zero values present).")

    print("=== End Report ===\n")


def build_player_week_features(save: bool = True) -> pd.DataFrame:
    """
    Full pipeline:
    1. Load raw parquets
    2. Build base player-week table
    3. Add custom fantasy points
    4. Add opponent defense features
    5. Add game context (Vegas, weather, rest, bye week)
    6. Add injury features
    7. Add lag + rolling features
    8. Add season cumulative features
    9. Validate and optionally save
    """
    weekly, rosters, snaps, schedules, injuries = load_raw_data()

    df = prepare_base_weekly(weekly, rosters, snaps=snaps, offensive_only=False)
    df = add_custom_league_points(df)
    df = add_opponent_defense_features(df)

    if schedules is not None:
        print("Adding game context features (Vegas, weather, rest, bye week)...")
        df = add_game_context_features(df, schedules)
    else:
        print("schedules.parquet not found — skipping game context features")

    if injuries is not None:
        print("Adding injury features...")
        df = add_injury_features(df, injuries)
    else:
        print("injuries.parquet not found — skipping injury features")

    df = add_lag_and_rolling_features(df)
    df = add_simple_season_features(df)

    validate_features(df)

    if save:
        out_path = OUT_DIR / "player_week_features.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Saved: {out_path} (shape={df.shape})")

    return df


if __name__ == "__main__":
    build_player_week_features()
