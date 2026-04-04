# src/nfl_predict/train_model.py

from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool

from nfl_predict.model_registry import ModelRegistry

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Context / injury feature patterns included for every position
_CONTEXT_PATTERNS = [
    "total_line",
    "team_spread",
    "is_home",
    "rest_days",
    "bye_next_week",
    "injury_status",
    "practice_status",
    "is_dome",
    "temp",
    "wind",
]

# Position-specific stat patterns (substring match on lowercase column name)
_POSITION_PATTERNS: dict[str, list[str]] = {
    "QB": [
        "passing_",
        "passing",
        "interception",
        "sack",
        "rushing_",
        "rushing",
        "air_yards",
        "epa",
        "fantasy_points",
    ],
    "RB": [
        "rushing_",
        "rushing",
        "carry",
        "snap",
        "target",
        "reception",
        "fantasy_points",
    ],
    "WR": [
        "receiv",
        "target",
        "air_yards",
        "target_share",
        "wopr",
        "snap",
        "fantasy_points",
    ],
    "TE": [
        "receiv",
        "target",
        "snap",
        "fantasy_points",
    ],
    "K": [
        "fg_",
        "pat_",
        "fgm",
        "fg_long",
        "fg_made",
        "fg_att",
        "fantasy_points",
    ],
    "DST": ["def_", "sack", "interception", "def_tds", "fantasy_points"],
}

# Columns always included regardless of position (excluding identifiers)
_ALWAYS_ALLOW = {
    "season",
    "week",
    "position_group",
    "season_type",
    "games_played_roll3",
    "games_played_roll5",
    "games_played_roll8",
}

# Columns always excluded (identifiers, current-week leakage, strings)
_DROP_EXACT = {
    "target_points_next_week",
    "fantasy_points_custom",
    "fantasy_points_ppr",
    "player_id",
    "gsis_id",
    "pfr_player_id",
    "espn_id",
    "yahoo_id",
    "fantasypros_id",
    "sleeper_id",
    "rotowire_id",
    "sportradar_id",
    "player_display_name",
    "player_name",
    "first_name",
    "last_name",
    "recent_team",
    "opponent_team",
    "position",
    "headshot_url",
    "status",
}


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "player_week_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run features.py first.")
    return pd.read_parquet(path)


def add_target_next_week(df: pd.DataFrame) -> pd.DataFrame:
    """Create target column: next week's fantasy points for each player."""
    df = df.sort_values(["player_id", "season", "week"]).copy()

    if "fantasy_points_custom" not in df.columns:
        raise KeyError(
            "Column 'fantasy_points_custom' missing. Check features.py output."
        )

    df["target_points_next_week"] = df.groupby("player_id", group_keys=False)[
        "fantasy_points_custom"
    ].shift(-1)

    # Drop last week per player (no target available)
    return df[df["target_points_next_week"].notna()].copy()


def filter_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    return df[df["position"] == position].copy()


def _get_feature_cols(df: pd.DataFrame, position: str) -> list[str]:
    """
    Select feature columns for a given position by pattern matching.
    Includes position-specific stat patterns + universal context/injury patterns.
    Falls back to all numeric columns if no patterns match.
    """
    pos = (position or "").upper()
    stat_patterns = _POSITION_PATTERNS.get(pos, [])
    all_patterns = stat_patterns + _CONTEXT_PATTERNS

    exclude_id_prefixes = {"name", "id", "gsis", "pfr", "espn", "yahoo", "sleeper"}

    chosen: set[str] = set()
    for c in df.columns:
        if c in _DROP_EXACT:
            continue
        if c in _ALWAYS_ALLOW:
            chosen.add(c)
            continue
        lower = c.lower()
        # Skip obvious identifier columns
        if (
            any(token in lower for token in exclude_id_prefixes)
            and c not in _ALWAYS_ALLOW
        ):
            continue
        for p in all_patterns:
            if p in lower:
                chosen.add(c)
                break

    # Keep only numeric columns
    numeric = [c for c in sorted(chosen) if pd.api.types.is_numeric_dtype(df[c])]

    # Fallback: use all non-dropped numeric columns
    if not numeric:
        numeric = [
            c
            for c in df.columns
            if c not in _DROP_EXACT and pd.api.types.is_numeric_dtype(df[c])
        ]

    return numeric


def train_position_model(
    df: pd.DataFrame,
    position: str,
    registry: ModelRegistry | None = None,
) -> str | None:
    """
    Train a CatBoost model for a single position.

    Uses the last season as the validation set; all prior seasons for training.
    If a ModelRegistry is provided the model is versioned and the champion is
    updated; otherwise the flat models/{pos}_catboost.cbm path is written
    directly (backward-compatible fallback).

    Returns the version_id string if registry is provided, else None.
    """
    df_pos = df[df["position"] == position].copy()
    print(f"\n{'=' * 55}")
    print(f"  Training {position} model  (rows={len(df_pos):,})")
    print(f"{'=' * 55}")

    if df_pos.empty:
        print(f"  No data for {position}, skipping.")
        return None

    target_col = "target_points_next_week"
    max_season = int(df_pos["season"].max())
    train_df = df_pos[df_pos["season"] < max_season].copy()
    valid_df = df_pos[df_pos["season"] == max_season].copy()

    print(f"  Train seasons : {sorted(int(s) for s in train_df['season'].unique())}")
    print(f"  Valid season  : {max_season}")
    print(f"  Train rows    : {len(train_df):,}   Valid rows: {len(valid_df):,}")

    feature_cols = _get_feature_cols(train_df, position)
    cat_cols = [c for c in ["position_group", "season_type"] if c in feature_cols]

    # Add any remaining non-numeric columns as categoricals
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(train_df[c]) and c not in cat_cols:
            cat_cols.append(c)

    # season/week must never be categorical
    cat_cols = [c for c in cat_cols if c not in ("season", "week")]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[target_col].copy()

    for c in cat_cols:
        X_train[c] = X_train[c].astype("string").fillna("__NA__")
        X_valid[c] = X_valid[c].astype("string").fillna("__NA__")

    print(f"  Features      : {len(feature_cols)}  |  Categoricals: {cat_cols}")

    train_pool = Pool(X_train, y_train, cat_features=cat_cols if cat_cols else None)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols if cat_cols else None)

    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        iterations=2000,
        od_type="Iter",
        od_wait=100,
        random_seed=42,
        verbose=100,
    )
    model.fit(train_pool, eval_set=valid_pool)

    pred_valid = model.predict(valid_pool)
    valid_mae = float((abs(pred_valid - y_valid)).mean())
    print(f"  Validation MAE: {valid_mae:.4f} PPR points")

    meta = {
        "position": position,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "train_seasons": sorted(int(s) for s in train_df["season"].unique()),
        "valid_season": max_season,
        "valid_mae": valid_mae,
    }

    if registry is not None:
        return registry.register(position, model, meta)
    else:
        # Backward-compatible flat save
        model_path = MODEL_DIR / f"{position.lower()}_catboost.cbm"
        meta_path = MODEL_DIR / f"{position.lower()}_catboost_meta.json"
        model.save_model(str(model_path))
        pd.Series(meta).to_json(meta_path, indent=2)
        print(f"  Saved {model_path}")
        return None


def main(positions: list[str] | None = None, use_registry: bool = True) -> None:
    """
    Train models for all (or specified) positions.

    Parameters
    ----------
    positions    : list of positions to train; defaults to all in the data
    use_registry : if True, register each trained model in ModelRegistry
    """
    df = load_features()
    print("Loaded features:", df.shape)

    df = add_target_next_week(df)
    print("After adding target:", df.shape)

    if positions is None:
        positions = sorted(df["position"].dropna().unique().tolist())

    registry = ModelRegistry() if use_registry else None
    print("Training models for positions:", positions)

    for pos in positions:
        train_position_model(df, pos, registry=registry)

    if registry is not None:
        print("\nFinal model registry:")
        for pos in positions:
            registry.compare(pos)


if __name__ == "__main__":
    main()
