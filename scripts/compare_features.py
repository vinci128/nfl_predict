"""
Compare model MAE: baseline features vs. enhanced features (new context signals).

Trains a CatBoost WR model twice on the same train/validation split:
  - baseline: drops all new context columns before training
  - enhanced: uses all features including Vegas, weather, rest, injuries, bye week

Run from the project root:
    uv run python scripts/compare_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool

# Ensure src/ is on the path when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nfl_predict.features import build_player_week_features
from nfl_predict.train_model import add_target_next_week

# New context columns added by the enhanced pipeline
NEW_CONTEXT_COLS = [
    "team_spread",
    "total_line",
    "temp",
    "wind",
    "is_dome",
    "is_home",
    "rest_days",
    "bye_next_week",
    "injury_status",
    "practice_status_enc",
]


def _train_and_eval(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label: str,
) -> float:
    """Train a CatBoost regressor and return validation MAE."""
    target_col = "target_points_next_week"

    drop_exact = [
        target_col,
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
        "position",
    ]
    drop_patterns = [
        "fg_made_list",
        "fg_missed_list",
        "fg_made_distance",
        "fg_missed_distance",
    ]

    def _feature_cols(df: pd.DataFrame) -> list[str]:
        cols = [c for c in df.columns if c not in drop_exact]
        cols = [c for c in cols if not any(p in c for p in drop_patterns)]
        # Keep only numeric columns
        return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    feature_cols = _feature_cols(train_df)
    cat_cols = [
        c for c in ["position_group", "season_type", "team"] if c in feature_cols
    ]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[target_col].copy()

    for c in cat_cols:
        X_train[c] = X_train[c].astype(str).fillna("__NA__")
        X_valid[c] = X_valid[c].astype(str).fillna("__NA__")

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_cols)

    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        iterations=2000,
        od_type="Iter",
        od_wait=100,
        random_seed=42,
        verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool)

    preds = model.predict(valid_pool)
    mae = float((abs(preds - y_valid)).mean())
    print(f"[{label}] features={len(feature_cols):3d}  val_MAE={mae:.4f}")

    # Top 10 feature importances
    fi = (
        pd.Series(
            model.get_feature_importance(),
            index=feature_cols,
        )
        .sort_values(ascending=False)
        .head(10)
    )
    print(f"  Top features: {fi.index.tolist()}")
    return mae


def main() -> None:
    print("Building feature set...")
    df = build_player_week_features(save=True)

    print("Adding prediction target...")
    df = add_target_next_week(df)

    # Focus on WR for a clean, fast comparison
    position = "WR"
    df_pos = df[df["position"] == position].copy()
    max_season = df_pos["season"].max()
    train_df = df_pos[df_pos["season"] < max_season].copy()
    valid_df = df_pos[df_pos["season"] == max_season].copy()

    print(f"\nPosition: {position}")
    print(f"Train seasons: {sorted(train_df['season'].unique())}")
    print(f"Validation season: {max_season}")
    print(f"Train rows: {len(train_df)}, Validation rows: {len(valid_df)}\n")

    # --- Baseline: drop all new context columns (and their derivatives) ---
    baseline_drop = [
        c
        for c in train_df.columns
        if any(c == base or c.startswith(base + "_") for base in NEW_CONTEXT_COLS)
    ]
    train_baseline = train_df.drop(
        columns=[c for c in baseline_drop if c in train_df.columns]
    )
    valid_baseline = valid_df.drop(
        columns=[c for c in baseline_drop if c in valid_df.columns]
    )

    print("=" * 55)
    print("BASELINE (no game context / injury features)")
    print("=" * 55)
    mae_baseline = _train_and_eval(train_baseline, valid_baseline, "baseline")

    # --- Enhanced: use all features ---
    print("\n" + "=" * 55)
    print("ENHANCED (Vegas + weather + rest + injuries + bye week)")
    print("=" * 55)
    mae_enhanced = _train_and_eval(train_df, valid_df, "enhanced")

    # --- Summary ---
    delta = mae_baseline - mae_enhanced
    pct = delta / mae_baseline * 100
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"  Baseline MAE : {mae_baseline:.4f}")
    print(f"  Enhanced MAE : {mae_enhanced:.4f}")
    print(f"  Improvement  : {delta:+.4f}  ({pct:+.2f}%)")
    if delta > 0:
        print("  ✓ Enhanced features reduce prediction error.")
    elif delta < 0:
        print("  ✗ New features did not help on this validation set.")
    else:
        print("  = No change.")


if __name__ == "__main__":
    main()
