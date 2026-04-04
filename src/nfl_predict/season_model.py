"""
Season-total CatBoost models for fantasy football draft projections.

Trains one model per position × quantile (p10 / p50 / p90) using
end-of-season feature snapshots built by season_features.py.

The target is the player's *total* fantasy points in the following season.
Models are saved to models/{pos}_season_{label}.cbm and optionally registered
in the ModelRegistry under position keys like "WR_SEASON_P50".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool

from nfl_predict.model_registry import ModelRegistry
from nfl_predict.season_features import (
    build_all_inference_rows,
    build_season_snapshot,
    load_features,
    load_rosters,
)

MODEL_DIR = Path("models")

POSITIONS = ["QB", "RB", "WR", "TE", "K"]
QUANTILES = [0.1, 0.5, 0.9]

# Position-specific column patterns for feature selection
_POS_PATTERNS: dict[str, list[str]] = {
    "QB": [
        "passing",
        "interception",
        "sack",
        "rushing",
        "air_yards",
        "epa",
        "fantasy_points",
    ],
    "RB": ["rushing", "carry", "snap", "target", "reception", "fantasy_points"],
    "WR": [
        "receiv",
        "target",
        "air_yards",
        "target_share",
        "wopr",
        "snap",
        "fantasy_points",
    ],
    "TE": ["receiv", "target", "snap", "fantasy_points"],
    "K": ["fg_", "pat_", "fgm", "fg_long", "fg_made", "fg_att", "fantasy_points"],
}

_UNIVERSAL = ["age_at_season_start", "years_exp", "games_played"]

_DROP_EXACT = {
    "player_id",
    "gsis_id",
    "player_display_name",
    "player_name",
    "recent_team",
    "team",
    "position",
    "season",
    "season_total_pts_next",
    "season_total_pts_current",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _season_registry_key(position: str, quantile: float | None = None) -> str:
    """Registry key for a season model, e.g. 'WR_SEASON_P50'."""
    if quantile is None:
        return f"{position.upper()}_SEASON"
    return f"{position.upper()}_SEASON_P{int(quantile * 100)}"


def _quantile_label(quantile: float | None) -> str:
    return f"p{int(quantile * 100)}" if quantile is not None else "rmse"


def _get_season_feature_cols(df: pd.DataFrame, position: str) -> list[str]:
    """
    Select numeric feature columns for the season model.

    Includes:
    - Columns matching position-specific stat patterns
    - Universal features (age, experience, games played)

    Excludes identifier and target columns.
    """
    patterns = _POS_PATTERNS.get(position.upper(), ["fantasy_points"])
    drop_id_prefixes = (
        "gsis",
        "pfr",
        "espn",
        "yahoo",
        "sleeper",
        "rotowire",
        "sportradar",
    )

    chosen: list[str] = []
    for col in df.columns:
        if col in _DROP_EXACT:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        lower = col.lower()
        if any(lower.startswith(p) for p in drop_id_prefixes):
            continue
        # Always include universal features
        if any(u in lower for u in _UNIVERSAL):
            chosen.append(col)
            continue
        # Position-specific patterns
        if any(p in lower for p in patterns):
            chosen.append(col)

    return sorted(set(chosen))


# ---------------------------------------------------------------------------
# Training data builder
# ---------------------------------------------------------------------------


def build_training_data(
    position: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build train / validation DataFrames for the season model.

    Split: train = all seasons before max_season, valid = max_season.
    Only rows with a valid (> 0) next-season target are included.

    Returns
    -------
    df_train, df_valid, feature_cols
    """
    df_weekly = load_features()
    rosters = load_rosters()
    snapshot = build_season_snapshot(df_weekly, rosters=rosters)

    pos_snap = snapshot[
        (snapshot["position"] == position.upper())
        & snapshot["season_total_pts_next"].notna()
        & (snapshot["season_total_pts_next"] > 0)
    ].copy()

    if len(pos_snap) < 50:
        raise ValueError(
            f"Not enough training data for {position}: {len(pos_snap)} rows "
            f"(need ≥ 50)."
        )

    feature_cols = _get_season_feature_cols(pos_snap, position)
    feature_cols = [c for c in feature_cols if c in pos_snap.columns]

    max_season = int(pos_snap["season"].max())
    df_train = pos_snap[pos_snap["season"] < max_season].copy()
    df_valid = pos_snap[pos_snap["season"] == max_season].copy()

    return df_train, df_valid, feature_cols


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_season_model(
    position: str,
    quantile: float | None = None,
    registry: ModelRegistry | None = None,
    iterations: int = 500,
    depth: int = 4,
) -> str | None:
    """
    Train a CatBoost season-total model for a position.

    Parameters
    ----------
    position  : "WR", "RB", "QB", "TE", or "K"
    quantile  : None → RMSE loss; 0.1 / 0.5 / 0.9 → quantile regression
    registry  : if provided, version and register the model
    iterations: CatBoost max iterations
    depth     : tree depth (shallow = less overfitting on small datasets)

    Returns
    -------
    version_id if registered, else None
    """
    df_train, df_valid, feature_cols = build_training_data(position)

    target = "season_total_pts_next"
    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train[target]
    X_valid = df_valid[feature_cols].fillna(0)
    y_valid = df_valid[target]

    loss_function = f"Quantile:alpha={quantile}" if quantile is not None else "RMSE"

    model = CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        loss_function=loss_function,
        l2_leaf_reg=5,
        verbose=False,
        random_seed=42,
        early_stopping_rounds=50,
    )

    model.fit(
        Pool(X_train, label=y_train),
        eval_set=Pool(X_valid, label=y_valid),
    )

    preds = model.predict(X_valid)
    mae = float((preds - y_valid).abs().mean())
    rmse = float(((preds - y_valid) ** 2).mean() ** 0.5)

    label = _quantile_label(quantile)
    print(
        f"  [{position}] season ({label}) — "
        f"val MAE={mae:.1f}  RMSE={rmse:.1f}  "
        f"n_train={len(y_train)}  n_valid={len(y_valid)}"
    )

    # Always persist to a flat file (primary load path for predict_season)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    flat_path = MODEL_DIR / f"{position.lower()}_season_{label}.cbm"
    model.save_model(str(flat_path))

    if registry is not None:
        reg_key = _season_registry_key(position, quantile)
        meta = {
            "model_type": "season",
            "feature_cols": feature_cols,
            "cat_cols": [],
            "train_seasons": sorted(int(s) for s in df_train["season"].unique()),
            "valid_season": int(df_train["season"].max()) + 1,
            "valid_mae": mae,
            "valid_rmse": rmse,
            "quantile": quantile,
            "quantile_label": label,
            "position": position.upper(),
            "flat_model_path": str(flat_path),
        }
        return registry.register(
            position=reg_key,
            model=model,
            meta=meta,
            auto_promote=True,
        )

    return None


def train_all_quantiles(
    position: str,
    registry: ModelRegistry | None = None,
    iterations: int = 500,
) -> dict[str, str | None]:
    """
    Train p10, p50, and p90 models for a single position.

    Returns a dict mapping label → version_id (None when no registry).
    """
    results: dict[str, str | None] = {}
    for q in QUANTILES:
        label = _quantile_label(q)
        print(f"\nTraining {position} season model ({label})...")
        version_id = train_season_model(
            position, quantile=q, registry=registry, iterations=iterations
        )
        results[label] = version_id
    return results


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_season(
    position: str,
    as_of_season: int,
    quantiles: list[float] | None = None,
) -> pd.DataFrame:
    """
    Generate full-season projections for all active players of a position.

    Loads pre-trained quantile models from the flat model paths and applies
    them to end-of-``as_of_season`` feature snapshots.

    Parameters
    ----------
    position    : "WR", "RB", "QB", "TE", or "K"
    as_of_season: most recently completed season (features source)
    quantiles   : quantiles to predict; defaults to [0.1, 0.5, 0.9]

    Returns
    -------
    DataFrame with columns: player_id, player_name, team, position,
    projected_season, proj_p10, proj_p50, proj_p90
    """
    if quantiles is None:
        quantiles = QUANTILES

    df_weekly = load_features()
    rosters = load_rosters()
    inference = build_all_inference_rows(
        df_weekly,
        as_of_season=as_of_season,
        position=position,
        rosters=rosters,
    )

    if inference.empty:
        print(f"  No players found for {position} in season {as_of_season}")
        return pd.DataFrame()

    # Get feature cols aligned with training
    _, _, feature_cols = build_training_data(position)
    feature_cols = [c for c in feature_cols if c in inference.columns]
    X = inference[feature_cols].fillna(0)

    # Start result with identifier columns
    id_cols = [
        c
        for c in (
            "player_id",
            "player_display_name",
            "player_name",
            "recent_team",
            "position",
            "season",
        )
        if c in inference.columns
    ]
    projections = inference[id_cols].copy()

    # Normalize display columns
    if "player_display_name" in projections.columns:
        projections["player_name"] = projections["player_display_name"]
        projections.drop(columns=["player_display_name"], inplace=True)
    if "recent_team" in projections.columns:
        projections.rename(columns={"recent_team": "team"}, inplace=True)

    projections["projected_season"] = as_of_season + 1

    # Load and apply each quantile model
    for q in quantiles:
        label = _quantile_label(q)
        model_path = MODEL_DIR / f"{position.lower()}_season_{label}.cbm"
        col_name = f"proj_p{int(q * 100)}"

        if not model_path.exists():
            print(
                f"  Model not found: {model_path}. Run `nfl-predict draft-prep` first."
            )
            projections[col_name] = float("nan")
            continue

        m = CatBoostRegressor()
        m.load_model(str(model_path))
        projections[col_name] = m.predict(X).clip(min=0).round(1)

    return projections


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    positions: list[str] | None = None,
    use_registry: bool = True,
    iterations: int = 500,
) -> None:
    """Train season projection models for all (or selected) positions."""
    registry = ModelRegistry() if use_registry else None
    pos_list = positions or POSITIONS

    for pos in pos_list:
        print(f"\n{'=' * 50}")
        print(f"  Season model: {pos}")
        print(f"{'=' * 50}")
        try:
            train_all_quantiles(pos, registry=registry, iterations=iterations)
        except ValueError as e:
            print(f"  Skipping {pos}: {e}")
