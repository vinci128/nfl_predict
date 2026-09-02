"""
Season-total CatBoost models for fantasy football draft projections.

Trains one model per position × quantile (p10 / p50 / p90) using
end-of-season feature snapshots built by season_features.py.

The target is the player's *total* fantasy points in the following season.
Models are saved to models/{pos}_season_{label}.cbm and optionally registered
in the ModelRegistry under position keys like "WR_SEASON_P50".
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool

from nfl_predict.model_registry import ModelRegistry
from nfl_predict.season_features import (
    TARGET_COLS,
    build_all_inference_rows,
    build_season_snapshot,
    load_features,
    load_injuries,
    load_rosters,
)

MODEL_DIR = Path("models")

POSITIONS = ["QB", "RB", "WR", "TE", "K"]
QUANTILES = [0.1, 0.5, 0.9]

# Season total, plus the two halves it decomposes into: scoring rate and
# availability. The total is modelled directly rather than as ppg x games —
# the median of a product is not the product of the medians, so multiplying
# the component medians is a biased estimator (measurably worse MAE for QB).
# The component models exist to report *why* a projection is what it is:
# an elite rate paired with a low games estimate reads very differently on a
# draft board than a mediocre rate over a full season, though both can
# produce the same total.
TOTAL_TARGET = "season_total_pts_next"
PPG_TARGET = "season_ppg_next"
GAMES_TARGET = "games_played_next"
TARGETS = (TOTAL_TARGET, PPG_TARGET, GAMES_TARGET)

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
    # IDP. Tackle volume is the dominant term in this scoring, and tackle
    # volume is mostly a function of snaps and role, so defensive snap share
    # matters as much for a linebacker as target share does for a receiver.
    "LB": ["def_", "tackle", "sack", "snap", "fantasy_points"],
    "DL": ["def_", "tackle", "sack", "qb_hit", "snap", "fantasy_points"],
    "DB": ["def_", "tackle", "pass_defended", "interception", "snap", "fantasy_points"],
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
    *TARGET_COLS,
}

# Maximum regular-season games a player can be projected for. 17 in the modern
# schedule; a mid-season trade between teams with different bye weeks can push
# the observed count to 18, so clip predictions rather than observations.
MAX_GAMES = 17


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_TARGET_SLUGS = {
    TOTAL_TARGET: "total",
    PPG_TARGET: "ppg",
    GAMES_TARGET: "games",
}


def _target_slug(target: str) -> str:
    """Short name used in model filenames and registry keys."""
    try:
        return _TARGET_SLUGS[target]
    except KeyError:
        raise ValueError(
            f"target must be one of {sorted(_TARGET_SLUGS)}, got {target!r}"
        ) from None


def _season_registry_key(
    position: str, quantile: float | None = None, target: str = PPG_TARGET
) -> str:
    """Registry key for a season model, e.g. 'WR_SEASON_PPG_P50'."""
    base = f"{position.upper()}_SEASON_{_target_slug(target).upper()}"
    if quantile is None:
        return base
    return f"{base}_P{int(quantile * 100)}"


def _quantile_label(quantile: float | None) -> str:
    return f"p{int(quantile * 100)}" if quantile is not None else "rmse"


def _model_dir(league: str | None = None) -> Path:
    """Where this league's season models live.

    Models are namespaced by league because the target — fantasy points under
    that league's rules — differs between them. A model trained on one
    league's scoring produces confidently wrong numbers for another.
    """
    from nfl_predict.leagues import get_profile

    return get_profile(league).model_dir


def _model_paths(
    position: str, target: str, label: str, league: str | None = None
) -> tuple[Path, Path]:
    """Return (model_path, meta_path) for a position/target/quantile."""
    stem = f"{position.lower()}_season_{_target_slug(target)}_{label}"
    d = _model_dir(league)
    return d / f"{stem}.cbm", d / f"{stem}_meta.json"


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
    league: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build train / validation DataFrames for the season model.

    Split: train = all seasons before max_season, valid = max_season.
    Only rows with a valid (> 0) next-season target and at least one game
    played next season are included — per-game rate is undefined otherwise.

    Returns
    -------
    df_train, df_valid, feature_cols
    """
    df_weekly = load_features(league)
    rosters = load_rosters()
    snapshot = build_season_snapshot(df_weekly, rosters=rosters)

    pos_snap = snapshot[
        (snapshot["position"] == position.upper())
        & snapshot["season_total_pts_next"].notna()
        & (snapshot["season_total_pts_next"] > 0)
        & (snapshot["games_played_next"] > 0)
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
    target: str = PPG_TARGET,
    league: str | None = None,
) -> str | None:
    """
    Train a CatBoost season model for a position.

    Three targets are modelled independently: the season total (the number
    the draft board ranks on), the scoring rate, and games played. See the
    TARGETS comment for why the total is not derived from the other two.

    Parameters
    ----------
    position  : "WR", "RB", "QB", "TE", or "K"
    quantile  : None → RMSE loss; 0.1 / 0.5 / 0.9 → quantile regression
    registry  : if provided, version and register the model
    iterations: CatBoost max iterations
    depth     : tree depth (shallow = less overfitting on small datasets)
    target    : one of ``TARGETS``

    Returns
    -------
    version_id if registered, else None
    """
    _target_slug(target)  # validates

    df_train, df_valid, feature_cols = build_training_data(position, league)

    X_train = df_train[feature_cols].fillna(0)
    y_train = df_train[target]
    X_valid = df_valid[feature_cols].fillna(0)
    y_valid = df_valid[target]

    # A rate observed over 2 games is far noisier than one over 17. Weighting
    # by games played is the variance-correct fix (var of a mean scales 1/n)
    # and keeps the short seasons in the data, unlike a minimum-games cutoff.
    w_train = df_train[GAMES_TARGET] if target == PPG_TARGET else None
    w_valid = df_valid[GAMES_TARGET] if target == PPG_TARGET else None

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
        Pool(X_train, label=y_train, weight=w_train),
        eval_set=Pool(X_valid, label=y_valid, weight=w_valid),
    )

    preds = model.predict(X_valid)
    mae = float((preds - y_valid).abs().mean())
    rmse = float(((preds - y_valid) ** 2).mean() ** 0.5)

    label = _quantile_label(quantile)
    kind = _target_slug(target)
    print(
        f"  [{position}] season {kind} ({label}) — "
        f"val MAE={mae:.2f}  RMSE={rmse:.2f}  "
        f"n_train={len(y_train)}  n_valid={len(y_valid)}"
    )

    # Always persist to a flat file (primary load path for predict_season),
    # plus a meta sidecar recording the exact feature set the model was
    # trained on — predict_season must use this, not a freshly recomputed
    # feature list, or columns silently misalign after a data update.
    _model_dir(league).mkdir(exist_ok=True, parents=True)
    flat_path, flat_meta_path = _model_paths(position, target, label, league)
    model.save_model(str(flat_path))
    flat_meta_path.write_text(
        json.dumps(
            {
                "position": position.upper(),
                "quantile": quantile,
                "target": target,
                "feature_cols": feature_cols,
                "valid_mae": mae,
                "valid_rmse": rmse,
            },
            indent=2,
        )
    )

    if registry is not None:
        reg_key = _season_registry_key(position, quantile, target)
        meta = {
            "model_type": f"season_{_target_slug(target)}",
            "target": target,
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
    league: str | None = None,
) -> dict[str, str | None]:
    """
    Train p10/p50/p90 rate and availability models for a single position.

    Returns a dict mapping "{target_slug}_{label}" → version_id
    (None when no registry).
    """
    results: dict[str, str | None] = {}
    for target in TARGETS:
        kind = _target_slug(target)
        for q in QUANTILES:
            label = _quantile_label(q)
            print(f"\nTraining {position} season {kind} model ({label})...")
            results[f"{kind}_{label}"] = train_season_model(
                position,
                quantile=q,
                registry=registry,
                iterations=iterations,
                target=target,
                league=league,
            )
    return results


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_season(
    position: str,
    as_of_season: int,
    quantiles: list[float] | None = None,
    league: str | None = None,
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
        injuries=load_injuries(),
    )

    if inference.empty:
        print(f"  No players found for {position} in season {as_of_season}")
        return pd.DataFrame()

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
            # Injury-report summary from the source season — carried through
            # for display on the draft board, not used by any model.
            "inj_weeks_out",
            "inj_weeks_on_report",
            "inj_primary",
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

    # Load and apply each quantile model, aligning features to the exact
    # set the model was trained on (from its meta sidecar). Falls back to
    # recomputing from training data for models saved before sidecars existed.
    fallback_cols: list[str] | None = None

    def _apply(target: str, q: float) -> pd.Series | None:
        nonlocal fallback_cols
        label = _quantile_label(q)
        model_path, meta_path = _model_paths(position, target, label, league)

        if not model_path.exists():
            print(
                f"  Model not found: {model_path}. Run `nfl-predict draft-prep` first."
            )
            return None

        if meta_path.exists():
            feature_cols = json.loads(meta_path.read_text())["feature_cols"]
        else:
            if fallback_cols is None:
                print(
                    f"  No meta sidecar for {model_path.name} — recomputing feature "
                    "columns from training data (retrain with `nfl-predict draft-prep` "
                    "to pin them)."
                )
                _, _, fallback_cols = build_training_data(position, league)
            feature_cols = fallback_cols

        missing = [c for c in feature_cols if c not in inference.columns]
        if missing:
            print(
                f"  WARN: {len(missing)} training features missing from inference "
                f"data for {model_path.name} (filled with 0): {missing[:5]}"
            )
        X = inference.reindex(columns=feature_cols, fill_value=0).fillna(0)

        m = CatBoostRegressor()
        m.load_model(str(model_path))
        return pd.Series(m.predict(X), index=inference.index)

    for q in quantiles:
        pct = int(q * 100)

        total = _apply(TOTAL_TARGET, q)
        projections[f"proj_p{pct}"] = (
            float("nan") if total is None else total.clip(lower=0).round(1).to_numpy()
        )

        ppg = _apply(PPG_TARGET, q)
        projections[f"proj_ppg_p{pct}"] = (
            float("nan") if ppg is None else ppg.clip(lower=0).round(2).to_numpy()
        )

        games = _apply(GAMES_TARGET, q)
        projections[f"proj_games_p{pct}"] = (
            float("nan")
            if games is None
            else games.clip(lower=0, upper=MAX_GAMES).round(1).to_numpy()
        )

    return _enforce_quantile_order(projections, quantiles)


def _enforce_quantile_order(
    projections: pd.DataFrame, quantiles: list[float]
) -> pd.DataFrame:
    """
    Stop a low quantile predicting above a high one.

    The three quantiles are separate models with no constraint between them,
    so for a player they disagree about they can cross: Davante Adams came out
    with a p90 below his p50, and he ranks inside the top 30 on every board --
    a visible "ceiling lower than median" on the draft UI.

    The median is what VOR ranks on and the only one walk-forward validated, so
    it is left alone and the band is clamped around it. Sorting all three would
    re-rank players on the say-so of the least reliable model.
    """
    if 0.5 not in quantiles:
        return projections

    for stat in ("proj", "proj_ppg", "proj_games"):
        mid = f"{stat}_p50"
        if mid not in projections.columns:
            continue
        for q in quantiles:
            pct = int(q * 100)
            col = f"{stat}_p{pct}"
            if col == mid or col not in projections.columns:
                continue
            if q < 0.5:
                projections[col] = projections[[col, mid]].min(axis=1)
            else:
                projections[col] = projections[[col, mid]].max(axis=1)

    return projections


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    positions: list[str] | None = None,
    use_registry: bool = True,
    iterations: int = 500,
    league: str | None = None,
) -> None:
    """Train season projection models for all (or selected) positions."""
    from nfl_predict.leagues import get_profile

    profile = get_profile(league)
    registry = ModelRegistry(profile.registry_path) if use_registry else None
    # Only train what this league can actually start.
    pos_list = positions or [p for p in profile.roster.positions if p != "DST"]
    print(f"Training season models for {profile.name} [{profile.key}]: {pos_list}")

    for pos in pos_list:
        print(f"\n{'=' * 50}")
        print(f"  Season model: {pos}")
        print(f"{'=' * 50}")
        try:
            train_all_quantiles(
                pos, registry=registry, iterations=iterations, league=profile.key
            )
        except ValueError as e:
            print(f"  Skipping {pos}: {e}")
