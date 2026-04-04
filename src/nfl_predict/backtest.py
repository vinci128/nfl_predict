"""
Walk-forward backtesting for NFL fantasy prediction models.

For each test season, trains a fresh CatBoost model on all prior seasons
and evaluates predictions against actual results, comparing against three
naive baselines:
  - last_week  : use previous week's actual score
  - roll3      : 3-week rolling mean
  - season_mean: season-to-date mean

Results are saved to backtest_results/{position}.parquet and a JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from nfl_predict.metrics import compute_metrics, print_metrics_table, top_n_precision

RESULTS_DIR = Path("backtest_results")

# Columns excluded when building the feature matrix for backtest folds
_DROP_EXACT = frozenset(
    [
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
        "position",
        "headshot_url",
        "status",
        "opponent_team",
        "recent_team",
    ]
)
_DROP_PATTERNS = frozenset(
    ["fg_made_list", "fg_missed_list", "fg_made_distance", "fg_missed_distance"]
)


# ---------------------------------------------------------------------------
# Baseline predictors (no training required)
# ---------------------------------------------------------------------------


def _baseline_last_week(df: pd.DataFrame) -> pd.Series:
    col = "fantasy_points_custom_lag1"
    return df[col].fillna(0) if col in df.columns else pd.Series(0.0, index=df.index)


def _baseline_roll3(df: pd.DataFrame) -> pd.Series:
    col = "fantasy_points_custom_roll3"
    return df[col].fillna(0) if col in df.columns else pd.Series(0.0, index=df.index)


def _baseline_season_mean(df: pd.DataFrame) -> pd.Series:
    col = "fantasy_points_custom_season_mean"
    return df[col].fillna(0) if col in df.columns else pd.Series(0.0, index=df.index)


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------


def _select_features(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in _DROP_EXACT
        and not any(p in c for p in _DROP_PATTERNS)
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def _prep_pool(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    target_col: str | None = None,
) -> Pool:
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("__NA__")
    y = df[target_col] if target_col and target_col in df.columns else None
    return Pool(X, label=y, cat_features=cat_cols)


def _train_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    iterations: int,
) -> tuple[CatBoostRegressor, list[str], np.ndarray]:
    """Train one walk-forward fold. Returns (model, feature_cols, predictions)."""
    target = "target_points_next_week"
    feature_cols = _select_features(train_df)
    cat_cols = [c for c in ["position_group", "season_type"] if c in feature_cols]

    train_pool = _prep_pool(train_df, feature_cols, cat_cols, target)
    valid_pool = _prep_pool(test_df, feature_cols, cat_cols, target)

    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        iterations=iterations,
        od_type="Iter",
        od_wait=50,
        random_seed=42,
        verbose=False,
    )
    model.fit(train_pool, eval_set=valid_pool)
    preds = model.predict(valid_pool)
    return model, feature_cols, preds


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------


def run_walk_forward_backtest(
    df: pd.DataFrame,
    position: str,
    test_seasons: list[int] | None = None,
    min_train_seasons: int = 4,
    model_iterations: int = 500,
    top_n: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """
    Walk-forward cross-validation across historical seasons.

    For each test season, trains a fresh model on all prior seasons and
    predicts every player-week in the test season. Results are compared
    against actual scores and three baselines.

    Parameters
    ----------
    df               : feature DataFrame with target_points_next_week column
    position         : e.g. "WR"
    test_seasons     : seasons to test on; defaults to all seasons with
                       >= min_train_seasons of prior history
    min_train_seasons: minimum training seasons required per fold
    model_iterations : CatBoost iterations per fold (lower = faster)
    top_n            : N for top-N precision metric

    Returns
    -------
    results_df : player-week predictions + actuals + baselines
    summary    : aggregate and per-season metrics dict
    """
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    target_col = "target_points_next_week"
    if target_col not in df.columns:
        raise ValueError(
            f"Column '{target_col}' not found. Call add_target_next_week() first."
        )

    df_pos = df[df["position"] == position].copy()
    all_seasons = sorted(int(s) for s in df_pos["season"].unique())

    if test_seasons is None:
        test_seasons = [
            s for s in all_seasons if all_seasons.index(s) >= min_train_seasons
        ]

    print(f"\n{'=' * 60}")
    print(f"  Walk-forward backtest: {position}")
    print(f"  Test seasons : {test_seasons}")
    print(f"  Iterations   : {model_iterations}")
    print(f"{'=' * 60}")

    fold_results: list[pd.DataFrame] = []

    for test_season in test_seasons:
        train_seasons = [s for s in all_seasons if s < test_season]
        if len(train_seasons) < min_train_seasons:
            print(f"  [{test_season}] skip — only {len(train_seasons)} train seasons")
            continue

        train_df = df_pos[
            df_pos["season"].isin(train_seasons) & df_pos[target_col].notna()
        ].copy()
        test_df = df_pos[
            (df_pos["season"] == test_season) & df_pos[target_col].notna()
        ].copy()

        if test_df.empty:
            print(f"  [{test_season}] skip — no test rows with targets")
            continue

        print(
            f"  [{test_season}] train={len(train_df):,}  test={len(test_df):,}",
            end="  ",
            flush=True,
        )

        _, feature_cols, model_preds = _train_fold(train_df, test_df, model_iterations)

        fold_mae = float(np.abs(model_preds - test_df[target_col].values).mean())
        print(f"fold_MAE={fold_mae:.4f}")

        row = test_df[["player_id", "season", "week", target_col]].copy()
        row = row.rename(columns={target_col: "actual_points"})

        for extra_col in ["player_display_name", "player_name", "recent_team"]:
            if extra_col in test_df.columns:
                row[extra_col] = test_df[extra_col].values

        row["model_pred"] = model_preds
        row["baseline_last_week"] = _baseline_last_week(test_df).values
        row["baseline_roll3"] = _baseline_roll3(test_df).values
        row["baseline_season_mean"] = _baseline_season_mean(test_df).values
        row["n_train_seasons"] = len(train_seasons)
        fold_results.append(row)

    if not fold_results:
        raise ValueError(f"No backtest folds produced for position={position}.")

    results_df = pd.concat(fold_results, ignore_index=True)
    y_true = results_df["actual_points"]

    # Aggregate metrics
    agg: dict[str, dict] = {
        "model": compute_metrics(y_true, results_df["model_pred"]),
        "baseline_last_week": compute_metrics(y_true, results_df["baseline_last_week"]),
        "baseline_roll3": compute_metrics(y_true, results_df["baseline_roll3"]),
        "baseline_season_mean": compute_metrics(
            y_true, results_df["baseline_season_mean"]
        ),
    }

    # Per-week top-N precision (mean across all weeks)
    week_precisions: list[float] = []
    for _, grp in results_df.groupby(["season", "week"]):
        if len(grp) >= top_n:
            week_precisions.append(
                top_n_precision(grp["actual_points"], grp["model_pred"], top_n)
            )
    agg["model"][f"top{top_n}_precision"] = (
        float(np.mean(week_precisions)) if week_precisions else float("nan")
    )

    # Per-season breakdown (keyed by season string for JSON compatibility)
    per_season: dict[str, dict] = {}
    for season_key, grp in results_df.groupby("season"):
        per_season[str(season_key)] = {
            "model": compute_metrics(grp["actual_points"], grp["model_pred"]),
            "baseline_last_week": compute_metrics(
                grp["actual_points"], grp["baseline_last_week"]
            ),
            "n": len(grp),
        }

    summary = {
        **agg,
        "per_season": per_season,
        "position": position,
        "test_seasons": test_seasons,
    }

    # Persist results
    out_parquet = RESULTS_DIR / f"{position.lower()}.parquet"
    results_df.to_parquet(out_parquet, index=False)

    out_json = RESULTS_DIR / f"{position.lower()}_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    _print_backtest_summary(summary, position, top_n)
    return results_df, summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_backtest_summary(summary: dict, position: str, top_n: int = 10) -> None:
    printable = {
        k: v
        for k, v in summary.items()
        if k not in ("per_season", "position", "test_seasons")
    }
    print_metrics_table(printable, title=f"Backtest results — {position}")

    model_mae = summary["model"]["mae"]
    best_baseline_mae = min(
        summary["baseline_last_week"]["mae"],
        summary["baseline_roll3"]["mae"],
        summary["baseline_season_mean"]["mae"],
    )
    improvement_pct = (best_baseline_mae - model_mae) / best_baseline_mae * 100
    print(f"  Model vs best baseline: {improvement_pct:+.2f}%")
    if f"top{top_n}_precision" in summary.get("model", {}):
        print(
            f"  Top-{top_n} weekly precision: {summary['model'][f'top{top_n}_precision']:.3f}"
        )

    print("\n  Per-season model MAE vs last-week baseline:")
    for season, m in sorted(summary.get("per_season", {}).items()):
        model_s = m["model"]["mae"]
        bl_s = m["baseline_last_week"]["mae"]
        delta = bl_s - model_s
        print(
            f"    {season}  model={model_s:.4f}  baseline_lw={bl_s:.4f}  Δ={delta:+.4f}  n={m['n']}"
        )


def load_backtest_results(position: str) -> tuple[pd.DataFrame, dict]:
    """Load previously saved backtest results for a position."""
    parquet_path = RESULTS_DIR / f"{position.lower()}.parquet"
    json_path = RESULTS_DIR / f"{position.lower()}_summary.json"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No backtest results for {position}. Run `nfl-predict backtest` first."
        )

    results_df = pd.read_parquet(parquet_path)
    summary = json.loads(json_path.read_text()) if json_path.exists() else {}
    return results_df, summary
