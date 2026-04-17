"""Shared evaluation metrics for fantasy point predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Compute regression metrics for fantasy point predictions.

    Returns:
        mae      – Mean Absolute Error (primary metric)
        rmse     – Root Mean Squared Error
        r2       – Coefficient of determination
        spearman – Spearman rank correlation (ranking quality)
        n        – Number of samples used
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]

    if len(yt) == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
            "n": 0,
        }

    mae = float(np.abs(yt - yp).mean())
    rmse = float(np.sqrt(((yt - yp) ** 2).mean()))

    ss_tot = ((yt - yt.mean()) ** 2).sum()
    r2 = float(1 - ((yt - yp) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")

    # Spearman via pandas (no scipy dependency)
    spearman = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))

    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spearman, "n": int(len(yt))}


def top_n_precision(
    y_true: pd.Series,
    y_pred: pd.Series,
    n: int = 10,
) -> float:
    """
    Fraction of the true top-N scorers that appear in the predicted top-N.
    Measures ranking quality for lineup decisions.
    """
    n = min(n, len(y_true))
    if n == 0:
        return float("nan")
    true_top = set(y_true.nlargest(n).index)
    pred_top = set(y_pred.nlargest(n).index)
    return len(true_top & pred_top) / n


def print_metrics_table(results: dict[str, dict[str, float]], title: str = "") -> None:
    """Pretty-print a {method: metrics_dict} comparison table."""
    if title:
        print(f"\n{'=' * 68}")
        print(f"  {title}")
    print(f"{'=' * 68}")
    print(f"{'Method':<26} {'MAE':>7} {'RMSE':>7} {'R²':>7} {'Spearman':>10} {'N':>6}")
    print("-" * 68)
    for method, m in results.items():
        print(
            f"{method:<26} "
            f"{m['mae']:7.4f} "
            f"{m['rmse']:7.4f} "
            f"{m['r2']:7.4f} "
            f"{m['spearman']:10.4f} "
            f"{m['n']:6d}"
        )
    print()
