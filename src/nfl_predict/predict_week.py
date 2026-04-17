# src/nfl_predict/predict_week.py

import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from catboost import CatBoostRegressor, Pool

app = typer.Typer()

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------
# Default SEASON & WEEK utilities
# -----------------------------------------------------------


def get_default_season_and_week(today: datetime.date | None = None):
    """
    Automatically computes the current season and week based on the NFL calendar.
    Works for all future seasons.
    """

    if today is None:
        today = datetime.date.today()
    year = today.year

    # If we're in January -> previous year's season
    season = year - 1 if today.month < 3 else year

    # Opening game: first Thursday of September on or after the 5th (day after Labor Day week)
    d = datetime.date(season, 9, 1)
    while d.day < 5 or d.weekday() != 3:  # weekday 3 = Thursday
        d += datetime.timedelta(days=1)
    season_start = d

    # Compute relative week
    if today < season_start:
        # offseason: week 1
        week = 1
    else:
        days_since_start = (today - season_start).days
        week = days_since_start // 7 + 1

    # Realistic cap (max 22 weeks including playoffs)
    week = max(1, min(22, week))

    return season, week


def get_default_season_and_week_from_data(df: pd.DataFrame):
    """
    Uses available data to choose a default season/week:
    - season = most recent season in data
    - week = last week in data + 1
    """
    if df.empty:
        raise ValueError(
            "player_week_features is empty, cannot determine defaults."
        )

    max_season = int(df["season"].max())
    df_season = df[df["season"] == max_season]
    max_week = int(df_season["week"].max())

    target_week = max_week + 1
    return max_season, target_week


# -----------------------------------------------------------
# Data loading
# -----------------------------------------------------------


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "player_week_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Have you already run features.py?")
    df = pd.read_parquet(path)
    return df


def load_model_and_meta(position: str = "WR"):
    model_path = MODEL_DIR / f"{position.lower()}_catboost.cbm"
    meta_path = MODEL_DIR / f"{position.lower()}_catboost_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    model = CatBoostRegressor()
    model.load_model(model_path)

    meta = pd.read_json(meta_path, typ="series")
    feature_cols = list(meta["feature_cols"])
    cat_cols = list(meta["cat_cols"])

    return model, feature_cols, cat_cols, meta


# -----------------------------------------------------------
# Build inference dataset
# -----------------------------------------------------------


def build_inference_dataset(
    df: pd.DataFrame,
    season: int,
    target_week: int,
    position: str,
    feature_cols,
    cat_cols,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses features from week (target_week - 1) to predict target_week.
    If the feature week is partial (some players don't yet have a row
    for that week), we use each player's last available row with
    week <= feature_week (fallback to the most recent available week).
    """

    feature_week = target_week - 1
    if feature_week < 1:
        raise ValueError(
            "Cannot predict week 1: at least one prior week is required."
        )

    df_season = df[df["season"] == season].copy()
    df_pos = df_season[df_season["position"] == position].copy()

    # Consider all rows up to feature_week (inclusive). For players
    # who don't yet have a row in feature_week, we take their last
    # available row (e.g. week-1, week-2, ...)
    df_up_to = df_pos[df_pos["week"] <= feature_week].copy()

    if df_up_to.empty:
        raise ValueError(
            f"No features found for season={season} up to week={feature_week}. "
            "Download more recent data?"
        )

    # Choose the player identifier: prefer `player_id`, otherwise
    # `player_display_name` or `player_name`.
    id_col = None
    for cand in ["player_id", "player_display_name", "player_name"]:
        if cand in df_up_to.columns:
            id_col = cand
            break

    if id_col is None:
        # No identifier column available; group by index and take the last row
        # (unlikely but handled)
        idx = df_up_to.groupby(df_up_to.index)["week"].idxmax()
        df_feat = df_up_to.loc[idx].copy()
    else:
        idx = df_up_to.groupby(id_col)["week"].idxmax()
        df_feat = df_up_to.loc[idx].copy()

    # Warn if not all players come from feature_week (partial week)
    weeks_used = sorted(df_feat["week"].unique())
    if not (len(weeks_used) == 1 and weeks_used[0] == feature_week):
        # Not an error: expected behaviour when the week is partial
        print(
            f"Note: some players are using data from earlier weeks. "
            f"Weeks used for features: {weeks_used[:5]}{'...' if len(weeks_used) > 5 else ''}"
        )

    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise KeyError(f"Missing features during inference: {missing}")

    X = df_feat[feature_cols].copy()

    # Normalize categorical columns. Also detect any non-numeric
    # feature columns that weren't listed in meta's `cat_cols` and
    # treat them as categorical to avoid CatBoost trying to convert
    # list-like strings (e.g. '39;54') to floats.
    import pandas as _pd

    inferred_cat_cols = list(cat_cols) if cat_cols is not None else []
    for c in feature_cols:
        if (
            c in X.columns
            and not _pd.api.types.is_numeric_dtype(X[c])
            and c not in inferred_cat_cols
        ):
            inferred_cat_cols.append(c)

    for c in inferred_cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("__NA__")

    # return inferred_cat_cols alongside X if caller needs it? We keep
    # original signature but the caller (`make_predictions`) will compute
    # cat indices from X columns where necessary.

    return df_feat, X


# -----------------------------------------------------------
# Predictions
# -----------------------------------------------------------


def make_predictions(
    df_feat: pd.DataFrame,
    X: pd.DataFrame,
    model: CatBoostRegressor,
    cat_cols,
    season: int,
    target_week: int,
    position: str,
) -> pd.DataFrame:
    cat_idx = [X.columns.get_loc(c) for c in cat_cols if c in X.columns]
    pool = Pool(X, cat_features=cat_idx if cat_idx else None)
    preds = model.predict(pool)

    # Player name column
    name_col = None
    for cand in ["player_display_name", "player_name"]:
        if cand in df_feat.columns:
            name_col = cand
            break

    team_col = None
    for cand in ["recent_team", "team"]:
        if cand in df_feat.columns:
            team_col = cand
            break

    out = pd.DataFrame(
        {
            "season": season,
            "feature_week": df_feat["week"].values,
            "predicted_week": target_week,
            "player_id": df_feat.get(
                "player_id", pd.Series([None] * len(df_feat))
            ).values,
            "player_name": df_feat[name_col].values if name_col else None,
            "team": df_feat[team_col].values if team_col else None,
            "position": position,
            "expected_ppr_points": preds,
        }
    )

    return out.sort_values("expected_ppr_points", ascending=False).reset_index(
        drop=True
    )


# -----------------------------------------------------------
# Typer CLI
# -----------------------------------------------------------
def _clean_option(value: Any) -> Any:
    """
    If a Typer OptionInfo arrives (from incorrect calls), treat it as None.
    This makes us resilient if someone accidentally calls the Typer command from code.
    """
    from typer.models import OptionInfo

    if isinstance(value, OptionInfo):
        return None
    return value


def run_predictions(
    season: int | None = None,
    week: int | None = None,
    position: str = "WR",
):
    """
    Core function: runs predictions without depending on Typer.
    Can be called from the Typer CLI or from other Python modules.
    """

    # Guard against OptionInfo in case of incorrect calls
    season = _clean_option(season)
    week = _clean_option(week)
    position = _clean_option(position) or "WR"

    # 1) Load features
    df = load_features()

    # 2) If season/week are missing, use data-based defaults
    if season is None or week is None:
        data_season, data_week = get_default_season_and_week_from_data(df)
        season = season or data_season
        week = week or data_week

    season = int(season)
    week = int(week)

    print("\n=== Fantasy Predictions ===")
    print(f"Season (default from data): {season}")
    print(f"Target week: {week} (uses week {week - 1} as feature)")
    print(f"Position: {position}\n")

    model, feature_cols, cat_cols, meta = load_model_and_meta(position=position)

    df_feat, X = build_inference_dataset(
        df=df,
        season=season,
        target_week=week,
        position=position,
        feature_cols=feature_cols,
        cat_cols=cat_cols,
    )

    print(f"Found {len(df_feat)} {position} players for week {week - 1}")

    preds = make_predictions(
        df_feat=df_feat,
        X=X,
        model=model,
        cat_cols=cat_cols,
        season=season,
        target_week=week,
        position=position,
    )

    out_path = OUT_DIR / f"predictions_{position.lower()}_{season}_week{week}.csv"
    preds.to_csv(out_path, index=False)

    print(f"\nSaved CSV to: {out_path}\n")
    print("Top 10:\n")
    print(
        preds[["player_name", "team", "expected_ppr_points"]]
        .head(10)
        .to_string(index=False)
    )

    return preds


# -----------------------------------------------------------
# Typer CLI wrapper
# -----------------------------------------------------------


@app.command()
def predict(
    season: int | None = typer.Option(
        None,
        help="NFL season. Default = most recent season in data.",
    ),
    week: int | None = typer.Option(
        None,
        help="Week to predict. Default = next week relative to data.",
    ),
    position: str = typer.Option("WR", help="Position: WR, RB, QB, TE"),
):
    """Fantasy next-week prediction (Typer wrapper)."""
    run_predictions(season=season, week=week, position=position)


if __name__ == "__main__":
    app()
