# src/nfl_predict/predict_week.py

import datetime
from pathlib import Path
from typing import Optional

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

def get_default_season_and_week():
    """
    Calcola automaticamente la season e la week correnti in base al calendario NFL.
    Funziona per tutte le stagioni future.
    """

    today = datetime.date.today()
    year = today.year

    # Se siamo a gennaio -> stagione dell'anno precedente
    if today.month < 3:
        season = year - 1
    else:
        season = year

    # Inizio NFL 2024 (regola: primo TNF)
    # Per stagioni future basterà aggiornare questa data o potremmo
    # costruire un file con il calendario. Per ora semplice.
    season_start = datetime.date(2024, 9, 5)  # Opening game

    # Calcolo week relativa
    if today < season_start:
        # offseason: week 1
        week = 1
    else:
        days_since_start = (today - season_start).days
        week = days_since_start // 7 + 1

    # Limitazione realistica (max 22 weeks incluse playoffs)
    week = max(1, min(22, week))

    return season, week

def get_default_season_and_week_from_data(df: pd.DataFrame):
    """
    Usa i dati disponibili per scegliere una default season/week:
    - season = stagione più recente nei dati
    - week = ultima week nei dati + 1
    """
    if df.empty:
        raise ValueError("player_week_features è vuoto, impossibile determinare i default.")

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
        raise FileNotFoundError(f"{path} non trovato. Hai già eseguito features.py?")
    df = pd.read_parquet(path)
    return df


def load_model_and_meta(position: str = "WR"):
    model_path = MODEL_DIR / f"{position.lower()}_catboost.cbm"
    meta_path = MODEL_DIR / f"{position.lower()}_catboost_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadati non trovati: {meta_path}")

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
) -> pd.DataFrame:
    """
    Usiamo le feature della settimana (target_week - 1) per prevedere target_week.
    """

    feature_week = target_week - 1
    if feature_week < 1:
        raise ValueError("Non posso prevedere la week 1: serve almeno una week precedente.")

    df_season = df[df["season"] == season].copy()
    df_pos = df_season[df_season["position"] == position].copy()
    df_feat = df_pos[df_pos["week"] == feature_week].copy()

    if df_feat.empty:
        raise ValueError(
            f"Non esistono feature per season={season}, week={feature_week}. "
            "Scaricare i dati più recenti?"
        )

    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise KeyError(f"Feature mancanti durante inference: {missing}")

    X = df_feat[feature_cols].copy()

    # Normalize categorical columns. Also detect any non-numeric
    # feature columns that weren't listed in meta's `cat_cols` and
    # treat them as categorical to avoid CatBoost trying to convert
    # list-like strings (e.g. '39;54') to floats.
    import pandas as _pd

    inferred_cat_cols = list(cat_cols) if cat_cols is not None else []
    for c in feature_cols:
        if c in X.columns and not _pd.api.types.is_numeric_dtype(X[c]):
            if c not in inferred_cat_cols:
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

    # Nome giocatore
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
            "player_id": df_feat.get("player_id", pd.Series([None] * len(df_feat))).values,
            "player_name": df_feat[name_col].values if name_col else None,
            "team": df_feat[team_col].values if team_col else None,
            "position": position,
            "expected_ppr_points": preds,
        }
    )

    return out.sort_values("expected_ppr_points", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------
# Typer CLI
# -----------------------------------------------------------
def _clean_option(value: any) -> any:
    """
    Se arriva un Typer OptionInfo (per chiamate errate), lo tratta come None.
    Così siamo robusti anche se qualcuno chiama per sbaglio il comando Typer da codice.
    """
    from typer.models import OptionInfo

    if isinstance(value, OptionInfo):
        return None
    return value


def run_predictions(
    season: Optional[int] = None,
    week: Optional[int] = None,
    position: str = "WR",
):
    """
    Funzione core: fa le previsioni senza dipendere da Typer.
    Può essere chiamata sia da CLI Typer sia da altri moduli Python.
    """

    # Protezione da OptionInfo nel caso qualcuno chiami male
    season = _clean_option(season)
    week = _clean_option(week)
    position = _clean_option(position) or "WR"

    # 1) Carico le features
    df = load_features()

    # 2) Se mancano season/week, usa i default basati sui dati
    if season is None or week is None:
        data_season, data_week = get_default_season_and_week_from_data(df)
        season = season or data_season
        week = week or data_week

    season = int(season)
    week = int(week)

    print(f"\n=== Fantasy Predictions ===")
    print(f"Season (default dai dati): {season}")
    print(f"Target week: {week} (usa week {week-1} come feature)")
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

    print(f"Trovati {len(df_feat)} giocatori {position} per week {week-1}")

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

    print(f"\nSalvato CSV in: {out_path}\n")
    print("Top 10:\n")
    print(preds[["player_name", "team", "expected_ppr_points"]].head(10).to_string(index=False))

    return preds


# -----------------------------------------------------------
# Typer CLI wrapper
# -----------------------------------------------------------

@app.command()
def predict(
    season: Optional[int] = typer.Option(
        None,
        help="Season NFL. Default = stagione più recente nei dati.",
    ),
    week: Optional[int] = typer.Option(
        None,
        help="Week da prevedere. Default = prossima week rispetto ai dati.",
    ),
    position: str = typer.Option("WR", help="Posizione: WR, RB, QB, TE"),
):
    """Previsione fantasy next-week (wrapper Typer)."""
    run_predictions(season=season, week=week, position=position)


if __name__ == "__main__":
    app()