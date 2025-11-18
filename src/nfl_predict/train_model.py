# src/nfl_predict/train_model.py

from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "player_week_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} non trovato. Hai già eseguito features.py?")
    df = pd.read_parquet(path)
    return df


def add_target_next_week(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()

    if "fantasy_points_ppr" not in df.columns:
        raise KeyError(
            "Colonna 'fantasy_points_ppr' non trovata in player_week_features. "
            "Controlla che weekly_stats la includa e che features.py non l'abbia droppata."
        )

    # target = punti PPR della settimana successiva per lo stesso giocatore
    df["target_points_next_week"] = (
        df.groupby("player_id", group_keys=False)["fantasy_points_ppr"].shift(-1)
    )

    # Rimuovi righe senza target (ultima partita di ogni player)
    df = df[~df["target_points_next_week"].isna()].copy()

    return df


def filter_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    return df[df["position"] == position].copy()


def train_wr_model(df_wr: pd.DataFrame):
    # Usa ultima stagione come validazione
    max_season = df_wr["season"].max()
    train_df = df_wr[df_wr["season"] < max_season].copy()
    valid_df = df_wr[df_wr["season"] == max_season].copy()

    print(f"Training seasons: {sorted(train_df['season'].unique())}")
    print(f"Validation season: {max_season}")
    print(f"Train shape: {train_df.shape}, Valid shape: {valid_df.shape}")

    target_col = "target_points_next_week"

    # --- 1) Colonne da escludere hard-coded --------------------
    drop_exact = [
        target_col,
        "fantasy_points_ppr",  # punti della week corrente
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
        "headshot_url",   # inutile come feature
    ]

    drop_exact = [c for c in drop_exact if c in train_df.columns]

    def is_name_or_id(col: str) -> bool:
        lower = col.lower()
        return ("name" in lower or "id" in lower) and (col not in ["season", "week"])

    drop_pattern = [c for c in train_df.columns if is_name_or_id(c)]

    drop_cols = sorted(set(drop_exact) | set(drop_pattern))

    print("Dropping columns:", drop_cols)

    # --- 2) Costruisci il set di feature ------------------------
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_col].copy()

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[target_col].copy()

    # --- 3) Identifica categoriali vs numeriche -----------------
    # Categorical: tutte le non-numeriche
    cat_cols = list(
        X_train.select_dtypes(include=["object", "string", "category"]).columns
    )

    # forza team come categorica se non lo fosse già
    for c in ["team", "season_type", "position_group"]:
        if c in X_train.columns and c not in cat_cols:
            cat_cols.append(c)

    # assicurati che season/week NON siano categoriali
    for c in ["season", "week"]:
        if c in cat_cols:
            cat_cols.remove(c)

    # normalizza le categorical: string + fillna
    for c in cat_cols:
        X_train[c] = X_train[c].astype("string").fillna("__NA__")
        X_valid[c] = X_valid[c].astype("string").fillna("__NA__")

    # CatBoost vuole gli INDICI delle colonne categoriche
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    print(f"Using {len(feature_cols)} features")
    print("Sample features:", feature_cols[:10])
    print(f"Categorical columns ({len(cat_cols)}):", cat_cols)

    train_pool = Pool(X_train, y_train, cat_features=cat_idx if cat_idx else None)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_idx if cat_idx else None)

    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        iterations=2000,
        od_type="Iter",
        od_wait=100,  # early stopping
        random_seed=42,
        verbose=100,
    )

    model.fit(train_pool, eval_set=valid_pool)

    pred_valid = model.predict(valid_pool)
    valid_mae = (abs(pred_valid - y_valid)).mean()
    print(f"Validation MAE (PPR points): {valid_mae:.3f}")

    model_path = MODEL_DIR / "wr_catboost.cbm"
    model.save_model(model_path)
    print(f"WR model salvato in: {model_path}")

    meta = {
        "position": "WR",
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "train_seasons": sorted(map(int, train_df["season"].unique())),
        "valid_season": int(max_season),
        "valid_mae": float(valid_mae),
    }
    pd.Series(meta).to_json(MODEL_DIR / "wr_catboost_meta.json", indent=2)
    print("Metadati WR salvati.")


def main():
    df = load_features()
    print("Loaded features:", df.shape)

    df = add_target_next_week(df)
    print("After adding target:", df.shape)

    # Per ora facciamo solo WR
    df_wr = filter_position(df, "WR")
    print("WR subset:", df_wr.shape)

    train_wr_model(df_wr)


if __name__ == "__main__":
    main()
