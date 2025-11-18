from pathlib import Path
from typing import List, Optional

import pandas as pd

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Rolling windows (in partite) che vuoi usare come feature
ROLLING_WINDOWS = [3, 5, 8]


def _load_parquet(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} non trovato. Hai eseguito fetch_nfl_data.py?")
    return pd.read_parquet(path)


def load_raw_data():
    """Carica i parquet principali creati da fetch_nfl_data.py."""
    weekly = _load_parquet("weekly_stats")
    rosters = _load_parquet("rosters")
    try:
        snaps = _load_parquet("snap_counts")
    except FileNotFoundError:
        snaps = None
    return weekly, rosters, snaps


def prepare_base_weekly(
    weekly: pd.DataFrame,
    rosters: pd.DataFrame,
    snaps: Optional[pd.DataFrame] = None,
    offensive_only: bool = True,
) -> pd.DataFrame:
    """
    Crea una tabella base player-week con info anagrafiche e, opzionalmente,
    snap counts offensivi.
    """

def prepare_base_weekly(
    weekly: pd.DataFrame,
    rosters: pd.DataFrame,
    snaps: pd.DataFrame | None = None,
    offensive_only: bool = True,
) -> pd.DataFrame:
    # Copie locali per sicurezza
    weekly = weekly.copy()
    rosters = rosters.copy()

    # ------------------------------------------------------------------
    # 1) Armonizza gli ID giocatore tra weekly e rosters
    #    - nflreadpy usa "gsis_id" nei rosters
    #    - le weekly hanno già "player_id" tipo "00-0019596"
    # ------------------------------------------------------------------
    if "player_id" not in rosters.columns:
        if "gsis_id" in rosters.columns:
            rosters["player_id"] = rosters["gsis_id"]
        else:
            raise KeyError(
                "Rosters non ha né 'player_id' né 'gsis_id'. "
                "Controlla le colonne di rosters.parquet."
            )

    # ------------------------------------------------------------------
    # 2) Armonizza le colonne team/recent_team
    #    - in weekly: preferiamo usare "recent_team"
    #    - in rosters: usiamo "team"
    # ------------------------------------------------------------------
    if "recent_team" not in weekly.columns and "team" in weekly.columns:
        weekly.rename(columns={"team": "recent_team"}, inplace=True)

    if "team" not in rosters.columns and "recent_team" in rosters.columns:
        rosters.rename(columns={"recent_team": "team"}, inplace=True)

    # ------------------------------------------------------------------
    # 3) Ora possiamo selezionare le colonne minime dal roster
    # ------------------------------------------------------------------
    rosters_min = rosters[
        [c for c in ["player_id", "position", "team", "season", "week"] if c in rosters.columns]
    ].copy()

    # se non hai week in rosters, deduplica solo per (season, player_id)
    if "week" in rosters_min.columns:
        rosters_min = (
            rosters_min
            .sort_values(["season", "week"])
            .drop_duplicates(["season", "player_id"], keep="last")
        )
    else:
        rosters_min = (
            rosters_min
            .sort_values(["season"])
            .drop_duplicates(["season", "player_id"], keep="last")
        )

    # ------------------------------------------------------------------
    # 4) Merge con weekly sui campi comuni (player_id + season)
    # ------------------------------------------------------------------
    merge_keys = ["player_id"]
    if "season" in weekly.columns and "season" in rosters_min.columns:
        merge_keys.append("season")

    df = weekly.merge(rosters_min, on=merge_keys, how="left", suffixes=("", "_roster"))

    # Se la posizione nel weekly è mancante, usa quella del roster
    if "position" in weekly.columns:
        df["position"] = df["position"].fillna(df["position_roster"])
    else:
        df["position"] = df["position_roster"]

    df.drop(columns=[c for c in ["position_roster"] if c in df.columns], inplace=True)

    # Filtro solo ruoli offensivi per fantasy (QB, RB, WR, TE) nella v1
    if offensive_only:
        df = df[df["position"].isin(["QB", "RB", "WR", "TE"])].copy()

    # --- Merge con snap counts (se disponibili) ---
    if snaps is not None:
        # Tiene solo colonne rilevanti degli snap
        if "player_id" in snaps.columns:
            snap_cols = [
                "season",
                "week",
                "player_id",
            ]
            for c in ["offense_snaps", "offense_pct"]:
                if c in snaps.columns:
                    snap_cols.append(c)

            snaps_min = snaps[snap_cols].copy()
            df = df.merge(
                snaps_min,
                on=["season", "week", "player_id"],
                how="left",
            )

            # Rinominazioni comode
            if "offense_pct" in df.columns:
                df.rename(columns={"offense_pct": "snap_pct_offense"}, inplace=True)
            if "offense_snaps" in df.columns:
                df.rename(columns={"offense_snaps": "snaps_offense"}, inplace=True)
    else:
        print("no player_id available")

    # Ordina per giocatore/tempo
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    return df


def _select_stat_columns(df: pd.DataFrame) -> List[str]:
    """
    Seleziona un set di colonne di stats da usare per rolling features,
    prendendo solo quelle effettivamente presenti nel DataFrame.
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
        # fantasy
        "fantasy_points_ppr",
        # usage from snaps
        "snaps_offense",
        "snap_pct_offense",
    ]

    return [c for c in candidate_cols if c in df.columns]


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge:
    - lag1 delle stats principali (valore settimana precedente)
    - rolling mean sulle ultime N partite (usando SOLO valori passati per evitare leakage)
    """

    df = df.copy()
    stat_cols = _select_stat_columns(df)

    # Lag 1 per tutte le stats
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grouped = df.groupby("player_id", group_keys=False)

    for col in stat_cols:
        lag_col = f"{col}_lag1"
        df[lag_col] = grouped[col].shift(1)

    # Rolling features sulle stats laggate (quindi solo passato)
    for col in stat_cols:
        lag_col = f"{col}_lag1"
        if lag_col not in df.columns:
            continue

        for w in ROLLING_WINDOWS:
            roll_col = f"{col}_roll{w}"
            df[roll_col] = (
                grouped[lag_col]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

    # Numero di partite giocate recentemente (conteggio non-null della fantasy_points_ppr_lag1)
    if "fantasy_points_ppr_lag1" in df.columns:
        for w in ROLLING_WINDOWS:
            col = f"games_played_roll{w}"
            df[col] = (
                grouped["fantasy_points_ppr_lag1"]
                .rolling(window=w, min_periods=1)
                .count()
                .reset_index(level=0, drop=True)
            )

    return df


def add_simple_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge statistiche cumulative di stagione (espansive) fino alla settimana precedente.
    """
    df = df.copy()
    stat_cols = _select_stat_columns(df)

    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grouped = df.groupby(["player_id", "season"], group_keys=False)

    for col in stat_cols:
        # Usare valori laggati per evitare leakage (se esistono)
        src_col = f"{col}_lag1" if f"{col}_lag1" in df.columns else col

        cum_sum_col = f"{col}_season_cum"
        cum_mean_col = f"{col}_season_mean"

        df[cum_sum_col] = grouped[src_col].cumsum()
        df[cum_mean_col] = grouped[src_col].expanding(min_periods=1).mean().reset_index(
            level=[0, 1], drop=True
        )

    return df


def build_player_week_features(save: bool = True) -> pd.DataFrame:
    """
    Pipeline completa:
    - carica i raw parquet
    - crea base player-week
    - aggiunge lag + rolling features
    - aggiunge cumulative di stagione
    - salva su parquet
    """
    weekly, rosters, snaps = load_raw_data()

    df = prepare_base_weekly(weekly, rosters, snaps=snaps, offensive_only=True)
    df = add_lag_and_rolling_features(df)
    df = add_simple_season_features(df)

    # Rimuovi le righe senza almeno una partita precedente (se vuoi
    # usare solo esempi con history > 0 per il training).
    # Altrimenti puoi tenerle e lasciare che il modello gestisca i NaN.
    # Qui per ora le teniamo.
    if save:
        out_path = OUT_DIR / "player_week_features.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Salvato: {out_path} (shape={df.shape})")

    return df


if __name__ == "__main__":
    build_player_week_features()

