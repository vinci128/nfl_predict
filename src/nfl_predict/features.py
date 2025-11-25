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
    snaps: pd.DataFrame | None = None,
    offensive_only: bool = False,
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
    else:
        print("Keeping all positions (filtering positions that have no point).")
        df = df[~df["position"].isin(["LS", "NT", "DL", "OL"])].copy()

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
        # kicking (FG / PAT) - useful for K models
        "fg_made",
        "fg_att",
        "fg_missed",
        "fg_long",
        "fg_pct",
        "fg_made_0_19",
        "fg_made_20_29",
        "fg_made_30_39",
        "fg_made_40_49",
        "fg_made_50_59",
        "fg_made_60_",
        "fg_missed_0_19",
        "fg_missed_20_29",
        "fg_missed_30_39",
        "fg_missed_40_49",
        "fg_missed_50_59",
        "fg_missed_60_",
        "fg_made_list",
        "fg_missed_list",
        "fg_made_distance",
        "fg_missed_distance",
        "pat_made",
        "pat_att",
        "pat_missed",
        "pat_blocked",
        "pat_pct",
    ]

    # Keep only candidate columns that exist in df and are numeric.
    present = [c for c in candidate_cols if c in df.columns]
    numeric = [c for c in present if pd.api.types.is_numeric_dtype(df[c])]
    # If some columns are present but non-numeric (e.g. 'fg_made_list'),
    # we skip them for lag/rolling calculations to avoid aggregation errors.
    return numeric


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



def add_custom_league_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola i punti fantasy secondo le regole della tua lega NFL.com:

    Passing Yards: 1 per 10
    Passing TD: 4
    INT lanciate: -2
    Rushing Yards: 1 per 10
    Rushing TD: 6
    Receptions: 1
    Receiving Yards: 1 per 10
    Receiving TD: 6
    Kick/Punt Return TD: 6
    Fumble Recovered for TD: 6
    Fumbles Lost: -2
    2pt Conversion (qualsiasi): 2

    Kicking:
    PAT made: 1
    FG 0-19: 3
    FG 20-29: 3
    FG 30-39: 3
    FG 40-49: 4
    FG 50+: 5

    NB: per ora non calcola i punti Defense/ST di squadra (DST).
    """

    pts = pd.Series(0.0, index=df.index, dtype="float64")

    def col(name: str) -> pd.Series:
        # helper comodo: se la colonna non esiste, restituisce 0
        if name in df.columns:
            return df[name].fillna(0)
        return pd.Series(0, index=df.index, dtype="float64")

    # ----- OFFENSE -----

    # Passing
    pts += 0.1 * col("passing_yards")          # 1 pt / 10 yards
    # in nflreadpy spesso è "passing_tds" e "interceptions"
    pts += 4.0 * col("passing_tds")
    pts += -2.0 * (col("interceptions") + col("passing_interceptions"))

    # Rushing
    pts += 0.1 * col("rushing_yards")
    pts += 6.0 * col("rushing_tds")

    # Receiving
    pts += 1.0 * col("receptions")
    pts += 0.1 * col("receiving_yards")
    pts += 6.0 * col("receiving_tds")

    # Return TD (kickoff + punt + altri ST TD)
    return_tds = (
        col("kick_return_tds") +
        col("punt_return_tds") +
        col("special_teams_tds")
    )
    pts += 6.0 * return_tds

    # Fumble recovered for TD (se presente)
    pts += 6.0 * (
        col("fumble_recovery_tds") +
        col("defense_tds")  # nel dubbio, meglio includerlo
    )

    # Fumbles lost (varie declinazioni)
    fumbles_lost = (
        col("fumbles_lost") +
        col("rushing_fumbles_lost") +
        col("receiving_fumbles_lost") +
        col("sack_fumbles_lost")
    )
    pts += -2.0 * fumbles_lost

    # 2-point conversions (passing / rushing / receiving)
    two_pt = (
        col("passing_2pt_conversions") +
        col("rushing_2pt_conversions") +
        col("receiving_2pt_conversions") +
        col("two_point_conversions")
    )
    pts += 2.0 * two_pt

    # ----- KICKING -----

    # ----- KICKING (range approximation) -----

    fg_made = (
        col("field_goals_made") +
        col("fg_made") +
        col("fgm")  # fallback nel caso appaia nei tuoi dati
    )

    fg_long = (
        col("field_goals_longest") +
        col("fg_long") +
        col("fg_longest")
    )
    # Stima punteggio FG
    # Preferiamo usare i bucket per distanza se presenti (colonne come
    # fg_made_30_39, fg_made_40_49, fg_made_50_59, fg_made_60_).
    # Se i bucket non sono disponibili, usiamo una fallback basata su
    # `fg_long` ma assegnando a TUTTE le FG lo stesso punteggio (es. 5*x
    # per fg_made quando fg_long>=50) — questo evita il bug precedente
    # che faceva 5 + (n-1)*4.

    fg_points = pd.Series(0.0, index=df.index)

    # bucket columns (may not exist)
    fg0 = col("fg_made_0_19")
    fg20 = col("fg_made_20_29")
    fg30 = col("fg_made_30_39")
    fg40 = col("fg_made_40_49")
    fg50 = col("fg_made_50_59")
    fg60 = col("fg_made_60_")

    # If any bucket column exists (non-zero in at least some rows), use buckets
    if (fg0.sum() + fg20.sum() + fg30.sum() + fg40.sum() + fg50.sum() + fg60.sum()) > 0:
        fg_points = (
            3 * (fg0 + fg20 + fg30)
            + 4 * fg40
            + 5 * (fg50 + fg60)
        )
    else:
        # fallback: use fg_long to assign a per-FG value (avoid 5+(n-1)*4 bug)
        mask_50 = fg_long >= 50
        fg_points[mask_50] = 5 * fg_made[mask_50]

        mask_40 = (fg_long >= 40) & (fg_long < 50)
        fg_points[mask_40] = 4 * fg_made[mask_40]

        mask_30 = (fg_long >= 30) & (fg_long < 40)
        fg_points[mask_30] = 3 * fg_made[mask_30]

        mask_20 = (fg_long >= 20) & (fg_long < 30)
        fg_points[mask_20] = 3 * fg_made[mask_20]

        mask_0 = (fg_long < 20)
        fg_points[mask_0] = 3 * fg_made[mask_0]

    pts += fg_points

    # PAT (extra points)
    pat_made = (
        col("extra_points_made") +
        col("xpmade") +
        col("pat_made")
    )
    pts += pat_made * 1

    df["fantasy_points_custom"] = pts
    return df


def add_opponent_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature che rappresentano la "forza difensiva" della squadra avversaria
    misurata come punti fantasy concessi a ciascuna posizione (e totale) nelle
    settimane precedenti. Per ogni (season, week, opponent_team, position) calcoliamo
    i punti concessi nella settimana e poi costruiamo lag1 + rolling mean (3/5/8)
    per usare come feature d'input al modello.
    """

    df = df.copy()

    if "opponent_team" not in df.columns:
        print("Nessuna colonna 'opponent_team' trovata — salto le feature difensive dell'avversario.")
        return df

    if "fantasy_points_custom" not in df.columns:
        print("Nessuna colonna 'fantasy_points_custom' trovata — assicurati di chiamare add_custom_league_points prima.")
        return df

    # Punti concessi dall'avversario per posizione nella specifica settimana
    team_allowed = (
        df.groupby(["season", "week", "opponent_team", "position"], dropna=False)["fantasy_points_custom"]
        .sum()
        .reset_index(name="points_allowed")
        .rename(columns={"opponent_team": "team"})
    )

    # Ordinamento e calcolo lag/rolling per (team, position)
    team_allowed = team_allowed.sort_values(["team", "position", "season", "week"]).reset_index(drop=True)
    grouped = team_allowed.groupby(["team", "position"], group_keys=False)

    team_allowed["points_allowed_lag1"] = grouped["points_allowed"].shift(1)

    for w in ROLLING_WINDOWS:
        col = f"points_allowed_roll{w}"
        team_allowed[col] = (
            grouped["points_allowed_lag1"].rolling(window=w, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        )

    # Aggiungiamo anche il totale (tutte le posizioni insieme) per dar misura complessiva
    total_allowed = (
        df.groupby(["season", "week", "opponent_team"], dropna=False)["fantasy_points_custom"]
        .sum()
        .reset_index(name="points_allowed_total")
        .rename(columns={"opponent_team": "team"})
    )

    total_allowed = total_allowed.sort_values(["team", "season", "week"]).reset_index(drop=True)
    gtot = total_allowed.groupby(["team"], group_keys=False)
    total_allowed["points_allowed_total_lag1"] = gtot["points_allowed_total"].shift(1)
    for w in ROLLING_WINDOWS:
        total_allowed[f"points_allowed_total_roll{w}"] = (
            gtot["points_allowed_total_lag1"].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    # Rinominiamo le colonne per evitare collisioni al merge
    rename_map = {"points_allowed_lag1": "opp_points_allowed_lag1"}
    for w in ROLLING_WINDOWS:
        rename_map[f"points_allowed_roll{w}"] = f"opp_points_allowed_roll{w}"
        rename_map[f"points_allowed_total_roll{w}"] = f"opp_points_allowed_total_roll{w}"
    rename_map["points_allowed_total_lag1"] = "opp_points_allowed_total_lag1"

    team_allowed = team_allowed.rename(columns={k: v for k, v in rename_map.items() if k in team_allowed.columns})
    total_allowed = total_allowed.rename(columns={k: v for k, v in rename_map.items() if k in total_allowed.columns})

    # Merge delle stats position-specific
    merge_cols = ["season", "week", "team", "position"] + [v for k, v in rename_map.items() if k.startswith("points_allowed_roll")]
    merge_cols = [c for c in merge_cols if c in team_allowed.columns]

    df = df.merge(
        team_allowed[["season", "week", "team", "position"] + [c for c in team_allowed.columns if c.startswith("opp_")]],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "team", "position"],
        how="left",
    )

    # Merge delle stats totali
    df = df.merge(
        total_allowed[["season", "week", "team"] + [c for c in total_allowed.columns if c.startswith("opp_")]],
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "team"],
        how="left",
        suffixes=(None, "_tot"),
    )

    # Pulisci colonne di join duplicate
    for c in ["team", "team_tot"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

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

    df = prepare_base_weekly(weekly, rosters, snaps=snaps, offensive_only=False)
    df = add_custom_league_points(df)      # ⬅️ QUI

    # Add opponent defense features (points allowed to positions / total)
    df = add_opponent_defense_features(df)

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

