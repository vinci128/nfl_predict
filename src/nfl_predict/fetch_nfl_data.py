import nfl_data_py as nfl
import pandas as pd
from pathlib import Path

from datetime import date

CURRENT_YEAR = date.today().year
SEASONS = list(range(2015, CURRENT_YEAR + 1))


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def save_parquet(df: pd.DataFrame, name: str):
    path = DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"Saved {name} -> {path} shape={df.shape}")

def fetch_weekly_stats():
    print("Downloading weekly player stats...")
    df = nfl.import_weekly_data(SEASONS)
    print(f"weekly_stats: {df.shape}")
    save_parquet(df, "weekly_stats")

def fetch_rosters():
    print("Downloading rosters (seasonal)...")
    df = nfl.import_seasonal_rosters(SEASONS)
    print(f"rosters: {df.shape}")

    # --- Normalizzazione colonne "sporche" ---------------------
    # 1) jersey_number -> intero nullable
    if "jersey_number" in df.columns:
        df["jersey_number"] = (
            pd.to_numeric(df["jersey_number"], errors="coerce")
              .astype("Int64")
        )

    # 2) draft_number -> intero nullable
    if "draft_number" in df.columns:
        df["draft_number"] = (
            pd.to_numeric(df["draft_number"], errors="coerce")
              .astype("Int64")
        )

    # (Opzionale: puoi farlo anche per altre colonne tipo draft_round, draft_pick, ecc.)
    for col in ["draft_round", "draft_pick", "years_exp"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                  .astype("Int64")
            )
    # -----------------------------------------------------------

    save_parquet(df, "rosters")



def fetch_pbp():
    print("Downloading play-by-play (this is heavy)...")
    df = nfl.import_pbp_data(SEASONS)
    #df.to_parquet(DATA_DIR / "pbp.parquet", index=False)
    save_parquet(df, "pbp")
    print(f"pbp: {df.shape}")

def fetch_snap_counts():
    print("Downloading snap counts...")
    df = nfl.import_snap_counts(SEASONS)
    #df.to_parquet(DATA_DIR / "snap_counts.parquet", index=False)
    save_parquet(df,"snap_counts")
    print(f"snap counts: {df.shape}")

def fetch_team_schedules():
    print("Downloading schedules...")
    df = nfl.import_schedules(SEASONS)
    #df.to_parquet(DATA_DIR / "schedules.parquet", index=False)
    save_parquet(df,"schedules")
    print(f"schedules: {df.shape}")

def fetch_betting_lines():
    print("Downloading betting lines...")
    df = nfl.import_betting_data(SEASONS)
    #df.to_parquet(DATA_DIR / "betting_lines.parquet", index=False)
    save_parquet(df,"betting_lines")
    print(f"betting lines: {df.shape}")

def main():
    fetch_weekly_stats()
    fetch_rosters()
    #fetch_pbp()              # opzionale ma utile per metriche avanzate
    fetch_snap_counts()
    fetch_team_schedules()
    #fetch_betting_lines()

if __name__ == "__main__":
    main()


