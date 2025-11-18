from __future__ import annotations

import nflreadpy as nfl
import pandas as pd
from pathlib import Path

from datetime import date

# === Config base paths ===
# Adatta se nel tuo progetto usavi già un BASE_DIR diverso
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)



def save_parquet(df: pd.DataFrame, name: str) -> None:
    """Salva un DataFrame in data/<name>.parquet."""
    path = DATA_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {name} -> {path} shape={df.shape}")

# === Seasons helper via nflreadpy ===

def get_seasons(first_season: int = 2015) -> list[int]:
    """
    Usa nflreadpy per capire la season corrente e crea la lista [first_season..current].
    """
    current_season = nfl.get_current_season()  # int, es. 2025
    seasons = list(range(first_season, current_season + 1))
    print(f"Using seasons: {seasons}")
    return seasons


SEASONS = get_seasons()

# === Fetch functions usando nflreadpy (Polars -> Pandas) ===

def fetch_weekly_stats() -> None:
    """
    Weekly player stats combinati (passing, rushing, receiving, fumbles, PPR fantasy, ecc.)
    """
    print("Downloading weekly player stats via nflreadpy...")
    pl_df = nfl.load_player_stats(seasons=SEASONS)
    df = pl_df.to_pandas()
    print(f"weekly_stats: {df.shape}")
    save_parquet(df, "weekly_stats")


def fetch_rosters() -> None:
    """
    Roster stagionali – sostituisce import_seasonal_rosters / import_rosters.
    """
    print("Downloading seasonal rosters via nflreadpy...")
    pl_df = nfl.load_rosters(seasons=SEASONS)
    df = pl_df.to_pandas()
    print(f"rosters: {df.shape}")
    save_parquet(df, "rosters")


def fetch_snap_counts() -> None:
    """
    Snap counts per player-week.
    """
    print("Downloading snap counts via nflreadpy...")
    pl_df = nfl.load_snap_counts(seasons=SEASONS)
    df = pl_df.to_pandas()
    print(f"snap_counts: {df.shape}")
    save_parquet(df, "snap_counts")


def fetch_team_schedules() -> None:
    """
    Schedules/game info – sostituisce import_schedules.
    """
    print("Downloading schedules via nflreadpy...")
    pl_df = nfl.load_schedules(seasons=SEASONS)
    df = pl_df.to_pandas()
    print(f"schedules: {df.shape}")
    save_parquet(df, "schedules")


def main() -> None:
    """
    Entrypoint usato dal tuo CLI (nfl-predict update-all).
    """
    fetch_weekly_stats()
    fetch_rosters()
    fetch_snap_counts()
    fetch_team_schedules()


if __name__ == "__main__":
    main()