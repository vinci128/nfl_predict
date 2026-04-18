from __future__ import annotations

import datetime
import json
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

# === Config base paths ===
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = DATA_DIR / ".fetch_manifest.json"

# === Validation schemas ===

SCHEMA_REQUIREMENTS: dict[str, dict] = {
    "weekly_stats": {
        "required_cols": ["player_id", "season", "week"],
        "numeric_ranges": {
            "passing_yards": (0, 1000),
            "rushing_yards": (-50, 500),
            "receiving_yards": (-50, 500),
        },
    },
    "rosters": {"required_cols": ["season"]},
    "snap_counts": {"required_cols": ["season", "week"]},
    "schedules": {"required_cols": ["season", "week", "home_team", "away_team"]},
    "injuries": {"required_cols": ["season", "week", "gsis_id"]},
}


def validate_dataframe(df: pd.DataFrame, name: str) -> None:
    """Validate a fetched DataFrame. Raises ValueError on hard errors, warns on soft issues."""
    if df.empty:
        raise ValueError(f"{name}: DataFrame is empty after fetch")

    schema = SCHEMA_REQUIREMENTS.get(name, {})

    missing_cols = [c for c in schema.get("required_cols", []) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: missing required columns {missing_cols}")

    for col, (lo, hi) in schema.get("numeric_ranges", {}).items():
        if col in df.columns:
            ser = pd.to_numeric(df[col], errors="coerce")
            n_out = int(((ser < lo) | (ser > hi)).sum())
            if n_out > 0:
                print(f"  WARN {name}.{col}: {n_out} values outside [{lo}, {hi}]")

    for col in schema.get("required_cols", []):
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            if null_pct > 20:
                print(f"  WARN {name}.{col}: {null_pct:.1f}% null")

    seasons_str = (
        str(sorted(df["season"].unique())) if "season" in df.columns else "n/a"
    )
    print(f"  Validated {name}: shape={df.shape}, seasons={seasons_str}")


# === Manifest / incremental helpers ===


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _seasons_to_fetch(
    name: str, all_seasons: list[int], current_season: int
) -> list[int]:
    """Return seasons that need fetching: missing historicals + always current season."""
    manifest = _load_manifest()
    fetched = set(manifest.get(name, {}).get("seasons", []))
    return sorted(s for s in all_seasons if s == current_season or s not in fetched)


def _merge_and_save(name: str, new_df: pd.DataFrame) -> None:
    """Merge new seasons into existing parquet (replacing overlapping seasons), then save."""
    path = DATA_DIR / f"{name}.parquet"
    if path.exists() and "season" in new_df.columns:
        existing = pd.read_parquet(path)
        new_seasons = set(new_df["season"].unique())
        existing = existing[~existing["season"].isin(new_seasons)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False)
    print(f"  Saved {name} -> {path} shape={combined.shape}")


def _update_manifest(name: str, seasons: list[int]) -> None:
    manifest = _load_manifest()
    prev = set(manifest.get(name, {}).get("seasons", []))
    prev.update(seasons)
    manifest[name] = {
        "seasons": sorted(prev),
        "last_updated": datetime.datetime.now().isoformat(),
    }
    _save_manifest(manifest)


# === Seasons helper ===


def get_seasons(first_season: int = 2015) -> list[int]:
    """Compute the full list of seasons [first_season … current]."""
    current_season = nfl.get_current_season()
    seasons = list(range(first_season, current_season + 1))
    print(f"Using seasons: {seasons}")
    return seasons


SEASONS = get_seasons()
CURRENT_SEASON: int = nfl.get_current_season()


# === Incremental fetch functions ===


def fetch_weekly_stats() -> None:
    """Weekly player stats (passing, rushing, receiving, fantasy, etc.)."""
    to_fetch = _seasons_to_fetch("weekly_stats", SEASONS, CURRENT_SEASON)
    if not to_fetch:
        print("weekly_stats: all seasons cached — skipping")
        return
    print(f"Fetching weekly_stats for seasons: {to_fetch}")
    df = nfl.load_player_stats(seasons=to_fetch).to_pandas()
    validate_dataframe(df, "weekly_stats")
    _merge_and_save("weekly_stats", df)
    _update_manifest("weekly_stats", to_fetch)


def fetch_rosters() -> None:
    """Seasonal rosters with player-team-position mappings."""
    to_fetch = _seasons_to_fetch("rosters", SEASONS, CURRENT_SEASON)
    if not to_fetch:
        print("rosters: all seasons cached — skipping")
        return
    print(f"Fetching rosters for seasons: {to_fetch}")
    df = nfl.load_rosters(seasons=to_fetch).to_pandas()
    validate_dataframe(df, "rosters")
    _merge_and_save("rosters", df)
    _update_manifest("rosters", to_fetch)


def fetch_snap_counts() -> None:
    """Player snap participation per week."""
    to_fetch = _seasons_to_fetch("snap_counts", SEASONS, CURRENT_SEASON)
    if not to_fetch:
        print("snap_counts: all seasons cached — skipping")
        return
    print(f"Fetching snap_counts for seasons: {to_fetch}")
    df = nfl.load_snap_counts(seasons=to_fetch).to_pandas()
    validate_dataframe(df, "snap_counts")
    _merge_and_save("snap_counts", df)
    _update_manifest("snap_counts", to_fetch)


def fetch_team_schedules() -> None:
    """Game schedules including Vegas lines and weather conditions."""
    to_fetch = _seasons_to_fetch("schedules", SEASONS, CURRENT_SEASON)
    if not to_fetch:
        print("schedules: all seasons cached — skipping")
        return
    print(f"Fetching schedules for seasons: {to_fetch}")
    df = nfl.load_schedules(seasons=to_fetch).to_pandas()
    validate_dataframe(df, "schedules")
    _merge_and_save("schedules", df)
    _update_manifest("schedules", to_fetch)


def fetch_injuries() -> None:
    """Weekly injury reports (report_status, practice_status)."""
    to_fetch = _seasons_to_fetch("injuries", SEASONS, CURRENT_SEASON)
    if not to_fetch:
        print("injuries: all seasons cached — skipping")
        return
    print(f"Fetching injuries for seasons: {to_fetch}")
    df = nfl.load_injuries(seasons=to_fetch).to_pandas()
    validate_dataframe(df, "injuries")
    _merge_and_save("injuries", df)
    _update_manifest("injuries", to_fetch)


def main() -> None:
    """Entrypoint used by the CLI (nfl-predict update-all)."""
    fetch_weekly_stats()
    fetch_rosters()
    fetch_snap_counts()
    fetch_team_schedules()
    fetch_injuries()


if __name__ == "__main__":
    main()
