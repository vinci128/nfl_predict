from __future__ import annotations

import datetime
import json
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

# === Config base paths ===
# CWD-relative, matching features.py / train_model.py / predict_week.py —
# run all commands from the project root (or the container WORKDIR).
DATA_DIR = Path("data")
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
    "team_stats": {"required_cols": ["season", "week", "team"]},
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


def _align_dtypes(
    stored: pd.DataFrame, fresh: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile column types between a stored parquet and a fresh load.

    nflverse changes column types between releases — jersey_number and
    draft_number were strings in older loads and are floats now. Concatenating
    the two then yields an object column holding both, which pyarrow refuses to
    write, and the entire refresh dies on a column no model reads.

    The fresh load wins wherever the stored values convert cleanly; where they
    do not, both sides become strings, which always round-trips.
    """
    stored = stored.copy()
    fresh = fresh.copy()

    for col in stored.columns.intersection(fresh.columns):
        stored_type, fresh_type = stored[col].dtype, fresh[col].dtype
        if stored_type == fresh_type:
            continue

        stored_obj = pd.api.types.is_object_dtype(stored_type)
        fresh_obj = pd.api.types.is_object_dtype(fresh_type)
        if stored_obj == fresh_obj:
            # Both numeric (int vs float, say). Concat widens these safely.
            continue

        text, number = (stored, fresh) if stored_obj else (fresh, stored)
        converted = pd.to_numeric(text[col], errors="coerce")
        if converted.notna().sum() == text[col].notna().sum():
            text[col] = converted.astype(number[col].dtype, errors="ignore")
        else:
            stored[col] = stored[col].astype("string")
            fresh[col] = fresh[col].astype("string")

    return stored, fresh


def _merge_and_save(name: str, new_df: pd.DataFrame) -> None:
    """Merge new seasons into existing parquet (replacing overlapping seasons), then save."""
    path = DATA_DIR / f"{name}.parquet"
    if path.exists() and "season" in new_df.columns:
        existing = pd.read_parquet(path)
        new_seasons = set(new_df["season"].unique())
        existing = existing[~existing["season"].isin(new_seasons)]
        existing, new_df = _align_dtypes(existing, new_df)
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


# === Incremental fetch functions ===
#
# Seasons are computed lazily in main() (not at import time) so importing
# this module — e.g. via the CLI — has no side effects and never goes stale
# in a long-running process.

_LOADERS = {
    "weekly_stats": nfl.load_player_stats,
    "rosters": nfl.load_rosters,
    "snap_counts": nfl.load_snap_counts,
    "schedules": nfl.load_schedules,
    "injuries": nfl.load_injuries,
    # Team-level defensive stats — the D/ST projection's only source.
    "team_stats": nfl.load_team_stats,
}


def fetch_dataset(name: str, seasons: list[int], current_season: int) -> None:
    """Fetch one dataset incrementally (missing historicals + current season)."""
    to_fetch = _seasons_to_fetch(name, seasons, current_season)
    if not to_fetch:
        print(f"{name}: all seasons cached — skipping")
        return
    print(f"Fetching {name} for seasons: {to_fetch}")
    df = _LOADERS[name](seasons=to_fetch).to_pandas()
    validate_dataframe(df, name)
    _merge_and_save(name, df)
    _update_manifest(name, to_fetch)


def main() -> None:
    """Entrypoint used by the CLI (nfl-predict update-all)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seasons = get_seasons()
    current_season = seasons[-1]
    for name in _LOADERS:
        fetch_dataset(name, seasons, current_season)


if __name__ == "__main__":
    main()
