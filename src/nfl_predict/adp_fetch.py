"""
ADP (Average Draft Position) data fetching and normalisation.

Supports two sources:
- Sleeper public API  (no auth required)
- FantasyPros JSON    (no auth required for free tier)

Output is always a normalised CSV with columns:
    player_name, position, team, adp

Usage
-----
    from nfl_predict.adp_fetch import fetch_adp, save_adp_csv

    df = fetch_adp(source="sleeper", scoring="ppr")
    save_adp_csv(df)                        # → data/adp_current.csv

CLI
---
    nfl-predict fetch-adp [--source sleeper|fantasypros] [--scoring ppr|half|std]
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
DEFAULT_ADP_PATH = DATA_DIR / "adp_current.csv"

# Sleeper: public player endpoint (no key needed)
_SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"

# FantasyPros JSON for PPR ADP (free, no key)
_FP_ADP_URLS = {
    "ppr": "https://api.fantasypros.com/public/v2/json/nfl/{year}/consensus-rankings"
    "?type=ros&scoring=PPR&position=ALL&limit=300",
    "half": "https://api.fantasypros.com/public/v2/json/nfl/{year}/consensus-rankings"
    "?type=ros&scoring=HALF&position=ALL&limit=300",
    "std": "https://api.fantasypros.com/public/v2/json/nfl/{year}/consensus-rankings"
    "?type=ros&scoring=STD&position=ALL&limit=300",
}


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------


def normalise_name(name: str) -> str:
    """Lowercase, strip suffix (Jr/Sr/II/III/IV), collapse whitespace."""
    name = name.strip().lower()
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)\.?$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


# ---------------------------------------------------------------------------
# Sleeper source
# ---------------------------------------------------------------------------


def fetch_from_sleeper(
    scoring: str = "ppr",
    season: int | None = None,
    positions: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch ADP data from the Sleeper public API.

    Sleeper exposes all NFL player metadata at /v1/players/nfl.
    ADP is embedded in each player object as ``fantasy_positions``,
    ``search_rank``, and ``years_exp``.  We use ``search_rank`` as an
    ADP proxy (Sleeper's own consensus ranking).

    Parameters
    ----------
    scoring   : "ppr", "half", or "std" (Sleeper has one ranking; kept for API compat)
    season    : ignored (Sleeper ranking is always current)
    positions : filter to these positions (default: QB/RB/WR/TE/K)

    Returns
    -------
    DataFrame with columns: player_name, position, team, adp
    """
    import json
    import urllib.request

    if positions is None:
        positions = ["QB", "RB", "WR", "TE", "K"]

    print("Fetching player data from Sleeper API…")
    req = urllib.request.Request(
        _SLEEPER_PLAYERS_URL,
        headers={"User-Agent": "nfl-predict/1.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data: dict = json.loads(resp.read())

    rows = []
    for _pid, player in data.items():
        pos = player.get("position", "")
        if pos not in positions:
            continue
        rank = player.get("search_rank")
        if not rank or rank > 9000:  # Sleeper uses 9999999 for unranked
            continue
        name = player.get("full_name") or player.get("search_full_name") or ""
        team = player.get("team") or "FA"
        rows.append(
            {
                "player_name": name,
                "position": pos,
                "team": team,
                "adp": float(rank),
                "_source": "sleeper",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("  Warning: no ranked players returned from Sleeper.")
        return df

    df = df.sort_values("adp").reset_index(drop=True)
    print(f"  Fetched {len(df)} ranked players from Sleeper.")
    return df[["player_name", "position", "team", "adp"]]


# ---------------------------------------------------------------------------
# FantasyPros source  (fallback)
# ---------------------------------------------------------------------------


def fetch_from_fantasypros(
    scoring: str = "ppr",
    season: int | None = None,
) -> pd.DataFrame:
    """
    Fetch consensus rankings from the FantasyPros public JSON endpoint.

    Note: FantasyPros may require a key for some endpoints; this uses the
    publicly accessible JSON feed that powers their free rankings pages.
    Falls back gracefully to an empty DataFrame on 4xx errors.

    Parameters
    ----------
    scoring : "ppr", "half", or "std"
    season  : season year (defaults to current)

    Returns
    -------
    DataFrame with columns: player_name, position, team, adp
    """
    import json
    import urllib.request
    from datetime import datetime

    if season is None:
        season = datetime.now().year

    url_template = _FP_ADP_URLS.get(scoring, _FP_ADP_URLS["ppr"])
    url = url_template.format(year=season)

    print(f"Fetching rankings from FantasyPros ({scoring})…")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nfl-predict/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  FantasyPros fetch failed: {e}")
        return pd.DataFrame({"player_name": [], "position": [], "team": [], "adp": []})

    players = data.get("players", [])
    rows = []
    for rank, p in enumerate(players, start=1):
        rows.append(
            {
                "player_name": p.get("player_name", ""),
                "position": p.get("player_position_id", ""),
                "team": p.get("player_team_id", "FA"),
                "adp": float(p.get("rank_ave", rank)),
                "_source": "fantasypros",
            }
        )

    df = pd.DataFrame(rows)
    print(f"  Fetched {len(df)} ranked players from FantasyPros.")
    return df[["player_name", "position", "team", "adp"]] if not df.empty else df


# ---------------------------------------------------------------------------
# Synthetic fallback (for testing / offline use)
# ---------------------------------------------------------------------------


def generate_synthetic_adp(
    board_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic ADP data from an existing draft board CSV.

    Adds slight random noise to ``overall_rank`` to simulate real ADP
    variation (±10 picks).  Useful for offline testing and CI.

    Parameters
    ----------
    board_path : path to draft_board CSV; auto-detected from outputs/ if None

    Returns
    -------
    DataFrame with columns: player_name, position, team, adp
    """
    import glob
    import random

    random.seed(42)

    if board_path is None:
        boards = sorted(glob.glob("outputs/draft_board_*.csv"))
        if not boards:
            raise FileNotFoundError(
                "No draft board CSV found. Run `nfl-predict board` first."
            )
        board_path = boards[-1]

    board = pd.read_csv(board_path)
    df = board[["player_name", "position", "team", "overall_rank"]].copy()
    # Add ±10 pick noise, clamp to positive
    noise = [random.gauss(0, 5) for _ in range(len(df))]
    df["adp"] = (df["overall_rank"] + noise).clip(lower=1.0).round(1)
    df = df.drop(columns=["overall_rank"]).sort_values("adp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_adp(
    source: str = "sleeper",
    scoring: str = "ppr",
    season: int | None = None,
    fallback_to_synthetic: bool = True,
) -> pd.DataFrame:
    """
    Fetch ADP data from the specified source.

    Parameters
    ----------
    source      : "sleeper", "fantasypros", or "synthetic"
    scoring     : "ppr", "half", or "std"
    season      : season year for FantasyPros (ignored by Sleeper/synthetic)
    fallback_to_synthetic : if the live fetch fails, fall back to synthetic ADP

    Returns
    -------
    Normalised DataFrame: player_name, position, team, adp
    """
    df = pd.DataFrame()

    if source == "synthetic":
        return generate_synthetic_adp()

    try:
        if source == "sleeper":
            df = fetch_from_sleeper(scoring=scoring, season=season)
        elif source == "fantasypros":
            df = fetch_from_fantasypros(scoring=scoring, season=season)
        else:
            raise ValueError(
                f"Unknown ADP source: {source!r}. Use 'sleeper', 'fantasypros', or 'synthetic'."
            )
    except Exception as e:
        print(f"  Live ADP fetch failed ({e})")
        df = pd.DataFrame()

    if df.empty and fallback_to_synthetic:
        print("  Falling back to synthetic ADP (rank + noise).")
        try:
            df = generate_synthetic_adp()
        except FileNotFoundError as e:
            print(f"  Synthetic fallback also failed: {e}")

    return df


def save_adp_csv(
    df: pd.DataFrame,
    path: str | Path | None = None,
) -> Path:
    """
    Save ADP DataFrame to CSV.

    Parameters
    ----------
    df   : DataFrame from fetch_adp()
    path : output path (defaults to data/adp_current.csv)

    Returns
    -------
    Path to the written file.
    """
    if path is None:
        path = DEFAULT_ADP_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  ADP saved → {path}  ({len(df)} players)")
    return path
