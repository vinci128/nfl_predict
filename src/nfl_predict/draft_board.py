"""
Draft board generation for fantasy football.

Assembles per-position season projections from season_model.py,
computes Value Over Replacement (VOR), assigns overall and positional
tiers, and exports the result to CSV or JSON.

Typical usage
-------------
    from nfl_predict.draft_board import build_draft_board, export_draft_board

    board = build_draft_board(as_of_season=2024)
    export_draft_board(board, season=2025)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nfl_predict.season_model import POSITIONS, predict_season

OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# League settings
# ---------------------------------------------------------------------------


@dataclass
class DraftSettings:
    """
    Fantasy league configuration used for VOR calculation.

    All fields correspond to standard ESPN / Yahoo league defaults but can
    be overridden for custom formats (e.g. superflex, 3-WR leagues).
    """

    league_size: int = 12
    qb_starters: int = 1
    rb_starters: int = 2
    wr_starters: int = 2
    te_starters: int = 1
    k_starters: int = 1
    flex_spots: int = 1  # RB / WR / TE eligible
    # Extra buffer beyond the last starter (accounts for bye weeks / handcuffs)
    replacement_buffer: int = 3
    scoring: str = "custom"

    def replacement_ranks(self) -> dict[str, int]:
        """
        Return the replacement-level rank for each position.

        Replacement player = the first undrafted player at that spot, roughly
        (starters_per_team × league_size) + flex allocation + buffer.
        """
        n = self.league_size
        buf = self.replacement_buffer
        flex_share = self.flex_spots * n // 3  # approx flex distributed across RB/WR/TE
        return {
            "QB": n * self.qb_starters + buf,
            "RB": n * self.rb_starters + flex_share + buf,
            "WR": n * self.wr_starters + flex_share + buf,
            "TE": n * self.te_starters + buf,
            "K": n * self.k_starters + buf,
        }


# ---------------------------------------------------------------------------
# VOR calculation
# ---------------------------------------------------------------------------


def compute_vor(
    projections: pd.DataFrame,
    settings: DraftSettings | None = None,
) -> pd.DataFrame:
    """
    Add Value Over Replacement (VOR) to a projections DataFrame.

    VOR = proj_p50 − replacement_player_proj_p50 for each player's position.

    Parameters
    ----------
    projections : DataFrame with at least [position, proj_p50] columns
    settings    : DraftSettings (defaults to standard 12-team league)

    Returns
    -------
    Copy of projections with 'vor' and 'replacement_baseline' columns added.
    """
    if settings is None:
        settings = DraftSettings()
    if "proj_p50" not in projections.columns:
        raise ValueError("projections must have a 'proj_p50' column.")

    replacement_ranks = settings.replacement_ranks()
    df = projections.copy()

    vor = pd.Series(0.0, index=df.index, dtype=float)
    baseline = pd.Series(0.0, index=df.index, dtype=float)

    for pos, rank in replacement_ranks.items():
        mask = df["position"] == pos
        if not mask.any():
            continue
        sorted_pts = (
            df.loc[mask, "proj_p50"].sort_values(ascending=False).reset_index(drop=True)
        )
        repl_idx = min(rank - 1, len(sorted_pts) - 1)
        repl_pts = float(sorted_pts.iloc[repl_idx])

        vor[mask] = df.loc[mask, "proj_p50"] - repl_pts
        baseline[mask] = repl_pts

    df["vor"] = vor.round(1)
    df["replacement_baseline"] = baseline.round(1)
    return df


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------


def assign_tiers(
    df: pd.DataFrame,
    n_tiers: int = 8,
) -> pd.DataFrame:
    """
    Assign overall and positional tiers based on VOR.

    Overall tiers bucket all players together (tier 1 = elite regardless of
    position).  Positional tiers rank within each position separately.

    Parameters
    ----------
    df      : DataFrame with 'vor' and 'position' columns
    n_tiers : number of overall tier buckets (default 8)

    Returns
    -------
    Copy of df with 'tier' (overall) and 'pos_tier' (positional) columns.
    """
    if "vor" not in df.columns:
        raise ValueError("DataFrame must have a 'vor' column. Run compute_vor() first.")

    df = df.copy()
    n = len(df)
    tier_size = max(1, n // n_tiers)

    # Overall tier (1 = top tier)
    ranked = df["vor"].sort_values(ascending=False)
    tier_series = pd.Series(n_tiers, index=df.index, dtype=int)
    for t in range(n_tiers):
        start = t * tier_size
        end = (t + 1) * tier_size if t < n_tiers - 1 else n
        tier_series[ranked.iloc[start:end].index] = t + 1
    df["tier"] = tier_series

    # Positional tier (within each position, 1 = top)
    pos_tier = pd.Series(1, index=df.index, dtype=int)
    for pos in df["position"].unique():
        mask = df["position"] == pos
        pos_df = df[mask]
        n_pos = len(pos_df)
        n_pos_tiers = min(n_tiers, max(1, n_pos // 3))
        pos_tier_size = max(1, n_pos // n_pos_tiers)
        pos_ranked = pos_df["vor"].sort_values(ascending=False)
        for t in range(n_pos_tiers):
            start = t * pos_tier_size
            end = (t + 1) * pos_tier_size if t < n_pos_tiers - 1 else n_pos
            pos_tier[pos_ranked.iloc[start:end].index] = t + 1
    df["pos_tier"] = pos_tier

    return df


# ---------------------------------------------------------------------------
# Draft board assembly
# ---------------------------------------------------------------------------


def build_draft_board(
    as_of_season: int,
    positions: list[str] | None = None,
    adp_path: str | None = None,
    settings: DraftSettings | None = None,
) -> pd.DataFrame:
    """
    Build the full draft board from per-position season projections.

    Parameters
    ----------
    as_of_season : most recently completed season; projects as_of_season + 1
    positions    : positions to include (default: QB / RB / WR / TE / K)
    adp_path     : optional CSV path with ADP data (columns: player_name, adp)
    settings     : DraftSettings for VOR calculation

    Returns
    -------
    DataFrame sorted by VOR (descending), with overall_rank and pos_rank.
    """
    pos_list = positions or POSITIONS
    if settings is None:
        settings = DraftSettings()

    print(
        f"\nBuilding {as_of_season + 1} draft board (features from {as_of_season})..."
    )

    all_projs: list[pd.DataFrame] = []
    for pos in pos_list:
        print(f"  Projecting {pos}...")
        proj = predict_season(pos, as_of_season=as_of_season)
        if not proj.empty:
            all_projs.append(proj)

    if not all_projs:
        raise ValueError(
            "No projections generated. Run `nfl-predict draft-prep` first."
        )

    board = pd.concat(all_projs, ignore_index=True)

    # Ensure all quantile columns exist
    for col in ("proj_p10", "proj_p50", "proj_p90"):
        if col not in board.columns:
            board[col] = float("nan")

    # VOR and tiers
    board = compute_vor(board, settings=settings)
    board = assign_tiers(board)

    # Rankings
    board = board.sort_values("vor", ascending=False).reset_index(drop=True)
    board["overall_rank"] = board.index + 1
    board["pos_rank"] = (
        board.groupby("position")["vor"].rank(ascending=False, method="min").astype(int)
    )

    # Optional ADP merge
    if adp_path:
        board = _merge_adp(board, adp_path)

    return board


def _merge_adp(board: pd.DataFrame, adp_path: str) -> pd.DataFrame:
    """
    Merge ADP data from a CSV file into the draft board.

    Expected CSV columns: player_name (or 'name'), adp (or 'avg_pick').
    Uses fuzzy matching (difflib) to handle name discrepancies like
    Jr./Sr./III suffixes and minor spelling variants.
    """
    import difflib

    try:
        adp_df = pd.read_csv(adp_path)
    except Exception as e:
        print(f"  Could not load ADP file {adp_path}: {e}")
        return board

    adp_df.columns = [c.lower().strip() for c in adp_df.columns]
    name_col = next(
        (c for c in ("player_name", "name", "playername") if c in adp_df.columns),
        None,
    )
    adp_col = next(
        (c for c in ("adp", "avg_pick", "average_pick") if c in adp_df.columns),
        None,
    )

    if name_col is None or adp_col is None:
        print(
            f"  ADP CSV must have a name column and an adp column. "
            f"Found: {list(adp_df.columns)}"
        )
        return board

    def _normalise(s: str) -> str:
        import re

        return re.sub(r"\s+(jr|sr|ii|iii|iv)\.?$", "", s.lower().strip())

    adp_lookup = {
        _normalise(n): v
        for n, v in zip(adp_df[name_col], adp_df[adp_col], strict=False)
    }
    board_names = board["player_name"].map(_normalise)

    adp_vals: list[float] = []
    for name in board_names:
        if name in adp_lookup:
            adp_vals.append(adp_lookup[name])
        else:
            matches = difflib.get_close_matches(
                name, adp_lookup.keys(), n=1, cutoff=0.85
            )
            adp_vals.append(adp_lookup[matches[0]] if matches else float("nan"))

    board["adp"] = adp_vals
    board["adp_rank"] = pd.to_numeric(board["adp"], errors="coerce")
    board["value_vs_adp"] = (board["adp_rank"] - board["overall_rank"]).round(0)
    return board


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_draft_board(
    board: pd.DataFrame,
    out_path: str | None = None,
    fmt: str = "csv",
    season: int | None = None,
) -> Path:
    """
    Export the draft board to CSV or JSON.

    Parameters
    ----------
    board    : DataFrame from build_draft_board()
    out_path : explicit output path (auto-generated from season if None)
    fmt      : "csv" or "json"
    season   : projected season year for the auto-generated filename

    Returns
    -------
    Path object pointing to the written file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if out_path is None:
        ext = ".csv" if fmt == "csv" else ".json"
        season_str = str(season) if season else "latest"
        out_path = str(OUTPUT_DIR / f"draft_board_{season_str}{ext}")

    path = Path(out_path)

    display_cols = [
        "overall_rank",
        "tier",
        "pos_rank",
        "pos_tier",
        "player_name",
        "position",
        "team",
        "proj_p10",
        "proj_p50",
        "proj_p90",
        "vor",
        "replacement_baseline",
        "projected_season",
    ]
    optional_cols = ["adp", "adp_rank", "value_vs_adp"]
    cols = display_cols + [c for c in optional_cols if c in board.columns]
    export_df = board[[c for c in cols if c in board.columns]].copy()

    for col in ("proj_p10", "proj_p50", "proj_p90", "vor"):
        if col in export_df.columns:
            export_df[col] = export_df[col].round(1)

    if fmt == "csv":
        export_df.to_csv(path, index=False)
    elif fmt == "json":
        projected_season = (
            int(board["projected_season"].iloc[0])
            if "projected_season" in board.columns
            else None
        )
        result: dict = {
            "projected_season": projected_season,
            "generated_at": pd.Timestamp.now().isoformat(),
            "tiers": {
                str(t): export_df.loc[export_df["tier"] == t, "player_name"].tolist()
                for t in sorted(export_df["tier"].unique())
            },
            "players": json.loads(export_df.to_json(orient="records")),
        }
        path.write_text(json.dumps(result, indent=2))
    else:
        raise ValueError(f"Unknown format '{fmt}'. Use 'csv' or 'json'.")

    print(f"  Draft board saved → {path}")
    return path
