"""
Team defence / special teams projections.

D/ST scores off a different table than players do: team-level defensive
counting stats, plus points and yards allowed, which have to be derived by
joining each game against its opponent.

Why this is not a CatBoost quantile family like every other position: measured
over 2015-2024, the correlation between a defence's fantasy points in one
season and the next is r = 0.32 — about 10% of the variance. There are 32 rows
per season to learn from. A gradient-boosted model on that is fitting noise; a
shrinkage estimator (regress next season on prior season, pooled across all
year pairs) is the honest form of the same information, and its residual
spread gives the p10/p90 band directly.

The practical consequence for a draft: D/ST projections are nearly flat, so
the board will correctly refuse to spend an early pick on one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nfl_predict.leagues import LeagueProfile, ScoringRules, get_profile

DATA_DIR = Path("data")

# Normal quantiles for the p10/p90 band around the fitted median.
_Z90 = 1.2815515655446004


def _load(name: str) -> pd.DataFrame | None:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def _regular_season(df: pd.DataFrame) -> pd.DataFrame:
    """Drop postseason rows. Fantasy seasons are regular season only."""
    for col in ("season_type", "game_type"):
        if col in df.columns:
            return df[df[col].astype(str).str.upper().isin(["REG", "REGULAR"])].copy()
    return df.copy()


def points_allowed(schedules: pd.DataFrame) -> pd.DataFrame:
    """
    Points each team conceded, per game, from the scoreboard.

    A team's points allowed is its *opponent's* score, so each game yields two
    rows. Games without a final score (unplayed, or a fetch mid-season) are
    dropped rather than scored as a shutout.
    """
    sch = _regular_season(schedules)
    home = sch[["season", "week", "home_team", "away_score"]].rename(
        columns={"home_team": "team", "away_score": "points_allowed"}
    )
    away = sch[["season", "week", "away_team", "home_score"]].rename(
        columns={"away_team": "team", "home_score": "points_allowed"}
    )
    both = pd.concat([home, away], ignore_index=True)
    return both.dropna(subset=["points_allowed"]).reset_index(drop=True)


def yards_allowed(team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Yards each team conceded, per game.

    Taken from the opponent's own offensive line in the same game — team_stats
    reports what a team gained, not what it gave up.
    """
    ts = _regular_season(team_stats)
    gained = ts[["passing_yards", "rushing_yards"]].fillna(0).sum(axis=1)
    out = ts[["season", "week", "team"]].copy()
    out["yards_allowed"] = gained
    # Attach to the opponent: these are the yards *they* allowed.
    return out.rename(columns={"team": "opponent_team"})


def build_dst_weekly(
    team_stats: pd.DataFrame,
    schedules: pd.DataFrame,
    rules: ScoringRules,
) -> pd.DataFrame:
    """One row per team-game, with the league's D/ST points for that game."""
    ts = _regular_season(team_stats)
    df = ts.merge(points_allowed(schedules), on=["season", "week", "team"], how="left")
    df = df.merge(
        yards_allowed(team_stats), on=["season", "week", "opponent_team"], how="left"
    )
    # A game with no scoreline cannot be scored: the points-allowed ladder is
    # most of a D/ST's week, and treating a missing score as 0 would hand out
    # the shutout bonus.
    df = df.dropna(subset=["points_allowed"]).copy()
    df["dst_points"] = rules.score(df)
    return df


def dst_season_totals(
    team_stats: pd.DataFrame,
    schedules: pd.DataFrame,
    rules: ScoringRules,
) -> pd.DataFrame:
    """Season D/ST points and games played, per team-season."""
    weekly = build_dst_weekly(team_stats, schedules, rules)
    out = weekly.groupby(["season", "team"], as_index=False).agg(
        dst_points=("dst_points", "sum"),
        games=("dst_points", "size"),
    )
    return out.sort_values(["season", "dst_points"], ascending=[True, False])


def fit_shrinkage(totals: pd.DataFrame) -> tuple[float, float, float, int]:
    """
    Fit next-season points on prior-season points, pooled over all year pairs.

    Returns (slope, intercept, residual_sd, n_pairs). The slope is the shrinkage
    factor: well below 1, meaning a great defence is expected to regress most of
    the way back to average.
    """
    wide = totals.pivot(index="team", columns="season", values="dst_points")
    seasons = sorted(c for c in wide.columns)
    pairs = []
    for prev, nxt in zip(seasons, seasons[1:], strict=False):
        if nxt - prev != 1:
            continue
        pair = wide[[prev, nxt]].dropna()
        pairs.append(pd.DataFrame({"prior": pair[prev], "next": pair[nxt]}))

    if not pairs:
        raise ValueError("need at least two consecutive seasons of D/ST history")

    data = pd.concat(pairs, ignore_index=True)
    slope, intercept = np.polyfit(data["prior"], data["next"], 1)
    resid = data["next"] - (slope * data["prior"] + intercept)
    return float(slope), float(intercept), float(resid.std(ddof=2)), len(data)


def project_dst(
    as_of_season: int,
    league: str | LeagueProfile | None = None,
    team_stats: pd.DataFrame | None = None,
    schedules: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Project every team's D/ST for `as_of_season + 1`.

    Output matches `season_model.predict_season` so the rows concatenate
    straight onto the draft board.
    """
    profile = get_profile(league)
    rules = profile.dst_scoring
    if rules is None:
        return pd.DataFrame()

    if team_stats is None:
        team_stats = _load("team_stats")
    if schedules is None:
        schedules = _load("schedules")
    if team_stats is None or schedules is None:
        print(
            "  D/ST skipped: team_stats.parquet or schedules.parquet missing. "
            "Run `nfl-predict update-all`."
        )
        return pd.DataFrame()
    if team_stats.empty or schedules.empty:
        print("  D/ST skipped: team_stats or schedules is empty.")
        return pd.DataFrame()

    totals = dst_season_totals(team_stats, schedules, rules)
    history = totals[totals["season"] <= as_of_season]
    if history["season"].nunique() < 2:
        print("  D/ST skipped: not enough season history to fit.")
        return pd.DataFrame()

    slope, intercept, resid_sd, n_pairs = fit_shrinkage(history)
    print(
        f"  D/ST shrinkage fit on {n_pairs} team-season pairs: "
        f"next = {slope:.2f} x prior + {intercept:.1f}  (resid sd {resid_sd:.1f})"
    )

    latest = history[history["season"] == as_of_season].copy()
    p50 = slope * latest["dst_points"] + intercept

    out = pd.DataFrame(
        {
            "player_id": "DST_" + latest["team"].astype(str),
            "player_name": latest["team"].astype(str) + " D/ST",
            "team": latest["team"].astype(str),
            "position": "DST",
            "season": as_of_season,
            "projected_season": as_of_season + 1,
            "proj_p10": (p50 - _Z90 * resid_sd).round(1),
            "proj_p50": p50.round(1),
            "proj_p90": (p50 + _Z90 * resid_sd).round(1),
        }
    )
    return out.sort_values("proj_p50", ascending=False).reset_index(drop=True)
