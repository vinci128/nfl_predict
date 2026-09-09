"""Week-1 opponent adjustment, built from the prior season's defences.

Why this module exists at all
-----------------------------
`predict_week` cannot run in week 1 — it predicts week *W* from week *W-1*
features and raises below week 2 — and in any case the rolling opponent
columns it would use (`opp_points_allowed_roll5`) need games that have not
been played yet. So the only projection available for week 1 is the season
model's per-game rate (`proj_ppg_p50`), which knows nothing about who the
player actually faces.

This fills exactly that gap, and nothing else. From week 2 the weekly model
consumes the opponent features itself and this module should not be layered
on top of it.

What the numbers are
--------------------
Measured on 2016-2025, on the identical estimator this applies — a defence's
*prior-season* points allowed per game to a position, against a player's
actual points in weeks 1-4 of the following season, relative to his own
prior-season rate:

    pos   slope (pts per z)   t
    QB          1.775        4.47
    RB          0.539        3.49
    WR          0.177        1.37
    TE          0.155        1.04
    LB          0.119        1.29
    DL          0.194        3.37
    DB          0.148        2.03

Only QB and RB are applied. WR/TE/LB miss significance outright. DL and DB
clear it but move a player by ~0.3 points across the whole range of
defences, which cannot change a lineup and would only add noise to it — the
adjustment is deliberately not applied where it cannot matter.

Using a prior *season* rather than a rolling in-season window is not the
compromise it looks like. Measured on the same data, a defence's rate
against a position correlates r=0.15-0.33 with its own rate the next season,
against r=0.14-0.30 for the first half of a season against its own second
half. Season-to-season carryover is as strong as within-season stability;
nine games is simply a small sample.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")

# Points per standard deviation of opponent strength. Positions absent here
# are left untouched — see the module docstring for why.
MATCHUP_COEF: dict[str, float] = {"QB": 1.775, "RB": 0.539}

# A z-score beyond this is a small-sample artefact rather than a real
# defence, and would otherwise hand a QB several points on the strength of
# one lopsided prior season.
Z_CLIP = 2.0


def prior_season_dvp(features: pd.DataFrame, season: int) -> pd.DataFrame:
    """Rate each defence by what it allowed per game to each position.

    Uses ``season - 1``, scored under whatever league the feature table was
    built for — points allowed is league-specific, so this must be handed
    that league's own table.

    Returns columns ``team, position, dvp, dvp_z``, where ``team`` is the
    *defence*.
    """
    prior = season - 1
    d = features[
        (features["season"] == prior)
        & (features["week"] <= 18)
        & features["fantasy_points_custom"].notna()
        & features["opponent_team"].notna()
    ]
    if d.empty:
        return pd.DataFrame(columns=pd.Index(["team", "position", "dvp", "dvp_z"]))

    allowed = (
        d.groupby(["opponent_team", "position"])["fantasy_points_custom"]
        .sum()
        .reset_index(name="allowed")
    )
    games = d.groupby("opponent_team")["week"].nunique().reset_index(name="games")
    dvp = allowed.merge(games, on="opponent_team")
    dvp = dvp[dvp["games"] > 0]
    dvp["dvp"] = dvp["allowed"] / dvp["games"]
    dvp["dvp_z"] = dvp.groupby("position")["dvp"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() else 0.0
    )
    dvp["dvp_z"] = dvp["dvp_z"].fillna(0.0).clip(-Z_CLIP, Z_CLIP)
    return dvp.rename(columns={"opponent_team": "team"})[
        ["team", "position", "dvp", "dvp_z"]
    ]


def week_opponents(season: int, week: int) -> pd.DataFrame:
    """Who each team plays in one week, from the schedule.

    Returns ``team, opponent, is_home, spread_line, total_line`` with one row
    per team. A team on bye is simply absent.
    """
    path = DATA_DIR / "schedules.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `nfl-predict update-all`.")
    s = pd.read_parquet(path)
    g = s[(s["season"] == season) & (s["week"] == week)]
    if g.empty:
        return pd.DataFrame(
            columns=pd.Index(
                ["team", "opponent", "is_home", "spread_line", "total_line"]
            )
        )

    cols = ["spread_line", "total_line"]
    for c in cols:
        if c not in g.columns:
            g[c] = pd.NA

    home = g.rename(columns={"home_team": "team", "away_team": "opponent"}).assign(
        is_home=True
    )
    away = g.rename(columns={"away_team": "team", "home_team": "opponent"}).assign(
        is_home=False
    )
    out = pd.concat([home, away], ignore_index=True)
    return out[["team", "opponent", "is_home", *cols]]


def apply_matchup(
    players: pd.DataFrame,
    features: pd.DataFrame,
    season: int,
    week: int,
    points_col: str = "expected_ppr_points",
) -> pd.DataFrame:
    """Adjust a week-1 projection for the opponent each player faces.

    ``players`` needs ``position``, ``nfl_team`` and ``points_col``. Adds:

    ``opponent``      who they face (NA on a bye or an unknown team)
    ``dvp_z``         opponent strength, +ve = concedes more than average
    ``matchup_delta`` points added, 0 where the position is not adjusted
    ``points_col``    adjusted in place; the original is kept as
                      ``<points_col>_base``

    A player whose team is missing from the schedule keeps his base
    projection and gets a delta of 0 — never a silent drop.
    """
    out = players.copy()
    base = f"{points_col}_base"
    out[base] = out[points_col]
    out["opponent"] = pd.NA
    out["dvp_z"] = 0.0
    out["matchup_delta"] = 0.0

    sched = week_opponents(season, week)
    if sched.empty:
        return out

    dvp = prior_season_dvp(features, season)
    opp = dict(zip(sched["team"], sched["opponent"], strict=False))
    z = dict(
        zip(
            zip(dvp["team"], dvp["position"], strict=False),
            dvp["dvp_z"],
            strict=False,
        )
    )

    for i, row in out.iterrows():
        team = row.get("nfl_team")
        pos = row.get("position")
        if not team or team not in opp:
            continue
        out.at[i, "opponent"] = opp[team]
        coef = MATCHUP_COEF.get(pos)
        zz = z.get((opp[team], pos), 0.0)
        out.at[i, "dvp_z"] = round(float(zz), 2)
        if coef is None:
            continue
        delta = coef * float(zz)
        out.at[i, "matchup_delta"] = round(delta, 2)
        out.at[i, points_col] = round(float(row[points_col]) + delta, 1)

    return out
