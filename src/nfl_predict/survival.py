"""Lineup choice for an elimination league, where only last place matters.

Royal Rumble is not head-to-head. All eight teams share one matchup each
week, there are seven matchup periods for eight teams, `playoffTeamCount` is
0, and every team carries an `eliminationMatchupPeriod` and an
`eliminationProbabilityStatus` (DANGER for the team in last, CAUTION for the
rest). One team goes out per week until one is left.

That changes the objective. `lineup.espn_optimal` maximises expected points,
which is right when you are trying to outscore one specific opponent. Here
points above survival buy nothing: the only question is whether somebody
else finishes below you. So this module maximises **P(not last)** instead,
which is a different lineup whenever the two disagree.

They disagree in a specific and useful way. Comfortably clear of the cut,
the highest-mean lineup is often *worse* than a lower-mean, lower-variance
one, because it widens the tail that reaches down past a rival. Near the
bottom it inverts: variance is the only thing that can still save you, and
the safe lineup is the one that loses slowly. Neither behaviour is
reachable from an expected-points objective, which has no notion of a
threshold at all.

How the spread is estimated
---------------------------
A player's week-to-week SD is fitted per position from that league's own
history, as ``SD = a + b * mean`` over quartiles of his 8-week average, and
cached per league. The fit has to be per league because the target is
fantasy points under that league's rules — Royal Rumble pays 0.04/yd
passing where Ludopathy pays a bucketed 1/10, so a QB's spread is not the
same number in both.

Measured on Royal Rumble, 2021-2025:

    QB   SD ~ 7.6 + 0.03*mean   (essentially flat — a 21-pt QB is no more
                                 volatile in absolute terms than a 7-pt one)
    RB   SD ~ 4.1 + 0.29*mean
    WR   SD ~ 3.7 + 0.33*mean
    TE   SD ~ 2.5 + 0.43*mean   (scales steepest)
    K    SD ~ 4.7 - 0.01*mean   (flat)

So "bench the volatile one" is not a uniform instruction: at QB and K the
projection carries no variance information at all, and the choice there
should be made on the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Positions with no history in a league's feature table — D/ST, which
# `dst.py` models separately and never writes weekly rows for. Proportional
# to the mean, in the middle of the range the fitted positions occupy.
_FALLBACK = (3.0, 0.35)

# Enough draws that P(last) is stable to well under a percentage point,
# which is finer than any decision made on it.
DEFAULT_SIMS = 20_000

_MODEL_CACHE: dict[str, dict[str, tuple[float, float]]] = {}


def variance_model(league: str | None = None) -> dict[str, tuple[float, float]]:
    """Fit ``SD = a + b * mean`` per position from one league's own history.

    Cached per artifact key: the fit is over that league's scoring, and two
    leagues that score differently get different numbers.
    """
    from nfl_predict.leagues import get_profile

    profile = get_profile(league)
    key = profile.artifact_key
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    f = pd.read_parquet(profile.features_path)
    d = f[(f["season"] >= 2021) & (f["week"] <= 18)]
    d = d[d["fantasy_points_custom"].notna() & d["fantasy_points_custom_roll8"].notna()]
    d = d[d["fantasy_points_custom_roll8"] > 1]

    model: dict[str, tuple[float, float]] = {}
    for pos, s in d.groupby("position"):
        if len(s) < 400:
            continue
        try:
            q = pd.qcut(
                s["fantasy_points_custom_roll8"], 4, labels=False, duplicates="drop"
            )
        except ValueError:
            continue
        xs, ys = [], []
        for _, g in s.groupby(q):
            xs.append(g["fantasy_points_custom_roll8"].mean())
            ys.append(g["fantasy_points_custom"].std())
        if len(xs) < 2 or any(pd.isna(y) for y in ys):
            continue
        b, a = np.polyfit(xs, ys, 1)
        # A negative slope is real at QB and K (flat variance, noise in the
        # fit) but must never drive SD below a sane floor at high means.
        model[str(pos)] = (float(a), float(b))

    _MODEL_CACHE[key] = model
    return model


def player_sd(
    position: str, mean: float, model: dict[str, tuple[float, float]]
) -> float:
    """Standard deviation for one player-week, floored at a quarter of the mean.

    The floor matters at QB, where the fitted slope is ~0: without it a
    30-point projection would carry the same spread as a 7-point one and the
    optimiser would treat an elite QB as risk-free.
    """
    a, b = model.get(position, _FALLBACK)
    return float(max(a + b * mean, 0.25 * mean, 1.0))


def simulate_last_place(
    teams: dict[str, dict], sims: int = DEFAULT_SIMS, seed: int | None = 0
) -> dict[str, float]:
    """P(each team finishes last) given settled points and unplayed spreads.

    ``teams`` maps a team id to ``{"settled": float, "means": [...],
    "sds": [...]}`` — what is already banked, and the mean/SD of each starter
    still to play. Returns a probability per team; they sum to 1 up to ties.
    """
    rng = np.random.default_rng(seed)
    ids = list(teams)
    totals = np.empty((len(ids), sims))
    for i, tid in enumerate(ids):
        t = teams[tid]
        total = np.full(sims, float(t["settled"]))
        for mu, sd in zip(t["means"], t["sds"], strict=True):
            # Truncated at zero: a fantasy week can be negative in principle
            # but a normal tail that deep is a modelling artefact, not a risk
            # anyone is exposed to.
            total += np.maximum(rng.normal(mu, sd, sims), 0.0)
        totals[i] = total

    lowest = totals.argmin(axis=0)
    counts = np.bincount(lowest, minlength=len(ids))
    return {tid: float(counts[i]) / sims for i, tid in enumerate(ids)}


def league_teams(league: str | None = None, week: int | None = None) -> dict[str, dict]:
    """Every team's settled points and unplayed starter distributions.

    Read from each team's *current* lineup, which is the only thing anyone's
    final score can come from. A rival's bench is irrelevant: he may improve
    it before kickoff, but we cannot see that and guessing at it would make
    our own choice worse, not better.
    """
    from nfl_predict.espn_fantasy import EspnFantasyClient
    from nfl_predict.leagues import fantasy_position, get_profile

    profile = get_profile(league)
    model = variance_model(profile.key)
    rows = EspnFantasyClient.from_env(profile.key).fetch_rosters(week=week)

    teams: dict[str, dict] = {}
    for r in rows:
        if r["lineup_slot"] in ("BE", "IR"):
            continue
        t = teams.setdefault(
            r["team_id"],
            {
                "name": r["team_name"],
                "settled": 0.0,
                "means": [],
                "sds": [],
                "is_mine": bool(r["is_mine"]),
            },
        )
        if r["actual_points"] is not None:
            t["settled"] += r["actual_points"]
            continue
        mu = r["projected_points"] or 0.0
        pos = fantasy_position(r["position"]) or r["position"]
        t["means"].append(mu)
        t["sds"].append(player_sd(pos, mu, model))
    return teams


def _as_team(players: list[dict], model: dict[str, tuple[float, float]]) -> dict:
    """Turn one candidate lineup into a simulate_last_place entry."""
    settled = sum(p["expected_ppr_points"] or 0.0 for p in players if p.get("locked"))
    live = [p for p in players if not p.get("locked")]
    return {
        "settled": settled,
        "means": [p["expected_ppr_points"] or 0.0 for p in live],
        "sds": [
            player_sd(p["position"], p["expected_ppr_points"] or 0.0, model)
            for p in live
        ],
    }


def survival_optimal(
    league: str | None = None, week: int | None = None, sims: int = DEFAULT_SIMS
):
    """Pick the lineup that minimises P(finishing last).

    Starts from the highest-mean lineup and accepts single swaps that lower
    the probability, until none does. Returns
    ``(starters, bench, p_last_before, p_last_after)``.
    """
    from nfl_predict.leagues import get_profile
    from nfl_predict.lineup import espn_optimal

    profile = get_profile(league)
    model = variance_model(profile.key)
    starters, bench, _out, _total = espn_optimal(profile.key, week)
    others = {
        tid: t for tid, t in league_teams(profile.key, week).items() if not t["is_mine"]
    }

    def p_last(cand: list[dict]) -> float:
        board = dict(others)
        board["__me__"] = _as_team(cand, model)
        return simulate_last_place(board, sims=sims)["__me__"]

    flex = set(profile.roster.flex_positions)

    def eligible(player: dict, slot: str) -> bool:
        return player["position"] == slot or (
            slot == "FLEX" and player["position"] in flex
        )

    best, best_p = list(starters), p_last(starters)
    start_p = best_p
    pool = list(bench)

    improved = True
    while improved:
        improved = False
        for i, s in enumerate(best):
            if s.get("locked"):
                continue  # his game has kicked off; the slot cannot change
            for j, b in enumerate(pool):
                if b.get("locked") or not eligible(b, s["slot"]):
                    continue
                cand = list(best)
                cand[i] = {**b, "slot": s["slot"]}
                p = p_last(cand)
                if p < best_p - 1e-4:
                    best, best_p = cand, p
                    pool = list(pool)
                    pool[j] = {**s, "slot": "BN"}
                    improved = True
                    break
            if improved:
                break

    return best, pool, start_p, best_p
