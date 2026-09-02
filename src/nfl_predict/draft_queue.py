"""
Build an ESPN autodraft queue from a league's draft board.

ESPN drafts from your queue when you miss a pick, falling back to its own
rankings when the queue runs dry. So the queue is the only way the model's
opinion reaches a pick you are not present for -- which matters when two of
these leagues draft two hours apart on the same evening.

A queue is not just the board's top N. Ranked purely by VOR the first fifteen
names are almost all running backs, and ESPN would happily take six of them
before a quarterback. `build_queue` walks the board in VOR order but caps each
position at the roster target, so the list stays draftable however early it is
reached.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_predict.leagues import LeagueProfile, get_profile

# Enough names to survive a full draft: every pick another team makes can
# remove one of ours, so a queue the length of the roster empties immediately.
DEFAULT_DEPTH = 75

# How many candidates to queue per roster slot. One name per slot is not a
# queue: by the time ESPN reaches it, other teams have taken most of them.
DEPTH_PER_TARGET = 3


def build_queue(
    board: pd.DataFrame,
    profile: str | LeagueProfile | None = None,
    depth: int = DEFAULT_DEPTH,
    drafted: set[str] | None = None,
) -> pd.DataFrame:
    """
    Order a board into an autodraft queue.

    Players are taken in VOR order, so ESPN always reaches for the best one
    still available. Each position is capped at `DEPTH_PER_TARGET` times its
    roster target: capping at the target itself would leave the queue with no
    alternative once those few names are gone, and every pick the other teams
    make can take one. Kickers and defences are held to the tail — they are
    worth about as much as each other, so reaching one early wastes a pick.
    """
    profile = get_profile(profile)
    targets = dict(profile.roster_targets)

    board = board.copy()
    if drafted:
        board = board[~board["player_name"].isin(drafted)]
    board = board.sort_values("overall_rank")

    late = {"K", "DST"}
    tail = sum(targets.get(p, 0) for p in late)

    taken: dict[str, int] = {}
    rows: list[pd.Series] = []

    for _, row in board.iterrows():
        pos = row["position"]
        if pos in late:
            continue
        target = targets.get(pos, 0)
        if not target:
            continue
        # Enough alternatives that a run on the position cannot empty the
        # queue, without letting one position crowd out the rest.
        limit = max(target * DEPTH_PER_TARGET, target + 3)
        if taken.get(pos, 0) >= limit:
            continue
        taken[pos] = taken.get(pos, 0) + 1
        rows.append(row)
        if len(rows) >= depth - tail:
            break

    # Append the best kicker and defence at the tail, where they belong.
    for pos in ("DST", "K"):
        if targets.get(pos):
            best = board[board["position"] == pos].head(targets[pos])
            rows.extend(r for _, r in best.iterrows())

    queue = pd.DataFrame(rows).reset_index(drop=True)
    queue["queue_rank"] = pd.RangeIndex(1, len(queue) + 1)
    return queue[["queue_rank", *(c for c in queue.columns if c != "queue_rank")]]


def render_queue(queue: pd.DataFrame, profile: LeagueProfile) -> str:
    """A numbered list to read while filling ESPN's queue UI."""
    counts = queue["position"].value_counts().to_dict()
    summary = "  ".join(f"{p}{counts[p]}" for p in sorted(counts))

    lines = [
        f"{profile.name} — autodraft queue ({len(queue)} players)",
        f"  {summary}",
        "",
        f"  {'#':>3}  {'player':<26}{'pos':<5}{'team':<6}{'proj':>7}{'vor':>7}{'adp':>7}",
    ]
    for _, r in queue.iterrows():
        adp = f"{r['adp']:.0f}" if pd.notna(r.get("adp")) else "-"
        lines.append(
            f"  {int(r['queue_rank']):>3}  {r['player_name']:<26}"
            f"{r['position']:<5}{str(r['team']):<6}"
            f"{r['proj_p50']:>7.1f}{r['vor']:>7.1f}{adp:>7}"
        )
    return "\n".join(lines)


def write_queue(
    queue: pd.DataFrame, profile: LeagueProfile, out_dir: str | Path = "outputs"
) -> Path:
    """Save the queue beside the board it came from."""
    path = Path(out_dir) / f"draft_queue_{profile.season}_{profile.key}.csv"
    cols = [
        c
        for c in (
            "queue_rank",
            "player_name",
            "position",
            "team",
            "proj_p50",
            "proj_games_p50",
            "vor",
            "adp",
        )
        if c in queue.columns
    ]
    queue[cols].to_csv(path, index=False)
    return path
