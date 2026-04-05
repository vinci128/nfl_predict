"""
Live draft assistant for fantasy football.

Tracks the state of an in-progress snake or auction draft, marks players
as drafted, and suggests best-available players based on VOR and roster
needs.  State is persisted to JSON so a session can survive restarts.

Optional LLM advisor: pass ``--llm`` on the CLI (or call
``get_llm_suggestion`` directly) to get natural-language reasoning from
Claude about which pick to make.  Requires ``anthropic`` to be installed:

    pip install "nfl-predict[draft]"
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

OUTPUT_DIR = Path("outputs")

# Default starter roster slots (standard ESPN/Yahoo)
_DEFAULT_SLOTS: dict[str, int] = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "K": 1,
    "FLEX": 1,  # RB/WR/TE
}

# Minimum bench depth targets (starters + bench)
_DEFAULT_ROSTER_TARGETS: dict[str, int] = {
    "QB": 2,
    "RB": 5,
    "WR": 5,
    "TE": 2,
    "K": 1,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PickRecord:
    """One pick entry in the draft log."""

    overall_pick: int
    round: int
    pick_in_round: int
    player_name: str
    position: str
    team: str
    proj_p50: float
    vor: float
    drafter: str  # "me" or a team name / number


@dataclass
class DraftState:
    """
    Full state of an ongoing fantasy draft.

    Attributes
    ----------
    board           : Full draft board DataFrame (loaded once from CSV/build).
    available       : Subset of board not yet drafted.
    picks           : All picks made so far (in order).
    my_picks        : Only the user's picks.
    my_roster       : {position: [player_name, ...]} for the user's team.
    league_size     : Number of teams in the league.
    draft_position  : User's pick slot (1-based, for snake positioning).
    current_pick    : Overall pick number for the next pick.
    state_path      : Path to persist state JSON.
    """

    board: pd.DataFrame
    available: pd.DataFrame
    picks: list[PickRecord] = field(default_factory=list)
    my_picks: list[PickRecord] = field(default_factory=list)
    my_roster: dict[str, list[str]] = field(
        default_factory=lambda: {pos: [] for pos in _DEFAULT_ROSTER_TARGETS}
    )
    league_size: int = 12
    draft_position: int = 1
    current_pick: int = 1
    state_path: Path = field(default_factory=lambda: OUTPUT_DIR / "draft_state.json")


# ---------------------------------------------------------------------------
# State initialisation
# ---------------------------------------------------------------------------


def init_draft_state(
    board: pd.DataFrame,
    league_size: int = 12,
    draft_position: int = 1,
    state_path: Path | None = None,
) -> DraftState:
    """
    Initialise a fresh DraftState from a draft board.

    Parameters
    ----------
    board          : DataFrame from build_draft_board()
    league_size    : Number of teams
    draft_position : User's draft slot (1 = first pick)
    state_path     : Where to persist state; defaults to outputs/draft_state.json
    """
    if state_path is None:
        state_path = OUTPUT_DIR / "draft_state.json"

    return DraftState(
        board=board.copy(),
        available=board.copy(),
        league_size=league_size,
        draft_position=draft_position,
        state_path=state_path,
    )


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def mark_drafted(
    state: DraftState,
    player_name: str,
    drafter: str = "other",
    player_id: str | None = None,
) -> DraftState:
    """
    Record a pick and remove the player from the available board.

    Parameters
    ----------
    state       : Current DraftState
    player_name : Player's display name (case-insensitive, partial match OK)
    drafter     : "me" for user's pick, or any label for an opponent
    player_id   : Optional exact player_id for unambiguous lookup (preferred
                  when available, e.g. from the board UI buttons)

    Returns
    -------
    Updated DraftState (mutates in-place and returns self).
    """
    avail = state.available
    name_col = "player_name"

    # 1. Exact player_id match (most reliable — no name collision possible)
    if player_id and "player_id" in avail.columns:
        match = avail[avail["player_id"] == player_id]
        if not match.empty:
            pass  # found it
        else:
            player_id = None  # fall through to name matching

    if not player_id or "player_id" not in avail.columns:
        # 2. Exact name match
        match = avail[avail[name_col].str.lower() == player_name.lower()]
        # 3. Case-insensitive substring
        if match.empty:
            match = avail[
                avail[name_col].str.lower().str.contains(player_name.lower(), na=False)
            ]

    if match.empty:
        raise ValueError(
            f"Player '{player_name}' not found in available board. "
            f"Check spelling or use a unique substring."
        )
    if len(match) > 1:
        names = match[name_col].tolist()
        raise ValueError(
            f"Ambiguous match for '{player_name}': {names}. Use a more specific name."
        )

    row = match.iloc[0]
    n_picks = len(state.picks)
    rnd = (n_picks // state.league_size) + 1
    pick_in_rnd = (n_picks % state.league_size) + 1

    record = PickRecord(
        overall_pick=state.current_pick,
        round=rnd,
        pick_in_round=pick_in_rnd,
        player_name=str(row[name_col]),
        position=str(row.get("position", "")),
        team=str(row.get("team", "")),
        proj_p50=float(row.get("proj_p50", 0.0)),
        vor=float(row.get("vor", 0.0)),
        drafter=drafter,
    )

    state.picks.append(record)
    state.current_pick += 1

    if drafter == "me":
        state.my_picks.append(record)
        pos = record.position
        state.my_roster.setdefault(pos, []).append(record.player_name)

    # Remove from available board
    state.available = state.available[
        state.available[name_col] != row[name_col]
    ].reset_index(drop=True)

    return state


def undo_last_pick(state: DraftState) -> DraftState:
    """
    Reverse the most recent pick.

    Removes the last entry from picks, restores the player to the available
    board, updates my_roster if it was the user's pick, and decrements the
    pick counter.  A no-op if no picks have been recorded yet.

    Returns
    -------
    Updated DraftState.
    """
    if not state.picks:
        return state

    last = state.picks.pop()

    if last.drafter == "me":
        # Also remove from my_picks
        state.my_picks = [
            p for p in state.my_picks if p.overall_pick != last.overall_pick
        ]
        # Remove from my_roster
        roster_slot = state.my_roster.get(last.position, [])
        if last.player_name in roster_slot:
            roster_slot.remove(last.player_name)

    # Restore the player row from the full board
    board_row = state.board[state.board["player_name"] == last.player_name]
    if not board_row.empty:
        state.available = (
            pd.concat([state.available, board_row], ignore_index=True)
            .sort_values("overall_rank")
            .reset_index(drop=True)
        )

    state.current_pick = max(1, state.current_pick - 1)
    return state


def suggest_best_available(
    state: DraftState,
    needs: list[str] | None = None,
    n: int = 5,
) -> pd.DataFrame:
    """
    Return the top-N available players by VOR, with optional positional weighting.

    If ``needs`` is provided (e.g. ``["RB", "WR"]``), players at those
    positions are boosted to the top before falling back to pure VOR order.

    Parameters
    ----------
    state : Current DraftState
    needs : Positions to prioritise (e.g. from roster analysis)
    n     : Number of suggestions to return

    Returns
    -------
    DataFrame of top-N recommendations with key columns.
    """
    avail = state.available.copy()
    if avail.empty:
        return pd.DataFrame()

    display_cols = [
        c
        for c in (
            "overall_rank",
            "tier",
            "player_name",
            "position",
            "team",
            "proj_p10",
            "proj_p50",
            "proj_p90",
            "vor",
            "pos_rank",
        )
        if c in avail.columns
    ]

    if needs:
        needs_upper = [p.upper() for p in needs]
        priority = avail[avail["position"].isin(needs_upper)].sort_values(
            "vor", ascending=False
        )
        rest = avail[~avail["position"].isin(needs_upper)].sort_values(
            "vor", ascending=False
        )
        ordered = pd.concat([priority, rest], ignore_index=True)
    else:
        ordered = avail.sort_values("vor", ascending=False)

    return ordered[display_cols].head(n).reset_index(drop=True)


def analyse_roster_needs(state: DraftState) -> list[str]:
    """
    Identify positions where the user's roster is below the target depth.

    Returns a list of positions (most urgent first) that still need filling.
    """
    needs: list[tuple[int, str]] = []
    for pos, target in _DEFAULT_ROSTER_TARGETS.items():
        current = len(state.my_roster.get(pos, []))
        deficit = target - current
        if deficit > 0:
            needs.append((deficit, pos))

    # Sort by deficit (descending) so the most needed position comes first
    needs.sort(reverse=True)
    return [pos for _, pos in needs]


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------


def render_board(
    state: DraftState,
    n: int = 50,
    show_drafted: bool = False,
) -> str:
    """
    Render a formatted ASCII draft board.

    Shows the top-N available players plus a summary of user's roster.

    Parameters
    ----------
    state        : Current DraftState
    n            : Number of available players to display
    show_drafted : If True, also list recently drafted players at the bottom
    """
    lines: list[str] = []

    # Header
    pick = state.current_pick
    rnd = ((pick - 1) // state.league_size) + 1
    lines.append("=" * 80)
    lines.append(
        f"  DRAFT BOARD — Pick #{pick}  Round {rnd}  "
        f"({len(state.available)} players available)"
    )
    lines.append("=" * 80)

    # Available players table
    avail = state.available.sort_values("vor", ascending=False).head(n)
    col_fmt = "{:>4}  {:<28}  {:>4}  {:>5}  {:>7}  {:>7}  {:>7}  {:>7}"
    lines.append(
        col_fmt.format(
            "Rank", "Player", "Pos", "Tier", "Floor", "Median", "Ceil", "VOR"
        )
    )
    lines.append("-" * 80)

    for _, row in avail.iterrows():
        lines.append(
            col_fmt.format(
                int(row.get("overall_rank", 0)),
                str(row.get("player_name", ""))[:28],
                str(row.get("position", "")),
                int(row.get("tier", 0)),
                f"{row.get('proj_p10', 0.0):.0f}",
                f"{row.get('proj_p50', 0.0):.0f}",
                f"{row.get('proj_p90', 0.0):.0f}",
                f"{row.get('vor', 0.0):.1f}",
            )
        )

    # My roster summary
    lines.append("")
    lines.append("  MY ROSTER")
    lines.append("  " + "-" * 40)
    for pos in ("QB", "RB", "WR", "TE", "K"):
        players = state.my_roster.get(pos, [])
        target = _DEFAULT_ROSTER_TARGETS.get(pos, 0)
        status = "✓" if len(players) >= target else f"need {target - len(players)}"
        player_str = ", ".join(players) if players else "(empty)"
        lines.append(f"  {pos:<4} [{status}]  {player_str}")

    # Recent picks
    if show_drafted and state.picks:
        lines.append("")
        lines.append("  RECENT PICKS (last 10)")
        lines.append("  " + "-" * 40)
        for p in state.picks[-10:]:
            marker = "← ME" if p.drafter == "me" else ""
            lines.append(
                f"  #{p.overall_pick:>3}  {p.player_name:<28}  {p.position}  "
                f"{p.drafter}  {marker}"
            )

    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_state(state: DraftState) -> None:
    """Persist draft state to JSON (board and available stored as CSV strings)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "league_size": state.league_size,
        "draft_position": state.draft_position,
        "current_pick": state.current_pick,
        "my_roster": state.my_roster,
        "picks": [asdict(p) for p in state.picks],
        "my_picks": [asdict(p) for p in state.my_picks],
        "board_csv": state.board.to_csv(index=False),
        "available_csv": state.available.to_csv(index=False),
    }
    state.state_path.write_text(json.dumps(payload, indent=2))


def load_state(path: Path | str | None = None) -> DraftState:
    """
    Load a previously saved DraftState from JSON.

    Parameters
    ----------
    path : JSON file path; defaults to outputs/draft_state.json
    """
    if path is None:
        path = OUTPUT_DIR / "draft_state.json"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No draft state found at {path}. Run `nfl-predict draft-start` first."
        )
    payload = json.loads(path.read_text())

    import io

    board = pd.read_csv(io.StringIO(payload["board_csv"]))
    available = pd.read_csv(io.StringIO(payload["available_csv"]))

    state = DraftState(
        board=board,
        available=available,
        picks=[PickRecord(**p) for p in payload["picks"]],
        my_picks=[PickRecord(**p) for p in payload["my_picks"]],
        my_roster=payload["my_roster"],
        league_size=payload["league_size"],
        draft_position=payload["draft_position"],
        current_pick=payload["current_pick"],
        state_path=path,
    )
    return state


# ---------------------------------------------------------------------------
# Optional LLM advisor (requires `pip install anthropic`)
# ---------------------------------------------------------------------------


def get_llm_suggestion(
    state: DraftState,
    needs: list[str] | None = None,
    n_context: int = 25,
    api_key: str | None = None,
) -> str:
    """
    Ask Claude for a natural-language draft recommendation.

    Serialises the draft state into a structured prompt and calls the
    Anthropic API.  Returns a text explanation of the recommended pick.

    Parameters
    ----------
    state     : Current DraftState
    needs     : Positional needs (auto-detected if None)
    n_context : Number of top-available players to include in prompt
    api_key   : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)

    Requires
    --------
    anthropic >= 0.20.0  (install with: pip install "nfl-predict[draft]")
    """
    try:
        import anthropic  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise ImportError(
            "The 'anthropic' package is required for LLM suggestions. "
            "Install it with: pip install anthropic"
        ) from exc

    if needs is None:
        needs = analyse_roster_needs(state)

    top_available = suggest_best_available(state, n=n_context)
    my_picks_summary = [
        f"{p.player_name} ({p.position}, round {p.round})" for p in state.my_picks
    ]

    pick = state.current_pick
    rnd = ((pick - 1) // state.league_size) + 1

    prompt = textwrap.dedent(f"""
        You are an expert fantasy football draft advisor.

        DRAFT CONTEXT:
        - Overall pick: #{pick}, Round {rnd}
        - League size: {state.league_size} teams
        - My draft position: #{state.draft_position}
        - Positional needs (most urgent first): {needs or "balanced"}

        MY CURRENT ROSTER:
        {json.dumps(state.my_roster, indent=2)}

        MY PICKS SO FAR:
        {chr(10).join(f"  - {p}" for p in my_picks_summary) or "  (none yet)"}

        TOP {n_context} AVAILABLE PLAYERS (sorted by VOR — Value Over Replacement):
        {top_available.to_string(index=False)}

        Based on this context, recommend the single best pick I should make right
        now and explain why in 2-3 sentences.  Consider roster construction,
        positional scarcity, upside vs. floor, and Value Over Replacement.
        Be direct — give me a name and a reason.
    """).strip()

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
