"""
NFL Fantasy Agent powered by Claude Opus 4.6.

Manages your NFL.com Fantasy team autonomously:
- Sets optimal weekly lineups
- Claims waiver wire players
- Analyzes and acts on trade offers
- Assists during drafts

Usage:
    from nfl_predict.nfl_agent import run_agent
    asyncio.run(run_agent())

Or via CLI:
    nfl-predict agent
    nfl-predict agent --task "Check and set my lineup for this week"
    nfl-predict agent --draft   # Run in draft mode (requires nfl-sync in another terminal)
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import anthropic
from dotenv import load_dotenv

from .fantasy_client import NFLFantasyClient
from .predict_week import run_predictions

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DESTRUCTIVE_TOOLS = {"set_lineup", "claim_waiver_player", "accept_trade", "decline_trade", "propose_trade", "make_draft_pick"}

SYSTEM_PROMPT = """You are an expert NFL Fantasy Football manager operating autonomously on behalf of the user.

Your goal is to maximize fantasy points and win the league using data-driven decisions backed by machine learning predictions.

## Available capabilities
- View and analyze the current roster
- Get ML-model predictions (PPR scoring) for all skill positions (QB, RB, WR, TE, K)
- Browse the waiver wire for available players
- View current week's matchup
- Set the optimal starting lineup
- Claim waiver wire players (add/drop)
- Analyze, accept, or decline trade offers
- Propose trades to other teams

## Decision guidelines

### Lineup
- Never start players listed as Out or IR
- Be cautious with Doubtful players (start only if clearly best option)
- For Questionable players, weigh the risk against the backup option
- Start the player with the highest predicted PPR points at each position
- Consider matchups when predictions are close (within 1-2 points)

### Waiver wire
- Prioritize players who recently became clear starters or have an increased role
- Consider season-long value, not just this week
- Don't drop healthy starters for a one-week streaming play unless the upside is huge
- Target handcuffs for your star RBs if they're available

### Trades
- Evaluate trades based on projected remaining-season value, not just one week
- Factor in bye weeks and injury history
- Be willing to sell high on players who had outlier weeks
- Buy low on injured players returning soon with a clear role

## Workflow
1. Always gather information first (roster, predictions, waiver wire, matchup, trade offers)
2. Analyze the data thoroughly
3. Explain your reasoning before taking action
4. Take actions one at a time (each destructive action requires user confirmation)
5. After acting, summarize what was done and what the impact should be

## Scoring system (PPR variant)
- Passing: 1pt / 10 yds, 4pt TD, -2pt INT
- Rushing: 1pt / 10 yds, 6pt TD
- Receiving: 1pt / reception, 1pt / 10 yds, 6pt TD
- Kicking: PAT 1pt, FG 0-39yd 3pt, FG 40-49yd 4pt, FG 50+yd 5pt

Always be concise, data-driven, and decisive."""


DRAFT_SYSTEM_PROMPT = """You are an expert NFL Fantasy Football draft assistant operating in real-time during a draft.

IMPORTANT: You are running alongside `nfl-sync` in a separate terminal.
- `nfl-sync` is the SOLE process that records picks into draft_state.json.
- You NEVER record picks yourself. You only READ state and MAKE picks when it's your turn.
- Always read get_local_draft_state first to see the current board before acting.

## Your workflow each loop
1. Call get_local_draft_state — see current pick number, my roster, top available players.
2. Call get_draft_turn_status — check live on NFL.com if it is currently our pick turn.
3. If NOT our turn: note who was just picked (compare pick count to last loop) and wait.
4. If OUR TURN: call get_draft_rankings for 1-2 key positions, choose the best pick,
   explain in 1-2 sentences, then call make_draft_pick.

## Draft strategy
- Rounds 1-3: Elite RBs and WRs by VOR. Avoid QB/TE/K.
- Rounds 4-6: Fill RB/WR depth; top TE (Kelce/Andrews tier) if available.
- Rounds 7-9: QB1, WR depth, TE if not taken.
- Rounds 10-12: Handcuffs for your RBs, upside WRs.
- Rounds 13+: Streamers, K, DEF.

Be fast and decisive. Give a player name and a one-sentence reason."""


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def get_tools(draft_mode: bool = False) -> list[dict]:
    common_tools = [
        {
            "name": "get_my_roster",
            "description": (
                "Get the current team roster from NFL.com Fantasy. "
                "Returns all players with position, injury status, lineup slot, and projected points."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_waiver_wire",
            "description": (
                "Get available players on the waiver wire from NFL.com Fantasy. "
                "Optionally filter by position."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "string",
                        "description": "Filter by position (optional)",
                        "enum": ["QB", "RB", "WR", "TE", "K", "DEF"],
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max players to return (default 40)",
                        "default": 40,
                    },
                },
                "required": [],
            },
        },
        {
            "name": "get_matchup",
            "description": "Get the current week matchup: scores, opponent, projected totals.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_trade_offers",
            "description": "Get all pending incoming trade offers.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_league_teams",
            "description": "Get all teams in the league (names and IDs), useful for proposing trades.",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "set_lineup",
            "description": (
                "DESTRUCTIVE — Sets the starting lineup for the current week. "
                "Requires user confirmation. "
                "Provide the list of player IDs to start and a clear explanation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "starter_player_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Player IDs to set as starters",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Why these players were chosen (shown to user for confirmation)",
                    },
                },
                "required": ["starter_player_ids", "explanation"],
            },
        },
        {
            "name": "claim_waiver_player",
            "description": (
                "DESTRUCTIVE — Add a player from waivers and drop a player from the roster. "
                "Requires user confirmation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "add_player_id": {
                        "type": "string",
                        "description": "Player ID to add from waivers",
                    },
                    "add_player_name": {
                        "type": "string",
                        "description": "Name of player to add (shown to user)",
                    },
                    "drop_player_id": {
                        "type": "string",
                        "description": "Player ID to drop from roster",
                    },
                    "drop_player_name": {
                        "type": "string",
                        "description": "Name of player to drop (shown to user)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this move improves the team",
                    },
                },
                "required": [
                    "add_player_id", "add_player_name",
                    "drop_player_id", "drop_player_name", "reasoning",
                ],
            },
        },
        {
            "name": "accept_trade",
            "description": (
                "DESTRUCTIVE — Accept a trade offer. Requires user confirmation. "
                "Only use after thorough analysis showing this benefits the team."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "trade_id": {"type": "string", "description": "Trade offer ID"},
                    "reasoning": {
                        "type": "string",
                        "description": "Why accepting this trade is good",
                    },
                },
                "required": ["trade_id", "reasoning"],
            },
        },
        {
            "name": "decline_trade",
            "description": "DESTRUCTIVE — Decline a trade offer. Requires user confirmation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "trade_id": {"type": "string"},
                    "reasoning": {"type": "string", "description": "Why declining is correct"},
                },
                "required": ["trade_id", "reasoning"],
            },
        },
        {
            "name": "propose_trade",
            "description": (
                "DESTRUCTIVE — Propose a trade to another team. Requires user confirmation. "
                "Specify which players to offer and which to request."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "target_team_id": {
                        "type": "string",
                        "description": "Team ID of the trade partner",
                    },
                    "target_team_name": {
                        "type": "string",
                        "description": "Name of the trade partner team (shown to user)",
                    },
                    "offer_player_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Player IDs you will give up",
                    },
                    "offer_player_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of players you will give up (shown to user)",
                    },
                    "request_player_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Player IDs you want to receive",
                    },
                    "request_player_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of players you want to receive (shown to user)",
                    },
                    "message": {
                        "type": "string",
                        "description": "Optional message to include with the trade offer",
                        "default": "",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Why this trade makes sense for both teams",
                    },
                },
                "required": [
                    "target_team_id", "target_team_name",
                    "offer_player_ids", "offer_player_names",
                    "request_player_ids", "request_player_names",
                    "reasoning",
                ],
            },
        },
    ]

    if draft_mode:
        draft_tools = [
            {
                "name": "get_local_draft_state",
                "description": (
                    "Read the current draft state from draft_state.json, kept up to date by "
                    "`nfl-sync` running in another terminal. Returns available players (with "
                    "VOR/projections), my current roster, pick count, and positional needs. "
                    "Use this as your primary source of draft board information — do not "
                    "poll NFL.com for pick history."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "top_n": {
                            "type": "integer",
                            "description": "Number of top available players to return (default 50)",
                            "default": 50,
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "get_draft_rankings",
                "description": (
                    "Get VOR-based season-total draft rankings for a specific position from the "
                    "local draft board. Returns overall_rank, tier, proj_p10, proj_p50, proj_p90, "
                    "vor, pos_rank for available (undrafted) players only. Use for deep position "
                    "analysis before committing to a pick."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "position": {
                            "type": "string",
                            "description": "Position to get rankings for",
                            "enum": ["QB", "RB", "WR", "TE", "K"],
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Number of players to return (default 30)",
                            "default": 30,
                        },
                    },
                    "required": ["position"],
                },
            },
            {
                "name": "get_draft_turn_status",
                "description": (
                    "Check live on NFL.com whether it is currently our turn to pick. "
                    "Returns is_my_turn, pick_number, and time_remaining. "
                    "Only call this to confirm it is our turn before calling make_draft_pick — "
                    "do not use it to track pick history (use get_local_draft_state instead)."
                ),
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "make_draft_pick",
                "description": (
                    "DESTRUCTIVE — Draft a specific player on NFL.com. "
                    "Only call after get_draft_turn_status confirms it is our turn. "
                    "Requires user confirmation."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "player_id": {"type": "string", "description": "Player ID to draft"},
                        "player_name": {
                            "type": "string",
                            "description": "Player name (shown to user for confirmation)",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why this is the best pick",
                        },
                    },
                    "required": ["player_id", "player_name", "reasoning"],
                },
            },
        ]
        return draft_tools  # draft mode only needs draft tools, not in-season tools
    else:
        weekly_tool = {
            "name": "get_player_predictions",
            "description": (
                "Get ML model predictions (PPR points) for players at a specific position "
                "for the upcoming week. Use this to identify who to start and who to pick up."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "string",
                        "description": "Position to get predictions for",
                        "enum": ["QB", "RB", "WR", "TE", "K"],
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top players to return (default 40)",
                        "default": 40,
                    },
                },
                "required": ["position"],
            },
        }
        return [weekly_tool] + common_tools


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


async def execute_tool(
    name: str, tool_input: dict, nfl_client: NFLFantasyClient
) -> Any:
    """Dispatch a tool call to the appropriate implementation."""

    # --- Draft-mode read tools (read from local state, not NFL.com) ---

    if name == "get_local_draft_state":
        top_n = int(tool_input.get("top_n", 50))
        try:
            from .draft_assistant import analyse_roster_needs, load_state, suggest_best_available
            state = load_state()
            needs = analyse_roster_needs(state)
            top_available = suggest_best_available(state, needs=needs, n=top_n)
            return {
                "current_pick": state.current_pick,
                "picks_made": len(state.picks),
                "my_roster": state.my_roster,
                "positional_needs": needs,
                "top_available": top_available.to_dict(orient="records"),
            }
        except FileNotFoundError:
            return {"error": "No draft_state.json found. Run `nfl-predict draft-start` first."}

    if name == "get_draft_rankings":
        position = tool_input.get("position", "").upper()
        top_n = int(tool_input.get("top_n", 30))
        try:
            from .draft_assistant import load_state
            state = load_state()
            avail = state.available.copy()
            if position:
                avail = avail[avail["position"] == position]
            cols = [c for c in (
                "overall_rank", "pos_rank", "tier", "player_name", "position",
                "team", "proj_p10", "proj_p50", "proj_p90", "vor",
            ) if c in avail.columns]
            return avail.sort_values("vor", ascending=False).head(top_n)[cols].to_dict(orient="records")
        except FileNotFoundError:
            return {"error": "Draft state not found. Run nfl-predict draft-start and nfl-sync first."}

    if name == "get_draft_turn_status":
        return await nfl_client.get_draft_turn_info()

    # --- In-season read tools ---

    if name == "get_my_roster":
        roster = await nfl_client.get_roster()
        return [asdict(p) for p in roster]

    if name == "get_player_predictions":
        position = tool_input["position"]
        top_n = int(tool_input.get("top_n", 40))
        try:
            df = run_predictions(position=position.lower())
            if df is not None and not df.empty:
                cols = [c for c in ["player_name", "team", "position", "expected_ppr_points"] if c in df.columns]
                return df[cols].head(top_n).to_dict(orient="records")
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}
        return []

    if name == "get_waiver_wire":
        position = tool_input.get("position")
        limit = int(tool_input.get("limit", 40))
        players = await nfl_client.get_waiver_wire(position=position, limit=limit)
        return [asdict(p) for p in players]

    if name == "get_matchup":
        matchup = await nfl_client.get_matchup()
        return asdict(matchup) if matchup else {"error": "Could not retrieve matchup"}

    if name == "get_trade_offers":
        offers = await nfl_client.get_trade_offers()
        return [asdict(o) for o in offers]

    if name == "get_league_teams":
        return await nfl_client.get_league_teams()

    # --- Destructive tools ---

    if name == "set_lineup":
        return await nfl_client.set_lineup(tool_input["starter_player_ids"])

    if name == "claim_waiver_player":
        return await nfl_client.claim_waiver_player(
            tool_input["add_player_id"],
            tool_input["drop_player_id"],
        )

    if name == "accept_trade":
        return await nfl_client.accept_trade(tool_input["trade_id"])

    if name == "decline_trade":
        return await nfl_client.decline_trade(tool_input["trade_id"])

    if name == "propose_trade":
        return await nfl_client.propose_trade(
            target_team_id=tool_input["target_team_id"],
            offer_player_ids=tool_input["offer_player_ids"],
            request_player_ids=tool_input["request_player_ids"],
            message=tool_input.get("message", ""),
        )

    if name == "make_draft_pick":
        return await nfl_client.make_draft_pick(tool_input["player_id"])

    raise ValueError(f"Unknown tool: {name}")


# ---------------------------------------------------------------------------
# Human-in-the-loop confirmation
# ---------------------------------------------------------------------------


def confirm_action(tool_name: str, tool_input: dict) -> bool:
    """Print a summary of a destructive action and ask the user to confirm."""
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  ACTION REQUIRED: {tool_name.upper().replace('_', ' ')}")
    print(sep)

    if tool_name == "set_lineup":
        print(f"  Starters: {tool_input.get('starter_player_ids', [])}")
        print(f"  Reason:   {tool_input.get('explanation', '')}")

    elif tool_name == "claim_waiver_player":
        print(f"  ADD:    {tool_input.get('add_player_name', tool_input.get('add_player_id'))}")
        print(f"  DROP:   {tool_input.get('drop_player_name', tool_input.get('drop_player_id'))}")
        print(f"  Reason: {tool_input.get('reasoning', '')}")

    elif tool_name in ("accept_trade", "decline_trade"):
        action = "Accept" if tool_name == "accept_trade" else "Decline"
        print(f"  {action} trade ID: {tool_input.get('trade_id', '')}")
        print(f"  Reason: {tool_input.get('reasoning', '')}")

    elif tool_name == "propose_trade":
        print(f"  Target: {tool_input.get('target_team_name', tool_input.get('target_team_id'))}")
        print(f"  Offering:  {tool_input.get('offer_player_names', tool_input.get('offer_player_ids'))}")
        print(f"  Requesting:{tool_input.get('request_player_names', tool_input.get('request_player_ids'))}")
        print(f"  Reason: {tool_input.get('reasoning', '')}")

    elif tool_name == "make_draft_pick":
        print(f"  Draft: {tool_input.get('player_name', tool_input.get('player_id'))}")
        print(f"  Reason: {tool_input.get('reasoning', '')}")

    print(sep)
    try:
        answer = input("  Confirm? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    print()
    return answer == "y"


# ---------------------------------------------------------------------------
# Tool format converters
# ---------------------------------------------------------------------------


def _tools_for_openai(tools: list[dict]) -> list[dict]:
    """Convert Claude-style tool defs to OpenAI/Ollama format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Backend-specific agent loops
# ---------------------------------------------------------------------------


async def _claude_loop(
    client: anthropic.Anthropic,
    system: str,
    tools: list[dict],
    messages: list[dict],
    nfl_client: "NFLFantasyClient",
    auto_confirm: bool,
    max_turns: int,
    draft_mode: bool,
) -> None:
    turn = 0
    while turn < max_turns:
        turn += 1
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=16_000,
            thinking={"type": "adaptive"},
            system=system,
            tools=tools,
            messages=messages,
        )

        for block in response.content:
            if hasattr(block, "text") and block.text:
                print(f"\n[Agent]\n{block.text}\n")

        if response.stop_reason == "end_turn":
            print("[Agent] Task complete.")
            break
        if response.stop_reason != "tool_use":
            print(f"[Agent] Stopped: {response.stop_reason}")
            break

        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tb in tool_blocks:
            result = await _handle_tool_call(tb.name, tb.input, tb.id, nfl_client, auto_confirm)
            tool_results.append({"type": "tool_result", "tool_use_id": tb.id, "content": result})

        messages.append({"role": "user", "content": tool_results})

        if draft_mode:
            await asyncio.sleep(5)

    if turn >= max_turns:
        print(f"[Agent] Reached max turns ({max_turns}).")


async def _ollama_loop(
    client: Any,
    model: str,
    system: str,
    tools: list[dict],
    messages: list[dict],
    nfl_client: "NFLFantasyClient",
    auto_confirm: bool,
    max_turns: int,
    draft_mode: bool,
) -> None:
    oai_tools = _tools_for_openai(tools)
    full_messages = [{"role": "system", "content": system}] + messages

    turn = 0
    while turn < max_turns:
        turn += 1
        response = client.chat.completions.create(
            model=model,
            tools=oai_tools,
            messages=full_messages,
        )

        msg = response.choices[0].message
        finish = response.choices[0].finish_reason

        if msg.content:
            print(f"\n[Agent]\n{msg.content}\n")

        full_messages.append(msg)

        if finish == "stop" or not msg.tool_calls:
            print("[Agent] Task complete.")
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                inp = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                inp = {}

            result = await _handle_tool_call(name, inp, tc.id, nfl_client, auto_confirm)

            full_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        if draft_mode:
            await asyncio.sleep(5)

    if turn >= max_turns:
        print(f"[Agent] Reached max turns ({max_turns}).")


async def _handle_tool_call(
    name: str,
    inp: dict,
    call_id: str,
    nfl_client: "NFLFantasyClient",
    auto_confirm: bool,
) -> str:
    """Execute a single tool call, with confirmation for destructive ones. Returns JSON string."""
    print(f"[Tool → {name}] {json.dumps(inp, indent=2)[:300]}")

    if name in DESTRUCTIVE_TOOLS and not auto_confirm:
        if not confirm_action(name, inp):
            return json.dumps({"status": "cancelled", "message": "User cancelled."})

    try:
        result = await execute_tool(name, inp, nfl_client)
        result_str = json.dumps(result, default=str)
        print(f"[Result] {result_str[:500]}{'...' if len(result_str) > 500 else ''}")
        return result_str
    except Exception as e:
        print(f"[Error] {name}: {e}")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------


async def run_agent(
    task: Optional[str] = None,
    auto_confirm: bool = False,
    draft_mode: bool = False,
    max_turns: int = 30,
) -> None:
    """
    Run the NFL Fantasy agent.

    Backend is selected via LLM_BACKEND env var:
      LLM_BACKEND=claude   → Claude Opus 4.6 (needs ANTHROPIC_API_KEY)
      LLM_BACKEND=ollama   → local Ollama (needs OLLAMA_MODEL, default: qwen2.5:32b)

    In draft mode, nfl-sync must be running in a separate terminal to keep
    draft_state.json up to date. The agent reads from that file and only
    touches NFL.com to confirm pick turns and submit picks.

    Args:
        task: What the agent should do. Defaults to full weekly management.
        auto_confirm: Skip confirmation prompts for destructive actions.
        draft_mode: Run in draft-assistant mode (reads from draft_state.json).
        max_turns: Max agent turns before stopping.
    """
    backend = os.getenv("LLM_BACKEND", "ollama").lower()

    email = os.getenv("NFL_EMAIL")
    password = os.getenv("NFL_PASSWORD")
    league_id = os.getenv("NFL_LEAGUE_ID")
    team_id = os.getenv("NFL_TEAM_ID")

    missing = [
        var for var, val in [
            ("NFL_EMAIL", email), ("NFL_PASSWORD", password),
            ("NFL_LEAGUE_ID", league_id), ("NFL_TEAM_ID", team_id),
        ] if not val
    ]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}. Check your .env file."
        )

    if task is None:
        if draft_mode:
            task = (
                "I'm in my fantasy draft. nfl-sync is running in another terminal and keeping "
                "draft_state.json up to date. Use get_local_draft_state to see the board, "
                "get_draft_turn_status to check if it's my turn, and make_draft_pick when ready. "
                "Keep watching until the draft ends."
            )
        else:
            task = (
                "Fully manage my NFL Fantasy team for this week:\n"
                "1. Review my current roster and identify any injury concerns\n"
                "2. Get ML predictions for all relevant positions (WR, RB, QB, TE, K)\n"
                "3. Set the optimal starting lineup based on predictions and injury status\n"
                "4. Check the waiver wire and claim any high-value players if needed\n"
                "5. Review and respond to any pending trade offers\n"
                "6. Summarize all actions taken and the expected impact"
            )

    tools = get_tools(draft_mode=draft_mode)
    system = DRAFT_SYSTEM_PROMPT if draft_mode else SYSTEM_PROMPT
    headless = os.getenv("NFL_BROWSER_HEADLESS", "false").lower() in ("1", "true", "yes")

    # --- Build LLM client ---
    if backend == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
        llm = anthropic.Anthropic(api_key=api_key)
        model_name = "Claude Opus 4.6"
        ollama_model = None
    else:
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for Ollama support. "
                "Install it with: uv sync --extra ollama"
            ) from exc
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:32b")
        llm = _openai.OpenAI(base_url=f"{ollama_host}/v1", api_key="ollama")
        model_name = ollama_model

    print("\n" + "=" * 55)
    print(f"  NFL FANTASY AGENT — {model_name}")
    print("=" * 55)
    print(f"  Backend:      {backend}")
    print(f"  Task: {task[:70]}{'...' if len(task) > 70 else ''}")
    print(f"  Auto-confirm: {auto_confirm}")
    print(f"  Draft mode:   {draft_mode}")
    if draft_mode:
        print("  NOTE: Reads from draft_state.json (maintained by nfl-sync)")
    print("=" * 55 + "\n")

    async with NFLFantasyClient(
        email=email, password=password,
        league_id=league_id, team_id=team_id,
        headless=headless,
    ) as nfl_client:

        if backend == "claude":
            await _claude_loop(
                llm, system, tools,
                [{"role": "user", "content": task}],
                nfl_client, auto_confirm, max_turns, draft_mode,
            )
        else:
            await _ollama_loop(
                llm, ollama_model, system, tools,
                [{"role": "user", "content": task}],
                nfl_client, auto_confirm, max_turns, draft_mode,
            )


# ---------------------------------------------------------------------------
# Draft mode: polling loop
# ---------------------------------------------------------------------------


async def run_draft_agent(auto_confirm: bool = False) -> None:
    """
    Continuously monitor the draft and make picks when it's our turn.

    Requires `nfl-sync` running in a separate terminal to keep draft_state.json
    current. This agent only reads state and submits picks — it does not record
    picks independently.

    Runs until the draft completes or the user interrupts (Ctrl+C).
    """
    print("\n[Draft Agent] Starting. Ensure `nfl-predict nfl-sync` is running in another terminal.")
    print("[Draft Agent] Press Ctrl+C to stop.\n")
    try:
        await run_agent(draft_mode=True, auto_confirm=auto_confirm, max_turns=100)
    except KeyboardInterrupt:
        print("\n[Draft Agent] Stopped by user.")
