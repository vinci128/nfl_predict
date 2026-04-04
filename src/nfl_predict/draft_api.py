"""
FastAPI router for the live fantasy draft web UI.

Endpoints
---------
GET  /draft                   - setup page (or redirect if session active)
POST /draft/start             - initialise a draft session
GET  /draft/board             - main live draft page
POST /draft/pick              - record a pick; returns updated partials
GET  /draft/partials/board    - available-players table fragment (htmx)
GET  /draft/partials/roster   - my-roster sidebar fragment (htmx)
GET  /draft/partials/suggest  - best-available suggestions fragment (htmx)
POST /draft/llm-suggest       - Claude pick recommendation (Phase 3)
POST /draft/reset             - wipe session and return to setup
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from nfl_predict.draft_assistant import (
    analyse_roster_needs,
    init_draft_state,
    load_state,
    mark_drafted,
    save_state,
    suggest_best_available,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

STATE_PATH = Path("outputs/draft_state.json")
BOARDS_GLOB = "outputs/draft_board_*.csv"

router = APIRouter(prefix="/draft", tags=["draft"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_exists() -> bool:
    return STATE_PATH.exists()


def _load_or_404():
    if not _state_exists():
        raise HTTPException(
            status_code=404,
            detail="No active draft session. Go to /draft to start one.",
        )
    return load_state(STATE_PATH)


def _available_boards() -> list[str]:
    return sorted(glob.glob(BOARDS_GLOB))


def _llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _state_to_dict(state: Any) -> dict:
    """Serialize key state info for templates."""
    needs = analyse_roster_needs(state)
    suggestions = suggest_best_available(state, needs=needs, n=5)

    return {
        "current_pick": state.current_pick,
        "round": ((state.current_pick - 1) // state.league_size) + 1,
        "pick_in_round": ((state.current_pick - 1) % state.league_size) + 1,
        "league_size": state.league_size,
        "draft_position": state.draft_position,
        "n_available": len(state.available),
        "n_picks": len(state.picks),
        "my_roster": state.my_roster,
        "needs": needs,
        "recent_picks": [
            {
                "pick": p.overall_pick,
                "round": p.round,
                "player": p.player_name,
                "position": p.position,
                "team": p.team,
                "vor": round(p.vor, 1),
                "drafter": p.drafter,
                "is_mine": p.drafter == "me",
            }
            for p in reversed(state.picks[-15:])
        ],
        "suggestions": suggestions.to_dict(orient="records")
        if not suggestions.empty
        else [],
        "llm_available": _llm_available(),
    }


def _board_rows(state: Any, position_filter: str = "ALL") -> list[dict]:
    """Return available players as a list of dicts, optionally filtered."""
    avail = state.available.copy()
    if position_filter and position_filter != "ALL":
        avail = avail[avail["position"] == position_filter]
    avail = avail.sort_values("vor", ascending=False).head(150)
    return avail.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Setup page
# ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
async def draft_setup(request: Request):
    """Setup / landing page."""
    boards = _available_boards()
    return templates.TemplateResponse(
        "draft_setup.html",
        {
            "request": request,
            "boards": boards,
            "session_active": _state_exists(),
            "llm_available": _llm_available(),
        },
    )


# ---------------------------------------------------------------------------
# Start session
# ---------------------------------------------------------------------------


@router.post("/start")
async def draft_start(
    board_path: str = Form(...),
    league_size: int = Form(12),
    draft_position: int = Form(1),
):
    """Initialise a new draft session from a board CSV."""
    path = Path(board_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Board not found: {board_path}")

    board = pd.read_csv(path)
    state = init_draft_state(
        board,
        league_size=league_size,
        draft_position=draft_position,
        state_path=STATE_PATH,
    )
    save_state(state)
    return RedirectResponse(url="/draft/board", status_code=303)


# ---------------------------------------------------------------------------
# Main board page
# ---------------------------------------------------------------------------


@router.get("/board", response_class=HTMLResponse)
async def draft_board_page(request: Request, pos: str = "ALL"):
    """Full draft board page."""
    state = _load_or_404()
    ctx = _state_to_dict(state)
    ctx["board_rows"] = _board_rows(state, pos)
    ctx["pos_filter"] = pos
    ctx["positions"] = ["ALL", "QB", "RB", "WR", "TE", "K"]
    return templates.TemplateResponse("draft_board.html", {"request": request, **ctx})


# ---------------------------------------------------------------------------
# Partial: board table (htmx swap target)
# ---------------------------------------------------------------------------


@router.get("/partials/board", response_class=HTMLResponse)
async def board_partial(request: Request, pos: str = "ALL"):
    state = _load_or_404()
    rows = _board_rows(state, pos)
    return templates.TemplateResponse(
        "partials/board_table.html",
        {"request": request, "board_rows": rows, "pos_filter": pos},
    )


# ---------------------------------------------------------------------------
# Partial: roster sidebar
# ---------------------------------------------------------------------------


@router.get("/partials/roster", response_class=HTMLResponse)
async def roster_partial(request: Request):
    state = _load_or_404()
    ctx = _state_to_dict(state)
    return templates.TemplateResponse(
        "partials/roster_panel.html", {"request": request, **ctx}
    )


# ---------------------------------------------------------------------------
# Partial: suggestions
# ---------------------------------------------------------------------------


@router.get("/partials/suggest", response_class=HTMLResponse)
async def suggest_partial(request: Request):
    state = _load_or_404()
    needs = analyse_roster_needs(state)
    suggestions = suggest_best_available(state, needs=needs, n=6)
    return templates.TemplateResponse(
        "partials/suggestions.html",
        {
            "request": request,
            "suggestions": suggestions.to_dict(orient="records")
            if not suggestions.empty
            else [],
            "needs": needs,
        },
    )


# ---------------------------------------------------------------------------
# Record a pick
# ---------------------------------------------------------------------------


@router.post("/pick", response_class=HTMLResponse)
async def draft_pick(
    request: Request,
    player_name: str = Form(...),
    drafter: str = Form("other"),
    pos: str = Form("ALL"),
):
    """Record a pick and return the refreshed board + roster partials."""
    state = _load_or_404()

    try:
        state = mark_drafted(state, player_name, drafter=drafter)
    except ValueError as e:
        # Return an error banner that htmx can swap into #pick-error
        return HTMLResponse(
            f'<div id="pick-error" class="bg-red-100 border border-red-400 '
            f'text-red-700 px-4 py-2 rounded mb-2">{e}</div>',
            status_code=422,
        )

    save_state(state)

    # Return OOB (out-of-band) swaps for board + roster + suggestions + header
    ctx = _state_to_dict(state)
    ctx["board_rows"] = _board_rows(state, pos)
    ctx["pos_filter"] = pos
    ctx["positions"] = ["ALL", "QB", "RB", "WR", "TE", "K"]
    ctx["request"] = request

    return templates.TemplateResponse("partials/pick_response.html", ctx)


# ---------------------------------------------------------------------------
# LLM suggestion (Phase 3)
# ---------------------------------------------------------------------------


@router.post("/llm-suggest", response_class=HTMLResponse)
async def llm_suggest(request: Request):
    """Ask Claude for a pick recommendation (requires ANTHROPIC_API_KEY)."""
    if not _llm_available():
        return HTMLResponse(
            '<p class="text-gray-500">Set ANTHROPIC_API_KEY to enable AI advice.</p>'
        )

    state = _load_or_404()
    needs = analyse_roster_needs(state)

    try:
        from nfl_predict.draft_assistant import get_llm_suggestion

        advice = get_llm_suggestion(state, needs=needs, n_context=20)
    except ImportError:
        advice = "Install the 'anthropic' package to enable AI suggestions."
    except Exception as e:
        advice = f"Error: {e}"

    return templates.TemplateResponse(
        "partials/llm_advice.html",
        {"request": request, "advice": advice, "needs": needs},
    )


# ---------------------------------------------------------------------------
# Reset session
# ---------------------------------------------------------------------------


@router.post("/reset")
async def draft_reset():
    if STATE_PATH.exists():
        STATE_PATH.unlink()
    return RedirectResponse(url="/draft", status_code=303)
