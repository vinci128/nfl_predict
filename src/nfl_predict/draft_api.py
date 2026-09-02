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
POST /draft/reset             - wipe session and return to setup
"""

from __future__ import annotations

import glob
import html
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from nfl_predict.draft_assistant import (
    analyse_roster_needs,
    init_draft_state,
    load_state,
    mark_drafted,
    save_state,
    state_lock,
    suggest_best_available,
    undo_last_pick,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

BOARDS_GLOB = "outputs/draft_board_*.csv"
STATES_GLOB = "outputs/draft_state_*.json"

router = APIRouter(prefix="/draft", tags=["draft"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_path(league: str | None = None) -> Path:
    """Where a league's draft session lives.

    The same path the CLI uses, so `nfl-sync` in a second terminal writes the
    session this UI is reading. A hardcoded path here meant the two wrote to
    different files and picks pulled from ESPN never reached the browser.
    """
    from nfl_predict.leagues import get_profile

    return get_profile(league).state_path


def _active_state_path() -> Path:
    """The session this UI should act on.

    Normally the active league's, which is what the draft-day workflow sets.
    A session started from another league's board is still found, because the
    board picker offers every board regardless of the active league.
    """
    active = _state_path()
    if active.exists():
        return active

    existing = sorted(glob.glob(STATES_GLOB))
    return Path(existing[0]) if len(existing) == 1 else active


def _state_exists() -> bool:
    return _active_state_path().exists()


def _load_or_404():
    path = _active_state_path()
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="No active draft session. Go to /draft to start one.",
        )
    return load_state(path)


def _available_boards() -> list[str]:
    return sorted(glob.glob(BOARDS_GLOB))


def _board_league(board_path: str) -> str:
    """League key for a board file (`draft_board_{season}_{league}.csv`).

    Boards are name-encoded with the league profile key, so the league a
    session runs under is read back from the selected file rather than
    hardcoded to the default. Falls back to the default league when the
    filename doesn't match any known key.
    """
    from nfl_predict.leagues import DEFAULT_LEAGUE, league_keys

    stem = Path(board_path).stem
    for key in league_keys():
        if stem.endswith(f"_{key}"):
            return key
    return DEFAULT_LEAGUE


def _position_tabs(state) -> list[str]:
    """Board filter tabs for this draft's league.

    Driven by the league profile rather than a fixed list, so an IDP league
    shows LB/DL tabs and a D/ST league shows DST.
    """
    from nfl_predict.leagues import get_profile

    return ["ALL", *get_profile(getattr(state, "league", None)).roster.positions]


def _state_to_dict(state: Any) -> dict:
    """Serialize key state info for templates."""
    needs = analyse_roster_needs(state)
    suggestions = suggest_best_available(state, needs=needs, n=5)

    round_num = ((state.current_pick - 1) // state.league_size) + 1
    pick_in_round = ((state.current_pick - 1) % state.league_size) + 1
    is_my_turn = (round_num % 2 == 1 and pick_in_round == state.draft_position) or (
        round_num % 2 == 0
        and pick_in_round == state.league_size - state.draft_position + 1
    )

    return {
        "current_pick": state.current_pick,
        "round": round_num,
        "pick_in_round": pick_in_round,
        "league_size": state.league_size,
        "draft_position": state.draft_position,
        "is_my_turn": is_my_turn,
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
        "suggestions": _records(suggestions) if not suggestions.empty else [],
    }


def _records(df: pd.DataFrame) -> list[dict]:
    """
    DataFrame -> template-ready dicts, with NaN replaced by None.

    `to_dict` leaves NaN in place, and NaN is truthy in Jinja — so a missing
    value passes an `{% if row.col %}` guard and then renders as the literal
    string "nan". None fails the guard and hits the intended fallback.
    """
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


_BANNER_TONES = {
    "error": "bg-red-100 border-red-400 text-red-700",
    "warn": "bg-yellow-100 border-yellow-400 text-yellow-700",
    "info": "bg-blue-100 border-blue-400 text-blue-700",
}


def _sync_banner(synced: int, errors: list[str]) -> tuple[str, str]:
    """
    Build the banner text and tone for a completed sync.

    Unmatched players are named rather than counted: the user has to record
    those picks by hand, so they need to know which ones.
    """
    message = f"Synced {synced} pick{'' if synced == 1 else 's'} from ESPN."
    if errors:
        message += f" {len(errors)} not matched: " + "; ".join(errors)
    return message, _BANNER_TONES["warn" if errors else "info"]


def _banner_response(
    request: Request,
    state: Any,
    pos: str,
    message: str,
    tone: str = "error",
) -> HTMLResponse:
    """
    Return a message banner *and* the re-rendered board.

    The forms that produce these messages target #board-section with
    hx-swap="outerHTML", so a response containing only a banner replaces the
    whole table and the board disappears mid-draft. The banner therefore
    rides along out-of-band while the swap payload stays the board itself.

    Status must be 2xx: htmx does not swap non-2xx responses, so returning
    4xx here would silently discard the message and leave the user with no
    feedback at all.
    """
    banner = (
        f'<div id="pick-error" hx-swap-oob="true" class="border px-4 py-2 '
        f'rounded mb-2 {_BANNER_TONES[tone]}">{html.escape(message)}</div>'
    )
    board = templates.get_template("partials/board_table.html").render(
        request=request,
        board_rows=_board_rows(state, pos),
        pos_filter=pos,
    )
    return HTMLResponse(banner + board)


def _board_rows(state: Any, position_filter: str = "ALL") -> list[dict]:
    """Return available players as a list of dicts, optionally filtered."""
    avail = state.available.copy()
    if position_filter and position_filter != "ALL":
        avail = avail[avail["position"] == position_filter]
    avail = avail.sort_values("vor", ascending=False).head(150)
    return _records(avail)


# ---------------------------------------------------------------------------
# Setup page
# ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
async def draft_setup(request: Request):
    """Setup / landing page.

    Defaults come from the active league profile rather than fixed values —
    neither league has 12 teams, so a hardcoded default was wrong for both and
    would have thrown the snake order off from the first pick.
    """
    from nfl_predict.leagues import get_profile

    profile = get_profile()
    boards = _available_boards()

    # Preselect this league's board, so the picker opens on the one the rest
    # of the page is describing.
    selected = next(
        (b for b in boards if _board_league(b) == profile.key),
        boards[-1] if boards else "",
    )

    return templates.TemplateResponse(
        "draft_setup.html",
        {
            "request": request,
            "active_tab": "Draft",
            "boards": boards,
            "selected_board": selected,
            "league_name": profile.name,
            "league_size": profile.roster.league_size,
            "session_active": _state_exists(),
        },
    )


# ---------------------------------------------------------------------------
# Start session
# ---------------------------------------------------------------------------


@router.post("/start")
async def draft_start(
    board_path: str = Form(...),
    league_size: int | None = Form(None),
    draft_position: int = Form(1),
):
    """Initialise a new draft session from a board CSV."""
    from nfl_predict.leagues import get_profile

    # Only accept boards from the known outputs/ glob — the path arrives from
    # a form field, so don't let it read arbitrary files on the server.
    if board_path not in _available_boards():
        raise HTTPException(status_code=400, detail=f"Board not found: {board_path}")

    league = _board_league(board_path)
    if not league_size or league_size < 2:
        league_size = get_profile(league).roster.league_size

    state_path = _state_path(league)
    board = pd.read_csv(Path(board_path))
    state = init_draft_state(
        board,
        league_size=league_size,
        draft_position=draft_position,
        state_path=state_path,
        league=league,
    )
    with state_lock(state_path):
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
    ctx["positions"] = _position_tabs(state)
    return templates.TemplateResponse(
        "draft_board.html", {"request": request, "active_tab": "Draft", **ctx}
    )


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
            "suggestions": _records(suggestions) if not suggestions.empty else [],
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
    player_id: str = Form(""),
):
    """Record a pick and return the refreshed board + roster partials."""
    with state_lock(_active_state_path()):
        state = _load_or_404()

        try:
            state = mark_drafted(
                state,
                player_name,
                drafter=drafter,
                player_id=player_id or None,
            )
        except ValueError as e:
            return _banner_response(request, state, pos, str(e))

        save_state(state)

    # Return OOB (out-of-band) swaps for board + roster + suggestions + header
    ctx = _state_to_dict(state)
    ctx["board_rows"] = _board_rows(state, pos)
    ctx["pos_filter"] = pos
    ctx["positions"] = _position_tabs(state)
    ctx["request"] = request

    return templates.TemplateResponse("partials/pick_response.html", ctx)


# ---------------------------------------------------------------------------
# Reset session
# ---------------------------------------------------------------------------


@router.post("/reset")
async def draft_reset():
    path = _active_state_path()
    if path.exists():
        path.unlink()
    return RedirectResponse(url="/draft", status_code=303)


# ---------------------------------------------------------------------------
# Undo last pick
# ---------------------------------------------------------------------------


@router.post("/undo", response_class=HTMLResponse)
async def draft_undo(request: Request, pos: str = Form("ALL")):
    """Reverse the last recorded pick."""
    with state_lock(_active_state_path()):
        state = _load_or_404()

        if not state.picks:
            return _banner_response(
                request, state, pos, "No picks to undo.", tone="warn"
            )

        state = undo_last_pick(state)
        save_state(state)

    ctx = _state_to_dict(state)
    ctx["board_rows"] = _board_rows(state, pos)
    ctx["pos_filter"] = pos
    ctx["positions"] = _position_tabs(state)
    ctx["request"] = request

    return templates.TemplateResponse("partials/pick_response.html", ctx)


# ---------------------------------------------------------------------------
# Autocomplete player names
# ---------------------------------------------------------------------------


@router.get("/autocomplete", response_class=HTMLResponse)
async def autocomplete(q: str = Query(default="")):
    """
    Return <option> elements for the player-name datalist.
    htmx swaps these into #player-suggestions on each keyup.
    """
    if not _state_exists() or not q:
        return HTMLResponse("")
    state = load_state(_active_state_path())
    avail = state.available
    names = avail["player_name"]
    # regex=False: user keystrokes like '(' or '.' must not be treated as regex
    matches = avail[names.str.contains(q, case=False, na=False, regex=False)].head(10)
    # Each option carries data-player-id so JS can populate the hidden field
    options = "".join(
        '<option value="{name}" data-pid="{pid}">'.format(
            name=html.escape(str(row["player_name"]), quote=True),
            pid=html.escape(str(row.get("player_id", "")), quote=True),
        )
        for _, row in matches.iterrows()
    )
    return HTMLResponse(options)


# ---------------------------------------------------------------------------
# NFL Fantasy live sync
# ---------------------------------------------------------------------------


@router.get("/nfl-sync-status")
async def nfl_sync_status():
    """Check whether live draft sync is configured, and by which provider."""
    from nfl_predict.draft_sync import available_providers

    providers = available_providers()
    return JSONResponse({"available": bool(providers), "providers": providers})


@router.post("/nfl-sync", response_class=HTMLResponse)
async def nfl_sync(request: Request, pos: str = Form("ALL")):
    """
    Pull the latest picks from ESPN Fantasy and record any new ones.

    Uses the ESPN provider (ESPN_LEAGUE_ID, plus ESPN_S2/ESPN_SWID for
    a private league). Returns the same OOB swap as /draft/pick so the
    board refreshes automatically.
    """
    from nfl_predict.draft_sync import DraftSyncError, make_client
    from nfl_predict.espn_fantasy import EspnFantasyError

    state = _load_or_404()
    n_recorded_at_fetch = len(state.picks)

    try:
        client = make_client()
        new_picks = client.fetch_new_picks(
            already_recorded=n_recorded_at_fetch,
        )
    except (DraftSyncError, EspnFantasyError) as e:
        return _banner_response(request, state, pos, f"Draft sync error: {e}")

    if not new_picks:
        return _banner_response(
            request, state, pos, "No new picks since last sync.", tone="info"
        )

    errors: list[str] = []
    with state_lock(_active_state_path()):
        # Re-load under the lock: nfl-sync (CLI) may have recorded picks between
        # our fetch and now — skip any that were already applied.
        state = _load_or_404()
        already_applied = len(state.picks) - n_recorded_at_fetch
        if already_applied > 0:
            new_picks = new_picks[already_applied:]
        for pick in new_picks:
            try:
                state = mark_drafted(
                    state,
                    pick["player_name"],
                    drafter="me" if pick.get("is_mine") else "other",
                    player_id=pick.get("player_id") or None,
                )
            except ValueError as e:
                errors.append(str(e))

        save_state(state)

    ctx = _state_to_dict(state)
    ctx["board_rows"] = _board_rows(state, pos)
    ctx["pos_filter"] = pos
    ctx["positions"] = _position_tabs(state)
    ctx["request"] = request
    # Rendered as an out-of-band banner by pick_response.html. Without this
    # the board just changes under the user with no confirmation, and picks
    # ESPN reported but we could not match are dropped in silence.
    ctx["sync_message"], ctx["sync_tone_class"] = _sync_banner(
        len(new_picks) - len(errors), errors
    )

    return templates.TemplateResponse("partials/pick_response.html", ctx)
