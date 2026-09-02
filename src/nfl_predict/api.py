from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from nfl_predict.draft_api import router as draft_router
from nfl_predict.leagues import league_nav as _league_nav
from nfl_predict.predict_week import get_default_season_and_week, run_predictions
from nfl_predict.weekly_api import router as weekly_router

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# The nav's league switcher is on every page, so the helper is a Jinja
# global rather than something each endpoint has to remember to pass.
# Jinja types this map from its own builtins, so it needs widening.
_jinja_globals: dict[str, Any] = templates.env.globals
_jinja_globals["league_nav"] = _league_nav

app = FastAPI(title="nfl-predict API", version="0.1.0")


@app.middleware("http")
async def _apply_league_cookie(request: Request, call_next):
    """
    Pin each request to the viewer's chosen league.

    Both leagues are served from one process, so the league cannot be a
    process-wide setting: it is read per request and every get_profile() call
    below picks it up without being passed it explicitly.
    """
    from nfl_predict.leagues import reset_active_league, set_active_league

    token = set_active_league(request.cookies.get("league"))
    try:
        return await call_next(request)
    finally:
        reset_active_league(token)


@app.get("/league/{key}")
async def switch_league(key: str, request: Request):
    """Switch the active league and return to the page the viewer came from."""
    from urllib.parse import urlparse

    from nfl_predict.leagues import get_profile

    try:
        profile = get_profile(key)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    # Only the path of the referer is reused, never its host, so this cannot
    # be turned into an open redirect.
    back = urlparse(request.headers.get("referer", "")).path or "/"
    response = RedirectResponse(url=back, status_code=303)
    response.set_cookie(
        "league", profile.key, max_age=31_536_000, samesite="lax", httponly=True
    )
    return response


# Draft UI router
app.include_router(draft_router)

# Season-long weekly lineup UI router
app.include_router(weekly_router)

# Serve a small single-page app from `static/`
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class PredictRequest(BaseModel):
    season: int | None = None
    week: int | None = None
    position: str | None = "WR"
    top_n: int | None = 20


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """Run predictions for the requested position/week/season.

    This endpoint uses the same `run_predictions` helper as the CLI and
    returns the top-N predicted players as JSON. If `season` or `week` are
    omitted they are inferred from the processed data (same behaviour as CLI).
    """

    try:
        # write_csv=False: API requests shouldn't leave CSV files behind.
        df = run_predictions(
            season=req.season,
            week=req.week,
            position=req.position or "WR",
            write_csv=False,
        )
    except FileNotFoundError as exc:
        # Missing model / feature parquet — a setup problem, not a server bug
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        # Don't leak internals to the client; log server-side instead.
        import logging

        logging.getLogger(__name__).exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    # convert DataFrame to JSON-serializable list of records
    try:
        records = df.to_dict(orient="records")
    except Exception:
        # If df is not a DataFrame or unexpectedly structured, return raw repr
        return {"result": str(df)}

    top_n = max(0, int(req.top_n or 0))
    return {"count": len(records), "predictions": records[:top_n]}


@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    """
    Landing page tying the two tools together.

    Both are read-only summaries of on-disk state, so a missing board, an
    absent roster file or an unfinished prediction run shows as a hint on the
    card rather than an error page.
    """
    from nfl_predict.draft_api import _available_boards, _state_exists
    from nfl_predict.weekly_api import _latest_prediction_files, _load_rosters

    boards = _available_boards()
    current_pick = None
    if _state_exists():
        try:
            from nfl_predict.draft_api import _active_state_path
            from nfl_predict.draft_assistant import load_state

            current_pick = load_state(_active_state_path()).current_pick
        except Exception:
            # A corrupt or half-written state file must not take down the
            # landing page — the draft card just loses its pick number.
            current_pick = None

    try:
        season, week = get_default_season_and_week()
    except Exception:
        season = week = None

    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "active_tab": None,
            "n_boards": len(boards),
            "session_active": _state_exists(),
            "current_pick": current_pick,
            "n_teams": len(_load_rosters()),
            "has_predictions": bool(_latest_prediction_files()),
            "season": season,
            "week": week,
        },
    )


if __name__ == "__main__":
    # Lightweight runner for development; prefer `uvicorn` in production.
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
