"""
FastAPI router for the season-long "weekly" league beta.

No accounts: a shared link lets anyone in the league pick their team from a
dropdown and see that team's optimal lineup + expected points for the
upcoming week, built from the same CatBoost weekly predictions used by the
CLI (`nfl-predict predict`).

Endpoints
---------
GET  /weekly              - team picker page
GET  /weekly/team/{team}  - lineup + bench for one team, current week
POST /weekly/update       - start the data + prediction refresh
GET  /weekly/update       - poll refresh progress (htmx partial)
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from nfl_predict.draft_assistant import _DEFAULT_SLOTS
from nfl_predict.predict_week import get_default_season_and_week

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _timeago(ts: float | None) -> str:
    """Render a unix timestamp as a short relative age (`4m ago`)."""
    if not ts:
        return "never"
    secs = max(0, int(time.time() - ts))
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


templates.env.filters["timeago"] = _timeago

ROSTERS_PATH = Path("data/league_rosters.json")
OUT_DIR = Path("outputs")

router = APIRouter(prefix="/weekly", tags=["weekly"])


# ---------------------------------------------------------------------------
# Data refresh
# ---------------------------------------------------------------------------

# Positions the weekly page needs a prediction file for.
UPDATE_POSITIONS = ("QB", "RB", "WR", "TE", "K")

# Progress of the refresh, shared between the worker thread and the pollers.
# A refresh takes minutes, so the request that starts it cannot wait for it.
_update_state: dict = {
    "status": "idle",  # idle | running | done | error
    "step": "",
    "started_at": None,
    "finished_at": None,
    "error": None,
}
_update_lock = threading.Lock()


def _update_steps() -> list[tuple[str, Callable[[], object]]]:
    """
    The refresh pipeline, as named steps.

    Model training is deliberately absent: it is far slower than the rest and
    the weekly page only needs current data scored by the existing models.
    Retraining stays a CLI decision (`nfl-predict update-all`).
    """
    from nfl_predict import features, fetch_nfl_data, predict_week

    steps: list[tuple[str, Callable[[], object]]] = [
        ("Fetching NFL data", fetch_nfl_data.main),
        ("Building features", features.build_player_week_features),
    ]
    for pos in UPDATE_POSITIONS:
        steps.append(
            (
                f"Predicting {pos}",
                lambda p=pos: predict_week.run_predictions(position=p),
            )
        )
    return steps


def _run_update(steps: list[tuple[str, Callable[[], object]]] | None = None) -> None:
    """Run the refresh, recording progress in ``_update_state``."""
    try:
        for label, fn in steps if steps is not None else _update_steps():
            with _update_lock:
                _update_state["step"] = label
            fn()
        with _update_lock:
            _update_state.update(
                status="done", step="", finished_at=time.time(), error=None
            )
    except Exception as exc:  # noqa: BLE001 - surfaced to the page, not swallowed
        traceback.print_exc()
        with _update_lock:
            _update_state.update(
                status="error",
                step="",
                finished_at=time.time(),
                error=f"{type(exc).__name__}: {exc}",
            )


def start_update(
    steps: list[tuple[str, Callable[[], object]]] | None = None,
) -> bool:
    """
    Start a refresh unless one is already running.

    Returns True if this call started it. The guard matters: the pipeline
    writes the same parquet and CSV files from every step, so two concurrent
    runs would race on them.
    """
    with _update_lock:
        if _update_state["status"] == "running":
            return False
        _update_state.update(
            status="running",
            step="Starting",
            started_at=time.time(),
            finished_at=None,
            error=None,
        )
    threading.Thread(target=_run_update, args=(steps,), daemon=True).start()
    return True


def update_status() -> dict:
    """Snapshot of the refresh state, plus when predictions were last written."""
    with _update_lock:
        state = dict(_update_state)

    files = _latest_prediction_files()
    state["predictions_at"] = max(f.stat().st_mtime for f in files) if files else None
    state["n_prediction_files"] = len(files)
    if state["started_at"] and state["status"] == "running":
        state["elapsed"] = int(time.time() - state["started_at"])
    else:
        state["elapsed"] = None
    return state


# ---------------------------------------------------------------------------
# Rosters
# ---------------------------------------------------------------------------


def _load_rosters() -> dict[str, list[dict]]:
    if not ROSTERS_PATH.exists():
        return {}
    with open(ROSTERS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("teams", {})


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def _latest_prediction_files() -> list[Path]:
    """Same convention as lineup.py: predictions_<pos>_<season>_week<week>.csv,
    picking the most recent season/week per position."""
    import re

    pred_files = list(OUT_DIR.glob("predictions_*.csv"))
    pattern = re.compile(
        r"predictions_(?P<position>[A-Za-z]+)_(?P<season>\d{4})_week(?P<week>\d+)\.csv"
    )
    records = []
    for path in pred_files:
        m = pattern.fullmatch(path.name)
        if not m:
            continue
        records.append(
            {
                "position": m.group("position").upper(),
                "season": int(m.group("season")),
                "week": int(m.group("week")),
                "path": path,
            }
        )
    if not records:
        return []

    df_files = pd.DataFrame(records)
    df_files = df_files.sort_values(["position", "season", "week"])
    latest = df_files.groupby("position", as_index=False).tail(1)
    return list(latest["path"])


def _load_latest_predictions() -> pd.DataFrame:
    """Load and concatenate the most recent predictions_*.csv per position.

    Returns an empty DataFrame if no prediction files exist yet.
    """
    files = _latest_prediction_files()
    if not files:
        return pd.DataFrame()

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        if "position" in df.columns:
            df["position"] = df["position"].str.upper()
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Lineup optimizer
# ---------------------------------------------------------------------------


def _build_lineup(merged: pd.DataFrame) -> tuple[list[dict], list[dict], float]:
    """Greedy optimal-lineup selection using _DEFAULT_SLOTS (QB1/RB2/WR2/TE1/FLEX1/K1).

    Fills required single-position slots first (highest expected points),
    then fills FLEX from the best remaining RB/WR/TE, then bench = everyone
    else. Returns (starters, bench, total_starter_points).
    """
    remaining = merged.sort_values("expected_ppr_points", ascending=False).copy()
    starters: list[dict] = []
    used_index: set[int] = set()

    for pos in ("QB", "RB", "WR", "TE", "K"):
        n = _DEFAULT_SLOTS.get(pos, 0)
        pool = remaining[
            (remaining["position"] == pos) & (~remaining.index.isin(used_index))
        ]
        for _, row in pool.head(n).iterrows():
            starters.append({**row.to_dict(), "slot": pos})
            used_index.add(row.name)

    flex_n = _DEFAULT_SLOTS.get("FLEX", 0)
    flex_pool = remaining[
        remaining["position"].isin(["RB", "WR", "TE"])
        & (~remaining.index.isin(used_index))
    ]
    for _, row in flex_pool.head(flex_n).iterrows():
        starters.append({**row.to_dict(), "slot": "FLEX"})
        used_index.add(row.name)

    bench_df = remaining[~remaining.index.isin(used_index)]
    bench = [{**row.to_dict(), "slot": "BN"} for _, row in bench_df.iterrows()]

    total = sum(s.get("expected_ppr_points", 0) or 0 for s in starters)
    return starters, bench, round(total, 1)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_class=HTMLResponse)
async def weekly_setup(request: Request):
    """Team picker page."""
    rosters = _load_rosters()
    season, week = get_default_season_and_week()
    return templates.TemplateResponse(
        "weekly_setup.html",
        {
            "request": request,
            "active_tab": "Weekly",
            "team_names": sorted(rosters.keys()),
            "season": season,
            "week": week,
            "has_rosters": bool(rosters),
            **update_status(),
        },
    )


@router.get("/team/{team_name}", response_class=HTMLResponse)
async def weekly_team(request: Request, team_name: str):
    """Optimal lineup + bench for one team's current week."""
    rosters = _load_rosters()
    if team_name not in rosters:
        raise HTTPException(status_code=404, detail=f"Unknown team: {team_name}")

    roster = pd.DataFrame(rosters[team_name])
    season, week = get_default_season_and_week()
    preds = _load_latest_predictions()

    if preds.empty:
        return templates.TemplateResponse(
            "weekly_team.html",
            {
                "request": request,
                "active_tab": "Weekly",
                "team_name": team_name,
                "team_names": sorted(rosters.keys()),
                "season": season,
                "week": week,
                "no_predictions": True,
                "starters": [],
                "bench": [],
                "total": 0,
            },
        )

    join_keys = ["player_name", "position"]
    if "team" in preds.columns and "team" in roster.columns:
        join_keys.append("team")

    merged = preds.merge(roster, on=join_keys, how="inner")

    unmatched = roster[~roster["player_name"].isin(merged["player_name"])][
        "player_name"
    ].tolist()

    starters, bench, total = _build_lineup(merged) if not merged.empty else ([], [], 0)

    return templates.TemplateResponse(
        "weekly_team.html",
        {
            "request": request,
            "active_tab": "Weekly",
            "team_name": team_name,
            "team_names": sorted(rosters.keys()),
            "season": season,
            "week": week,
            "no_predictions": False,
            "starters": starters,
            "bench": bench,
            "total": total,
            "unmatched": unmatched,
        },
    )


# ---------------------------------------------------------------------------
# Data refresh endpoints
# ---------------------------------------------------------------------------


def _update_partial(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "partials/update_status.html",
        {"request": request, **update_status()},
    )


@router.post("/update", response_class=HTMLResponse)
async def weekly_update(request: Request):
    """
    Kick off the data + prediction refresh.

    Returns the progress panel immediately; the work continues in a worker
    thread and the panel polls itself until it finishes.
    """
    start_update()
    return _update_partial(request)


@router.get("/update", response_class=HTMLResponse)
async def weekly_update_status(request: Request):
    """Progress panel — polled by htmx while a refresh is running."""
    return _update_partial(request)
