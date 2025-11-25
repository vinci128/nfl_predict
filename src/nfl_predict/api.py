from typing import Optional

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from nfl_predict.predict_week import run_predictions


app = FastAPI(title="nfl-predict API", version="0.1.0")

# Serve a small single-page app from `static/`
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class PredictRequest(BaseModel):
    season: Optional[int] = None
    week: Optional[int] = None
    position: Optional[str] = "WR"
    top_n: Optional[int] = 20


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
        df = run_predictions(season=req.season, week=req.week, position=req.position)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # convert DataFrame to JSON-serializable list of records
    try:
        records = df.to_dict(orient="records")
    except Exception:
        # If df is not a DataFrame or unexpectedly structured, return raw repr
        return {"result": str(df)}

    top_n = max(0, int(req.top_n or 0))
    return {"count": len(records), "predictions": records[:top_n]}


@app.get("/")
def read_index():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"status": "ok", "note": "Static UI not installed."}


if __name__ == "__main__":
    # Lightweight runner for development; prefer `uvicorn` in production.
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
