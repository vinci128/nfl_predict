from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nfl_predict.predict_week import run_predictions


app = FastAPI(title="nfl-predict API", version="0.1.0")


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


if __name__ == "__main__":
    # Lightweight runner for development; prefer `uvicorn` in production.
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
