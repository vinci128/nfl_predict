nfl_predict
===========

**Overview**:
- Small pipeline to fetch NFL data, build player-week features, train CatBoost models per position and produce next-week fantasy predictions using a custom league scoring.

**Quick Setup**:
- This project uses `uv` to manage / install dependencies for now. If you don't have `uv` installed, install it according to your environment. Then install project dependencies with:

```bash
# install project dependencies (using uv)
uv venv
source .venv/bin/activate
uv sync
```

- The repository uses a Typer CLI. After dependencies are installed you can run the CLI commands shown below.

**Entry points / Common commands**:
- Run the CLI (if a console script `nfl-predict` is installed):

```bash
# show available CLI commands
nfl-predict --help

# Full pipeline: fetch raw data, build features and retrain models
nfl-predict --fetch --train --position K

# Make predictions for a position (defaults to latest season/week from data)
nfl-predict --position K --season 2025 --week 12
```

To do inference on a set of players defined in csv file, useful to analyze one own team, one can use the suggest entry point.

```bash
suggest  [ROSTER_PATH]
```

- Alternatively you can run modules directly with Python:

```bash
python -m nfl_predict.fetch_nfl_data      # download raw parquet files
python -m nfl_predict.features            # build player_week_features.parquet
python -m nfl_predict.train_model         # train models (all positions)
python -m nfl_predict.predict_week        # run predictions
```

**Files & folders to know**:
- `data/processed/player_week_features.parquet` : processed player-week dataset used for training/inference
- `models/` : trained CatBoost model files and metadata (e.g. `k_catboost.cbm`, `k_catboost_meta.json`)
- `outputs/` : CSV prediction outputs (e.g. `predictions_k_2025_week12.csv`)
- `src/nfl_predict/` : pipeline source code (CLI, features, training, prediction)

**Notes on scoring**:
- Custom league scoring is implemented in `src/nfl_predict/features.py` (function `add_custom_league_points`). Current kicking rules implemented:
  - PAT made: 1 point
  - FG 0-19: 3 points
  - FG 20-29: 3 points
  - FG 30-39: 3 points
  - FG 40-49: 4 points
  - FG 50+: 5 points

**Development notes**:
- The training pipeline now selects position-relevant features per position to keep models focused and reduce noise.
- The pipeline also detects non-numeric feature columns (for example list-like strings such as `"39;54"`) and treats them as categorical so CatBoost won't try to coerce them to floats.

**TODO**:
- Add a production HTTP API to serve model predictions (suggest using `FastAPI`):
  - `/predict` endpoint to request predictions for a given `season`, `week`, and `position`
  - `/models` and `/health` endpoints
  - Containerize the API (Dockerfile + small entrypoint)

License: (none specified)
