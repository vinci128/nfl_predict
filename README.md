nfl_predict
===========

CatBoost-based NFL fantasy football pipeline with two modes:

1. **Weekly predictions** — point forecasts for upcoming game weeks (used during the season)
2. **Draft assistant** — season-total projections (p10/p50/p90) + VOR-based draft board + live draft UI (used at draft time)

Plus an **autonomous agent** (Claude or Ollama) that manages your NFL.com Fantasy team in-season and advises during the draft.

Stack: Python 3.12, CatBoost, pandas, FastAPI + htmx, Typer CLI, nflreadpy.

---

## Setup

Requires [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync                       # standard install (Claude backend)
uv sync --extra ollama        # also installs openai for local Ollama support

cp .env.example .env          # fill in credentials (see below)
```

---

## Scoring system

Custom league scoring (not standard PPR):

| Stat | Points |
|---|---|
| Passing yard | 0.1 |
| Passing TD | 6 |
| Rush/Rec yard | 0.1 |
| Rush/Rec TD | 6 |
| Reception | 1 (PPR) |
| FG 0–39 yd | 3 |
| FG 40–49 yd | 4 |
| FG 50+ yd | 5 |
| PAT | 1 |

Passing at 0.1 pts/yard (vs standard 0.04) makes QBs ~2.5× more valuable. Elite QB season totals of 500–750 pts are expected and correct.

---

## CLI commands

### Data & models

```bash
# Full refresh: fetch data + rebuild features + retrain all models
uv run nfl-predict update-all

# Individual steps
uv run nfl-predict update-all --no-train   # data only, skip training
uv run nfl-predict train --position QB     # retrain one position
uv run nfl-predict models                  # list registry
uv run nfl-predict promote <model-id>      # set a new champion model
uv run nfl-predict backtest                # walk-forward backtest vs baselines

# Weekly predictions
uv run nfl-predict predict --position WR
uv run nfl-predict predict --position WR --season 2025 --week 12
```

### Draft preparation

```bash
# Run once before draft day (trains p10/p50/p90 season models)
uv run nfl-predict draft-prep

# Pull ADP
uv run nfl-predict fetch-adp --source sleeper --scoring half

# Build the draft board
uv run nfl-predict board --league-size 12 --adp data/adp_current.csv --fmt csv
uv run nfl-predict board --fmt table           # quick terminal preview
uv run nfl-predict board --fmt table --superflex

# Season projections (terminal)
uv run nfl-predict project-season --position QB --top 20
```

### Live draft (terminal mode)

```bash
uv run nfl-predict draft-start --league-size 12 --draft-position 5
uv run nfl-predict draft-pick "Bijan Robinson" --mine   # your pick
uv run nfl-predict draft-pick "Drake Maye"              # opponent pick
```

### NFL.com auto-sync

Polls NFL.com and records picks automatically into `draft_state.json`:

```bash
# Requires: NFL_FANTASY_USERNAME, NFL_FANTASY_PASSWORD, NFL_FANTASY_LEAGUE_ID
uv run nfl-predict nfl-sync --interval 20
```

### AI agent

Manages your team autonomously (lineup, waivers, trades) or advises during the draft.

```bash
# In-season: full weekly management
LLM_BACKEND=claude uv run nfl-predict agent

# In-season: specific task
LLM_BACKEND=claude uv run nfl-predict agent --task "Check waiver wire for RB handcuffs"

# Draft day advisor (reads draft_state.json kept current by nfl-sync)
LLM_BACKEND=claude uv run nfl-predict agent --draft

# Local Ollama instead of Claude
LLM_BACKEND=ollama uv run nfl-predict agent --draft
```

`LLM_BACKEND=claude` requires `ANTHROPIC_API_KEY`. `LLM_BACKEND=ollama` requires `uv sync --extra ollama` and a running Ollama instance (set `OLLAMA_HOST` / `OLLAMA_MODEL` if non-default).

### Web UI

```bash
uvicorn nfl_predict.api:app --host 0.0.0.0 --port 8000
# → http://localhost:8000/draft
```

---

## Draft day workflow

### Night before

```bash
uv run nfl-predict update-all --no-train
uv run nfl-predict draft-prep
uv run nfl-predict fetch-adp --source sleeper --scoring half
uv run nfl-predict board --league-size 12 --adp data/adp_current.csv
```

### At the venue — four terminals

```
Terminal 1 (once, before draft starts):
  nfl-predict draft-start --league-size 12 --draft-position 5

Terminal 2 (runs all draft — mechanical pick recorder):
  nfl-predict nfl-sync --interval 20

Terminal 3 (runs all draft — AI advisor + auto-picker):
  LLM_BACKEND=claude nfl-predict agent --draft

Terminal 4 (optional — web UI for you and friends on same WiFi):
  uvicorn nfl_predict.api:app --host 0.0.0.0 --port 8000
```

`nfl-sync` is the sole process that writes picks to `draft_state.json`. The agent reads from that file and only touches NFL.com to confirm pick turns and submit picks — they don't conflict.

### During the draft (web UI)

- Type a player name and press Enter to record an opponent pick
- Toggle **Mine** before submitting for your own picks
- Click **↩ Undo** to reverse a miskey
- Use position tabs (QB / RB / WR / TE / K) to filter the board
- Click **Ask Claude** for pick advice (requires `ANTHROPIC_API_KEY`)

---

## Docker

```bash
# Draft day (local)
cp .env.example .env
docker compose --profile draft up
# → http://localhost:8000/draft
# Friends on same WiFi → http://<your-ip>:8000/draft
```

---

## Environment variables

```bash
# NFL.com credentials (for nfl-sync and agent --draft)
NFL_FANTASY_USERNAME=you@email.com
NFL_FANTASY_PASSWORD=yourpassword
NFL_FANTASY_LEAGUE_ID=12345678
NFL_FANTASY_TEAM_ID=3            # your team slot (1-based)

# Agent browser automation credentials (same account, different var names)
NFL_EMAIL=you@email.com
NFL_PASSWORD=yourpassword
NFL_LEAGUE_ID=12345678
NFL_TEAM_ID=3

# AI
ANTHROPIC_API_KEY=sk-ant-...     # Claude backend
OLLAMA_HOST=http://localhost:11434   # Ollama backend (optional)
OLLAMA_MODEL=qwen2.5:32b             # Ollama model (optional)

# Browser
NFL_BROWSER_HEADLESS=false       # set true for headless Playwright
```

---

## Development

```bash
uv run pytest tests/ -x -q      # 114 tests

# Pre-commit hooks run automatically on commit:
#   ruff --fix, ruff format, ty check
```

Branch strategy: work on `dev`, merge to `master` for tagged releases only.

---

## Deployment (VPS via GitHub CD)

Push to `master` or tag `v1.0.0` — GitHub Actions builds and pushes the Docker image to `ghcr.io/vinci128/nfl_predict`.

To enable automatic SSH deploy, add to GitHub → Settings → Secrets:
- **Variable** `SSH_HOST` — server IP or hostname
- **Variable** `SSH_USER` — SSH login user (default: `deploy`)
- **Secret** `SSH_PRIVATE_KEY` — private key authorized on the server

---

## Repository layout

```
src/nfl_predict/
  features.py          Weekly feature engineering
  fetch_nfl_data.py    nflreadpy wrapper (weekly stats, rosters, schedules, injuries)
  train_model.py       CatBoost weekly model training per position
  predict_week.py      Week-level prediction pipeline
  model_registry.py    Versioned model registry (JSON)
  backtest.py          Walk-forward backtest vs baselines
  metrics.py           MAE, RMSE, R², Spearman, top-N precision

  season_features.py   Player-season snapshots from weekly data
  season_model.py      CatBoost quantile regression (p10/p50/p90)
  draft_board.py       VOR calculation, tier assignment, CSV/JSON export
  draft_assistant.py   Live draft state — mark_drafted, undo, suggest, save/load
  adp_fetch.py         ADP from Sleeper / FantasyPros / synthetic fallback
  nfl_fantasy.py       NFL.com OAuth2 REST client — live draft pick polling
  fantasy_client.py    NFL.com Playwright browser automation (lineups, waivers, trades)
  nfl_agent.py         Claude / Ollama agent with tool_use
  draft_api.py         FastAPI router (/draft/*) with htmx partials
  api.py               Main FastAPI app
  cli.py               Typer CLI (nfl-predict)

data/                  Raw parquet files (gitignored, regenerated by update-all)
outputs/               Draft board CSV, draft state JSON (gitignored)
models/                Trained .cbm files + model_registry.json
tests/                 114 tests across bugs, draft phases 1–3
```
