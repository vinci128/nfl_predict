# CLAUDE.md — nfl_predict

This file gives Claude Code the context needed to work effectively in this repo.

## What this project does

`nfl_predict` is a CatBoost-based NFL fantasy football pipeline with two modes:

1. **Weekly predictions** — point forecasts for upcoming game weeks (used during the season)
2. **Draft assistant** — season-total projections (p10/p50/p90) + VOR-based draft board + live draft UI (used at draft time)

The stack: Python 3.12, CatBoost, pandas, FastAPI + htmx, Typer CLI, nflreadpy for data.

---

## Repository layout

```
src/nfl_predict/
  features.py          Weekly feature engineering (roll windows, season cumulative stats)
  fetch_nfl_data.py    nflreadpy wrapper — pulls weekly_stats, rosters, schedules, injuries
  train_model.py       CatBoost weekly model training per position
  predict_week.py      Week-level prediction pipeline
  model_registry.py    Versioned model registry (JSON), champion tracking
  backtest.py          Walk-forward backtest vs baselines

  season_features.py   Player-season snapshots from weekly data (last-week cumulative)
  season_model.py      CatBoost quantile regression (p10/p50/p90) for season totals
  draft_board.py       VOR calculation, tier assignment, CSV/JSON export
  draft_assistant.py   Live draft state — mark_drafted, undo, suggest, save/load JSON
  adp_fetch.py         ADP from Sleeper / FantasyPros / synthetic fallback
  nfl_fantasy.py       NFL.com OAuth2 client — live draft pick polling
  draft_api.py         FastAPI router (/draft/*) with htmx partials
  api.py               Main FastAPI app (mounts draft router)
  cli.py               Typer CLI entry point (nfl-predict)

  templates/
    draft_setup.html           Setup / landing page
    draft_board.html           Live draft board page
    partials/board_table.html  htmx swap target — available players table
    partials/pick_response.html OOB swap after each pick (board+roster+header)
    partials/roster_panel.html  My roster sidebar
    partials/suggestions.html   Best-available panel
    partials/llm_advice.html    Claude advisor response

data/
  weekly_stats.parquet    Raw NFL weekly player stats
  rosters.parquet         Player roster info (name, position, age, years_exp)
  injuries.parquet        Weekly injury report
  schedules.parquet       Game schedules
  snap_counts.parquet     Snap count data
  processed/
    player_week_features.parquet  Engineered features (roll windows, cumulative)
  adp_current.csv         Most recent ADP fetch

outputs/
  draft_board_YYYY.csv    Current draft board (rebuilt each year)
  draft_state.json        Live draft session state (persisted per pick)
  models/                 Trained .cbm model files

models/
  model_registry.json     Version registry

tests/
  test_bugs.py            Regression tests
  test_draft_phase1.py    season_features, season_model, draft_board (48 tests)
  test_draft_phase2.py    draft_assistant (32 tests)
  test_draft_phase3.py    adp_fetch, LLM endpoint, CLI (27 tests)
```

---

## Development workflow

### Branch strategy
- **`dev`** — all active development goes here (default branch)
- **`master`** — only for tagged releases; never commit directly
- Always work on `dev`. Push with `git push -u origin dev`.

### Run tests
```bash
uv run pytest tests/ -x -q
```
All 114 tests must pass before committing.

### Pre-commit hooks (run automatically on commit)
- `ruff --fix` — lint and auto-fix
- `ruff format` — format
- `ty check` — type checking

If pre-commit modifies files, **re-stage** the modified files and commit again. Never use `--no-verify`.

### Adding dependencies
```bash
uv add <package>          # runtime
uv add --dev <package>    # dev only
```
Always commit `pyproject.toml` and `uv.lock` together.

---

## Scoring system

The custom scoring (`add_custom_league_points` in `features.py`) uses:
- **Passing: 0.1 pts/yard** (not the standard 0.04) — this makes QBs 2.5× more valuable than in standard PPR
- Rushing: 0.1 pts/yard, 6 pts/TD
- Receiving: 1 pt/rec (PPR), 0.1 pts/yard, 6 pts/TD
- FG: 3/4/5 pts for 0–39/40–49/50+

Consequence: elite QB season totals of 500–750 pts are correct for this system. The `positional_scarcity` multiplier in `DraftSettings` (default QB=0.7) adjusts the board for real draft dynamics in 1-QB leagues.

---

## Key CLI commands

```bash
# Full data refresh + retrain
uv run nfl-predict update-all

# Draft preparation (run once per season)
uv run nfl-predict draft-prep                          # train p10/p50/p90 models
uv run nfl-predict fetch-adp --source sleeper          # pull ADP
uv run nfl-predict board --league-size 12 \
       --adp data/adp_current.csv --fmt csv            # build board
uv run nfl-predict board --fmt table                   # quick terminal preview
uv run nfl-predict board --fmt table --superflex       # superflex league

# Season projections
uv run nfl-predict project-season --position QB --top 20

# Live draft (terminal mode)
uv run nfl-predict draft-start --league-size 12 --draft-position 5
uv run nfl-predict draft-pick "Bijan Robinson" --mine
uv run nfl-predict draft-pick "Drake Maye"             # opponent pick

# NFL Fantasy auto-sync (run in a second terminal during draft)
# Requires: NFL_FANTASY_USERNAME, NFL_FANTASY_PASSWORD, NFL_FANTASY_LEAGUE_ID
uv run nfl-predict nfl-sync --interval 30

# Start the web UI
uvicorn nfl_predict.api:app --host 0.0.0.0 --port 8000
# → http://localhost:8000/draft
```

---

## Draft day workflow (real-world use)

### Night before
```bash
uv run nfl-predict update-all --no-train
uv run nfl-predict draft-prep
uv run nfl-predict fetch-adp --source sleeper --scoring half
uv run nfl-predict board --league-size 12 --adp data/adp_current.csv
```

### At the venue
```bash
export ANTHROPIC_API_KEY=sk-...          # optional — enables "Ask Claude" button
export NFL_FANTASY_USERNAME=you@email.com
export NFL_FANTASY_PASSWORD=yourpassword
export NFL_FANTASY_LEAGUE_ID=12345678
export NFL_FANTASY_TEAM_ID=3             # your team slot number

uvicorn nfl_predict.api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/draft in browser
# Friends on same WiFi can view at http://<your-ip>:8000/draft

# In a second terminal for auto-sync with NFL Fantasy:
uv run nfl-predict nfl-sync --interval 30
```

### During the draft
- Type a player name (or substring) and press Enter to record an opponent pick
- Toggle **Mine** checkbox before submitting for your own picks
- Click a row's **Fill** or **Mine** button to pre-fill the input (also sets player_id for exact match)
- Click **↩ Undo** to reverse the last pick (any miskey)
- Use position filter tabs (QB / RB / WR / TE / K) to narrow the board
- Click **Ask Claude** for AI pick advice (requires `ANTHROPIC_API_KEY`)
- **NFL Sync** button appears when NFL Fantasy credentials are set — pulls picks automatically

---

## Data model notes

### Season snapshot features
Built by `season_features.build_season_snapshot()`. Takes the **last week** of each player-season (which carries `season_cum` and `roll8` aggregates). Key columns used as features:

- `fantasy_points_custom_season_cum` — season total points so far
- `fantasy_points_custom_roll8_mean` — rolling 8-week average (per-game rate proxy)
- `games_played_season` — games played in that season (important for injury-affected players)
- `age_at_season_start`, `years_exp` — career stage signals

### Known model limitation
The season model predicts **season totals**, so a player who missed games due to injury (e.g. Burrow: 8 games, 30 pts/game) gets a deflated projection (262 pts) rather than rate-adjusted. The fix (per-game rate target × predicted games played) is planned as a medium-complexity improvement.

### Player name format
Raw feature data uses abbreviated names (`J.Allen`, `B.Robinson`). Roster data uses full names (`Josh Allen`, `Bijan Robinson`). The join between them is on `player_id` / `gsis_id` — never on name alone. Two players with the same abbreviated name (e.g. Brian Robinson and Bijan Robinson both appear as `B.Robinson`) are correctly separated by their distinct `player_id`.

---

## Architecture decisions

### Why CatBoost quantile regression?
Native support for `Quantile:alpha=` loss gives p10/p50/p90 in a single training call. No need for separate calibration. Handles the mixed categorical/numerical feature space well with minimal preprocessing.

### Why htmx (not React/Vue)?
The draft UI is server-rendered Jinja2 with htmx for partial updates. After each pick, the server returns an HTML fragment with `hx-swap-oob` attributes that update the board table, roster sidebar, pick counter header, and suggestions panel simultaneously — no JSON API, no client-side state.

### Why snake draft state in JSON?
`draft_state.json` survives server restarts, browser refreshes, hotspot drops. The board CSV is embedded as a CSV string so the full state round-trips through a single file. `outputs/` is gitignored.

### Positional scarcity in VOR
Raw VOR (`proj_p50 − replacement_baseline`) is mathematically correct but puts 12 QBs in the top 27 picks for this scoring system. `DraftSettings.positional_scarcity` applies a per-position multiplier *after* VOR is calculated. Default: `QB=0.7, TE=0.85, RB/WR=1.0, K=0.5`. Pass `--superflex` to set QB=1.0 for superflex leagues.

---

## Files to never edit directly
- `data/*.parquet` — regenerated by `update-all`
- `outputs/draft_board_*.csv` — regenerated by `board` command
- `outputs/draft_state.json` — managed by draft assistant at runtime
- `models/model_registry.json` — managed by `model_registry.py`
- `uv.lock` — managed by `uv`; commit alongside `pyproject.toml` changes
