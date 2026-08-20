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
  metrics.py           MAE, RMSE, R2, Spearman, top-N precision

  season_features.py   Player-season snapshots + injury-report summary per season
  season_model.py      CatBoost quantile regression — 3 targets (total / ppg / games)
                       x 3 quantiles = 9 models per position
  draft_board.py       VOR calculation, tier assignment, CSV/JSON export
  draft_assistant.py   Live draft state — mark_drafted, undo, suggest, save/load JSON
  adp_fetch.py         ADP from Sleeper / FantasyPros / synthetic fallback
  nfl_fantasy.py       NFL.com OAuth2 client — live draft pick polling
  draft_api.py         FastAPI router (/draft/*) with htmx partials
  weekly_api.py        FastAPI router (/weekly/*) — season-long league beta
  lineup.py            Standalone `suggest` CLI — optimal lineup from predictions
  api.py               Main FastAPI app (mounts draft + weekly routers)
  cli.py               Typer CLI entry point (nfl-predict)

  templates/
    draft_setup.html           Setup / landing page
    draft_board.html           Live draft board page
    partials/board_table.html  htmx swap target — available players table
    partials/pick_response.html OOB swap after each pick (board+roster+header)
    partials/roster_panel.html  My roster sidebar
    partials/suggestions.html   Best-available panel
    weekly_setup.html          /weekly team picker
    weekly_team.html           /weekly per-team lineup view

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

models/
  model_registry.json     Version registry
  {pos}_season_{target}_{q}.cbm       Trained season models + _meta.json sidecars

tests/
  test_bugs.py            Regression tests (9 tests)
  test_draft_phase1.py    season_features, season_model, draft_board (60 tests)
  test_draft_phase2.py    draft_assistant (34 tests)
  test_draft_phase3.py    adp_fetch, CLI (25 tests)
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
All 128 tests must pass before committing.

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
- **Passing: 0.1 pts/yard** (not the standard 0.04), 4 pts/TD, -2 pts/INT — the yardage rate makes QBs 2.5× more valuable than in standard PPR
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
uv run nfl-predict draft-prep                          # train season models
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
- **NFL Sync** button appears when NFL Fantasy credentials are set — pulls picks automatically

---

## Data model notes

### Season snapshot features
Built by `season_features.build_season_snapshot()`. Takes the **last week** of each player-season (which carries `season_cum` and `roll8` aggregates). Key columns used as features:

- `fantasy_points_custom_season_cum` — season total points so far
- `fantasy_points_custom_roll8` — rolling 8-week average (per-game rate proxy)
- `fantasy_points_custom_season_mean` — per-game rate over the season to date
- `games_played_season` — regular-season games the player appeared in
- `age_at_season_start`, `years_exp` — career stage signals

**Regular-season scope.** The snapshot drops postseason rows before aggregating (`regular_season_only=True`). Fantasy seasons are regular-season only, and playoff appearances track team quality rather than the player — leaving them in inflated 7.6% of player-season totals by a mean of +19 pts.

**`games_played_season` counts appearances, not scoring games.** A player who took the field but scored <= 0 — a QB with two picks and no TDs, a kicker with no attempts — still played. Counting only positive-scoring games undercounted 21% of QB seasons by up to 8 games and corrupts any per-game rate derived from it.

### Rate vs availability
A season total conflates *how good a player is* with *how much of him you get*. The season model reports both, as three independent CatBoost families per position (p10/p50/p90 each):

| Target | Column | Question |
|---|---|---|
| `season_total_pts_next` | `proj_p50` | expected season points (VOR ranks on this) |
| `season_ppg_next` | `proj_ppg_p50` | scoring rate in games actually played |
| `games_played_next` | `proj_games_p50` | availability |

A player who missed time now reads as "elite rate, low games" rather than one deflated number — e.g. Dak Prescott (8 games in 2024) projects 29.6 ppg over 11.7 games.

**These columns do not multiply back to `proj_p50`.** The product overstates the total in ~80% of rows (median ~10%): the rate model answers "when he plays", and players who miss time also score less while hurt. The total is modelled directly rather than derived, because the median of a product is not the product of the medians — multiplying component medians measurably worsened QB MAE.

The rate model is trained with `sample_weight = games_played_next`, since a rate observed over 2 games is far noisier than one over 17.

The draft UI shows `Rate` and `G` columns on the board table (gated on the columns being present, so sessions started from an older board CSV still render), with `G` amber under 13 games and red under 11. The Best Available panel is too narrow for both, so it shows a games badge only when the projection is under 13. **Low games does not imply injury** — it also covers committee roles and backups; the model predicts games, not the reason.

**Measured bias.** Walk-forward 2019–2024, mean residual for players with <12 games vs ≥12 games in the prior season: QB **+36 pts**, RB −2, WR +4, TE −4. The bias is essentially QB-only — passing at 0.1 pts/yard puts QBs at 30–45 ppg, so a missed game costs a QB ~3× what it costs a WR. Availability is also the dominant error term: substituting true games played into the projection cuts QB MAE from ~125 to ~57, while substituting the true rate only reaches ~84.

### Injury data is reported, not modelled
`injuries.parquet` is summarised per player-season by `build_injury_season_features` into `inj_weeks_out`, `inj_weeks_on_report`, `inj_weeks_dnp`, and `inj_primary` (body part). These reach the board CSV and the `G` column tooltip — **they are deliberately excluded from every model.**

Walk-forward 2019–2024 found no incremental predictive value on any of the three targets: deltas within ±1.3% MAE, mostly slightly *worse*, and unchanged when restricted to established starters (≥10 games). `games_played_season` is already a feature and is itself the strong injury proxy; the report detail adds nothing on top. Multi-season durability history (lagged games played, career injury burden) was also tested and added nothing.

Aggregate the report **directly**, never via the weekly feature table. The weekly merge in `features.py` attaches injury status to games the player *played*, so it structurally cannot see a week he was Out — 0% of `Out` rows have a weekly stat line, and only 50% of report rows are visible at all. `injury_status_season_cum` therefore means "played while listed", not "missed time". That weekly usage is still legitimate for the *weekly* model, where playing hurt predicts lower output.

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
