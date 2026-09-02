nfl_predict
===========

CatBoost-based NFL fantasy football pipeline with two modes:

1. **Weekly predictions** — point forecasts for upcoming game weeks (used during the season)
2. **Draft assistant** — season projections split into scoring rate and availability (p10/p50/p90 each) + VOR-based draft board + live draft UI (used at draft time)

Stack: Python 3.12, CatBoost, pandas, FastAPI + htmx, Typer CLI, nflreadpy.

---

## Setup

Requires [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync

cp .env.example .env          # fill in credentials (see below)
```

---

## Leagues

This repo serves **three real ESPN leagues**, two of which have incompatible
rules. Pick one with `--league` on any command, pin it for a session with
`export NFL_PREDICT_LEAGUE=hoh`, or choose it from the dropdown in the web UI.
Run `nfl-predict leagues` to see the live configuration.

| | **Ludopathy Bowl** (`ludopathy`) | **Hell or Highwater** (`hoh`) | **Royal Rumble** (`rumble`) |
|---|---|---|---|
| Teams | 10 | 14 | 8 |
| Roster | 21 (9 bench, 3 IR) | 16 (7 bench, 3 IR) | 14 (5 bench, 1 IR) |
| Starters | QB1 RB2 WR2 TE1 FLEX1 **LB3 DL1** K1 | QB1 RB2 WR2 TE1 FLEX1 **D/ST1** K1 | QB1 RB2 WR2 TE1 FLEX1 **D/ST1** K1 |
| Keepers | 6 per team | none | none |
| Private? | yes (needs cookies) | no | yes (needs cookies) |

### Scoring

Hell or Highwater and Royal Rumble run ESPN's default PPR, identical category
for category, so the column below covers both.

| Stat | Ludopathy | Hell or Highwater / Royal Rumble |
|---|---|---|
| Passing yards | 1 per 10 (floored) | 0.04 per yard |
| Passing TD / INT | 4 / **−4** | 4 / −2 |
| Sack taken (QB) | **−0.5** | not scored |
| Rush/Rec yards | 1 per 10 (floored) | 0.1 per yard |
| Rush/Rec TD | 6 | 6 |
| Reception | 1 (PPR) | 1 (PPR) |
| Fumble lost | −2 | −2 |
| 2-pt conversion | 2 (pass, rush) | 2 |
| FG 0–39 / 40–49 / 50–59 / 60+ | 3 / 3 / 5 / 6 | 3 / 4 / 5 / 6 |
| Missed FG | −1 | −1 |
| PAT | 1 | 1 |
| Game bonuses | 400 pass +4; 100/200 rush +2/+3; 100/200 rec +2/+3 | — |
| IDP | sack 4, INT 5, fumble 4, tackle 2.5 solo / 1.5 assist | not scored |

Every value was read off each league's ESPN settings page on 2026-09-02, not
assumed. Ludopathy's were wrong in seven places before that check — see
CLAUDE.md, "Verify scoring against ESPN, not against assumption".

The passing rate is the difference that matters. At 0.1/yard Ludopathy makes
QBs ~2.5× more valuable than standard, and elite QB season totals of 500–750
are correct there. The same seasons land near 300–430 in Hell or Highwater.

**A model trained on one league's scoring is confidently wrong for another**,
so every generated artifact is namespaced:

```
data/processed/{artifact_key}/player_week_features.parquet
models/{artifact_key}/…
outputs/draft_board_{season}_{league}.csv
outputs/draft_queue_{season}_{league}.csv
outputs/draft_state_{league}.json
```

Boards and draft state are per *league*, because VOR depends on league size and
two drafts must not share a session. Feature tables and models are per
*scoring*: Royal Rumble scores exactly as Hell or Highwater does, so it reuses
that fit rather than repeating it. At 8 teams against 14 the same player is
worth 88 VOR in one and 149 in the other — the projections are shared, the
valuations are not.

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
# Show configured leagues
uv run nfl-predict leagues

# Build a feature table per league (raw data is shared, scoring is not)
uv run nfl-predict features --all

# Run once before draft day, per league
# (trains total / rate / games models, p10/p50/p90 each)
uv run nfl-predict draft-prep --league hoh

# Pull ADP (shared across leagues). All three score a full point per
# reception, so this is `ppr` — `half` systematically undervalues
# high-volume slot receivers and pass-catching backs here.
uv run nfl-predict fetch-adp --source sleeper --scoring ppr

# Build the draft board — team count and scarcity come from the league
uv run nfl-predict board --league hoh --adp data/adp_current.csv --fmt csv
uv run nfl-predict board --league ludopathy --fmt table   # terminal preview

# Season projections (terminal)
uv run nfl-predict project-season --league hoh --position QB --top 20

# Autodraft queue: ESPN drafts from this on any pick you miss, and falls
# back to its own rankings once it empties. Paste the order into ESPN's
# Pick Queue before the draft starts.
uv run nfl-predict queue --league hoh
```

### Live draft (terminal mode)

```bash
uv run nfl-predict draft-start --league hoh --draft-position 5
uv run nfl-predict draft-pick "Bijan Robinson" --mine   # your pick
uv run nfl-predict draft-pick "Drake Maye"              # opponent pick
```

### ESPN auto-sync

Polls ESPN and records picks automatically into the league's draft state.
The league id comes from the profile; a **private** league also needs the
`ESPN_S2` and `ESPN_SWID` session cookies. Store them once — the prompt does
not echo, the file is written at mode 600, and every league is checked
afterwards so an expired cookie surfaces now rather than mid-draft:

```bash
uv run nfl-predict espn-login
uv run nfl-predict nfl-sync --league hoh --interval 20
```

Those cookies authenticate the whole ESPN account, not one league. They expire,
so run `espn-login` again near draft time.

### Web UI

```bash
uvicorn nfl_predict.api:app --host 0.0.0.0 --port 8000
# → http://localhost:8000/draft
```

The active league comes from a per-request cookie set by the dropdown in the
nav, so one server serves all three and two people can browse different
leagues at once. It changes the board, the models, the draft state and the
weekly lineups together.

---

## Draft day workflow

### Night before

```bash
export LEAGUE=hoh          # or ludopathy

uv run nfl-predict update-all --no-train --all
uv run nfl-predict draft-prep --league $LEAGUE
uv run nfl-predict fetch-adp --source sleeper --scoring half
uv run nfl-predict board --league $LEAGUE --adp data/adp_current.csv
```

For **Ludopathy**, fill in `data/keepers_ludopathy_2026.txt` once ESPN locks
keepers (one hour before the draft) and rebuild the board. `board` reads that
file automatically from the league profile. With 60 players kept, replacement
level moves a long way — the board is materially wrong without it.

### At the venue — three terminals

```
Terminal 1 (once, before draft starts):
  nfl-predict draft-start --league hoh --draft-position 5

Terminal 2 (runs all draft — mechanical pick recorder):
  nfl-predict nfl-sync --league hoh --interval 20

Terminal 3 (optional — web UI for you and friends on same WiFi):
  uvicorn nfl_predict.api:app --host 0.0.0.0 --port 8000
```

`nfl-sync` is the sole process that writes picks to the league's
`outputs/draft_state_{league}.json`.

### During the draft (web UI)

- Type a player name and press Enter to record an opponent pick
- Toggle **Mine** before submitting for your own picks
- Click **↩ Undo** to reverse a miskey
- Use position tabs to filter the board — they follow the league, so Ludopathy
  shows LB/DL and Hell or Highwater shows DST
- **Rate** is projected points per game, **G** projected games played — a low `G`
  (amber under 13, red under 11) marks limited availability, and its tooltip shows
  last season's injury record. Note Rate x G does not equal Median; see CLAUDE.md

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

Every league's ESPN id, team slot and size lives in its profile, so nothing
below is required to build boards. These are overrides and credentials.

```bash
# Pin a league for the whole shell (otherwise pass --league)
NFL_PREDICT_LEAGUE=hoh

# Private leagues only. Ludopathy and Royal Rumble are private; Hell or
# Highwater is public and needs neither. Prefer `nfl-predict espn-login`,
# which writes these without echoing them. They authenticate the whole ESPN
# account, not one league, and they expire — refresh before draft day.
ESPN_S2=...
ESPN_SWID={...}

# Optional overrides of the profile. These apply to EVERY league at once, so
# with more than one configured you usually want --league instead.
ESPN_LEAGUE_ID=1773102615
ESPN_TEAM_ID=9
ESPN_SEASON=2026
ESPN_LEAGUE_SIZE=10
```

---

## Development

```bash
uv run pytest tests/ -x -q      # 128 tests

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

  season_features.py   Player-season snapshots + injury-report summary
  season_model.py      CatBoost quantile regression — total / rate / games
  draft_board.py       VOR calculation, tier assignment, CSV/JSON export
  draft_assistant.py   Live draft state — mark_drafted, undo, suggest, save/load
  adp_fetch.py         ADP from Sleeper / FantasyPros / synthetic fallback
  draft_queue.py       ESPN autodraft queue built from a board
  leagues.py           League profiles — scoring, rosters, artifact paths
  dst.py               Team defence / special teams projection (shrinkage)
  espn_fantasy.py      ESPN API client — draft picks, rosters, box scores
  nfl_fantasy.py       NFL.com OAuth2 client — deprecated, folded into ESPN
  draft_sync.py        Chooses a live-sync provider
  draft_api.py         FastAPI router (/draft/*) with htmx partials
  weekly_api.py        FastAPI router (/weekly/*) — season-long league beta
  lineup.py            Standalone `suggest` CLI — optimal lineup from predictions
  api.py               Main FastAPI app (mounts draft + weekly routers)
  cli.py               Typer CLI (nfl-predict)

data/                  Raw parquet files (gitignored, regenerated by update-all)
outputs/               Draft board / queue CSV, draft state JSON (gitignored)
models/                Trained .cbm files + model_registry.json, per scoring
scripts/               One-off analyses (ESPN client diff, feature ablation)
tests/                 525 tests
```
