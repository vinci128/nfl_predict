# CLAUDE.md — nfl_predict

This file gives Claude Code the context needed to work effectively in this repo.

## What this project does

`nfl_predict` is a CatBoost-based NFL fantasy football pipeline with two modes:

1. **Weekly predictions** — point forecasts for upcoming game weeks (used during the season)
2. **Draft assistant** — season-total projections (p10/p50/p90) + VOR-based draft board + live draft UI (used at draft time)

The stack: Python 3.12, CatBoost, pandas, FastAPI + htmx, Typer CLI, nflreadpy for data.

It serves **three real ESPN leagues**, two of which have incompatible rules.
Everything that depends on scoring — feature tables, models, boards, draft
state — is namespaced by league. See "Leagues" below; run `nfl-predict leagues`
for the current configuration.

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
  draft_queue.py       ESPN autodraft queue built from a board
  espn_fantasy.py      ESPN API client — draft picks, rosters, box scores
  nfl_fantasy.py       NFL.com OAuth2 client — DEPRECATED, folded into ESPN
  draft_sync.py        Picks a live-sync provider (ESPN, or the legacy NFL one)
  leagues.py           League profiles — scoring rules, rosters, artifact paths
  dst.py               Team defence / special teams projections (shrinkage)
  draft_api.py         FastAPI router (/draft/*) with htmx partials
  weekly_api.py        FastAPI router (/weekly/*) — season-long league beta
  lineup.py            Standalone `suggest` CLI — optimal lineup from predictions
  api.py               Main FastAPI app (mounts draft + weekly routers)
  cli.py               Typer CLI entry point (nfl-predict)

  templates/
    base.html                  Shell: nav, league switcher
    home.html                  Landing page
    draft_setup.html           Setup / landing page
    draft_board.html           Live draft board page
    partials/board_table.html  htmx swap target — available players table
    partials/pick_response.html OOB swap after each pick (board+roster+header)
    partials/roster_panel.html  My roster sidebar
    partials/suggestions.html   Best-available panel
    partials/update_status.html Data-refresh progress panel
    weekly_setup.html          /weekly team picker
    weekly_team.html           /weekly per-team lineup view

scripts/
  compare_espn_clients.py  Diffs our ESPN client against the espn-api library
  compare_features.py      Model MAE with and without the context features

data/
  weekly_stats.parquet    Raw NFL weekly player stats
  rosters.parquet         Player roster info (name, position, age, years_exp)
  injuries.parquet        Weekly injury report
  schedules.parquet       Game schedules
  snap_counts.parquet     Snap count data
  processed/{artifact_key}/
    player_week_features.parquet  Engineered features (roll windows, cumulative)
  adp_current.csv         Most recent ADP fetch
  team_stats.parquet      Team-level stats (the D/ST projection's only source)
  keepers_ludopathy_2026.txt  Ludopathy keeper list (excluded from the board)

outputs/
  draft_board_YYYY_{league}.csv  Current draft board (rebuilt each year)
  draft_queue_YYYY_{league}.csv  ESPN autodraft queue, board order
  draft_state_{league}.json      Live draft session state (persisted per pick)

models/
  model_registry.json     Version registry
  {pos}_season_{target}_{q}.cbm       Trained season models + _meta.json sidecars

tests/                                                    525 tests total
  test_bugs.py              Regression tests for fixed defects (16)
  test_draft_phase1.py      season_features, season_model, draft_board (60)
  test_draft_phase2.py      draft_assistant (37)
  test_draft_phase3.py      adp_fetch, CLI (32)
  test_leagues.py           Scoring rules verified against ESPN, per league (84)
  test_espn_sync.py         ESPN client: picks, rosters, box scores (156)
  test_espn_login.py        Cookie storage, without echoing them (13)
  test_dst.py               Team defence shrinkage projection (21)
  test_draft_queue.py       Autodraft queue depth and ordering (16)
  test_projection_sanity.py Properties every league's board must hold (31)
  test_league_switcher.py   Per-request league from a cookie (15)
  test_draft_api_league.py  Web UI acts on the right league's session (14)
  test_draft_cli_league.py  CLI acts on the right league's session (9)
  test_weekly_update.py     Background data refresh (21)
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

### Pre-commit hooks
- `ruff --fix` — lint and auto-fix
- `ruff format` — format
- `ty check` — type checking

**Hooks are not installed by default** — a fresh clone has an empty `.git/hooks/`, so nothing runs on commit until you run `uv run pre-commit install`. Until then the config is inert and CI is the only gate.

If pre-commit modifies files, **re-stage** the modified files and commit again. Never use `--no-verify`.

### CI (`.github/workflows/ci.yml`)
Runs on push to `dev`/`master` and on PRs: `pre-commit run --all-files` → `pytest`.

CI runs the **same** `.pre-commit-config.yaml`, so the hook set is the single definition of the lint/format/type gates and is enforced whether or not a contributor installed the hooks. Reproduce a CI failure locally with:

```bash
uv run pre-commit run --all-files
uv run pytest tests/ -x -q
```

Note pre-commit lints *every* Python file, including `scripts/`, while the old CI step only covered `src/` and `tests/`.

### Adding dependencies
```bash
uv add <package>          # runtime
uv add --dev <package>    # dev only
```
Always commit `pyproject.toml` and `uv.lock` together.

---

## Leagues

`leagues.py` holds one `LeagueProfile` per league: scoring rules, roster
configuration, ESPN ids, keeper settings, and the artifact paths its outputs
live under. **Scoring is data, not code** — `ScoringRules` is applied by
`add_custom_league_points`, which no longer hardcodes any values.

| | **Ludopathy Bowl** (`ludopathy`) | **Hell or Highwater** (`hoh`) | **Royal Rumble** (`rumble`) |
|---|---|---|---|
| Teams / roster | 10 / 23 (10 bench, 3 IR) | 14 / 16 (7 bench, 3 IR) | 8 / 14 (5 bench, 1 IR) |
| Starters | QB1 RB2 WR2 TE1 FLEX1 **LB2 DL1 DB2** K1 | QB1 RB2 WR2 TE1 FLEX1 **D/ST1** K1 | QB1 RB2 WR2 TE1 FLEX1 **D/ST1** K1 |
| Passing yards | **1 pt / 10 yds** (floored) | **0.1 / yd** (continuous) | 0.04 / yd |
| Field goals | 3 / 3 / 5 / 6, −1 missed | 3 / 4 / 5 / 6, −1 missed | 3 / 4 / 5 / 6, −1 missed |
| 2-pt conversions | 2 (pass, rush) | 2 | 2 |
| Interception thrown | **−4** | **−4** | −2 |
| Sack taken (QB) | **−0.5** | **−0.5** | not scored |
| Game bonuses | 400 pass, 100/200 rush, 100/200 rec | none | none |
| Keepers | **6 per team** | none | none |
| ESPN league id | 1773102615 | 581348581 | 1546288813 |
| Private? | yes (needs cookies) | **no** | yes (needs cookies) |

**They did until 2026-09-03.** Hell or Highwater and Royal Rumble ran ESPN's
default PPR, shared one `ScoringRules` instance and therefore one fit. Hours
before the draft the Hell or Highwater commissioner moved passing to 0.1/yd,
an interception to −4, and added the −0.5 sack penalty. It now has its own
`_HOH_SCORING`, its own feature table and its own models; Royal Rumble keeps
`_ESPN_PPR_SCORING` and inherited the artifacts the two used to share.

Note the passing rates only *look* alike: Hell or Highwater is a continuous
0.1/yd, Ludopathy is the bucketed "every 10 yards = 1" that floors the
remainder. 327 yards is 32.7 in one and 32 in the other.

**Re-read the scoring on draft day.** A commissioner can change it at any
time, and nothing announces it. `EspnFantasyClient` plus the stat-id table in
this file is enough to diff it in a minute; doing that is what caught this,
40 minutes before the draft.

Because the training target *is* fantasy points, identical scoring means an
identical fit, so Royal Rumble declares `shares_artifacts_with="hoh"` and both
read one feature table and one set of models (`profile.artifact_key`). Fitting
twice was not just waste: the two runs disagreed on 288 of 690 players, by up
to 28 points, so the same player was worth two different numbers depending on
which league you asked. `_validate_shared_artifacts` refuses the declaration
if the two leagues' scoring differs, and `artifact_keys()` is what a
build-everything loop should iterate so no table is built twice.

Boards and draft state stay per-league regardless — VOR depends on league size,
and two drafts on one evening must not share a session file.

What separates them is size. At 8 teams Royal Rumble has by far the deepest
available pool, so replacement level is high and VOR gaps are much smaller —
Chase Brown is worth 142 VOR in the 14-team league and 88 in the 8-team one,
and kickers and defences come off the board ~26 picks earlier.

Consequence: elite QB season totals of 500–750 are correct for Ludopathy and
wrong for Hell or Highwater, where the same season lands near 300–430. A model
trained on one league's points is confidently wrong for the other, which is why
`models/` and `data/processed/` are namespaced by league key.

### Scoring rules are per-*game*, not per-season
ESPN's threshold bonuses ("100–199 yard rushing game") are per-game awards, so
`ScoringRules.score()` must run at weekly grain, before any aggregation.
Scoring a pre-summed season total pays each bonus once for the whole year.

### Two forms of yardage scoring
`per_unit` is a continuous rate (0.04/yd). `per_increment` is ESPN's bucketed
form ("every 10 yards = 1"), which **floors** the remainder — 347 passing yards
is 34 points, not 34.7. A stat scored under both would double-count, which
`ScoringRules.__post_init__` rejects.

### Verify scoring against ESPN, not against assumption
Both leagues' rules were checked against ESPN's settings page on 2026-09-02 and
Ludopathy's were wrong in seven places, every one of them an assumption nobody
had confirmed: 50-59 and 60+ field goals scored 0 rather than 5 and 6, missed
field goals and 2-point conversions were missing entirely, an interception cost
−2 rather than −4, the sack-taken penalty was absent, and the IDP categories
used the D/ST values.

Categories worth zero are omitted from ESPN's summary, so an absent row does
mean the category scores nothing — but a *present* row is the only evidence
that a value is what you think it is. Read the page:

    https://fantasy.espn.com/football/league/settings?leagueId=<id>&view=scoring

For a public league the same values come back from the API without a login,
which is easier to diff:

    view=mSettings -> settings.scoringSettings.scoringItems[]

### D/ST and IDP score off different tables
ESPN's settings page has a **Team Defense / Special Teams** section and a
separate **Defensive Players** section, and their values differ. In Ludopathy a
defender's sack is 4 where a defence's is 1, and his interception 5 where a
defence's is 2. Scoring IDPs off the D/ST column understated every linebacker
by more than half.

ESPN also pays a tackle under two categories at once: Total Tackles (TK, 1)
applies to every tackle, and Solo (TKS, 1.5) or Assisted (TKA, 0.5) applies on
top — so a solo tackle is worth 2.5 and an assist 1.5. `_LUDOPATHY_SCORING`
folds the pair together into one rate per stat.

### Known gaps
- Ludopathy's 40+/50+ yard TD pass bonuses (PTD40 +2, PTD50 +3) need play-level
  data and are not modelled. They are declared in `ScoringRules.unmodelled` and
  printed by `nfl-predict leagues` rather than silently dropped.
- Ludopathy's IDP scoring is taken from ESPN's Defensive Players table and is
  confirmed, but it has not yet been checked against a real box score. Do that
  after week 1 — `EspnFantasyClient.fetch_boxscore` returns ESPN's own
  `actual_points` per player, which is exactly the comparison to run.

---

## Key CLI commands

Almost every command takes `--league`. Omit it and the default (`ludopathy`)
applies; export `NFL_PREDICT_LEAGUE=hoh` to pin one for a whole session.

```bash
# What is configured
uv run nfl-predict leagues

# Full data refresh. Raw data is shared; scoring is not, so --all rebuilds
# one feature table per league.
uv run nfl-predict update-all --all
uv run nfl-predict features --all                      # features only

# Draft preparation (run once per season, per league)
uv run nfl-predict draft-prep --league hoh             # train season models
uv run nfl-predict fetch-adp --source sleeper          # pull ADP (shared)
uv run nfl-predict board --league hoh \
       --adp data/adp_current.csv --fmt csv            # build board
uv run nfl-predict board --league ludopathy --fmt table   # terminal preview

# Season projections
uv run nfl-predict project-season --league hoh --position QB --top 20

# Autodraft queue. ESPN drafts from your queue on a pick you miss, so this is
# what stands in for you if two drafts overlap or you step away.
uv run nfl-predict queue --league hoh

# Live draft (terminal mode) — team count comes from the league
uv run nfl-predict draft-start --league hoh --draft-position 5
uv run nfl-predict draft-pick "Bijan Robinson" --mine
uv run nfl-predict draft-pick "Drake Maye"             # opponent pick

# Store the cookies a private league needs. Prompts without echoing, writes
# .env at mode 600, then checks every configured league. Cookies expire, so
# run it near draft time.
uv run nfl-predict espn-login

# ESPN auto-sync (run in a second terminal during the draft).
# The league id comes from the profile; private leagues also need
# ESPN_S2 and ESPN_SWID.
uv run nfl-predict nfl-sync --league hoh --interval 30

# Start the web UI
uvicorn nfl_predict.api:app --host 0.0.0.0 --port 8000
# → http://localhost:8000/draft
```

Per-league artifacts:

```
data/processed/{league}/player_week_features.parquet
models/{league}/{pos}_season_{target}_{q}.cbm
outputs/draft_board_{season}_{league}.csv
outputs/draft_state_{league}.json
```

---

## Draft day workflow (real-world use)

Set `LEAGUE` first — every command below reads it.

### Night before
```bash
export LEAGUE=hoh          # or ludopathy

uv run nfl-predict update-all --no-train --all
uv run nfl-predict draft-prep --league $LEAGUE
uv run nfl-predict fetch-adp --source sleeper --scoring half
uv run nfl-predict board --league $LEAGUE --adp data/adp_current.csv
```

For **Ludopathy**, write the keeper list once ESPN locks it, one hour before
the draft, and rebuild:

```bash
uv run nfl-predict keepers --league ludopathy     # reads ESPN, writes the file
uv run nfl-predict board --league ludopathy --adp data/adp_current.csv
```

After the lock each team's roster *is* its keepers, which is what `keepers`
reads. Before it, rosters still hold last season's full squads — the command
counts players per team and refuses to write rather than dumping 220 names
into an exclusion list that would empty the board. 60 of the pool's best go,
so replacement level moves a long way; the board is wrong until you rebuild.

### At the venue
```bash
# Private leagues need cookies. Hell or Highwater is public; Ludopathy and
# Royal Rumble are not. Prompts without echoing and checks all three.
uv run nfl-predict espn-login

uvicorn nfl_predict.api:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/draft in browser
# Friends on same WiFi can view at http://<your-ip>:8000/draft

# In a second terminal for auto-sync with ESPN:
uv run nfl-predict nfl-sync --league hoh --interval 30
```

Pick the league from the **nav dropdown**, not an environment variable. The
active league is a per-request cookie, so one server can run two drafts at once
in two browser profiles and each keeps its own session file. `NFL_PREDICT_LEAGUE`
still works and is what the CLI reads.

**A session belongs to exactly one league.** `/draft` redirects to the board
when the active league has one, and `/draft/board` redirects to setup when it
does not, so switching league mid-evening lands on that league's draft rather
than a form or a 404. `_active_state_path` deliberately has no fallback to
"the only session on disk": it used to, and with a Hell or Highwater draft
running, switching to Royal Rumble showed that 14-team slate under the Royal
Rumble heading.

**Size and draft slot come from ESPN automatically.** The setup page paints the
league profile's own values, then replaces them with ESPN's on load — a
mistyped draft position throws the snake order off from the first pick, so it
is not left to typing. The `auto_fetch` flag is set only on that first paint;
the swapped-in response omits it, which is what stops the load trigger firing
forever. Rendering the profile's values first also means the form still
submits if ESPN never answers. The button beside the fields is a manual
re-read, for once a randomised order locks. `EspnFantasyClient.fetch_draft_setup` takes the size from
`settings.size` and the slot from the index of your team in
`draftSettings.pickOrder`. It also reports whether that slot is settled:
`orderType` is MANUAL or PREDETERMINED once a commissioner has fixed the order
(Ludopathy), and DRAFT_START when ESPN randomises it as the draft begins (Hell
or Highwater, Royal Rumble) — in which case re-read it once the draft opens.

### `.env` is only read by `EspnFantasyClient.from_env`
`espn-login` writes the ESPN cookies to `.env`, and for a while nothing read
that file: a private league answered 401 with working credentials sitting on
disk, and `espn-login --check` still passed because it had set them in its own
process. `from_env` now calls `_load_dotenv()` first, with `override=False` so
an exported variable still wins.

Tests must not inherit that file. `test_espn_sync.py` has an autouse fixture
that chdirs to a tmp directory, because a test asserting "no cookies
configured" would otherwise pick up live credentials and print them into the
failure output.

Load the **autodraft queue** into ESPN before the clock starts, so a pick you
miss still follows the board:

```bash
uv run nfl-predict queue --league hoh    # then paste the order into ESPN's Pick Queue
```

Verified against a live practice draft: ESPN drafts from the queue before
falling back to its own rankings.

### During the draft
- Type a player name (or substring) and press Enter to record an opponent pick
- Toggle **Opponent/Mine** before submitting for your own picks; it resets to
  Opponent after each pick, since most picks are not yours
- Click a row's **Fill** or **Mine** button to pre-fill the input (also sets player_id for exact match)
- Click **↩ Undo** to reverse the last pick (any miskey)
- Use position filter tabs to narrow the board — they follow the league, so
  Ludopathy shows LB/DL and Hell or Highwater shows DST
- **Sync** button appears when the league has an ESPN id — pulls picks automatically

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

**The three quantiles are separate models and can cross.** Nothing constrains
p10 <= p50 <= p90, so on a player the fits disagree about they invert — Davante
Adams came out with a p90 below his p50 while ranking inside the top 30 on
every board, which the UI renders as a ceiling lower than the median.
`season_model._enforce_quantile_order` clamps the band around the median rather
than sorting all three: p50 is what VOR ranks on and the only quantile with
walk-forward validation behind it, so re-ranking players on the say-so of the
least reliable model would be the worse trade.

**These columns do not multiply back to `proj_p50`.** The product overstates the total in ~80% of rows (median ~10%): the rate model answers "when he plays", and players who miss time also score less while hurt. The total is modelled directly rather than derived, because the median of a product is not the product of the medians — multiplying component medians measurably worsened QB MAE.

The rate model is trained with `sample_weight = games_played_next`, since a rate observed over 2 games is far noisier than one over 17.

The draft UI shows `Rate` and `G` columns on the board table (gated on the columns being present, so sessions started from an older board CSV still render), with `G` amber under 13 games and red under 11. The Best Available panel is too narrow for both, so it shows a games badge only when the projection is under 13. **Low games does not imply injury** — it also covers committee roles and backups; the model predicts games, not the reason.

**Measured bias.** Walk-forward 2019–2024, mean residual for players with <12 games vs ≥12 games in the prior season: QB **+36 pts**, RB −2, WR +4, TE −4. The bias is essentially QB-only — passing at 0.1 pts/yard puts QBs at 30–45 ppg, so a missed game costs a QB ~3× what it costs a WR. Availability is also the dominant error term: substituting true games played into the projection cuts QB MAE from ~125 to ~57, while substituting the true rate only reaches ~84.

### Injury data is reported, not modelled
`injuries.parquet` is summarised per player-season by `build_injury_season_features` into `inj_weeks_out`, `inj_weeks_on_report`, `inj_weeks_dnp`, and `inj_primary` (body part). These reach the board CSV and the `G` column tooltip — **they are deliberately excluded from every model.**

Walk-forward 2019–2024 found no incremental predictive value on any of the three targets: deltas within ±1.3% MAE, mostly slightly *worse*, and unchanged when restricted to established starters (≥10 games). `games_played_season` is already a feature and is itself the strong injury proxy; the report detail adds nothing on top. Multi-season durability history (lagged games played, career injury burden) was also tested and added nothing.

Aggregate the report **directly**, never via the weekly feature table. The weekly merge in `features.py` attaches injury status to games the player *played*, so it structurally cannot see a week he was Out — 0% of `Out` rows have a weekly stat line, and only 50% of report rows are visible at all. `injury_status_season_cum` therefore means "played while listed", not "missed time". That weekly usage is still legitimate for the *weekly* model, where playing hurt predicts lower output.

### IDP (Ludopathy only)
`position` in the feature table carries the **fantasy slot**, not the raw
roster label: DE/DT/NT collapse to `DL`, ILB/MLB/OLB to `LB`, CB/S/FS/SS to
`DB`, and the raw value is kept beside it as `position_raw`. A slot only
reaches the board if a league starts it — DB mapped correctly for months but
was dropped from every board until Ludopathy added the slot on 2026-09-03. Everything downstream compares against the
slot, so this mapping is what makes `LB` and `DL` models possible at all. The
mapping lives in `leagues.fantasy_position`; a position that maps to nothing
(offensive line, long snapper, punter) is dropped — it cannot fill a slot in
either league.

Scale check against 2025 under the corrected rules: an elite LB is ~390 points
and an elite DL ~230, next to ~600 for the top QB and ~190 for the top kicker.
IDP is not a late-round afterthought in this league — the best linebacker
ranks **26th overall** on the board and 15 of the top 60 are LB or DL. The
earlier figures here (~145 LB, ~80 DL, "well after the skill positions") came
from scoring IDPs off the D/ST table and are wrong.

### D/ST (Hell or Highwater only)
`dst.py` is deliberately **not** a CatBoost quantile family. Two derived inputs
drive most of the scoring and neither is in any stats table directly: points
allowed is the *opponent's* score (from `schedules`), and yards allowed are the
*opponent's* gains (from `team_stats`, joined on `opponent_team`). Get either
join backwards and the numbers still look plausible.

Measured 2015–2024, a defence's fantasy points correlate **r = 0.32** with the
next season's — about 10% of the variance, from 32 rows a year. So the
projection is a shrinkage fit (`next ≈ 0.29 × prior + 50`) with the p10/p90
band taken from the residual spread. The band (~82 points wide) is twice the
spread between the best and worst projected defence (~43), which is the honest
picture: the board will rank all 32 in a clump and let them fall to the end of
the draft.

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
