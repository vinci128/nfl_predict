from pathlib import Path

import pandas as pd
import typer

from nfl_predict import features, fetch_nfl_data, predict_week, train_model

app = typer.Typer(help="NFL fantasy prediction pipeline CLI.")
OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# leagues: show the configured league profiles
# ---------------------------------------------------------------------------


@app.command()
def leagues(
    league: str | None = typer.Option(None, help="Show only this league."),
) -> None:
    """List configured leagues: scoring, roster and artifact paths."""
    from nfl_predict.leagues import describe, get_profile, league_keys

    keys = [get_profile(league).key] if league else league_keys()
    for key in keys:
        profile = get_profile(key)
        print(describe(profile))
        print(f"  features        {profile.features_path}")
        print(f"  models          {profile.model_dir}")
        print(f"  board           {profile.board_path(profile.season)}")
        print()


# ---------------------------------------------------------------------------
# features: rebuild the player-week feature table for one league
# ---------------------------------------------------------------------------


@app.command(name="features")
def features_cmd(
    league: str | None = typer.Option(
        None, help="League profile key (see `nfl-predict leagues`)."
    ),
    all_leagues: bool = typer.Option(
        False, "--all", help="Rebuild for every configured league."
    ),
) -> None:
    """Build the player-week feature table, scored under a league's rules."""
    from nfl_predict.leagues import league_keys

    targets = league_keys() if all_leagues else [league]
    for key in targets:
        features.build_player_week_features(league=key)


# ---------------------------------------------------------------------------
# update-all: fetch → features → train → predict
# ---------------------------------------------------------------------------


@app.command()
def update_all(
    fetch: bool = typer.Option(True, help="Download updated NFL data."),
    train: bool = typer.Option(True, help="Retrain models after data update."),
    position: str | None = typer.Option(
        None, help="Position for predictions (default: all main positions)."
    ),
    league: str | None = typer.Option(
        None, help="League profile key (see `nfl-predict leagues`)."
    ),
    all_leagues: bool = typer.Option(
        False, "--all", help="Build features for every configured league."
    ),
) -> None:
    """Fetch data, build features, train models, and run predictions."""
    from nfl_predict.leagues import get_profile, league_keys

    if fetch:
        print(">> Fetching raw NFL data...")
        fetch_nfl_data.main()
        print(">> Building features...")
        # Raw data is shared; scoring is not. Each league gets its own table.
        for key in league_keys() if all_leagues else [league]:
            features.build_player_week_features(league=key)

    if train:
        print(">> Training models...")
        train_model.main()

    print(">> Making predictions...")
    profile = get_profile(league)
    main_positions = [p for p in profile.roster.positions if p != "DST"]
    targets = [position] if position else main_positions
    for pos in targets:
        print(f"   {pos}...")
        predict_week.run_predictions(position=pos)


# ---------------------------------------------------------------------------
# backtest: walk-forward evaluation with baseline comparison
# ---------------------------------------------------------------------------


@app.command()
def backtest(
    position: str = typer.Option(
        "WR", help="Position to backtest (e.g. WR, RB, QB, TE, K)."
    ),
    seasons: str | None = typer.Option(
        None,
        help="Comma-separated test seasons, e.g. '2021,2022,2023'. Defaults to auto.",
    ),
    min_train: int = typer.Option(
        4, help="Minimum training seasons required per fold."
    ),
    iterations: int = typer.Option(
        500, help="CatBoost iterations per fold (lower = faster)."
    ),
    update_registry: bool = typer.Option(
        True, help="Attach backtest MAE to the current champion in the registry."
    ),
) -> None:
    """Run walk-forward backtest and compare against baselines."""
    from nfl_predict.backtest import run_walk_forward_backtest
    from nfl_predict.model_registry import ModelRegistry

    test_seasons: list[int] | None = None
    if seasons:
        test_seasons = [int(s.strip()) for s in seasons.split(",")]

    df = train_model.load_features()
    df = train_model.add_target_next_week(df)

    _, summary = run_walk_forward_backtest(
        df,
        position=position,
        test_seasons=test_seasons,
        min_train_seasons=min_train,
        model_iterations=iterations,
    )

    if update_registry:
        registry = ModelRegistry()
        champion = registry.get_champion(position)
        if champion:
            registry.update_backtest(champion["version_id"], position, summary["model"])
            print(f"Updated backtest metrics for champion {champion['version_id']}")
        else:
            print("No champion found in registry — run `nfl-predict train` first.")


# ---------------------------------------------------------------------------
# predict: run weekly predictions for a position
# ---------------------------------------------------------------------------


@app.command()
def predict(
    position: str = typer.Option("WR", help="Position to predict (QB/RB/WR/TE/K)."),
    season: int | None = typer.Option(
        None, help="Season year (default: inferred from data)."
    ),
    week: int | None = typer.Option(
        None, help="Week number (default: inferred from data)."
    ),
) -> None:
    """Run weekly fantasy predictions for a position."""
    from nfl_predict import predict_week

    predict_week.run_predictions(position=position, season=season, week=week)


# ---------------------------------------------------------------------------
# train: train models and register them
# ---------------------------------------------------------------------------


@app.command()
def train(
    position: str | None = typer.Option(
        None, help="Train only this position (default: all)."
    ),
    no_registry: bool = typer.Option(
        False, help="Skip model registry (plain file save)."
    ),
) -> None:
    """Train position models and register them in the model registry."""
    positions = [position] if position else None
    train_model.main(positions=positions, use_registry=not no_registry)


# ---------------------------------------------------------------------------
# models: list registered model versions
# ---------------------------------------------------------------------------


@app.command()
def models(
    position: str | None = typer.Option(
        None, help="Filter by position (default: all)."
    ),
) -> None:
    """List registered model versions and their metrics."""
    from nfl_predict.model_registry import ModelRegistry

    registry = ModelRegistry()
    if position:
        registry.compare(position)
    else:
        for pos in ["QB", "RB", "WR", "TE", "K"]:
            versions = registry.list_versions(pos)
            if versions:
                registry.compare(pos)


# ---------------------------------------------------------------------------
# promote: set a specific version as the active champion
# ---------------------------------------------------------------------------


@app.command()
def promote(
    version_id: str = typer.Argument(
        help="Version ID to promote (e.g. wr_20250101_120000_abc12345)."
    ),
    position: str = typer.Option(
        ..., help="Position this model was trained on (e.g. WR)."
    ),
) -> None:
    """Promote a specific model version to the active champion."""
    from nfl_predict.model_registry import ModelRegistry

    registry = ModelRegistry()
    registry.promote(version_id, position)
    print(f"Champion for {position} is now: {version_id}")


# ---------------------------------------------------------------------------
# draft-prep: train season models for all positions
# ---------------------------------------------------------------------------


@app.command(name="draft-prep")
def draft_prep(
    position: str | None = typer.Option(
        None, help="Train only this position (default: all)."
    ),
    no_registry: bool = typer.Option(
        False, help="Skip model registry (plain file save)."
    ),
    iterations: int = typer.Option(500, help="CatBoost iterations per quantile model."),
    league: str | None = typer.Option(
        None, help="League profile key (see `nfl-predict leagues`)."
    ),
) -> None:
    """Train season-total projection models (p10/p50/p90) for drafts."""
    from nfl_predict.season_model import main as season_main

    positions = [position.upper()] if position else None
    season_main(
        positions=positions,
        use_registry=not no_registry,
        iterations=iterations,
        league=league,
    )


# ---------------------------------------------------------------------------
# project-season: show season projections for a position
# ---------------------------------------------------------------------------


@app.command(name="project-season")
def project_season(
    position: str = typer.Option("WR", help="Position to project (WR/RB/QB/TE/K)."),
    season: int | None = typer.Option(
        None,
        help="Season whose stats are used as features (default: most recent).",
    ),
    top: int = typer.Option(30, help="Number of players to display."),
    league: str | None = typer.Option(
        None, help="League profile key (see `nfl-predict leagues`)."
    ),
) -> None:
    """Show season projections (floor / median / ceiling) for a position."""
    from nfl_predict.season_features import load_features
    from nfl_predict.season_model import predict_season

    if season is None:
        df = load_features(league)
        season = int(df["season"].max())
        print(f"Using most recent season: {season}")

    proj = predict_season(position.upper(), as_of_season=season, league=league)
    if proj.empty:
        print("No projections available. Run `nfl-predict draft-prep` first.")
        raise typer.Exit(1)

    proj = proj.sort_values("proj_p50", ascending=False).head(top)
    proj["rank"] = range(1, len(proj) + 1)

    display = proj[
        [
            c
            for c in ("rank", "player_name", "team", "proj_p10", "proj_p50", "proj_p90")
            if c in proj.columns
        ]
    ]
    print(
        f"\n{position.upper()} projections for {season + 1} (features from {season})\n"
    )
    print(display.to_string(index=False))


# ---------------------------------------------------------------------------
# board: build and export the full draft board
# ---------------------------------------------------------------------------


@app.command()
def board(
    season: int | None = typer.Option(
        None,
        help="Features source season (default: most recent in data).",
    ),
    adp: str | None = typer.Option(
        None, help="Path to ADP CSV (columns: player_name, adp)."
    ),
    league: str | None = typer.Option(
        None, help="League profile key (see `nfl-predict leagues`)."
    ),
    league_size: int | None = typer.Option(
        None, help="Override the league's team count."
    ),
    fmt: str = typer.Option(
        "csv", help="Export format: 'csv', 'json', or 'table' (terminal preview only)."
    ),
    out: str | None = typer.Option(None, help="Output file path (auto if not set)."),
    positions: str | None = typer.Option(
        None, help="Comma-separated positions to include (default: all)."
    ),
    qb_scarcity: float | None = typer.Option(
        None,
        help="Override the QB VOR scarcity multiplier (1.0 = superflex).",
    ),
    superflex: bool = typer.Option(
        False, help="Superflex league: sets --qb-scarcity 1.0 automatically."
    ),
    exclude: str | None = typer.Option(
        None,
        help=(
            "Path to a list of players who cannot be drafted (keepers): "
            "one gsis ID or full name per line, # comments allowed. "
            "Defaults to the league profile's keeper file when it has one."
        ),
    ),
) -> None:
    """Build and export the full fantasy draft board with VOR and tiers."""
    from dataclasses import replace as dc_replace

    from nfl_predict.draft_board import (
        build_draft_board,
        export_draft_board,
        load_exclusions,
    )
    from nfl_predict.leagues import get_profile
    from nfl_predict.season_features import load_features

    if fmt not in ("csv", "json", "table"):
        print(f"Error: --fmt must be 'csv', 'json', or 'table'. Got: '{fmt}'")
        raise typer.Exit(1)

    profile = get_profile(league)
    print(f"League: {profile.name} [{profile.key}]")

    if season is None:
        df = load_features(profile.key)
        season = int(df["season"].max())
        print(f"Using most recent season as feature source: {season}")

    pos_list = [p.strip().upper() for p in positions.split(",")] if positions else None

    # Start from the league's own settings; the flags are overrides on top.
    settings = profile.to_draft_settings()
    if league_size is not None:
        settings = dc_replace(settings, league_size=league_size)
    if superflex or qb_scarcity is not None:
        scarcity = dict(settings.positional_scarcity)
        scarcity["QB"] = 1.0 if superflex else qb_scarcity
        settings = dc_replace(settings, positional_scarcity=scarcity)

    exclude_path = exclude or profile.keepers_path
    exclusions = load_exclusions(exclude_path) if exclude_path else None
    if exclusions is not None:
        print(f"Loaded {len(exclusions)} exclusions from {exclude_path}")

    draft_board = build_draft_board(
        as_of_season=season,
        positions=pos_list,
        adp_path=adp,
        settings=settings,
        exclude=exclusions,
        league=profile.key,
    )

    table_cols = [
        c
        for c in (
            "overall_rank",
            "tier",
            "player_name",
            "position",
            "pos_rank",
            "proj_p50",
            "vor",
            "adp",
        )
        if c in draft_board.columns
    ]

    if fmt == "table":
        # Terminal preview — no file export
        n = 40
        print(
            f"\nTop {n} overall (VOR) — {season + 1} draft board "
            f"[QB scarcity={settings.positional_scarcity.get('QB', 1.0)}]:\n"
        )
        print(draft_board[table_cols].head(n).to_string(index=False))
        return

    export_path = export_draft_board(
        draft_board,
        out_path=out
        or str(profile.board_path(season + 1, ".csv" if fmt == "csv" else ".json")),
        fmt=fmt,
        season=season + 1,
    )

    # Always print top 20 summary
    print(
        f"\nTop 20 overall (VOR) — {season + 1} draft board "
        f"[QB scarcity={settings.positional_scarcity.get('QB', 1.0)}]:\n"
    )
    print(draft_board[table_cols].head(20).to_string(index=False))
    print(f"\nFull board exported to: {export_path}")


# ---------------------------------------------------------------------------
# draft-start: initialise a live draft session
# ---------------------------------------------------------------------------


@app.command(name="draft-start")
def draft_start(
    season: int | None = typer.Option(
        None,
        help="Draft board season to load (default: most recent in outputs/).",
    ),
    league: str | None = typer.Option(
        None, help="League profile key (see `nfl-predict leagues`)."
    ),
    league_size: int | None = typer.Option(
        None, help="Override the league's team count."
    ),
    draft_position: int = typer.Option(1, help="Your draft slot (1-based)."),
    board_path: str | None = typer.Option(
        None, help="Path to draft board CSV (auto-detected if omitted)."
    ),
    state_path: str | None = typer.Option(
        None, help="Where to save draft state JSON (default: outputs/draft_state.json)."
    ),
) -> None:
    """Initialise a live draft session from a draft board CSV."""
    import glob as _glob
    from pathlib import Path

    from nfl_predict.draft_assistant import init_draft_state, render_board, save_state
    from nfl_predict.leagues import get_profile

    profile = get_profile(league)
    print(f"League: {profile.name} [{profile.key}]")

    # Locate board CSV — this league's, never another's.
    if board_path is None:
        if season is not None:
            board_path = str(profile.board_path(season))
        else:
            csvs = sorted(_glob.glob(f"outputs/draft_board_*_{profile.key}.csv"))
            if not csvs:
                print(
                    f"No draft board CSV found for {profile.key}. "
                    f"Run `nfl-predict board --league {profile.key}` first."
                )
                raise typer.Exit(1)
            board_path = csvs[-1]  # most recent
            print(f"Using board: {board_path}")

    if not Path(board_path).exists():
        print(f"Board file not found: {board_path}")
        raise typer.Exit(1)

    board = pd.read_csv(board_path)
    from pathlib import Path as _Path

    sp = _Path(state_path) if state_path else None
    state = init_draft_state(
        board,
        league_size=league_size,
        draft_position=draft_position,
        state_path=sp,
        league=profile.key,
    )
    save_state(state)

    print(render_board(state, n=30))
    print(f"\nDraft session started — state saved to {state.state_path}")
    print("Use `nfl-predict draft-pick <NAME>` to record each pick.")


# ---------------------------------------------------------------------------
# draft-pick: record a pick and get suggestions
# ---------------------------------------------------------------------------


@app.command(name="draft-pick")
def draft_pick(
    player: str = typer.Argument(
        help="Player name (or unique substring) being drafted."
    ),
    drafter: str = typer.Option(
        "other",
        help="Who made the pick: 'me' for your pick, or any label for opponents.",
    ),
    needs: str | None = typer.Option(
        None,
        help="Comma-separated positions to prioritise, e.g. 'RB,WR'. "
        "Auto-detected from your roster if omitted.",
    ),
    suggest: int = typer.Option(
        5, help="Number of best-available suggestions to show."
    ),
    show_board: bool = typer.Option(
        False, help="Redisplay the full board after the pick."
    ),
    state_path: str | None = typer.Option(
        None, help="Draft state JSON path (default: outputs/draft_state.json)."
    ),
) -> None:
    """Record a draft pick and show best-available suggestions."""
    from pathlib import Path as _Path

    from nfl_predict.draft_assistant import (
        analyse_roster_needs,
        load_state,
        mark_drafted,
        render_board,
        save_state,
        suggest_best_available,
    )

    sp = _Path(state_path) if state_path else None
    state = load_state(sp)

    # Mark the pick
    try:
        state = mark_drafted(state, player, drafter=drafter)
    except ValueError as e:
        print(f"Error: {e}")
        raise typer.Exit(1) from e

    last = state.picks[-1]
    marker = " ← YOUR PICK" if drafter == "me" else ""
    print(
        f"\nPick #{last.overall_pick}  R{last.round}.{last.pick_in_round}  "
        f"{last.player_name} ({last.position}, {last.team})  "
        f"VOR={last.vor:.1f}{marker}"
    )

    # Determine positional needs
    if needs:
        need_list = [n.strip().upper() for n in needs.split(",")]
    else:
        need_list = analyse_roster_needs(state)

    if need_list:
        print(f"Positional needs: {', '.join(need_list)}")

    # Best available
    suggestions = suggest_best_available(state, needs=need_list, n=suggest)
    if not suggestions.empty:
        print(f"\nTop {suggest} available:")
        cols = [
            c
            for c in ("player_name", "position", "proj_p50", "vor", "tier")
            if c in suggestions.columns
        ]
        print(suggestions[cols].to_string(index=False))

    if show_board:
        print()
        print(render_board(state, n=30, show_drafted=True))

    save_state(state)
    print(f"\n{len(state.available)} players remaining.")


# ---------------------------------------------------------------------------
# fetch-adp: pull ADP from Sleeper / FantasyPros / synthetic
# ---------------------------------------------------------------------------


@app.command(name="fetch-adp")
def fetch_adp_cmd(
    source: str = typer.Option(
        "sleeper",
        help="ADP source: 'sleeper', 'fantasypros', or 'synthetic'.",
    ),
    scoring: str = typer.Option("ppr", help="Scoring format: 'ppr', 'half', or 'std'."),
    out: str = typer.Option("data/adp_current.csv", help="Output CSV path."),
    no_fallback: bool = typer.Option(
        False, help="Do not fall back to synthetic ADP on live-fetch failure."
    ),
) -> None:
    """Fetch ADP data and save to CSV (use with `nfl-predict board --adp`)."""
    from nfl_predict.adp_fetch import fetch_adp, save_adp_csv

    df = fetch_adp(
        source=source,
        scoring=scoring,
        fallback_to_synthetic=not no_fallback,
    )
    if df.empty:
        print("No ADP data fetched.")
        raise typer.Exit(1)

    path = save_adp_csv(df, path=out)

    print(f"\nTop 10 ADP:\n{df.head(10).to_string(index=False)}")
    print(f"\nSaved {len(df)} players → {path}")


# ---------------------------------------------------------------------------
# nfl-sync: poll NFL Fantasy live draft and auto-record picks
# ---------------------------------------------------------------------------


@app.command(name="nfl-sync")
def nfl_sync_cmd(
    interval: int = typer.Option(30, help="Seconds between polls."),
    max_rounds: int = typer.Option(
        20, help="Stop polling after this many draft rounds."
    ),
    provider: str = typer.Option(
        "auto",
        help="Draft provider: 'espn' or 'auto'.",
    ),
    league: str | None = typer.Option(
        None, help="League profile key (see `nfl-predict leagues`)."
    ),
) -> None:
    """
    Poll the live draft and auto-record picks into the local state.

    ESPN (the only supported provider — NFL.com Fantasy moved to ESPN):
      ESPN_LEAGUE_ID, plus ESPN_S2 and ESPN_SWID for a private league.
      Optionally ESPN_TEAM_ID to identify your own picks as 'mine'.

    Run this in a separate terminal while the UI is open — it updates
    the same per-league draft state that the web UI reads.
    """
    from nfl_predict.draft_assistant import (
        load_state,
        mark_drafted,
        save_state,
        state_lock,
    )
    from nfl_predict.draft_sync import DraftSyncError, make_client, poll_draft
    from nfl_predict.leagues import get_profile

    profile = get_profile(league)
    print(f"League: {profile.name} [{profile.key}]")

    try:
        client = make_client(provider, league=profile.key)
    except DraftSyncError as e:
        print(f"Error: {e}")
        raise typer.Exit(1) from e

    state_path = profile.state_path
    if not state_path.exists():
        print("No active draft session. Run `nfl-predict draft-start` first.")
        raise typer.Exit(1)

    # Picks already in the local state — don't replay them on restart.
    already_recorded = len(load_state(state_path).picks)
    if already_recorded:
        print(f"Resuming: {already_recorded} picks already recorded locally.")

    def on_pick(pick: dict) -> None:
        # Lock the read-modify-write cycle: the web UI mutates the same file.
        with state_lock(state_path):
            state = load_state(state_path)
            try:
                updated = mark_drafted(
                    state,
                    pick["player_name"],
                    drafter="me" if pick.get("is_mine") else "other",
                    player_id=pick.get("player_id") or None,
                )
                save_state(updated)
                marker = " ← MINE" if pick.get("is_mine") else ""
                print(
                    f"  #{pick['overall_pick']:>3}  {pick['player_name']:<28} "
                    f"{pick.get('position', '')}{marker}"
                )
            except ValueError as e:
                print(f"  Warning: Could not record {pick['player_name']!r}: {e}")

    poll_draft(
        client,
        on_pick=on_pick,
        interval=interval,
        max_rounds=max_rounds,
        initial_recorded=already_recorded,
    )


# ---------------------------------------------------------------------------
# espn-login: store the session cookies a private league needs
# ---------------------------------------------------------------------------


@app.command(name="espn-login")
def espn_login(
    env_path: str = typer.Option(".env", help="File to write the cookies into."),
    check: bool = typer.Option(True, help="Try each private league afterwards."),
) -> None:
    """
    Store the ESPN session cookies that private leagues need.

    Prompts for `espn_s2` and `SWID` without echoing them, writes them to
    `.env` with owner-only permissions, and prints nothing back but their
    length. The values never reach the terminal scrollback, so they cannot be
    scrolled back to, copied out of a screen share, or read from a log.

    Get them from a logged-in browser: F12 -> Application -> Cookies ->
    https://fantasy.espn.com. They expire, so refresh them near draft time.
    """
    import getpass
    import os
    import stat
    from pathlib import Path as _Path

    from nfl_predict.espn_fantasy import _normalise_swid
    from nfl_predict.leagues import PROFILES

    print("ESPN session cookies (input is hidden).")
    print("  Chrome: F12 -> Application -> Cookies -> https://fantasy.espn.com\n")

    s2 = getpass.getpass("  espn_s2 : ").strip()
    swid = _normalise_swid(getpass.getpass("  SWID    : ").strip())

    if not s2 or not swid:
        print("\nBoth cookies are required; nothing was written.")
        raise typer.Exit(code=1)

    path = _Path(env_path)
    lines = path.read_text().splitlines() if path.exists() else []
    wanted = {"ESPN_S2": s2, "ESPN_SWID": swid}

    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key in wanted:
            out.append(f"{key}={wanted[key]}")
            seen.add(key)
        else:
            out.append(line)
    out += [f"{k}={v}" for k, v in wanted.items() if k not in seen]

    path.write_text("\n".join(out) + "\n")
    # The file now holds a live session token; keep it off other accounts.
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    print(f"\n  Saved to {path} (mode 600)")
    print(f"  espn_s2 {len(s2)} chars, SWID {len(swid)} chars")

    if not check:
        return

    # The prompt only set the process environment for this run's children, so
    # apply the values here to verify them without a reload.
    os.environ["ESPN_S2"] = s2
    os.environ["ESPN_SWID"] = swid

    from nfl_predict.espn_fantasy import EspnFantasyClient, EspnFantasyError

    print()
    for profile in PROFILES.values():
        if not profile.espn_league_id:
            continue
        try:
            client = EspnFantasyClient.from_env(profile.key)
            teams = len(client.get_league().get("teams") or [])
            print(f"  OK    {profile.name}: {teams} teams")
        except EspnFantasyError as e:
            print(f"  FAIL  {profile.name}: {e}")


if __name__ == "__main__":
    app()
