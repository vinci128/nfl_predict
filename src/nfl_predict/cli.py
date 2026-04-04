import pandas as pd
import typer

from nfl_predict import features, fetch_nfl_data, predict_week, train_model

app = typer.Typer(help="NFL fantasy prediction pipeline CLI.")

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
) -> None:
    """Fetch data, build features, train models, and run predictions."""
    if fetch:
        print(">> Fetching raw NFL data...")
        fetch_nfl_data.main()
        print(">> Building features...")
        features.build_player_week_features()

    if train:
        print(">> Training models...")
        train_model.main()

    print(">> Making predictions...")
    main_positions = ["WR", "RB", "QB", "TE", "K"]
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
) -> None:
    """Train season-total projection models (p10/p50/p90) for drafts."""
    from nfl_predict.season_model import main as season_main

    positions = [position.upper()] if position else None
    season_main(
        positions=positions, use_registry=not no_registry, iterations=iterations
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
) -> None:
    """Show season projections (floor / median / ceiling) for a position."""
    from nfl_predict.season_features import load_features
    from nfl_predict.season_model import predict_season

    if season is None:
        df = load_features()
        season = int(df["season"].max())
        print(f"Using most recent season: {season}")

    proj = predict_season(position.upper(), as_of_season=season)
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
    league_size: int = typer.Option(12, help="Number of teams in the league."),
    fmt: str = typer.Option("csv", help="Export format: 'csv' or 'json'."),
    out: str | None = typer.Option(None, help="Output file path (auto if not set)."),
    positions: str | None = typer.Option(
        None, help="Comma-separated positions to include (default: all)."
    ),
) -> None:
    """Build and export the full fantasy draft board with VOR and tiers."""
    from nfl_predict.draft_board import (
        DraftSettings,
        build_draft_board,
        export_draft_board,
    )
    from nfl_predict.season_features import load_features

    if season is None:
        df = load_features()
        season = int(df["season"].max())
        print(f"Using most recent season as feature source: {season}")

    pos_list = [p.strip().upper() for p in positions.split(",")] if positions else None
    settings = DraftSettings(league_size=league_size)

    draft_board = build_draft_board(
        as_of_season=season,
        positions=pos_list,
        adp_path=adp,
        settings=settings,
    )

    export_path = export_draft_board(
        draft_board, out_path=out, fmt=fmt, season=season + 1
    )

    # Print top 20 to terminal
    top = draft_board.head(20)
    print(f"\nTop 20 overall (VOR) — {season + 1} draft board:\n")
    cols = [
        c
        for c in (
            "overall_rank",
            "tier",
            "player_name",
            "position",
            "pos_rank",
            "proj_p50",
            "vor",
        )
        if c in top.columns
    ]
    print(top[cols].to_string(index=False))
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
    league_size: int = typer.Option(12, help="Number of teams in the league."),
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

    # Locate board CSV
    if board_path is None:
        if season is not None:
            board_path = f"outputs/draft_board_{season}.csv"
        else:
            csvs = sorted(_glob.glob("outputs/draft_board_*.csv"))
            if not csvs:
                print("No draft board CSV found. Run `nfl-predict board` first.")
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
    llm: bool = typer.Option(
        False, help="Ask Claude for a natural-language pick recommendation."
    ),
) -> None:
    """Record a draft pick and show best-available suggestions."""
    from pathlib import Path as _Path

    from nfl_predict.draft_assistant import (
        analyse_roster_needs,
        get_llm_suggestion,
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

    # Optional LLM advice
    if llm:
        print("\nAsking Claude for a recommendation...")
        try:
            advice = get_llm_suggestion(state, needs=need_list)
            print(f"\n{advice}")
        except ImportError as e:
            print(f"LLM unavailable: {e}")
        except Exception as e:
            print(f"LLM error: {e}")

    if show_board:
        print()
        print(render_board(state, n=30, show_drafted=True))

    save_state(state)
    print(f"\n{len(state.available)} players remaining.")


if __name__ == "__main__":
    app()
