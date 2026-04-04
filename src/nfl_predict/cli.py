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


if __name__ == "__main__":
    app()
