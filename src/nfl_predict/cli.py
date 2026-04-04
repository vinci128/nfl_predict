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


if __name__ == "__main__":
    app()
