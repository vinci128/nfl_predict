import asyncio
import sys
import typer


def _run(coro):
    """Run an async coroutine. Uses ProactorEventLoop on Windows (required by Playwright)."""
    if sys.platform == "win32":
        # ProactorEventLoop is required on Windows to launch subprocesses (the browser driver).
        # The 'Event loop is closed' message printed at exit is harmless cleanup noise.
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(coro)
from typing import Optional
from nfl_predict import (
    fetch_nfl_data,
    features,
    train_model,
    predict_week,
)

app = typer.Typer(help="NFL Fantasy prediction and agent CLI.")


@app.command()
def update_all(
    fetch: bool = typer.Option(True, help="Download updated data?"),
    train: bool = typer.Option(True, help="Retrain models after data update?"),
    position: Optional[str] = typer.Option(None, help="Position for predictions. Default = all main positions."),
):
    """Fetch data + build features (+ train) + predict in one shot."""
    print(">> Fetching raw NFL data...")
    if fetch:
        fetch_nfl_data.main()
        print(">> Building features...")
        features.build_player_week_features()
    if train:
        print(">> Training models...")
        train_model.main()
    print(">> Making predictions...")

    main_positions = ["WR", "RB", "QB", "TE", "K"]
    if position is None:
        for pos in main_positions:
            print(f">> Predictions for position: {pos}")
            predict_week.run_predictions(position=pos)
    else:
        predict_week.run_predictions(position=position)


@app.command()
def agent(
    task: Optional[str] = typer.Option(
        None,
        "--task", "-t",
        help=(
            "What the agent should do. "
            "Defaults to full weekly management (lineup + waivers + trades)."
        ),
    ),
    auto_confirm: bool = typer.Option(
        False,
        "--auto-confirm",
        help="Skip confirmation prompts for destructive actions. USE WITH CAUTION.",
    ),
    draft: bool = typer.Option(
        False,
        "--draft",
        help="Run in draft assistant mode — monitors the draft and makes picks.",
    ),
    max_turns: int = typer.Option(
        30,
        "--max-turns",
        help="Maximum agent turns before stopping.",
    ),
):
    """
    Run the Claude-powered NFL Fantasy agent.

    The agent logs into NFL.com Fantasy, analyses your team using ML predictions,
    and manages your roster (lineups, waivers, trades, drafts).

    Requires environment variables: NFL_EMAIL, NFL_PASSWORD, NFL_LEAGUE_ID,
    NFL_TEAM_ID, ANTHROPIC_API_KEY (set in .env file).

    NOTE: Automating NFL.com may violate their Terms of Service.
    Use on your own account at your own discretion.
    """
    from nfl_predict.nfl_agent import run_agent, run_draft_agent

    if draft:
        _run(run_draft_agent(auto_confirm=auto_confirm))
    else:
        _run(run_agent(task=task, auto_confirm=auto_confirm, max_turns=max_turns))


if __name__ == "__main__":
    app()
