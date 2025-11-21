import typer
from typing import Optional
from nfl_predict import (
    fetch_nfl_data,
    features,
    train_model,
    predict_week,
)  # se li rendi moduli importabili

app = typer.Typer(help="CLI per la pipeline NFL fantasy.")


@app.command()
def update_all(
    fetch: bool = typer.Option(True, help="Scaricare dati aggiornati?"),
    train: bool = typer.Option(True, help="Riallenare i modelli dopo l'update dati?"),
    position: Optional[str] = typer.Option(None, help="Posizione per le predizioni. Default = all main positions."),
):
    """Fetch + features (+ train) in un colpo solo."""
    print(">> Fetching raw NFL data...")
    if fetch:
        fetch_nfl_data.main()  # o chiama le funzioni interne
        print(">> Building features...")
        features.build_player_week_features()
    if train:
        print(">> Training models...")
        train_model.main()
    print(">> Making predictions...")

    # If no specific position provided, run predictions for main positions
    main_positions = ["WR", "RB", "QB", "TE", "K"]
    if position is None:
        for pos in main_positions:
            print(f">> Predictions for position: {pos}")
            predict_week.run_predictions(position=pos)
    else:
        predict_week.run_predictions(position=position)


if __name__ == "__main__":
    app()
