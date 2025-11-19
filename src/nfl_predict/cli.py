import typer
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
    position: str = typer.Option("WR", help="Posizione per le predizioni.")
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
    predict_week.run_predictions(position=position)  # esempio per RB


if __name__ == "__main__":
    app()
