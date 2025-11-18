import typer
from nfl_predict import fetch_nfl_data, features, train_model  # se li rendi moduli importabili

app = typer.Typer(help="CLI per la pipeline NFL fantasy.")

@app.command()
def update_all(train: bool = typer.Option(True, help="Riallenare i modelli dopo l'update dati?")):
    """Fetch + features (+ train) in un colpo solo."""
    print(">> Fetching raw NFL data...")
    fetch_nfl_data.main()     # o chiama le funzioni interne
    print(">> Building features...")
    features.build_player_week_features()
    if train:
        print(">> Training models...")
        train_model.main()

if __name__ == "__main__":
    app()
