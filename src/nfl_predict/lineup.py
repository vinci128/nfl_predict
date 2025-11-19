from pathlib import Path
import re

import pandas as pd
import typer

app = typer.Typer()
BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "outputs"


def _find_latest_prediction_files() -> list[Path]:
    """
    Cerca in OUT_DIR tutti i predictions_*.csv e per ogni posizione
    sceglie il file con season/week più recenti.

    Nome atteso: predictions_<pos>_<season>_week<week>.csv
    es: predictions_wr_2025_week12.csv
    """
    pred_files = list(OUT_DIR.glob("predictions_*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"Nessun file di predictions_*.csv trovato in {OUT_DIR}")

    pattern = re.compile(
        r"predictions_(?P<position>[A-Za-z]+)_(?P<season>\d{4})_week(?P<week>\d+)\.csv"
    )

    records = []
    for path in pred_files:
        m = pattern.fullmatch(path.name)
        if not m:
            # ignora file che non rispettano il pattern
            continue
        records.append(
            {
                "position": m.group("position").upper(),
                "season": int(m.group("season")),
                "week": int(m.group("week")),
                "path": path,
            }
        )

    if not records:
        raise RuntimeError(
            f"Trovati file in {OUT_DIR}, ma nessuno con nome del tipo "
            "'predictions_<pos>_<season>_week<week>.csv'"
        )

    df_files = pd.DataFrame(records)
    # ordina per position, season, week
    df_files = df_files.sort_values(["position", "season", "week"])
    # prendi l'ultima riga per ogni posizione (season/week max)
    latest = df_files.groupby("position", as_index=False).tail(1)

    return list(latest["path"])


@app.command()
def suggest(
    roster_path: str = typer.Argument("my_roster.csv", help="CSV con il tuo roster"),
    top_n: int = typer.Option(20, help="Quanti giocatori mostrare"),
):
    """
    Legge tutti i predictions_*.csv più recenti per ogni posizione in outputs/
    e suggerisce i tuoi giocatori ordinati per expected PPR.
    """
    # --- trova i file di predictions più aggiornati ---
    latest_files = _find_latest_prediction_files()
    typer.echo("Userò i seguenti file di predictions più recenti:")
    for p in latest_files:
        typer.echo(f"  - {p.name}")

    # --- carica e concatena tutte le predictions ---
    preds_list = []
    for path in latest_files:
        df = pd.read_csv(path)
        # normalizza posizione
        if "position" in df.columns:
            df["position"] = df["position"].str.upper()
        preds_list.append(df)

    preds = pd.concat(preds_list, ignore_index=True)

    # --- carica roster ---
    roster = pd.read_csv(roster_path)

    if "player_name" not in roster.columns or "position" not in roster.columns:
        raise ValueError(
            "Il roster deve contenere almeno le colonne: 'player_name' e 'position'. "
            "Opzionale ma consigliato: 'team'."
        )

    roster["position"] = roster["position"].str.upper()

    # --- chiavi di join dinamiche ---
    join_keys = ["player_name", "position"]
    if "team" in preds.columns and "team" in roster.columns:
        join_keys.append("team")

    merged = preds.merge(
        roster,
        on=join_keys,
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            "Nessuna corrispondenza tra predictions e roster. "
            "Controlla che i nomi dei giocatori (e i team, se usati) combacino."
        )

    merged = merged.sort_values("expected_ppr_points", ascending=False)

    print("\nI TUOI giocatori (ordinati per expected PPR):\n")
    cols = [c for c in ["player_name", "team", "position", "expected_ppr_points"] if c in merged.columns]
    print(merged[cols].head(top_n).to_string(index=False))


if __name__ == "__main__":
    app()
