"""
run_pipeline.py – Orchestration script for the AFL Tipping Model.

Runs the full data + model pipeline end-to-end in the correct order:
  1. fetch_data    – Download historical and current-season game data
  2. build_features – Engineer features from raw game data
  3. train_model   – Train the winner classifier and margin regressor
  4. generate_predictions – Generate predictions for the upcoming round
"""

import datetime
import os
import sys

import pandas as pd
import requests

# Allow importing sibling scripts both when run directly and as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_features import build_features, save_features  # noqa: E402
from scripts.app_config import load_model_config  # noqa: E402
from scripts.fetch_data import fetch_historical, save_data  # noqa: E402
from scripts.generate_predictions import (  # noqa: E402
    load_model,
    predict_round,
    save_predictions,
)
from scripts.train_model import save_models, train  # noqa: E402

_CFG = load_model_config()
BASE_URL = _CFG["data"]["base_url"]
USER_AGENT = _CFG["data"]["user_agent"]
REQUEST_TIMEOUT = int(_CFG["data"]["request_timeout"])
DEFAULT_START_YEAR = int(_CFG["pipeline"]["historical_start_year"])
DEFAULT_END_YEAR = int(_CFG["pipeline"]["historical_end_year"])


def run_pipeline(
    data_dir: str = "data",
    models_dir: str = "models",
    predictions_dir: str = "predictions",
    start_year: int | None = None,
    end_year: int | None = None,
) -> None:
    """Run the full AFL Tipping Model pipeline end-to-end.

    Steps
    -----
    1. Fetch historical game data and save to ``data_dir/historical_games.csv``.
    2. Build features from the raw data and save to ``data_dir/features.csv``.
    3. Train the LightGBM winner classifier and margin regressor, saving
       artifacts to ``models_dir``.
    4. Fetch the upcoming round for the current season and generate predictions,
       saving results to ``predictions_dir``.

    Args:
        data_dir: Directory for raw and feature CSV files.
        models_dir: Directory for trained model artefacts.
        predictions_dir: Directory for prediction output files.
        start_year: First historical season to fetch (inclusive). If ``None``,
            uses config ``pipeline.historical_start_year``.
        end_year: Last historical season to fetch (inclusive). If ``None``,
            uses config ``pipeline.historical_end_year``.
    """
    if start_year is None:
        start_year = DEFAULT_START_YEAR
    if end_year is None:
        end_year = DEFAULT_END_YEAR

    # ── Step 1: Fetch historical data ────────────────────────────────────────
    print(f"Step 1/4 – Fetching historical AFL data ({start_year}–{end_year})…")
    historical = fetch_historical(start_year=start_year, end_year=end_year)
    historical_path = os.path.join(data_dir, "historical_games.csv")
    save_data(historical, historical_path)
    print(f"  Saved {len(historical)} games to {historical_path}")

    # ── Step 2: Build features ───────────────────────────────────────────────
    print("Step 2/4 – Building features…")
    features = build_features(historical)
    features_path = os.path.join(data_dir, "features.csv")
    save_features(features, features_path)
    print(f"  Saved {len(features)} feature rows to {features_path}")

    # ── Step 3: Train model ──────────────────────────────────────────────────
    print("Step 3/4 – Training model…")
    clf, reg, col_means = train(features)
    save_models(clf, reg, col_means, models_dir=models_dir)

    # ── Step 4: Generate predictions ─────────────────────────────────────────
    print("Step 4/4 – Generating predictions…")
    current_year = datetime.date.today().year
    resp = requests.get(
        BASE_URL,
        params={"q": "games", "year": current_year},
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
    )
    resp.raise_for_status()
    all_games = resp.json().get("games", [])

    upcoming = [g for g in all_games if g.get("complete", 100) < 100]
    if not upcoming:
        print("  No upcoming games found – skipping predictions.")
        return

    current_round = min(g["round"] for g in upcoming)
    model = load_model(models_dir)
    predictions = predict_round(current_round, current_year, model, historical)
    save_predictions(predictions, current_round, output_dir=predictions_dir)
    print(predictions.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
