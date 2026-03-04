"""
generate_predictions.py – Inference pipeline for the AFL Tipping Model.

Fetches upcoming fixtures from the Squiggle API, engineers features from
historical data, runs the trained models, and outputs per-round predictions
to CSV and JSON files.
"""

import datetime
import json
import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

USER_AGENT="AFL Tipping Model (geoffmatheson@gmail.com)"

# Allow importing sibling scripts both when run directly and as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_features import FEATURE_COLS, build_features, build_game_features  # noqa: E402

BASE_URL = "https://api.squiggle.com.au/"


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class TippingPrediction:
    """Model output for a single match."""

    home_team: str
    away_team: str
    predicted_winner: str
    predicted_margin: float   # absolute margin
    win_probability: float    # probability that the *predicted_winner* wins


# ── Model wrapper ─────────────────────────────────────────────────────────────


class TippingModel:
    """Wraps a winner classifier and a margin regressor."""

    def __init__(self, classifier, regressor, col_means=None):
        self.classifier = classifier
        self.regressor = regressor
        self.col_means = col_means  # per-feature training means for NaN imputation

    def predict(
        self,
        features: np.ndarray,
        home_team: str,
        away_team: str,
    ) -> TippingPrediction:
        """Predict the winner and margin for a single match.

        Args:
            features: 2-D array of shape ``(1, n_features)``.
            home_team: Home team name.
            away_team: Away team name.

        Returns:
            :class:`TippingPrediction` with winner, margin, and probability.
        """
        win_prob = float(self.classifier.predict_proba(features)[0][1])
        margin = float(self.regressor.predict(features)[0])
        winner = home_team if margin >= 0 else away_team
        return TippingPrediction(
            home_team=home_team,
            away_team=away_team,
            predicted_winner=winner,
            predicted_margin=abs(margin),
            win_probability=win_prob if margin >= 0 else 1 - win_prob,
        )


# ── I/O helpers ───────────────────────────────────────────────────────────────


def load_model(models_dir: str = "models") -> TippingModel:
    """Load saved classifier, regressor, and imputation means from disk.

    Args:
        models_dir: Directory containing ``classifier.pkl``,
                    ``regressor.pkl``, and ``col_means.pkl``.

    Returns:
        A :class:`TippingModel` wrapping the loaded models.

    Raises:
        RuntimeError: If required model artefacts are missing or cannot be
            unpickled, with guidance on how to resolve the issue.
    """
    clf_path = os.path.join(models_dir, "classifier.pkl")
    reg_path = os.path.join(models_dir, "regressor.pkl")
    means_path = os.path.join(models_dir, "col_means.pkl")

    missing = [p for p in (clf_path, reg_path) if not os.path.exists(p)]
    if missing:
        searched_dir = os.path.abspath(models_dir)
        missing_names = ", ".join(os.path.basename(p) for p in missing)
        raise RuntimeError(
            f"Required model artefacts are missing: {missing_names}. "
            f"Searched directory: '{searched_dir}'. "
            "Run the training pipeline (fetch_data.py → build_features.py → "
            "train_model.py) to generate these files."
        )

    try:
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
        with open(reg_path, "rb") as f:
            reg = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, OSError) as exc:
        raise RuntimeError(
            f"Failed to load model artefacts from '{os.path.abspath(models_dir)}'. "
            "The files may be corrupted or incompatible with this environment. "
            "Try regenerating the models by rerunning the training pipeline."
        ) from exc

    col_means = None
    if os.path.exists(means_path):
        try:
            with open(means_path, "rb") as f:
                col_means = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as exc:
            raise RuntimeError(
                f"Found 'col_means.pkl' but failed to load it from "
                f"'{os.path.abspath(models_dir)}'. "
                "The file may be corrupted. "
                "Try regenerating the model artefacts."
            ) from exc

    return TippingModel(clf, reg, col_means=col_means)


def save_predictions(
    df: pd.DataFrame,
    round_number: int,
    output_dir: str = "predictions",
) -> None:
    """Save predictions to CSV and JSON files.

    Args:
        df: Predictions DataFrame.
        round_number: Round number used in the output filenames.
        output_dir: Directory to write files to.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"round_{round_number}.csv")
    json_path = os.path.join(output_dir, f"round_{round_number}.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    print(f"Predictions saved: {csv_path}, {json_path}")


# ── Core inference ────────────────────────────────────────────────────────────


def predict_round(
    round_number: int,
    season: int,
    model: TippingModel,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions for all matches in a given round.

    Args:
        round_number: The round to predict.
        season: The AFL season year.
        model: A fitted :class:`TippingModel`.
        historical_df: Historical completed game data used to build features.

    Returns:
        DataFrame with columns: ``round``, ``home_team``, ``away_team``,
        ``predicted_winner``, ``predicted_margin``, ``win_probability``.

    Raises:
        requests.HTTPError: If the Squiggle API request fails.
    """
    params = {"q": "games", "year": season, "round": round_number}
    response = requests.get(BASE_URL, params=params, timeout=30, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    games = response.json().get("games", [])

    # Determine imputation values consistent with training-time strategy.
    # Prefer the means stored on the model; if unavailable, compute from
    # the provided historical dataset to avoid arbitrary fillna(0) values.
    if model.col_means is not None:
        impute_values = model.col_means
    else:
        _hist_features = build_features(historical_df)
        impute_values = _hist_features[FEATURE_COLS].mean()

    results = []
    for game in games:
        feat_df = build_game_features(game, historical_df)
        X = feat_df[FEATURE_COLS].fillna(impute_values).values
        prediction = model.predict(X, home_team=game["hteam"], away_team=game["ateam"])
        results.append({
            "round": round_number,
            "home_team": prediction.home_team,
            "away_team": prediction.away_team,
            "predicted_winner": prediction.predicted_winner,
            "predicted_margin": round(prediction.predicted_margin, 1),
            "win_probability": round(prediction.win_probability, 3),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    historical = pd.read_csv("data/historical_games.csv")
    current_year = datetime.date.today().year

    # Determine the next incomplete round.
    resp = requests.get(
        BASE_URL, params={"q": "games", "year": current_year}, timeout=30, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    all_games = resp.json().get("games", [])

    upcoming = [g for g in all_games if g.get("complete", 100) < 100]
    if not upcoming:
        print("No upcoming games found.")
    else:
        current_round = min(g["round"] for g in upcoming)
        model = load_model("models")
        predictions = predict_round(current_round, current_year, model, historical)
        save_predictions(predictions, current_round)
        print(predictions.to_string(index=False))
