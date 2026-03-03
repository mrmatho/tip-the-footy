# AFL Tipping Model Plan

## Overview

This document outlines the plan to build a machine learning model that predicts AFL match results, including both the predicted winner and the winning margin.

---

## 1. Data Gathering

### 1.1 Historical Results

Historical AFL data provides the foundation for training the model. The primary data source is the [fitzRoy R package](https://github.com/jimmyday12/fitzRoy) and its Python equivalent, or the [AFL Tables](https://afltables.com) website. A suitable Python library for scraping is `squiggle_client` or `fitzRoy` (via `rpy2`), or direct API calls.

**Recommended source:** [Squiggle API](https://api.squiggle.com.au/) – free, well-maintained REST API for AFL data.

**Data to collect per match:**
| Field | Description |
|-------|-------------|
| `match_id` | Unique match identifier |
| `round` | Round number |
| `season` | Year |
| `venue` | Ground name |
| `home_team` | Home team name |
| `away_team` | Away team name |
| `home_score` | Final score – home team |
| `away_score` | Final score – away team |
| `margin` | `home_score - away_score` |
| `winner` | Team that won |
| `date` | Match date/time |

**Script outline – fetch historical data:**

```python
import requests
import pandas as pd

BASE_URL = "https://api.squiggle.com.au/"

def fetch_games(season: int) -> pd.DataFrame:
    """Fetch all games for a given AFL season from the Squiggle API."""
    params = {"q": "games", "year": season}
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    games = response.json().get("games", [])
    return pd.DataFrame(games)

def fetch_historical(start_year: int = 2010, end_year: int = 2024) -> pd.DataFrame:
    frames = [fetch_games(year) for year in range(start_year, end_year + 1)]
    return pd.concat(frames, ignore_index=True)
```

Historical data should be saved locally (e.g., `data/historical_games.csv`) to avoid repeated API calls.

### 1.2 Current Season Results

Current-season data is fetched from the same API but filtered to the current year. A scheduled job (e.g., a cron job or GitHub Actions workflow) should run weekly during the season to keep the dataset up to date.

```python
import datetime

def fetch_current_season() -> pd.DataFrame:
    current_year = datetime.date.today().year
    return fetch_games(current_year)
```

### 1.3 Feature Engineering

Raw scores are not available before a match is played. Instead, features are derived from *prior* match history:

| Feature | Description |
|---------|-------------|
| `home_win_rate_last_5` | Home team win rate over last 5 games |
| `away_win_rate_last_5` | Away team win rate over last 5 games |
| `home_avg_score_last_5` | Home team average score over last 5 games |
| `away_avg_score_last_5` | Away team average score over last 5 games |
| `home_avg_margin_last_5` | Home team average margin over last 5 games |
| `away_avg_margin_last_5` | Away team average margin over last 5 games |
| `head_to_head_win_rate` | Home team win rate vs. away team (all time) |
| `venue_home_win_rate` | Home team win rate at this venue |
| `days_since_last_game_home` | Rest days for home team |
| `days_since_last_game_away` | Rest days for away team |
| `home_ladder_position` | Current ladder position – home team |
| `away_ladder_position` | Current ladder position – away team |

---

## 2. Model Design

### 2.1 Targets

The model has two prediction targets:

1. **Winner** – binary classification (`home_win`: 1 if home team wins, 0 otherwise).
2. **Winning margin** – regression (`margin`: `home_score - away_score`, negative means away win).

### 2.2 Model Architecture

Two separate models are trained, one for each target:

| Model | Algorithm | Library |
|-------|-----------|---------|
| **Winner classifier** | Gradient Boosted Trees (XGBoost / LightGBM) | `xgboost` / `lightgbm` |
| **Margin regressor** | Gradient Boosted Trees (XGBoost / LightGBM) | `xgboost` / `lightgbm` |

Both models take the same feature set as input.

A single pipeline wrapper can combine both models:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class TippingPrediction:
    home_team: str
    away_team: str
    predicted_winner: str
    predicted_margin: float   # positive = home win by this margin
    win_probability: float    # probability that home team wins

class TippingModel:
    def __init__(self, classifier, regressor):
        self.classifier = classifier
        self.regressor = regressor

    def predict(self, features: np.ndarray, home_team: str, away_team: str) -> TippingPrediction:
        win_prob = self.classifier.predict_proba(features)[0][1]
        margin = self.regressor.predict(features)[0]
        winner = home_team if margin >= 0 else away_team
        return TippingPrediction(
            home_team=home_team,
            away_team=away_team,
            predicted_winner=winner,
            predicted_margin=abs(margin),
            win_probability=win_prob if margin >= 0 else 1 - win_prob,
        )
```

---

## 3. Training and Testing Setup

### 3.1 Data Splitting Strategy

AFL data is **time-series** in nature, so a simple random split would cause data leakage. Instead, use a **temporal split**:

- **Training set:** Seasons 2010–2022
- **Validation set:** Season 2023
- **Test set:** Season 2024

This reflects the real-world use case: predict future matches based on past data.

```
Timeline:
|---- Training ----|-- Validation --|-- Test --|-- Live --|
   2010 – 2022         2023             2024       2025+
```

For hyperparameter tuning, use **time-series cross-validation** (walk-forward validation):

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    # train and evaluate model
```

### 3.2 Evaluation Metrics

| Task | Primary Metric | Secondary Metric |
|------|---------------|-----------------|
| Classification (winner) | Accuracy | Log-loss, AUC-ROC |
| Regression (margin) | MAE (Mean Absolute Error) | RMSE |

Baseline comparison: a simple "always tip the home team" or "tip higher ladder position" rule should be beaten before the model is considered useful.

### 3.3 Training Script Outline

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error

def train(data: pd.DataFrame):
    # Temporal split
    train = data[data["season"] <= 2022]
    val   = data[data["season"] == 2023]
    test  = data[data["season"] == 2024]

    feature_cols = [...]   # list of engineered features
    target_clf   = "home_win"
    target_reg   = "margin"

    X_train, y_clf_train = train[feature_cols], train[target_clf]
    X_val,   y_clf_val   = val[feature_cols],   val[target_clf]
    X_test,  y_clf_test  = test[feature_cols],  test[target_clf]

    # Classification model
    clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, use_label_encoder=False)
    clf.fit(X_train, y_clf_train, eval_set=[(X_val, y_clf_val)], early_stopping_rounds=20)

    # Regression model
    reg = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05)
    reg.fit(X_train, train[target_reg], eval_set=[(X_val, val[target_reg])], early_stopping_rounds=20)

    # Evaluate on test set
    print("Test accuracy:", accuracy_score(y_clf_test, clf.predict(X_test)))
    print("Test MAE (margin):", mean_absolute_error(test[target_reg], reg.predict(X_test)))

    return clf, reg
```

---

## 4. Receiving Predictions (Inference)

### 4.1 Weekly Prediction Workflow

Before each round, the following steps run automatically:

1. **Fetch upcoming fixtures** from the Squiggle API.
2. **Build features** for each upcoming match using the latest historical data.
3. **Run the model** to produce a predicted winner and margin for every match.
4. **Output predictions** (CSV, JSON, or rendered HTML table).

```python
def predict_round(round_number: int, season: int, model: TippingModel) -> pd.DataFrame:
    """Generate predictions for all matches in a given round."""
    params = {"q": "games", "year": season, "round": round_number}
    games = requests.get(BASE_URL, params=params).json().get("games", [])
    results = []
    for game in games:
        features = build_features(game)          # engineer features for this match
        prediction = model.predict(
            features,
            home_team=game["hteam"],
            away_team=game["ateam"],
        )
        results.append({
            "round": round_number,
            "home_team": prediction.home_team,
            "away_team": prediction.away_team,
            "predicted_winner": prediction.predicted_winner,
            "predicted_margin": round(prediction.predicted_margin, 1),
            "win_probability": round(prediction.win_probability, 3),
        })
    return pd.DataFrame(results)
```

### 4.2 Output Formats

| Format | Use case |
|--------|----------|
| `predictions/round_{N}.csv` | Stored artifact for tracking accuracy over time |
| `predictions/round_{N}.json` | Machine-readable for downstream consumption |
| Console / README table | Human-readable summary |

### 4.3 Automated Scheduling (GitHub Actions)

A GitHub Actions workflow can automate the weekly prediction run:

```yaml
# .github/workflows/weekly_predictions.yml
name: Weekly AFL Predictions

on:
  schedule:
    - cron: "0 20 * * 3"   # Every Wednesday at 8 PM UTC (Thursday morning AEST)
  workflow_dispatch:        # Allow manual trigger

jobs:
  predict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: python scripts/generate_predictions.py
      - uses: actions/upload-artifact@v4
        with:
          name: predictions
          path: predictions/
```

---

## 5. Project Structure

```
tip-the-footy/
├── data/
│   ├── historical_games.csv      # Cached historical match data
│   └── features.csv              # Engineered feature dataset
├── models/
│   ├── classifier.pkl            # Saved winner classification model
│   └── regressor.pkl             # Saved margin regression model
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── scripts/
│   ├── fetch_data.py             # Download and cache AFL data
│   ├── build_features.py         # Feature engineering pipeline
│   ├── train_model.py            # Train and save models
│   └── generate_predictions.py   # Run inference for upcoming round
├── predictions/
│   └── round_<N>.csv             # Per-round prediction outputs
├── requirements.txt
├── AFL_TIPPING_MODEL_PLAN.md     # This file
└── README.md
```

---

## 6. Dependencies

```
# requirements.txt
requests>=2.31
pandas>=2.1
numpy>=1.26
scikit-learn>=1.4
xgboost>=2.0
lightgbm>=4.3
matplotlib>=3.8
seaborn>=0.13
jupyter>=1.0
```

---

## 7. Milestones

| # | Milestone | Description |
|---|-----------|-------------|
| 1 | **Data pipeline** | Fetch, clean, and cache historical data from Squiggle API |
| 2 | **Feature engineering** | Build rolling features and save to `data/features.csv` |
| 3 | **Baseline model** | Train a simple logistic regression as the accuracy baseline |
| 4 | **Full model** | Train XGBoost classifier + regressor; evaluate on 2024 holdout |
| 5 | **Inference pipeline** | `generate_predictions.py` script produces round predictions |
| 6 | **Automation** | GitHub Actions workflow runs predictions every Wednesday |
| 7 | **Tracking** | Compare predictions against actual results each week |
