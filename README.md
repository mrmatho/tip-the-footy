# tip-the-footy
Building an AFL tipping model - without really knowing what I'm doing

## Usage

1. Install dependencies:
   ```bash
   # install uv first if needed: https://docs.astral.sh/uv/getting-started/installation/
   uv sync
   ```

2. Run the full pipeline with the orchestration script:
   ```bash
   python scripts/run_pipeline.py
   ```

   Or run each step individually:
   ```bash
   python scripts/fetch_data.py
   python scripts/build_features.py
   python scripts/train_model.py
   python scripts/generate_predictions.py
   ```

Outputs are written to:
- `data/` for fetched data and engineered features
- `models/` for trained model artifacts
- `predictions/` for round predictions (CSV + JSON)

## Configuration

Runtime settings live in `config/model_config.toml`.

### Data/API
- `[data].base_url` – Squiggle API base URL.
- `[data].request_timeout` – HTTP timeout (seconds) for API calls.
- `[data].user_agent` – User-Agent header sent to Squiggle.

### Pipeline defaults
- `[pipeline].historical_start_year` – default start year for historical fetch.
- `[pipeline].historical_end_year` – default end year for historical fetch.

These are used when you run `python scripts/run_pipeline.py` without explicit
`start_year`/`end_year` arguments in code.

### Feature engineering
- `[features].rolling_window` – long rolling form window size.
- `[features].rolling_window_short` – short rolling form window size.

### Elo system
- `[elo].start_rating` – baseline Elo for new teams.
- `[elo].k_factor` – update magnitude after each match.
- `[elo].home_advantage` – Elo points added to home team in expected score.
- `[elo].season_reversion` – fraction of each team’s Elo pulled back toward
   `start_rating` at each season boundary.

### Training
- `[training.split].train_end_season` – seasons `<=` this value are training set.
- `[training.split].val_season` – validation season.
- `[training.split].test_season` – held-out test season.

LightGBM hyperparameters are configurable under:
- `[training.classifier]` (winner model)
- `[training.regressor]` (margin model)

Including:
`n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `subsample`,
`colsample_bytree`, `min_child_samples`, `random_state`, and `verbose`.

### Prediction output
- `[prediction].margin_round_decimals` – rounding precision for predicted margin.
- `[prediction].probability_round_decimals` – rounding precision for win probability.

After changing config values, rerun the pipeline to regenerate features/models
with the new settings.

### Recommended tuning workflow
1. **Tune Elo first**
   - Start with `[elo].k_factor` (e.g. 16, 20, 24, 28) and
     `[elo].home_advantage` (e.g. 30, 50, 70).
   - Keep `[elo].season_reversion` small (e.g. 0.05–0.20).
2. **Tune feature windows second**
   - Try nearby values for `[features].rolling_window` and
     `[features].rolling_window_short`.
   - Keep short window meaningfully smaller than long window.
3. **Tune model complexity third**
   - Adjust `[training.classifier]` and `[training.regressor]` settings such as
     `num_leaves`, `max_depth`, `min_child_samples`, and `learning_rate`.
4. **Only then tune split strategy**
   - Change `[training.split]` if you want a different validation/test period.

Suggested process: change one group at a time, run the full pipeline, and compare
metrics before making the next change.
