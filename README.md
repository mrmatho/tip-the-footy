# tip-the-footy
Building an AFL tipping model - without really knowing what I'm doing

## Usage

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the data + model pipeline from the repository root:
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
