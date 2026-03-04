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
