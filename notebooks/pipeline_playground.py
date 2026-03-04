import marimo

__generated_with = "0.11.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import contextlib
    import datetime
    import importlib
    import io
    import os
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    from scripts.app_config import load_model_config

    return contextlib, datetime, importlib, io, mo, os, pd, Path, load_model_config


@app.cell
def _(load_model_config, mo):
    load_model_config.cache_clear()
    cfg = load_model_config()

    start_year = mo.ui.slider(
        start=1990,
        stop=2030,
        step=1,
        value=int(cfg["pipeline"]["historical_start_year"]),
        label="Historical start year",
    )
    end_year = mo.ui.slider(
        start=1990,
        stop=2030,
        step=1,
        value=int(cfg["pipeline"]["historical_end_year"]),
        label="Historical end year",
    )

    rolling_window = mo.ui.slider(
        start=3,
        stop=15,
        step=1,
        value=int(cfg["features"]["rolling_window"]),
        label="Rolling window (long)",
    )
    rolling_window_short = mo.ui.slider(
        start=2,
        stop=10,
        step=1,
        value=int(cfg["features"]["rolling_window_short"]),
        label="Rolling window (short)",
    )

    elo_start = mo.ui.slider(
        start=1200,
        stop=1800,
        step=10,
        value=int(cfg["elo"]["start_rating"]),
        label="Elo start rating",
    )
    elo_k = mo.ui.slider(
        start=8,
        stop=60,
        step=1,
        value=int(cfg["elo"]["k_factor"]),
        label="Elo K-factor",
    )
    elo_home_adv = mo.ui.slider(
        start=0,
        stop=120,
        step=1,
        value=int(cfg["elo"]["home_advantage"]),
        label="Elo home advantage",
    )
    elo_reversion = mo.ui.slider(
        start=0.0,
        stop=0.50,
        step=0.01,
        value=float(cfg["elo"]["season_reversion"]),
        label="Elo season reversion",
    )

    train_end = mo.ui.slider(
        start=2000,
        stop=2030,
        step=1,
        value=int(cfg["training"]["split"]["train_end_season"]),
        label="Training split: train end season",
    )
    val_season = mo.ui.slider(
        start=2000,
        stop=2030,
        step=1,
        value=int(cfg["training"]["split"]["val_season"]),
        label="Training split: validation season",
    )
    test_season = mo.ui.slider(
        start=2000,
        stop=2030,
        step=1,
        value=int(cfg["training"]["split"]["test_season"]),
        label="Training split: test season",
    )

    n_estimators = mo.ui.slider(
        start=100,
        stop=1500,
        step=50,
        value=int(cfg["training"]["classifier"]["n_estimators"]),
        label="LightGBM n_estimators",
    )
    learning_rate = mo.ui.slider(
        start=0.01,
        stop=0.20,
        step=0.01,
        value=float(cfg["training"]["classifier"]["learning_rate"]),
        label="LightGBM learning_rate",
    )
    max_depth = mo.ui.slider(
        start=3,
        stop=12,
        step=1,
        value=int(cfg["training"]["classifier"]["max_depth"]),
        label="LightGBM max_depth",
    )
    num_leaves = mo.ui.slider(
        start=15,
        stop=127,
        step=1,
        value=int(cfg["training"]["classifier"]["num_leaves"]),
        label="LightGBM num_leaves",
    )
    subsample = mo.ui.slider(
        start=0.5,
        stop=1.0,
        step=0.05,
        value=float(cfg["training"]["classifier"]["subsample"]),
        label="LightGBM subsample",
    )
    colsample_bytree = mo.ui.slider(
        start=0.5,
        stop=1.0,
        step=0.05,
        value=float(cfg["training"]["classifier"]["colsample_bytree"]),
        label="LightGBM colsample_bytree",
    )
    min_child_samples = mo.ui.slider(
        start=5,
        stop=100,
        step=1,
        value=int(cfg["training"]["classifier"]["min_child_samples"]),
        label="LightGBM min_child_samples",
    )

    request_timeout = mo.ui.slider(
        start=5,
        stop=120,
        step=1,
        value=int(cfg["data"]["request_timeout"]),
        label="API timeout (seconds)",
    )

    run_pipeline_btn = mo.ui.button(label="Run full pipeline")
    run_baseline_btn = mo.ui.button(label="Run Baseline preset")
    run_conservative_btn = mo.ui.button(label="Run Conservative Elo preset")
    run_aggressive_btn = mo.ui.button(label="Run Aggressive Elo preset")

    mo.vstack(
        [
            mo.md("# AFL Pipeline Playground (marimo)"),
            mo.md(
                "Adjust parameters, then click **Run full pipeline**, or use one of "
                "the Elo preset buttons below. The notebook writes a temporary config "
                "and runs the full pipeline."
            ),
            mo.md("## Data and pipeline"),
            mo.hstack([start_year, end_year, request_timeout]),
            mo.md("## Elo"),
            mo.hstack([elo_start, elo_k, elo_home_adv, elo_reversion]),
            mo.md("## Feature windows"),
            mo.hstack([rolling_window, rolling_window_short]),
            mo.md("## Training split"),
            mo.hstack([train_end, val_season, test_season]),
            mo.md("## Model hyperparameters"),
            mo.hstack(
                [
                    n_estimators,
                    learning_rate,
                    max_depth,
                    num_leaves,
                    subsample,
                    colsample_bytree,
                    min_child_samples,
                ]
            ),
            mo.md("## Run"),
            run_pipeline_btn,
            mo.md("### Elo presets"),
            mo.hstack([run_baseline_btn, run_conservative_btn, run_aggressive_btn]),
        ]
    )

    return (
        colsample_bytree,
        elo_home_adv,
        elo_k,
        elo_reversion,
        elo_start,
        end_year,
        learning_rate,
        max_depth,
        min_child_samples,
        n_estimators,
        num_leaves,
        request_timeout,
        rolling_window,
        rolling_window_short,
        run_aggressive_btn,
        run_baseline_btn,
        run_conservative_btn,
        run_pipeline_btn,
        start_year,
        subsample,
        test_season,
        train_end,
        val_season,
    )


@app.cell
def _(
    Path,
    colsample_bytree,
    contextlib,
    datetime,
    elo_home_adv,
    elo_k,
    elo_reversion,
    elo_start,
    end_year,
    importlib,
    io,
    learning_rate,
    max_depth,
    min_child_samples,
    mo,
    n_estimators,
    num_leaves,
    os,
    pd,
    request_timeout,
    rolling_window,
    rolling_window_short,
    run_aggressive_btn,
    run_baseline_btn,
    run_conservative_btn,
    run_pipeline_btn,
    start_year,
    subsample,
    test_season,
    train_end,
    val_season,
):
    presets = {
        "baseline": {
            "elo_start": 1500.0,
            "elo_k": 20.0,
            "elo_home_adv": 50.0,
            "elo_reversion": 0.10,
        },
        "conservative": {
            "elo_start": 1500.0,
            "elo_k": 16.0,
            "elo_home_adv": 40.0,
            "elo_reversion": 0.15,
        },
        "aggressive": {
            "elo_start": 1500.0,
            "elo_k": 32.0,
            "elo_home_adv": 70.0,
            "elo_reversion": 0.05,
        },
    }

    run_mode = "manual"
    selected_preset = None
    if run_baseline_btn.value:
        run_mode = "preset"
        selected_preset = "baseline"
    elif run_conservative_btn.value:
        run_mode = "preset"
        selected_preset = "conservative"
    elif run_aggressive_btn.value:
        run_mode = "preset"
        selected_preset = "aggressive"
    elif not run_pipeline_btn.value:
        mo.md("Click **Run full pipeline** to execute with the selected settings.")
        return

    if run_mode == "preset":
        elo_start_value = float(presets[selected_preset]["elo_start"])
        elo_k_value = float(presets[selected_preset]["elo_k"])
        elo_home_adv_value = float(presets[selected_preset]["elo_home_adv"])
        elo_reversion_value = float(presets[selected_preset]["elo_reversion"])
    else:
        elo_start_value = float(elo_start.value)
        elo_k_value = float(elo_k.value)
        elo_home_adv_value = float(elo_home_adv.value)
        elo_reversion_value = float(elo_reversion.value)

    if int(start_year.value) > int(end_year.value):
        mo.md("Start year must be less than or equal to end year.")
        return

    if not (
        int(train_end.value) < int(val_season.value) < int(test_season.value)
    ):
        mo.md(
            "Training split must satisfy: train_end_season < val_season < test_season."
        )
        return

    if int(rolling_window_short.value) >= int(rolling_window.value):
        mo.md("Use a short rolling window smaller than the long rolling window.")
        return

    repo_root = Path.cwd()
    runtime_cfg = repo_root / "config" / "model_config.runtime.toml"

    toml_text = f"""
[data]
base_url = "https://api.squiggle.com.au/"
request_timeout = {int(request_timeout.value)}
user_agent = "AFL Tipping Model (geoffmatheson@gmail.com)"

[features]
rolling_window = {int(rolling_window.value)}
rolling_window_short = {int(rolling_window_short.value)}

[elo]
start_rating = {elo_start_value:.1f}
k_factor = {elo_k_value:.1f}
home_advantage = {elo_home_adv_value:.1f}
season_reversion = {elo_reversion_value:.2f}

[training.split]
train_end_season = {int(train_end.value)}
val_season = {int(val_season.value)}
test_season = {int(test_season.value)}

[training.classifier]
n_estimators = {int(n_estimators.value)}
learning_rate = {float(learning_rate.value):.2f}
max_depth = {int(max_depth.value)}
num_leaves = {int(num_leaves.value)}
subsample = {float(subsample.value):.2f}
colsample_bytree = {float(colsample_bytree.value):.2f}
min_child_samples = {int(min_child_samples.value)}
random_state = 42
verbose = -1

[training.regressor]
n_estimators = {int(n_estimators.value)}
learning_rate = {float(learning_rate.value):.2f}
max_depth = {int(max_depth.value)}
num_leaves = {int(num_leaves.value)}
subsample = {float(subsample.value):.2f}
colsample_bytree = {float(colsample_bytree.value):.2f}
min_child_samples = {int(min_child_samples.value)}
random_state = 42
verbose = -1

[prediction]
margin_round_decimals = 1
probability_round_decimals = 3

[pipeline]
historical_start_year = {int(start_year.value)}
historical_end_year = {int(end_year.value)}
""".strip() + "\n"

    runtime_cfg.write_text(toml_text, encoding="utf-8")

    os.environ["TIP_FOOTY_CONFIG"] = str(runtime_cfg)

    import scripts.app_config as app_config
    import scripts.build_features as build_features
    import scripts.fetch_data as fetch_data
    import scripts.generate_predictions as generate_predictions
    import scripts.run_pipeline as run_pipeline
    import scripts.train_model as train_model

    importlib.reload(app_config)
    app_config.load_model_config.cache_clear()
    importlib.reload(fetch_data)
    importlib.reload(build_features)
    importlib.reload(train_model)
    importlib.reload(generate_predictions)
    importlib.reload(run_pipeline)

    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        run_pipeline.run_pipeline(
            start_year=int(start_year.value),
            end_year=int(end_year.value),
        )

    log_text = output_buffer.getvalue()

    pred_dir = repo_root / "predictions"
    history_dir = repo_root / "notebooks"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / "pipeline_run_history.csv"
    csv_files = sorted(pred_dir.glob("round_*.csv"), key=lambda p: p.stat().st_mtime)

    run_label = (
        f"preset: {selected_preset}"
        if run_mode == "preset"
        else "manual slider settings"
    )

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if csv_files:
        latest = csv_files[-1]
        pred_df = pd.read_csv(latest)
        n_predictions = int(len(pred_df))
        avg_win_prob = float(pred_df["win_probability"].mean()) if n_predictions else float("nan")
        avg_margin = float(pred_df["predicted_margin"].mean()) if n_predictions else float("nan")

        run_record = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "mode": run_label,
                    "elo_k": elo_k_value,
                    "elo_home_advantage": elo_home_adv_value,
                    "elo_reversion": elo_reversion_value,
                    "rolling_window": int(rolling_window.value),
                    "rolling_window_short": int(rolling_window_short.value),
                    "train_end_season": int(train_end.value),
                    "val_season": int(val_season.value),
                    "test_season": int(test_season.value),
                    "n_predictions": n_predictions,
                    "avg_win_probability": round(avg_win_prob, 4),
                    "avg_predicted_margin": round(avg_margin, 3),
                    "prediction_file": latest.name,
                }
            ]
        )

        if history_path.exists():
            history = pd.read_csv(history_path)
            history = pd.concat([history, run_record], ignore_index=True)
        else:
            history = run_record
        history.to_csv(history_path, index=False)
        recent = history.tail(10).reset_index(drop=True)

        return mo.vstack(
            [
                mo.md(
                    f"### Pipeline complete\n"
                    f"Mode: **{run_label}**\n"
                    f"Using runtime config: `{runtime_cfg}`"
                ),
                mo.md("#### Run log"),
                mo.md(f"```text\n{log_text}\n```"),
                mo.md(f"#### Latest predictions: `{latest.name}`"),
                pred_df,
                mo.md("#### Recent runs comparison (latest 10)"),
                recent,
            ]
        )

    run_record = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "mode": run_label,
                "elo_k": elo_k_value,
                "elo_home_advantage": elo_home_adv_value,
                "elo_reversion": elo_reversion_value,
                "rolling_window": int(rolling_window.value),
                "rolling_window_short": int(rolling_window_short.value),
                "train_end_season": int(train_end.value),
                "val_season": int(val_season.value),
                "test_season": int(test_season.value),
                "n_predictions": 0,
                "avg_win_probability": float("nan"),
                "avg_predicted_margin": float("nan"),
                "prediction_file": "",
            }
        ]
    )

    if history_path.exists():
        history = pd.read_csv(history_path)
        history = pd.concat([history, run_record], ignore_index=True)
    else:
        history = run_record
    history.to_csv(history_path, index=False)
    recent = history.tail(10).reset_index(drop=True)

    return mo.vstack(
        [
            mo.md(
                f"### Pipeline complete\n"
                f"Mode: **{run_label}**\n"
                f"Using runtime config: `{runtime_cfg}`"
            ),
            mo.md("#### Run log"),
            mo.md(f"```text\n{log_text}\n```"),
            mo.md("No prediction CSV files found in `predictions/`."),
            mo.md("#### Recent runs comparison (latest 10)"),
            recent,
        ]
    )


if __name__ == "__main__":
    app.run()
