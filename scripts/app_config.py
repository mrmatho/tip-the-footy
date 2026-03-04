"""Application configuration loader for the AFL Tipping Model."""

from __future__ import annotations

import copy
import os
from functools import lru_cache

import tomllib


DEFAULT_CONFIG = {
    "data": {
        "base_url": "https://api.squiggle.com.au/",
        "request_timeout": 30,
        "user_agent": "AFL Tipping Model (geoffmatheson@gmail.com)",
    },
    "features": {
        "rolling_window": 5,
        "rolling_window_short": 3,
    },
    "elo": {
        "start_rating": 1500.0,
        "k_factor": 20.0,
        "home_advantage": 50.0,
        "season_reversion": 0.10,
    },
    "training": {
        "split": {
            "train_end_season": 2022,
            "val_season": 2023,
            "test_season": 2024,
        },
        "classifier": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "random_state": 42,
            "verbose": -1,
        },
        "regressor": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 5,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
            "random_state": 42,
            "verbose": -1,
        },
    },
    "prediction": {
        "margin_round_decimals": 1,
        "probability_round_decimals": 3,
    },
    "pipeline": {
        "historical_start_year": 2010,
        "historical_end_year": 2024,
    },
}


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update *base* with values from *override*."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _default_config_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "config", "model_config.toml")


@lru_cache(maxsize=1)
def load_model_config(config_path: str | None = None) -> dict:
    """Load application config from TOML and merge with defaults."""
    path = config_path or _default_config_path()
    merged = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path, "rb") as f:
            file_cfg = tomllib.load(f)
        _deep_update(merged, file_cfg)
    return merged
