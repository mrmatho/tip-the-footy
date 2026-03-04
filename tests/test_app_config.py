"""Tests for scripts/app_config.py."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.app_config import DEFAULT_CONFIG, load_model_config


class TestAppConfig:
    def test_loads_defaults_when_file_missing(self):
        cfg = load_model_config("C:/definitely-not-here/model_config.toml")
        assert cfg["elo"]["k_factor"] == DEFAULT_CONFIG["elo"]["k_factor"]
        assert cfg["elo"]["home_advantage"] == DEFAULT_CONFIG["elo"]["home_advantage"]

    def test_merges_override_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_config.toml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("[elo]\nk_factor = 32.0\n")
                f.write("[training.split]\ntrain_end_season = 2021\n")

            cfg = load_model_config(path)

        assert cfg["elo"]["k_factor"] == 32.0
        assert cfg["training"]["split"]["train_end_season"] == 2021
        # Unspecified keys should remain from defaults.
        assert cfg["elo"]["home_advantage"] == DEFAULT_CONFIG["elo"]["home_advantage"]

    def test_uses_tip_footy_config_env_var(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_config.toml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("[elo]\nhome_advantage = 65.0\n")

            old = os.environ.get("TIP_FOOTY_CONFIG")
            os.environ["TIP_FOOTY_CONFIG"] = path
            try:
                load_model_config.cache_clear()
                cfg = load_model_config()
            finally:
                if old is None:
                    os.environ.pop("TIP_FOOTY_CONFIG", None)
                else:
                    os.environ["TIP_FOOTY_CONFIG"] = old
                load_model_config.cache_clear()

        assert cfg["elo"]["home_advantage"] == 65.0
