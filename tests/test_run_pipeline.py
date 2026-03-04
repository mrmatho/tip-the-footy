"""Tests for scripts/run_pipeline.py – orchestration module."""

import os
import sys
import tempfile
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_HISTORICAL = pd.DataFrame(
    [
        {
            "id": 1,
            "round": 1,
            "season": 2020,
            "year": 2020,
            "venue": "MCG",
            "hteam": "Collingwood",
            "ateam": "Carlton",
            "hscore": 110,
            "ascore": 90,
            "date": "2020-03-19 19:30:00",
            "complete": 100,
        }
    ]
)

_SAMPLE_FEATURES = pd.DataFrame({"season": [2020], "home_win": [1], "margin": [20.0]})

_UPCOMING_GAME = {
    "id": 99,
    "round": 5,
    "year": 2025,
    "venue": "MCG",
    "hteam": "Collingwood",
    "ateam": "Carlton",
    "complete": 0,
}

_COMPLETED_GAME = dict(_UPCOMING_GAME, complete=100)


def _make_mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Tests for the top-level run_pipeline orchestration function."""

    def _patch_all(self, tmp_dir: str, upcoming_games: list):
        """Return a dict of patch targets that covers all external I/O."""
        mock_clf = MagicMock()
        mock_clf.predict.return_value = np.array([1])
        mock_clf.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_reg = MagicMock()
        mock_reg.predict.return_value = np.array([15.0])

        mock_model = MagicMock()
        mock_model.col_means = None
        mock_model.classifier = mock_clf
        mock_model.regressor = mock_reg

        return {
            "fetch": patch(
                "scripts.run_pipeline.fetch_historical",
                return_value=_SAMPLE_HISTORICAL,
            ),
            "save_data": patch("scripts.run_pipeline.save_data"),
            "build": patch(
                "scripts.run_pipeline.build_features",
                return_value=_SAMPLE_FEATURES,
            ),
            "save_feats": patch("scripts.run_pipeline.save_features"),
            "train": patch(
                "scripts.run_pipeline.train",
                return_value=(mock_clf, mock_reg, pd.Series(dtype=float)),
            ),
            "save_models": patch("scripts.run_pipeline.save_models"),
            "requests_get": patch(
                "scripts.run_pipeline.requests.get",
                return_value=_make_mock_response({"games": upcoming_games}),
            ),
            "load_model": patch(
                "scripts.run_pipeline.load_model",
                return_value=mock_model,
            ),
            "predict_round": patch(
                "scripts.run_pipeline.predict_round",
                return_value=pd.DataFrame(
                    [
                        {
                            "round": 5,
                            "home_team": "Collingwood",
                            "away_team": "Carlton",
                            "predicted_winner": "Collingwood",
                            "predicted_margin": 15.0,
                            "win_probability": 0.7,
                        }
                    ]
                ),
            ),
            "save_preds": patch("scripts.run_pipeline.save_predictions"),
        }

    def test_calls_fetch_historical(self):
        """run_pipeline should call fetch_historical with the configured years."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patches = self._patch_all(tmpdir, [_UPCOMING_GAME])
            with (
                patches["fetch"] as mock_fetch,
                patches["save_data"],
                patches["build"],
                patches["save_feats"],
                patches["train"],
                patches["save_models"],
                patches["requests_get"],
                patches["load_model"],
                patches["predict_round"],
                patches["save_preds"],
            ):
                run_pipeline(
                    data_dir=tmpdir,
                    models_dir=tmpdir,
                    predictions_dir=tmpdir,
                    start_year=2015,
                    end_year=2023,
                )
            mock_fetch.assert_called_once_with(start_year=2015, end_year=2023)

    def test_saves_historical_data(self):
        """run_pipeline should save historical data to the configured data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patches = self._patch_all(tmpdir, [_UPCOMING_GAME])
            with (
                patches["fetch"],
                patches["save_data"] as mock_save,
                patches["build"],
                patches["save_feats"],
                patches["train"],
                patches["save_models"],
                patches["requests_get"],
                patches["load_model"],
                patches["predict_round"],
                patches["save_preds"],
            ):
                run_pipeline(data_dir=tmpdir, models_dir=tmpdir, predictions_dir=tmpdir)
            saved_path = mock_save.call_args[0][1]
            assert saved_path == os.path.join(tmpdir, "historical_games.csv")

    def test_calls_build_features(self):
        """run_pipeline should call build_features with the historical DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patches = self._patch_all(tmpdir, [_UPCOMING_GAME])
            with (
                patches["fetch"],
                patches["save_data"],
                patches["build"] as mock_build,
                patches["save_feats"],
                patches["train"],
                patches["save_models"],
                patches["requests_get"],
                patches["load_model"],
                patches["predict_round"],
                patches["save_preds"],
            ):
                run_pipeline(data_dir=tmpdir, models_dir=tmpdir, predictions_dir=tmpdir)
            mock_build.assert_called_once()
            pd.testing.assert_frame_equal(
                mock_build.call_args[0][0].reset_index(drop=True),
                _SAMPLE_HISTORICAL.reset_index(drop=True),
            )

    def test_calls_train(self):
        """run_pipeline should pass features DataFrame to train()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patches = self._patch_all(tmpdir, [_UPCOMING_GAME])
            with (
                patches["fetch"],
                patches["save_data"],
                patches["build"],
                patches["save_feats"],
                patches["train"] as mock_train,
                patches["save_models"],
                patches["requests_get"],
                patches["load_model"],
                patches["predict_round"],
                patches["save_preds"],
            ):
                run_pipeline(data_dir=tmpdir, models_dir=tmpdir, predictions_dir=tmpdir)
            mock_train.assert_called_once()
            pd.testing.assert_frame_equal(
                mock_train.call_args[0][0].reset_index(drop=True),
                _SAMPLE_FEATURES.reset_index(drop=True),
            )

    def test_saves_models(self):
        """run_pipeline should save trained models to the configured models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patches = self._patch_all(tmpdir, [_UPCOMING_GAME])
            with (
                patches["fetch"],
                patches["save_data"],
                patches["build"],
                patches["save_feats"],
                patches["train"],
                patches["save_models"] as mock_save_models,
                patches["requests_get"],
                patches["load_model"],
                patches["predict_round"],
                patches["save_preds"],
            ):
                run_pipeline(data_dir=tmpdir, models_dir=tmpdir, predictions_dir=tmpdir)
            mock_save_models.assert_called_once()
            assert mock_save_models.call_args[1]["models_dir"] == tmpdir

    def test_calls_predict_round_with_upcoming_round(self):
        """run_pipeline should call predict_round with the correct upcoming round."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patches = self._patch_all(tmpdir, [_UPCOMING_GAME])
            with (
                patches["fetch"],
                patches["save_data"],
                patches["build"],
                patches["save_feats"],
                patches["train"],
                patches["save_models"],
                patches["requests_get"],
                patches["load_model"],
                patches["predict_round"] as mock_predict,
                patches["save_preds"],
            ):
                run_pipeline(data_dir=tmpdir, models_dir=tmpdir, predictions_dir=tmpdir)
            assert mock_predict.call_args[0][0] == _UPCOMING_GAME["round"]

    def test_saves_predictions(self):
        """run_pipeline should save predictions to the configured predictions directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patches = self._patch_all(tmpdir, [_UPCOMING_GAME])
            with (
                patches["fetch"],
                patches["save_data"],
                patches["build"],
                patches["save_feats"],
                patches["train"],
                patches["save_models"],
                patches["requests_get"],
                patches["load_model"],
                patches["predict_round"],
                patches["save_preds"] as mock_save_preds,
            ):
                run_pipeline(data_dir=tmpdir, models_dir=tmpdir, predictions_dir=tmpdir)
            mock_save_preds.assert_called_once()
            assert mock_save_preds.call_args[1]["output_dir"] == tmpdir

    def test_skips_predictions_when_no_upcoming_games(self):
        """run_pipeline should skip predictions gracefully when all games are complete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            patches = self._patch_all(tmpdir, [_COMPLETED_GAME])
            with (
                patches["fetch"],
                patches["save_data"],
                patches["build"],
                patches["save_feats"],
                patches["train"],
                patches["save_models"],
                patches["requests_get"],
                patches["load_model"],
                patches["predict_round"] as mock_predict,
                patches["save_preds"] as mock_save_preds,
            ):
                run_pipeline(data_dir=tmpdir, models_dir=tmpdir, predictions_dir=tmpdir)
            mock_predict.assert_not_called()
            mock_save_preds.assert_not_called()

    def test_pipeline_runs_steps_in_order(self):
        """run_pipeline should execute all four steps sequentially."""
        call_order = []

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch(
                    "scripts.run_pipeline.fetch_historical",
                    side_effect=lambda **_: call_order.append("fetch") or _SAMPLE_HISTORICAL,
                ),
                patch("scripts.run_pipeline.save_data"),
                patch(
                    "scripts.run_pipeline.build_features",
                    side_effect=lambda _: call_order.append("build") or _SAMPLE_FEATURES,
                ),
                patch("scripts.run_pipeline.save_features"),
                patch(
                    "scripts.run_pipeline.train",
                    side_effect=lambda _: call_order.append("train")
                    or (MagicMock(), MagicMock(), pd.Series(dtype=float)),
                ),
                patch("scripts.run_pipeline.save_models"),
                patch(
                    "scripts.run_pipeline.requests.get",
                    return_value=_make_mock_response({"games": [_UPCOMING_GAME]}),
                ),
                patch(
                    "scripts.run_pipeline.load_model",
                    return_value=MagicMock(col_means=None),
                ),
                patch(
                    "scripts.run_pipeline.predict_round",
                    side_effect=lambda *a, **kw: call_order.append("predict")
                    or pd.DataFrame(
                        [
                            {
                                "round": 5,
                                "home_team": "A",
                                "away_team": "B",
                                "predicted_winner": "A",
                                "predicted_margin": 10.0,
                                "win_probability": 0.6,
                            }
                        ]
                    ),
                ),
                patch("scripts.run_pipeline.save_predictions"),
            ):
                run_pipeline(data_dir=tmpdir, models_dir=tmpdir, predictions_dir=tmpdir)

        assert call_order == ["fetch", "build", "train", "predict"]
