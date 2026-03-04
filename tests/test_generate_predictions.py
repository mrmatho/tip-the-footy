"""Tests for scripts/generate_predictions.py – inference pipeline."""

import json
import os
import pickle
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_predictions import (
    BASE_URL,
    TippingModel,
    TippingPrediction,
    load_model,
    predict_round,
    save_predictions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(margin: float = 15.0, win_prob: float = 0.7) -> TippingModel:
    """Return a TippingModel backed by mock classifier / regressor."""
    clf = MagicMock()
    clf.predict_proba.return_value = np.array([[1 - win_prob, win_prob]])
    reg = MagicMock()
    reg.predict.return_value = np.array([margin])
    return TippingModel(clf, reg)


def _make_response(json_data: dict, status_code: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = Exception(
            f"HTTP Error {status_code}"
        )
    return mock_resp


SAMPLE_HISTORICAL = pd.DataFrame([{
    "id": 1, "round": 1, "season": 2023, "year": 2023,
    "venue": "MCG", "hteam": "Collingwood", "ateam": "Carlton",
    "hscore": 110, "ascore": 90,
    "date": "2023-03-16 19:30:00", "complete": 100,
}])

SAMPLE_UPCOMING = [{
    "id": 10, "round": 2, "year": 2023,
    "venue": "MCG", "hteam": "Collingwood", "ateam": "Carlton",
    "date": "2023-03-24 19:30:00", "complete": 0,
}]


# ---------------------------------------------------------------------------
# TippingPrediction
# ---------------------------------------------------------------------------


class TestTippingPrediction:
    def test_stores_all_fields(self):
        pred = TippingPrediction(
            home_team="A", away_team="B",
            predicted_winner="A", predicted_margin=10.0, win_probability=0.7,
        )
        assert pred.home_team == "A"
        assert pred.away_team == "B"
        assert pred.predicted_winner == "A"
        assert pred.predicted_margin == 10.0
        assert pred.win_probability == 0.7


# ---------------------------------------------------------------------------
# TippingModel.predict
# ---------------------------------------------------------------------------


class TestTippingModelPredict:
    def test_home_win_when_positive_margin(self):
        model = _make_mock_model(margin=15.0, win_prob=0.7)
        X = np.zeros((1, 12))
        pred = model.predict(X, "HomeTeam", "AwayTeam")
        assert pred.predicted_winner == "HomeTeam"
        assert pred.predicted_margin == pytest.approx(15.0)
        assert pred.win_probability == pytest.approx(0.7)

    def test_away_win_when_negative_margin(self):
        model = _make_mock_model(margin=-10.0, win_prob=0.4)
        X = np.zeros((1, 12))
        pred = model.predict(X, "HomeTeam", "AwayTeam")
        assert pred.predicted_winner == "AwayTeam"
        assert pred.predicted_margin == pytest.approx(10.0)
        # win_probability flips: 1 − 0.4 = 0.6
        assert pred.win_probability == pytest.approx(0.6)

    def test_returns_tipping_prediction_instance(self):
        model = _make_mock_model()
        pred = model.predict(np.zeros((1, 12)), "A", "B")
        assert isinstance(pred, TippingPrediction)

    def test_zero_margin_tips_home_team(self):
        model = _make_mock_model(margin=0.0, win_prob=0.5)
        pred = model.predict(np.zeros((1, 12)), "Home", "Away")
        assert pred.predicted_winner == "Home"


# ---------------------------------------------------------------------------
# Picklable stubs (MagicMock cannot be pickled)
# ---------------------------------------------------------------------------


class _ClfStub:
    """Minimal picklable classifier stub."""
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]] * len(X))


class _RegStub:
    """Minimal picklable regressor stub."""
    def predict(self, X):
        return np.array([5.0] * len(X))


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_returns_tipping_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "classifier.pkl"), "wb") as f:
                pickle.dump(_ClfStub(), f)
            with open(os.path.join(tmpdir, "regressor.pkl"), "wb") as f:
                pickle.dump(_RegStub(), f)
            model = load_model(tmpdir)
        assert isinstance(model, TippingModel)

    def test_loads_col_means_when_present(self):
        import pandas as pd
        col_means = pd.Series({"a": 0.5, "b": 0.3})
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "classifier.pkl"), "wb") as f:
                pickle.dump(_ClfStub(), f)
            with open(os.path.join(tmpdir, "regressor.pkl"), "wb") as f:
                pickle.dump(_RegStub(), f)
            with open(os.path.join(tmpdir, "col_means.pkl"), "wb") as f:
                pickle.dump(col_means, f)
            model = load_model(tmpdir)
        assert model.col_means is not None
        assert model.col_means["a"] == pytest.approx(0.5)

    def test_loaded_model_uses_correct_classifier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "classifier.pkl"), "wb") as f:
                pickle.dump(_ClfStub(), f)
            with open(os.path.join(tmpdir, "regressor.pkl"), "wb") as f:
                pickle.dump(_RegStub(), f)
            model = load_model(tmpdir)
        pred = model.predict(np.zeros((1, 12)), "A", "B")
        assert pred.win_probability == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# predict_round
# ---------------------------------------------------------------------------


class TestPredictRound:
    def test_returns_dataframe_with_predictions(self):
        model = _make_mock_model(margin=15.0)
        with patch("scripts.generate_predictions.requests.get") as mock_get:
            mock_get.return_value = _make_response({"games": SAMPLE_UPCOMING})
            df = predict_round(2, 2023, model, SAMPLE_HISTORICAL)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "predicted_winner" in df.columns
        assert "predicted_margin" in df.columns
        assert "win_probability" in df.columns

    def test_calls_correct_api_params(self):
        model = _make_mock_model()
        with patch("scripts.generate_predictions.requests.get") as mock_get:
            mock_get.return_value = _make_response({"games": []})
            predict_round(5, 2024, model, SAMPLE_HISTORICAL)
        mock_get.assert_called_once_with(
            BASE_URL,
            params={"q": "games", "year": 2024, "round": 5},
            timeout=30,
        )

    def test_empty_round_returns_empty_dataframe(self):
        model = _make_mock_model()
        with patch("scripts.generate_predictions.requests.get") as mock_get:
            mock_get.return_value = _make_response({"games": []})
            df = predict_round(1, 2023, model, SAMPLE_HISTORICAL)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_home_team_correctly_set(self):
        model = _make_mock_model(margin=15.0)
        with patch("scripts.generate_predictions.requests.get") as mock_get:
            mock_get.return_value = _make_response({"games": SAMPLE_UPCOMING})
            df = predict_round(2, 2023, model, SAMPLE_HISTORICAL)
        assert df.iloc[0]["home_team"] == "Collingwood"
        assert df.iloc[0]["away_team"] == "Carlton"

    def test_raises_on_http_error(self):
        model = _make_mock_model()
        with patch("scripts.generate_predictions.requests.get") as mock_get:
            mock_get.return_value = _make_response({}, status_code=500)
            with pytest.raises(Exception):
                predict_round(1, 2023, model, SAMPLE_HISTORICAL)


# ---------------------------------------------------------------------------
# save_predictions
# ---------------------------------------------------------------------------


class TestSavePredictions:
    _SAMPLE_DF = pd.DataFrame([{
        "round": 1, "home_team": "A", "away_team": "B",
        "predicted_winner": "A", "predicted_margin": 10.0,
        "win_probability": 0.7,
    }])

    def test_creates_csv_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_predictions(self._SAMPLE_DF, round_number=1, output_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "round_1.csv"))

    def test_creates_json_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_predictions(self._SAMPLE_DF, round_number=1, output_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "round_1.json"))

    def test_json_is_valid_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_predictions(self._SAMPLE_DF, round_number=1, output_dir=tmpdir)
            with open(os.path.join(tmpdir, "round_1.json")) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert data[0]["predicted_winner"] == "A"

    def test_csv_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_predictions(self._SAMPLE_DF, round_number=3, output_dir=tmpdir)
            loaded = pd.read_csv(os.path.join(tmpdir, "round_3.csv"))
            assert list(loaded["predicted_winner"]) == ["A"]

    def test_creates_output_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "new_predictions")
            assert not os.path.exists(out_dir)
            save_predictions(self._SAMPLE_DF, round_number=1, output_dir=out_dir)
            assert os.path.isdir(out_dir)
