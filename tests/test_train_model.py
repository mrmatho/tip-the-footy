"""Tests for scripts/train_model.py – model training module."""

import os
import pickle
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.build_features import FEATURE_COLS, TARGET_CLF, TARGET_REG
from scripts.train_model import save_models, train

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feature_data(
    n_train: int = 50,
    n_val: int = 10,
    n_test: int = 10,
) -> pd.DataFrame:
    """Generate synthetic feature data with the required column structure."""
    rng = np.random.default_rng(42)
    rows = n_train + n_val + n_test

    data = {col: rng.random(rows) for col in FEATURE_COLS}
    data["season"] = [2020] * n_train + [2023] * n_val + [2024] * n_test
    data[TARGET_CLF] = rng.integers(0, 2, size=rows).tolist()
    data[TARGET_REG] = (rng.random(rows) * 100 - 50).tolist()

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestTrain:
    def test_returns_two_models(self):
        data = _make_feature_data()
        clf, reg, col_means = train(data)
        assert clf is not None
        assert reg is not None

    def test_returns_col_means(self):
        data = _make_feature_data()
        clf, reg, col_means = train(data)
        assert isinstance(col_means, pd.Series)
        assert set(FEATURE_COLS).issubset(set(col_means.index))

    def test_classifier_can_predict(self):
        data = _make_feature_data()
        clf, _, _means = train(data)
        X = data[FEATURE_COLS].iloc[:3].values
        preds = clf.predict(X)
        assert len(preds) == 3

    def test_classifier_predict_proba_returns_two_classes(self):
        data = _make_feature_data()
        clf, _, _means = train(data)
        X = data[FEATURE_COLS].iloc[:3].values
        proba = clf.predict_proba(X)
        assert proba.shape == (3, 2)

    def test_regressor_can_predict(self):
        data = _make_feature_data()
        _, reg, _means = train(data)
        X = data[FEATURE_COLS].iloc[:3].values
        preds = reg.predict(X)
        assert len(preds) == 3

    def test_retains_rows_with_nan_features(self):
        """Training rows with NaN features should be kept and imputed, not dropped."""
        data = _make_feature_data()
        data_with_nan = data.copy()
        # Introduce NaN into a feature column for all training rows
        data_with_nan.loc[data_with_nan["season"] <= 2022, FEATURE_COLS[0]] = np.nan
        clf, reg, col_means = train(data_with_nan)
        assert clf is not None
        assert reg is not None

    def test_raises_on_empty_training_set(self):
        """train() should raise ValueError when no training rows exist."""
        data = _make_feature_data()
        data_no_train = data[data["season"] > 2022]
        with pytest.raises(ValueError, match="Training set is empty"):
            train(data_no_train)

    def test_works_without_val_or_test_data(self):
        """train() should succeed even when validation/test sets are empty."""
        data = _make_feature_data(n_train=50, n_val=0, n_test=0)
        data["season"] = 2020  # all training data
        clf, reg, col_means = train(data)
        assert clf is not None
        assert reg is not None


# ---------------------------------------------------------------------------
# save_models
# ---------------------------------------------------------------------------


class TestSaveModels:
    def test_creates_classifier_file(self):
        data = _make_feature_data()
        clf, reg, col_means = train(data)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_models(clf, reg, col_means, models_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "classifier.pkl"))

    def test_creates_regressor_file(self):
        data = _make_feature_data()
        clf, reg, col_means = train(data)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_models(clf, reg, col_means, models_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "regressor.pkl"))

    def test_creates_col_means_file(self):
        data = _make_feature_data()
        clf, reg, col_means = train(data)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_models(clf, reg, col_means, models_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "col_means.pkl"))

    def test_saved_classifier_is_loadable(self):
        data = _make_feature_data()
        clf, reg, col_means = train(data)
        X = data[FEATURE_COLS].iloc[:3].values
        with tempfile.TemporaryDirectory() as tmpdir:
            save_models(clf, reg, col_means, models_dir=tmpdir)
            with open(os.path.join(tmpdir, "classifier.pkl"), "rb") as f:
                loaded_clf = pickle.load(f)
            np.testing.assert_array_equal(clf.predict(X), loaded_clf.predict(X))

    def test_saved_regressor_is_loadable(self):
        data = _make_feature_data()
        clf, reg, col_means = train(data)
        X = data[FEATURE_COLS].iloc[:3].values
        with tempfile.TemporaryDirectory() as tmpdir:
            save_models(clf, reg, col_means, models_dir=tmpdir)
            with open(os.path.join(tmpdir, "regressor.pkl"), "rb") as f:
                loaded_reg = pickle.load(f)
            np.testing.assert_array_almost_equal(
                reg.predict(X), loaded_reg.predict(X)
            )

    def test_creates_models_directory(self):
        """save_models should create the target directory if it doesn't exist."""
        data = _make_feature_data()
        clf, reg, col_means = train(data)
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = os.path.join(tmpdir, "new_models_dir")
            assert not os.path.exists(models_dir)
            save_models(clf, reg, col_means, models_dir=models_dir)
            assert os.path.isdir(models_dir)
