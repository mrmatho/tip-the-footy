"""
train_model.py – Train and save the AFL Tipping Model.

Loads the engineered feature dataset, performs a temporal train/val/test
split, trains a LightGBM winner classifier and margin regressor, evaluates
them on the held-out 2024 season, and persists both models to disk.
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)

# Allow importing sibling scripts both when run directly and as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_features import FEATURE_COLS, TARGET_CLF, TARGET_REG  # noqa: E402
from scripts.app_config import load_model_config  # noqa: E402

_CFG = load_model_config()
_SPLIT_CFG = _CFG["training"]["split"]
_CLF_CFG = _CFG["training"]["classifier"]
_REG_CFG = _CFG["training"]["regressor"]


def train(data: pd.DataFrame):
    """Train winner classifier and margin regressor using a temporal split.

    Training set:  seasons ≤ 2022
    Validation set: season 2023  (used for validation/monitoring)
    Test set:       season 2024  (held-out evaluation)

    Args:
        data: Feature DataFrame produced by ``build_features``.

    Returns:
        Tuple of ``(classifier, regressor, col_means)`` – fitted LightGBM
        models and a :class:`pandas.Series` of per-feature training means
        used for NaN imputation at inference time.

    Raises:
        ValueError: If required feature columns are missing.
        ValueError: If the training set is empty after applying the split.
    """
    missing_features = [c for c in FEATURE_COLS if c not in data.columns]
    if missing_features:
        raise ValueError(
            "Training data is missing required feature columns: "
            f"{missing_features}. Rebuild features before training."
        )

    train_end_season = int(_SPLIT_CFG["train_end_season"])
    val_season = int(_SPLIT_CFG["val_season"])
    test_season = int(_SPLIT_CFG["test_season"])

    train_df = data[data["season"] <= train_end_season].dropna(
        subset=[TARGET_CLF, TARGET_REG]
    )
    val_df = data[data["season"] == val_season].dropna(
        subset=[TARGET_CLF, TARGET_REG]
    )
    test_df = data[data["season"] == test_season].dropna(
        subset=[TARGET_CLF, TARGET_REG]
    )

    if train_df.empty:
        raise ValueError("Training set is empty – check data ranges.")

    # Impute NaN features with training-set column means.
    col_means = train_df[FEATURE_COLS].mean()
    X_train = train_df[FEATURE_COLS].fillna(col_means)
    X_val = val_df[FEATURE_COLS].fillna(col_means) if not val_df.empty else X_train.iloc[:0]
    X_test = test_df[FEATURE_COLS].fillna(col_means) if not test_df.empty else X_train.iloc[:0]

    y_clf_train = train_df[TARGET_CLF]
    y_reg_train = train_df[TARGET_REG]
    y_clf_val = val_df[TARGET_CLF] if not val_df.empty else pd.Series(dtype=int)
    y_reg_val = val_df[TARGET_REG] if not val_df.empty else pd.Series(dtype=float)
    y_clf_test = test_df[TARGET_CLF] if not test_df.empty else pd.Series(dtype=int)
    y_reg_test = test_df[TARGET_REG] if not test_df.empty else pd.Series(dtype=float)

    # ── Classification model (winner prediction) ──────────────────────────────
    clf = lgb.LGBMClassifier(
        n_estimators=int(_CLF_CFG["n_estimators"]),
        learning_rate=float(_CLF_CFG["learning_rate"]),
        max_depth=int(_CLF_CFG["max_depth"]),
        num_leaves=int(_CLF_CFG["num_leaves"]),
        subsample=float(_CLF_CFG["subsample"]),
        colsample_bytree=float(_CLF_CFG["colsample_bytree"]),
        min_child_samples=int(_CLF_CFG["min_child_samples"]),
        random_state=int(_CLF_CFG["random_state"]),
        verbose=int(_CLF_CFG["verbose"]),
    )
    eval_set_clf = [(X_val, y_clf_val)] if not X_val.empty else None
    clf.fit(X_train, y_clf_train, eval_set=eval_set_clf)

    # ── Regression model (margin prediction) ──────────────────────────────────
    reg = lgb.LGBMRegressor(
        n_estimators=int(_REG_CFG["n_estimators"]),
        learning_rate=float(_REG_CFG["learning_rate"]),
        max_depth=int(_REG_CFG["max_depth"]),
        num_leaves=int(_REG_CFG["num_leaves"]),
        subsample=float(_REG_CFG["subsample"]),
        colsample_bytree=float(_REG_CFG["colsample_bytree"]),
        min_child_samples=int(_REG_CFG["min_child_samples"]),
        random_state=int(_REG_CFG["random_state"]),
        verbose=int(_REG_CFG["verbose"]),
    )
    eval_set_reg = [(X_val, y_reg_val)] if not X_val.empty else None
    reg.fit(X_train, y_reg_train, eval_set=eval_set_reg)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    if not X_test.empty:
        y_pred_clf = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred_reg = reg.predict(X_test)

        print("Test accuracy:", accuracy_score(y_clf_test, y_pred_clf))
        print("Test log-loss:", log_loss(y_clf_test, y_pred_proba))
        print("Test AUC-ROC:", roc_auc_score(y_clf_test, y_pred_proba))
        print("Test MAE (margin):", mean_absolute_error(y_reg_test, y_pred_reg))
    else:
        print("No test data available for evaluation.")

    return clf, reg, col_means


def save_models(clf, reg, col_means: pd.Series, models_dir: str = "models") -> None:
    """Persist trained models and imputation means to disk as pickle files.

    Args:
        clf: Trained winner classifier.
        reg: Trained margin regressor.
        col_means: Per-feature mean values from the training set, used for
                   NaN imputation at inference time.
        models_dir: Directory to save ``classifier.pkl``, ``regressor.pkl``,
                    and ``col_means.pkl`` in.
    """
    os.makedirs(models_dir, exist_ok=True)
    clf_path = os.path.join(models_dir, "classifier.pkl")
    reg_path = os.path.join(models_dir, "regressor.pkl")
    means_path = os.path.join(models_dir, "col_means.pkl")
    with open(clf_path, "wb") as f:
        pickle.dump(clf, f)
    with open(reg_path, "wb") as f:
        pickle.dump(reg, f)
    with open(means_path, "wb") as f:
        pickle.dump(col_means, f)
    print(f"Models saved to {models_dir}/")


if __name__ == "__main__":
    data = pd.read_csv("data/features.csv")
    print(f"Loaded {len(data)} rows for training.")
    clf, reg, col_means = train(data)
    save_models(clf, reg, col_means)
