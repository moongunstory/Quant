# models/regression/xgboost.py
"""
XGBoost regressor for price prediction.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
from xgboost import XGBRegressor


def create_xgboost_regressor(params: Dict[str, Any] = None) -> XGBRegressor:
    """
    Create XGBoost regressor with sensible defaults.

    Args:
        params: Override default parameters

    Returns:
        XGBRegressor instance
    """
    default_params = {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    if params:
        default_params.update(params)

    return XGBRegressor(**default_params)


def train_xgboost_regressor(
    model: XGBRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight_train: np.ndarray = None,
    sample_weight_val: np.ndarray = None,
    early_stopping_rounds: int = 50,
) -> XGBRegressor:
    """
    Train XGBoost regressor with early stopping.

    Args:
        model: XGBRegressor instance
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        sample_weight_train: Sample weights for training
        sample_weight_val: Sample weights for validation
        early_stopping_rounds: Early stopping patience

    Returns:
        Trained model
    """
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        sample_weight=[(sample_weight_val if sample_weight_val is not None else None)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    return model
