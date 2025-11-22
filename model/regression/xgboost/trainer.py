# model/regression/xgboost/trainer.py
"""
XGBoost training logic.
"""

from __future__ import annotations
import numpy as np
from xgboost import XGBRegressor


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
        sample_weight_eval_set=[sample_weight_val] if sample_weight_val is not None else None,
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    return model
