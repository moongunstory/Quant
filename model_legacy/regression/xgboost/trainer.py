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
    # XGBoost 2.0+ uses callbacks instead of early_stopping_rounds parameter
    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    except Exception as e:
        # Fallback: try old API or just fit without validation
        print(f"      Warning: Could not use validation set: {e}")
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight_train,
            verbose=False,
        )

    return model
