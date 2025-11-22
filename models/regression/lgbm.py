# models/regression/lgbm.py
"""
LightGBM regressor for price prediction.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
from lightgbm import LGBMRegressor, early_stopping


def create_lgbm_regressor(params: Dict[str, Any] = None) -> LGBMRegressor:
    """
    Create LightGBM regressor with sensible defaults.

    Args:
        params: Override default parameters

    Returns:
        LGBMRegressor instance
    """
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    if params:
        default_params.update(params)

    return LGBMRegressor(**default_params)


def train_lgbm_regressor(
    model: LGBMRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sample_weight_train: np.ndarray = None,
    sample_weight_val: np.ndarray = None,
    early_stopping_rounds: int = 50,
) -> LGBMRegressor:
    """
    Train LightGBM regressor with early stopping.

    Args:
        model: LGBMRegressor instance
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
    callbacks = []
    if early_stopping_rounds and early_stopping_rounds > 0:
        callbacks.append(
            early_stopping(
                stopping_rounds=early_stopping_rounds,
                first_metric_only=True,
                verbose=False
            )
        )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[sample_weight_val] if sample_weight_val is not None else None,
        callbacks=callbacks,
    )

    return model
