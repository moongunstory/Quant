# model/regression/lgbm/trainer.py
"""
LightGBM training logic.
"""

from __future__ import annotations
import numpy as np
from lightgbm import early_stopping, LGBMRegressor


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
