# model/regression/lgbm/model.py
"""
LightGBM regressor wrapper.
"""

from __future__ import annotations
from typing import Dict, Any
from lightgbm import LGBMRegressor


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
