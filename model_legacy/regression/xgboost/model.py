# model/regression/xgboost/model.py
"""
XGBoost regressor wrapper.
"""

from __future__ import annotations
from typing import Dict, Any
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
