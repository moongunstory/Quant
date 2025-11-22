# model/regression/config.py
"""
Configuration for regression-based prediction models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, Optional, List


@dataclass
class RegressionConfig:
    """Configuration for regression models."""

    # Data paths
    master_path: str = "data/processed/master_features_1h.parquet"
    model_dir: str = "data/models/regression"
    pred_log_path: str = "data/predictions/regression_predictions.parquet"
    report_dir: str = "data/reports"

    # Prediction horizons (in hours)
    horizons_hours: Tuple[int, ...] = (72, 168, 720, 2160)  # 3d, 7d, 30d, 90d

    # Training window
    window_days: int = 540  # Default 540 days of history

    # Horizon-specific windows (can be optimized by HPO)
    window_hours_map: Optional[Dict[int, int]] = None

    # Data columns
    timestamp_col: str = "timestamp"
    close_col: str = "fut_close"

    # Train/val split
    val_ratio: float = 0.2  # 20% for validation
    min_samples: int = 500  # Minimum samples needed for training

    # Sample weighting
    use_recent_weight: bool = True  # Give more weight to recent data
    use_class_weight: bool = False  # Not applicable for regression

    # Early stopping
    early_stopping_rounds: int = 50

    # Model saving
    save_models: bool = True
    skip_if_exists: bool = True  # Skip training if today's models exist

    # HPO integration
    hpo_best_config_path: str = "data/hpo/regression/best/best_config_regression.json"
    use_hpo_params: bool = True

    # Model parameters (can be overridden by HPO)
    lgbm_params_map: Optional[Dict[int, Dict]] = None
    xgboost_params_map: Optional[Dict[int, Dict]] = None

    def get_window_hours_for(self, horizon_hours: int) -> int:
        """Get training window for specific horizon."""
        if self.window_hours_map and horizon_hours in self.window_hours_map:
            return int(self.window_hours_map[horizon_hours])
        return self.window_days * 24

    def get_lgbm_params_for(self, horizon_hours: int) -> Dict:
        """Get LightGBM parameters for specific horizon."""
        if self.lgbm_params_map and horizon_hours in self.lgbm_params_map:
            return self.lgbm_params_map[horizon_hours]
        # Default parameters
        return {
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }

    def get_xgboost_params_for(self, horizon_hours: int) -> Dict:
        """Get XGBoost parameters for specific horizon."""
        if self.xgboost_params_map and horizon_hours in self.xgboost_params_map:
            return self.xgboost_params_map[horizon_hours]
        # Default parameters
        return {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
