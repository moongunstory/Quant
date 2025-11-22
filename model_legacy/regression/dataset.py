# model/regression/dataset.py
"""
Dataset preparation for regression models.
"""

from __future__ import annotations
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import RegressionConfig


def ensure_datetime_index(df: pd.DataFrame, cfg: RegressionConfig) -> pd.DataFrame:
    """Ensure DataFrame has datetime index."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if cfg.timestamp_col not in df.columns:
            raise ValueError(
                f"'{cfg.timestamp_col}' column not found and index is not DatetimeIndex"
            )
        df[cfg.timestamp_col] = pd.to_datetime(
            df[cfg.timestamp_col], utc=True, errors="coerce"
        )
        df = df.set_index(cfg.timestamp_col)

    return df.sort_index()


def coerce_numeric_features(
    df: pd.DataFrame,
    drop_cols: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Keep only numeric/boolean columns, drop specified columns.

    Args:
        df: Input DataFrame
        drop_cols: Columns to drop (targets, close price, etc.)

    Returns:
        feat_df: DataFrame with only numeric features
        feature_names: List of feature column names
    """
    work = df.copy()

    # Convert object columns to numeric if possible
    for col in work.columns:
        if col in drop_cols:
            continue
        s = work[col]
        if s.dtype == "object":
            work[col] = pd.to_numeric(s, errors="coerce")

    # Keep only numeric/boolean types, drop specified columns
    feat_df = (
        work
        .select_dtypes(include=[np.number, "bool"])
        .drop(columns=drop_cols, errors="ignore")
    )

    feature_names = feat_df.columns.tolist()
    return feat_df, feature_names


def build_regression_targets(
    df: pd.DataFrame,
    horizons_hours: List[int],
    cfg: RegressionConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build continuous return targets for regression.

    Args:
        df: Master features DataFrame (with datetime index)
        horizons_hours: List of prediction horizons in hours
        cfg: RegressionConfig

    Returns:
        df: DataFrame with target columns added
        target_cols: List of target column names
    """
    df = df.copy()

    if cfg.close_col not in df.columns:
        raise KeyError(f"Close price column '{cfg.close_col}' not found")

    close = df[cfg.close_col].astype(float)
    target_cols = []

    for h in horizons_hours:
        target_col = f"ret_{h}h"

        # Future return: (future_close - current_close) / current_close
        future_close = close.shift(-h)
        df[target_col] = (future_close - close) / close

        target_cols.append(target_col)

    return df, target_cols


def build_supervised_for_horizon(
    df: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    horizon_hours: int,
    cfg: RegressionConfig,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build supervised dataset for a specific horizon.

    Args:
        df: Master features DataFrame
        as_of_ts: Current timestamp
        horizon_hours: Prediction horizon
        cfg: RegressionConfig

    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (numpy array)
        feature_names: List of feature names
    """
    # Get training window
    window_hours = cfg.get_window_hours_for(horizon_hours)
    window_start_ts = as_of_ts - timedelta(hours=window_hours)
    last_label_ts = as_of_ts - timedelta(hours=horizon_hours)

    # Extract window
    df_win = df.loc[window_start_ts:last_label_ts].copy()
    if df_win.empty:
        raise ValueError(f"No data in window for horizon {horizon_hours}h")

    # Build target
    if cfg.close_col not in df_win.columns:
        raise KeyError(f"Close price column '{cfg.close_col}' not found")

    close = df_win[cfg.close_col].astype(float)
    future_close = close.shift(-horizon_hours)
    ret = (future_close - close) / close

    ret_col = f"ret_{horizon_hours}h"
    df_win[ret_col] = ret

    # Drop rows with NaN targets
    df_train = df_win.dropna(subset=[ret_col])
    if len(df_train) < cfg.min_samples:
        raise ValueError(
            f"Insufficient samples: {len(df_train)} < {cfg.min_samples}"
        )

    y = df_train[ret_col].to_numpy()

    # Extract features (drop target and close price)
    drop_cols = [ret_col, cfg.close_col]
    feat_df, feature_names = coerce_numeric_features(df_train, drop_cols=drop_cols)

    if not feature_names:
        raise ValueError(f"No numeric features available for horizon {horizon_hours}h")

    X = feat_df.to_numpy(dtype=float)

    return X, y, feature_names


def make_sample_weight(
    y: np.ndarray,
    cfg: RegressionConfig,
) -> np.ndarray:
    """
    Create sample weights (for recent data weighting).

    Args:
        y: Target array
        cfg: RegressionConfig

    Returns:
        sample_weight: Weight for each sample
    """
    n = len(y)
    if n == 0:
        return np.array([], dtype=float)

    # Recent data weighting (linear increase from 1.0 to 2.0)
    if cfg.use_recent_weight and n > 1:
        idx = np.arange(n, dtype=float)
        rel_pos = idx / (n - 1)  # 0.0 to 1.0
        weight = 1.0 + rel_pos  # 1.0 to 2.0
    else:
        weight = np.ones(n, dtype=float)

    return weight
