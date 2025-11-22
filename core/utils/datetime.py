# core/utils/datetime.py
"""
Date and time utilities.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


def ensure_datetime_index(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Ensure DataFrame has datetime index.

    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with DatetimeIndex
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if timestamp_col not in df.columns:
            raise ValueError(
                f"'{timestamp_col}' column not found and index is not DatetimeIndex"
            )
        df[timestamp_col] = pd.to_datetime(
            df[timestamp_col], utc=True, errors="coerce"
        )
        df = df.set_index(timestamp_col)

    return df.sort_index()


def is_file_stale(file_path: Path, staleness_hours: int) -> bool:
    """
    Check if a file is stale based on its modification time.

    Args:
        file_path: Path to file
        staleness_hours: Staleness threshold in hours

    Returns:
        True if file is stale or doesn't exist
    """
    if not file_path.exists():
        return True

    mtime = file_path.stat().st_mtime
    last_modified_dt = datetime.fromtimestamp(mtime)

    return (datetime.now() - last_modified_dt) > timedelta(hours=staleness_hours)


def apply_sliding_window(
    df: pd.DataFrame,
    window_days: int,
    timestamp_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    Apply sliding window to keep only recent data.

    Args:
        df: Input DataFrame
        window_days: Window size in days
        timestamp_col: Name of timestamp column

    Returns:
        Filtered DataFrame
    """
    cutoff = pd.Timestamp.now() - timedelta(days=window_days)
    return df[df[timestamp_col] >= cutoff].copy()
