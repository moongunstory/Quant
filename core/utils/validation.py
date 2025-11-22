# core/utils/validation.py
"""
Data validation utilities.
"""

from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
    source_name: str = "DataFrame"
) -> None:
    """
    Validate that DataFrame contains required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        source_name: Name of data source (for error messages)

    Raises:
        ValueError: If required columns are missing
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source_name} missing required columns: {missing}"
        )


def coerce_numeric(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Convert object columns to numeric where possible.

    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from conversion

    Returns:
        DataFrame with numeric columns converted
    """
    if exclude_cols is None:
        exclude_cols = []

    df = df.copy()

    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def check_missing_data(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> dict:
    """
    Check for missing data in DataFrame.

    Args:
        df: Input DataFrame
        threshold: Warning threshold (fraction of missing values)

    Returns:
        Dictionary with missing data statistics
    """
    missing_pct = df.isnull().mean()
    high_missing = missing_pct[missing_pct > threshold]

    return {
        'total_missing': df.isnull().sum().sum(),
        'missing_pct': missing_pct.to_dict(),
        'high_missing_cols': high_missing.to_dict() if not high_missing.empty else {}
    }
