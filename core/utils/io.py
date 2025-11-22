# core/utils/io.py
"""
Common I/O utilities for parquet files and data loading.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import pandas as pd


def read_parquet_safe(path: Path | str) -> Optional[pd.DataFrame]:
    """
    Safely read a parquet file.

    Args:
        path: Path to parquet file

    Returns:
        DataFrame if file exists and is not empty, None otherwise
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        df = pd.read_parquet(path)
        return df if not df.empty else None
    except Exception:
        return None


def save_parquet(
    df: pd.DataFrame,
    path: Path | str,
    compression: str = 'snappy',
    create_dir: bool = True
) -> None:
    """
    Save DataFrame to parquet file.

    Args:
        df: DataFrame to save
        path: Output path
        compression: Compression method
        create_dir: Create parent directory if it doesn't exist
    """
    path = Path(path)

    if create_dir:
        path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, index=False, compression=compression)


def merge_and_dedupe(
    new_df: pd.DataFrame,
    existing_df: Optional[pd.DataFrame],
    key_columns: List[str]
) -> pd.DataFrame:
    """
    Merge new data with existing, remove duplicates, and sort.

    Args:
        new_df: Newly collected data
        existing_df: Existing data (or None)
        key_columns: Columns to use for deduplication

    Returns:
        Merged and deduplicated DataFrame
    """
    if existing_df is None:
        return new_df

    merged = pd.concat([existing_df, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset=key_columns, keep='last')
    merged = merged.sort_values(key_columns[0]).reset_index(drop=True)

    return merged
