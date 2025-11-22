# process/quality_check.py
"""
Data quality validation module.
Detects missing values, time gaps, and potential data quality issues.
"""

from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def detect_time_gaps(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    expected_freq: str = "1H",
    max_gaps_to_report: int = 10
) -> List[Dict[str, Any]]:
    """
    Detect time gaps in a time series DataFrame.

    Args:
        df: DataFrame with time column
        time_col: Name of the time column
        expected_freq: Expected frequency (e.g., "1H" for hourly, "1D" for daily)
        max_gaps_to_report: Maximum number of gaps to report

    Returns:
        List of gap information dicts
    """
    if time_col not in df.columns or df.empty:
        return []

    df_sorted = df.sort_values(time_col).copy()
    time_series = pd.to_datetime(df_sorted[time_col])

    # Calculate differences
    diffs = time_series.diff()

    # Expected difference based on frequency
    expected_diff = pd.Timedelta(expected_freq)

    # Find gaps (differences larger than expected)
    gaps = diffs[diffs > expected_diff * 1.5]  # 1.5x tolerance

    gap_list = []
    for idx, gap in gaps.head(max_gaps_to_report).items():
        gap_list.append({
            "index": int(idx),
            "gap_size": str(gap),
            "timestamp": str(time_series.iloc[idx]),
        })

    return gap_list


def detect_outliers(
    df: pd.DataFrame,
    numeric_only: bool = True,
    iqr_multiplier: float = 3.0,
) -> Dict[str, int]:
    """
    Detect outliers using IQR method.

    Args:
        df: DataFrame to check
        numeric_only: Only check numeric columns
        iqr_multiplier: IQR multiplier for outlier detection (default 3.0)

    Returns:
        Dict mapping column name to number of outliers
    """
    outlier_counts = {}

    cols = df.select_dtypes(include=[np.number]).columns if numeric_only else df.columns

    for col in cols:
        if df[col].nunique() < 2:  # Skip constant columns
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outliers > 0:
            outlier_counts[col] = int(outliers)

    return outlier_counts


def validate_data_quality(
    df: pd.DataFrame,
    module_name: str,
    time_col: str = "timestamp",
    expected_freq: str = "1H",
) -> Dict[str, Any]:
    """
    Comprehensive data quality check.

    Args:
        df: DataFrame to validate
        module_name: Name of the data module (for reporting)
        time_col: Name of the time column
        expected_freq: Expected time frequency

    Returns:
        Dict containing quality metrics
    """
    if df.empty:
        return {
            "module": module_name,
            "status": "empty",
            "total_rows": 0,
        }

    report = {
        "module": module_name,
        "status": "ok",
        "total_rows": len(df),
        "total_columns": len(df.columns),
    }

    # Check missing values
    missing_counts = df.isnull().sum()
    missing_ratios = missing_counts / len(df)
    report["missing_ratios"] = {
        col: float(ratio)
        for col, ratio in missing_ratios.items()
        if ratio > 0
    }

    # Check for time gaps (if time column exists)
    if time_col in df.columns:
        gaps = detect_time_gaps(df, time_col, expected_freq)
        report["time_gaps"] = len(gaps)
        if gaps:
            report["gap_examples"] = gaps[:5]  # First 5 gaps

    # Check for outliers
    outliers = detect_outliers(df)
    report["outlier_columns"] = outliers

    # Warnings
    warnings = []
    if report.get("missing_ratios"):
        max_missing = max(report["missing_ratios"].values())
        if max_missing > 0.1:
            warnings.append(f"결측치 {max_missing*100:.1f}% 이상 발견")

    if report.get("time_gaps", 0) > 5:
        warnings.append(f"시간 갭 {report['time_gaps']}개 발견")

    if outliers:
        total_outliers = sum(outliers.values())
        if total_outliers > len(df) * 0.01:  # More than 1% outliers
            warnings.append(f"이상치 {total_outliers}개 발견 (전체의 {total_outliers/len(df)*100:.1f}%)")

    report["warnings"] = warnings

    # Print warnings
    if warnings:
        print(f"⚠️  [{module_name}] 데이터 품질 경고:")
        for warning in warnings:
            print(f"   - {warning}")
    else:
        print(f"✅ [{module_name}] 데이터 품질 정상")

    return report
