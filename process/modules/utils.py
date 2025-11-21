# process/modules/utils.py

from __future__ import annotations
import numpy as np
import pandas as pd


def ensure_sorted_datetime(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
    """timestamp 기반 정렬 + datetime 변환."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    df = df.sort_values(col)
    return df.reset_index(drop=True)


def ensure_sorted_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """date 기반 정렬 + 날짜 정규화."""
    df = df.copy()
    df[col] = pd.to_datetime(df[col]).dt.normalize()
    df = df.sort_values(col)
    return df.reset_index(drop=True)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """단순 RSI 구현 (충분히 실용적)."""
    series = series.astype(float)
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_rolling_zscore(
    s: pd.Series, window: int, prefix: str
) -> pd.DataFrame:
    """
    시리즈 하나에 대해 롤링 z-score 계산.
    반환: ['val', 'z'] 두 컬럼 가진 DataFrame
    """
    s = s.astype(float)
    roll = s.rolling(window)
    mean = roll.mean()
    std = roll.std()

    out = pd.DataFrame({f"{prefix}_val": s})
    out[f"{prefix}_z_{window}"] = (s - mean) / std
    return out


def compute_z_score(series: pd.Series, window: int, min_periods: int = 30) -> pd.Series:
    """
    Rolling Z-score 계산 (중복 코드 제거용 공통 함수).

    Args:
        series: 입력 시리즈
        window: Rolling window 크기
        min_periods: 최소 필요 데이터 수

    Returns:
        Z-score 시리즈
    """
    series = series.astype(float)
    roll = series.rolling(window, min_periods=min_periods)
    return (series - roll.mean()) / roll.std()
