# pipeline/processors/regime.py
"""
Market regime detection features.

Identifies bull/bear/sideways market conditions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .utils import ensure_sorted_datetime


def detect_regime_simple(
    df: pd.DataFrame,
    price_col: str = 'fut_close',
    ma_short: int = 50,
    ma_long: int = 200
) -> pd.DataFrame:
    """
    Simple regime detection based on moving averages.

    Regime definitions:
    - Bull (2): price > MA_short AND MA_short > MA_long
    - Bear (0): price < MA_long
    - Sideways (1): everything else

    Args:
        df: DataFrame with price_col
        price_col: Column name for price
        ma_short: Short MA window (default 50 hours)
        ma_long: Long MA window (default 200 hours)

    Returns:
        DataFrame with regime columns added
    """
    df = df.copy()

    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")

    # Ensure timestamp index
    if 'timestamp' in df.columns:
        df = ensure_sorted_datetime(df, 'timestamp')

    # Calculate moving averages
    df[f'ma_{ma_short}'] = df[price_col].rolling(ma_short).mean()
    df[f'ma_{ma_long}'] = df[price_col].rolling(ma_long).mean()

    # Regime classification
    df['regime'] = 1  # Default: sideways

    # Bull market: price above short MA, short MA above long MA
    bull_mask = (df[price_col] > df[f'ma_{ma_short}']) & (df[f'ma_{ma_short}'] > df[f'ma_{ma_long}'])
    df.loc[bull_mask, 'regime'] = 2

    # Bear market: price below long MA
    bear_mask = df[price_col] < df[f'ma_{ma_long}']
    df.loc[bear_mask, 'regime'] = 0

    # One-hot encode regimes
    df['regime_bear'] = (df['regime'] == 0).astype(int)
    df['regime_sideways'] = (df['regime'] == 1).astype(int)
    df['regime_bull'] = (df['regime'] == 2).astype(int)

    # Regime persistence (how long in current regime)
    df['regime_change'] = (df['regime'] != df['regime'].shift(1)).astype(int)
    df['regime_duration'] = df.groupby((df['regime_change'] == 1).cumsum()).cumcount() + 1

    return df


def add_volatility_regime(
    df: pd.DataFrame,
    price_col: str = 'fut_close',
    window: int = 24,
    lookback: int = 252
) -> pd.DataFrame:
    """
    Add volatility regime features.

    Classifies current volatility as:
    - Low: below 25th percentile
    - Normal: 25th-75th percentile
    - High: above 75th percentile

    Args:
        df: DataFrame with price_col
        price_col: Column name for price
        window: Window for volatility calculation (default 24h)
        lookback: Window for percentile calculation (default 252h H 10 days)

    Returns:
        DataFrame with volatility regime columns added
    """
    df = df.copy()

    # Calculate returns and volatility
    if 'log_ret_1h' not in df.columns:
        df['log_ret_1h'] = np.log(df[price_col] / df[price_col].shift(1))

    df['volatility'] = df['log_ret_1h'].rolling(window).std()

    # Calculate rolling percentiles
    df['vol_pct_25'] = df['volatility'].rolling(lookback).quantile(0.25)
    df['vol_pct_75'] = df['volatility'].rolling(lookback).quantile(0.75)

    # Classify volatility regime
    df['vol_regime'] = 1  # Default: normal

    low_vol_mask = df['volatility'] < df['vol_pct_25']
    df.loc[low_vol_mask, 'vol_regime'] = 0  # Low volatility

    high_vol_mask = df['volatility'] > df['vol_pct_75']
    df.loc[high_vol_mask, 'vol_regime'] = 2  # High volatility

    # One-hot encode
    df['vol_regime_low'] = (df['vol_regime'] == 0).astype(int)
    df['vol_regime_normal'] = (df['vol_regime'] == 1).astype(int)
    df['vol_regime_high'] = (df['vol_regime'] == 2).astype(int)

    return df
