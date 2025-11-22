# process/modules/sentiment.py
"""
Process sentiment data (Fear & Greed Index, CoinGecko market metrics) into features.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from .utils import ensure_sorted_datetime


def build_sentiment_features(
    df_fear_greed: Optional[pd.DataFrame] = None,
    df_coingecko: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build sentiment features from Fear & Greed Index and CoinGecko market data.

    Args:
        df_fear_greed: DataFrame with columns: timestamp, value (fear_greed_index), value_classification
        df_coingecko: DataFrame with columns: timestamp, market_cap, total_volume

    Returns:
        DataFrame with sentiment features (1h or daily frequency depending on input)
    """
    dfs_to_merge = []

    # ---- Fear & Greed Index ----
    if df_fear_greed is not None and not df_fear_greed.empty:
        fgi = ensure_sorted_datetime(df_fear_greed, "timestamp").copy()

        # Rename value column if needed
        if 'value' in fgi.columns:
            fgi = fgi.rename(columns={'value': 'fear_greed_index'})

        # Ensure numeric
        fgi['fear_greed_index'] = pd.to_numeric(fgi['fear_greed_index'], errors='coerce')

        # Z-score (30-day rolling)
        fgi['fgi_z_30d'] = (
            (fgi['fear_greed_index'] - fgi['fear_greed_index'].rolling(30).mean())
            / fgi['fear_greed_index'].rolling(30).std()
        )

        # Changes
        fgi['fgi_1d_change'] = fgi['fear_greed_index'].diff(1)
        fgi['fgi_7d_change'] = fgi['fear_greed_index'].diff(7)

        # Moving averages
        fgi['fgi_ma_7d'] = fgi['fear_greed_index'].rolling(7).mean()
        fgi['fgi_ma_30d'] = fgi['fear_greed_index'].rolling(30).mean()

        # Keep relevant columns
        fgi_cols = [
            'timestamp', 'fear_greed_index', 'fgi_z_30d',
            'fgi_1d_change', 'fgi_7d_change', 'fgi_ma_7d', 'fgi_ma_30d'
        ]
        fgi = fgi[[c for c in fgi_cols if c in fgi.columns]]

        dfs_to_merge.append(fgi)

    # ---- CoinGecko Market Metrics ----
    if df_coingecko is not None and not df_coingecko.empty:
        cg = ensure_sorted_datetime(df_coingecko, "timestamp").copy()

        # Market cap features
        if 'market_cap' in cg.columns:
            cg['market_cap'] = pd.to_numeric(cg['market_cap'], errors='coerce')

            # Log transform (market cap is very large)
            cg['log_market_cap'] = np.log(cg['market_cap'].replace(0, np.nan))

            # Changes
            cg['market_cap_1d_change'] = cg['market_cap'].pct_change(1)
            cg['market_cap_7d_change'] = cg['market_cap'].pct_change(7)

            # Z-score (30-day)
            cg['market_cap_z_30d'] = (
                (cg['market_cap'] - cg['market_cap'].rolling(30).mean())
                / cg['market_cap'].rolling(30).std()
            )

        # Volume features
        if 'total_volume' in cg.columns:
            cg['total_volume'] = pd.to_numeric(cg['total_volume'], errors='coerce')

            # Log transform
            cg['log_total_volume'] = np.log(cg['total_volume'].replace(0, np.nan))

            # Changes
            cg['volume_1d_change'] = cg['total_volume'].pct_change(1)
            cg['volume_7d_change'] = cg['total_volume'].pct_change(7)

            # Moving averages
            cg['volume_ma_7d'] = cg['total_volume'].rolling(7).mean()
            cg['volume_ma_30d'] = cg['total_volume'].rolling(30).mean()

            # Volume volatility
            cg['volume_volatility_7d'] = cg['volume_1d_change'].rolling(7).std()

        dfs_to_merge.append(cg)

    # Merge all sentiment dataframes
    if not dfs_to_merge:
        return pd.DataFrame()

    if len(dfs_to_merge) == 1:
        df_out = dfs_to_merge[0]
    else:
        df_out = dfs_to_merge[0]
        for df in dfs_to_merge[1:]:
            df_out = df_out.merge(df, on='timestamp', how='outer')

    df_out = df_out.sort_values('timestamp').reset_index(drop=True)

    return df_out
