# config/data.py
"""
Data-specific configuration (retention policies, update frequencies).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class DataRetentionConfig:
    """Data retention and update policies."""

    # Sliding window for re-fetchable data
    sliding_window_days: int = 540

    # Files that should be permanently kept (cannot re-fetch historical data)
    permanent_files: Tuple[str, ...] = (
        'oi_1h.parquet',
        'ls_ratio_top_1h.parquet',
        'news_raw.parquet',
    )

    # Update policies: (representative_file, staleness_hours)
    update_policies: Dict[str, Tuple[str, int]] = None

    def __post_init__(self):
        if self.update_policies is None:
            self.update_policies = {
                'binance': ('binance/ohlcv_futures_1h.parquet', 1),
                'news': ('news/news_raw.parquet', 2),
                'macro': ('macro/fred_dgs10.parquet', 24),
                'onchain': ('onchain/blockchain_com_n-transactions.parquet', 24),
                'derivatives': ('derivatives/deribit_btc_dvol.parquet', 24),
            }

    def should_apply_sliding_window(self, filename: str) -> bool:
        """Check if sliding window should be applied to this file."""
        return filename not in self.permanent_files
