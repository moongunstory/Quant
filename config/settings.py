# config/settings.py
"""
System-wide settings and constants.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SystemConfig:
    """Global system configuration."""

    # Update intervals
    update_interval_hours: int = 1  # Main loop update frequency

    # Data retention
    data_retention_days: int = 540  # 540-day sliding window for re-fetchable data

    # Retry settings
    retry_attempts: int = 3
    retry_backoff_factor: float = 1.0  # Exponential backoff: 1s, 2s, 4s

    # Logging
    log_level: str = "INFO"
    suppress_external_loggers: bool = True

    # Performance
    num_workers: int = 4  # For parallel data collection
    cache_enabled: bool = True

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    models_dir: Path = data_dir / "models"

    def __post_init__(self):
        """Ensure paths are Path objects."""
        self.project_root = Path(self.project_root)
        self.data_dir = Path(self.data_dir)
        self.raw_data_dir = Path(self.raw_data_dir)
        self.processed_data_dir = Path(self.processed_data_dir)
        self.models_dir = Path(self.models_dir)


@dataclass
class DataConfig:
    """Data collection and processing configuration."""

    # Permanent files (cannot re-fetch historical data)
    permanent_files: tuple = (
        'oi_1h.parquet',
        'ls_ratio_top_1h.parquet',
        'news_raw.parquet',
    )

    # Update policies (file, staleness_hours)
    update_policies: dict = None

    def __post_init__(self):
        if self.update_policies is None:
            self.update_policies = {
                'binance': ('binance/ohlcv_futures_1h.parquet', 1),
                'news': ('news/news_raw.parquet', 2),
                'macro': ('macro/fred_dgs10.parquet', 24),
                'onchain': ('onchain/blockchain_com_n-transactions.parquet', 24),
                'derivatives': ('derivatives/deribit_btc_dvol.parquet', 24),
            }
