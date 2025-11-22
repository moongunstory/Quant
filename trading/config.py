# trading/config.py
"""
Trading configuration.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class TradingConfig:
    """Configuration for trading system."""

    # Mode
    mode: Literal["paper", "live"] = "paper"

    # Binance API credentials (for live mode)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

    # Trading parameters
    symbol: str = "BTCUSDT"
    leverage: int = 3

    # Signal thresholds
    min_prediction_return: float = 0.01  # Only trade if |predicted return| > 1%

    # Position sizing (simple fixed percentage)
    position_size_pct: float = 0.20  # Use 20% of capital per trade

    # Risk management
    stop_loss_pct: float = 0.02  # 2% stop-loss
    take_profit_pct: float = 0.04  # 4% take-profit

    # Paper trading initial capital
    paper_initial_capital: float = 10000.0

    # Trading log paths
    paper_log_path: str = "data/trades/paper_trades.parquet"
    live_log_path: str = "data/trades/live_trades.parquet"

    def get_log_path(self) -> str:
        """Get appropriate log path based on mode."""
        return self.paper_log_path if self.mode == "paper" else self.live_log_path
