# trading/logger.py
"""
Trade logging to parquet files.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd


class TradeLogger:
    """
    Log all trades to parquet for analysis.

    Logs include:
    - Timestamp
    - Action (buy/sell/close)
    - Price
    - Quantity
    - P&L
    - Portfolio state
    """

    def __init__(self, log_path: str):
        """
        Initialize trade logger.

        Args:
            log_path: Path to parquet log file
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_trade(self, trade: Dict) -> None:
        """
        Append trade to log.

        Args:
            trade: Trade dict with keys: timestamp, action, price, btc_amount, etc.
        """
        df_new = pd.DataFrame([trade])

        if self.log_path.exists():
            df_existing = pd.read_parquet(self.log_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        df_combined.to_parquet(self.log_path, compression="zstd")

    def get_trades(self) -> pd.DataFrame:
        """
        Load all trades from log.

        Returns:
            DataFrame with all trades
        """
        if self.log_path.exists():
            return pd.read_parquet(self.log_path)
        return pd.DataFrame()

    def get_trade_count(self) -> int:
        """Get total number of trades."""
        df = self.get_trades()
        return len(df)

    def get_total_pnl(self) -> float:
        """Get total P&L from all closed trades."""
        df = self.get_trades()
        if "pnl" in df.columns:
            return df["pnl"].sum()
        return 0.0
