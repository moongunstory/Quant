# trading/execution/paper.py
"""
Paper trading simulator (virtual money).
"""

from __future__ import annotations
from typing import Dict, List, Literal
import pandas as pd


class PaperTrader:
    """
    Simulate trading without real money.

    Tracks:
    - Virtual capital (USDT)
    - Virtual position (BTC amount)
    - Entry price
    - Trade history
    - Equity curve
    """

    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize paper trader.

        Args:
            initial_capital: Starting capital in USDT
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0.0  # BTC amount (+ for long, - for short)
        self.entry_price = 0.0
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

    def execute(
        self,
        action: Literal["buy", "sell", "close", "hold"],
        size: float,  # Fraction of capital (for buy/sell) or BTC amount (for close)
        current_price: float,
        timestamp: pd.Timestamp,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> Dict:
        """
        Execute a paper trade.

        Args:
            action: "buy", "sell", "close", or "hold"
            size: Position size (fraction of capital or BTC amount)
            current_price: Current BTC price in USDT
            timestamp: Trade timestamp
            stop_loss: Stop-loss percentage (relative to entry)
            take_profit: Take-profit percentage (relative to entry)

        Returns:
            Trade result dict
        """
        if action == "hold":
            return {"status": "hold", "capital": self.capital, "position": self.position}

        if action == "buy":
            # Calculate BTC amount to buy
            usd_to_use = self.capital * size
            btc_amount = usd_to_use / current_price

            # Deduct from capital
            self.capital -= usd_to_use
            self.position += btc_amount
            self.entry_price = current_price

            trade = {
                "timestamp": timestamp,
                "action": "buy",
                "price": current_price,
                "btc_amount": btc_amount,
                "usd_amount": usd_to_use,
                "capital_after": self.capital,
                "position_after": self.position,
                "pnl": 0.0,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
            self.trades.append(trade)
            return trade

        elif action == "sell" or action == "close":
            # Close long position
            if self.position > 0:
                usd_received = self.position * current_price
                pnl = (current_price - self.entry_price) * self.position

                trade = {
                    "timestamp": timestamp,
                    "action": "sell",
                    "price": current_price,
                    "btc_amount": self.position,
                    "usd_amount": usd_received,
                    "pnl": pnl,
                    "capital_before": self.capital,
                }

                self.capital += usd_received
                self.position = 0.0
                self.entry_price = 0.0

                trade["capital_after"] = self.capital
                trade["position_after"] = self.position

                self.trades.append(trade)
                return trade

            else:
                return {"status": "no_position_to_close"}

        return {"status": "no_action"}

    def get_equity(self, current_price: float) -> float:
        """
        Total equity = cash + position value.

        Args:
            current_price: Current BTC price

        Returns:
            Total equity in USDT
        """
        position_value = self.position * current_price
        return self.capital + position_value

    def get_pnl(self, current_price: float) -> float:
        """
        Profit/Loss vs initial capital.

        Args:
            current_price: Current BTC price

        Returns:
            P&L in USDT
        """
        return self.get_equity(current_price) - self.initial_capital

    def get_return(self, current_price: float) -> float:
        """
        Return percentage.

        Args:
            current_price: Current BTC price

        Returns:
            Return as decimal (e.g., 0.15 = 15%)
        """
        return self.get_pnl(current_price) / self.initial_capital

    def get_trade_summary(self) -> Dict:
        """
        Get summary of all trades.

        Returns:
            Dict with trade statistics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
            }

        trades_with_pnl = [t for t in self.trades if "pnl" in t]
        total_pnl = sum(t["pnl"] for t in trades_with_pnl)
        winning = sum(1 for t in trades_with_pnl if t["pnl"] > 0)
        losing = sum(1 for t in trades_with_pnl if t["pnl"] < 0)

        return {
            "total_trades": len(trades_with_pnl),
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": winning / len(trades_with_pnl) if trades_with_pnl else 0.0,
            "total_pnl": total_pnl,
        }
