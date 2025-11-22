# trading/strategy.py
"""
Trading strategy logic.
"""

from __future__ import annotations
from typing import Dict, Tuple, Literal

from .config import TradingConfig


class SimpleStrategy:
    """
    Simple strategy based on predicted returns.

    Logic:
    - If predicted return > threshold: Long
    - If predicted return < -threshold: Short
    - Otherwise: Close position / Hold
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def generate_signal(
        self,
        predictions: Dict[int, float],  # {horizon_hours: predicted_return}
        current_position: float = 0.0,  # Current position (+ for long, - for short)
    ) -> Tuple[Literal["buy", "sell", "close", "hold"], float, float, float]:
        """
        Generate trading signal based on predictions.

        Args:
            predictions: Dict mapping horizon_hours to predicted return
            current_position: Current position size (BTC amount)

        Returns:
            action: "buy", "sell", "close", or "hold"
            size: Position size (as fraction of capital)
            stop_loss: Stop-loss percentage (relative to entry price)
            take_profit: Take-profit percentage (relative to entry price)
        """
        # Use shortest horizon for fastest reaction (e.g., 3-day = 72h)
        horizon_hours = min(predictions.keys())
        pred_return = predictions[horizon_hours]

        # Long signal: predicted return > threshold
        if pred_return > self.config.min_prediction_return:
            action = "buy"
            size = self.config.position_size_pct
            stop_loss = -self.config.stop_loss_pct
            take_profit = self.config.take_profit_pct

        # Short signal: predicted return < -threshold
        elif pred_return < -self.config.min_prediction_return:
            action = "sell"
            size = self.config.position_size_pct
            # For short: stop-loss and take-profit are reversed
            stop_loss = self.config.stop_loss_pct
            take_profit = -self.config.take_profit_pct

        # Weak signal: close position if open, otherwise hold
        else:
            if abs(current_position) > 0:
                action = "close"
                size = abs(current_position)
                stop_loss = 0.0
                take_profit = 0.0
            else:
                action = "hold"
                size = 0.0
                stop_loss = 0.0
                take_profit = 0.0

        return action, size, stop_loss, take_profit
