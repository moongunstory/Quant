# trading/execution/live.py
"""
Live trading executor for Binance Futures.

WARNING: This uses REAL MONEY! Test thoroughly in paper mode first.
"""

from __future__ import annotations
from typing import Dict, Literal
import pandas as pd

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("[WARNING] python-binance not installed. Live trading unavailable.")
    print("Install: pip install python-binance")


class BinanceExecutor:
    """
    Execute real trades on Binance Futures.

    WARNING: This uses real money!
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = "BTCUSDT",
        leverage: int = 3,
    ):
        """
        Initialize Binance executor.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            symbol: Trading symbol
            leverage: Leverage multiplier
        """
        if not BINANCE_AVAILABLE:
            raise ImportError("python-binance not installed. Cannot execute live trades.")

        self.client = Client(api_key, api_secret)
        self.symbol = symbol
        self.leverage = leverage

        # Set leverage
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            print(f"[Binance] Leverage set to {leverage}x for {symbol}")
        except Exception as e:
            print(f"[ERROR] Failed to set leverage: {e}")

    def execute(
        self,
        action: Literal["buy", "sell", "close", "hold"],
        size: float,  # Fraction of capital
        current_price: float,
        timestamp: pd.Timestamp,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> Dict:
        """
        Execute a real trade on Binance Futures.

        WARNING: This uses REAL MONEY!

        Args:
            action: "buy", "sell", "close", or "hold"
            size: Position size (fraction of capital)
            current_price: Current price (for reference)
            timestamp: Trade timestamp
            stop_loss: Stop-loss percentage (relative)
            take_profit: Take-profit percentage (relative)

        Returns:
            Trade result dict
        """
        if action == "hold":
            return {"status": "hold"}

        # Get account balance
        balance = self.get_balance()

        if action == "buy":
            # Calculate quantity
            usd_to_use = balance * size
            quantity = (usd_to_use * self.leverage) / current_price
            quantity = self._round_quantity(quantity)

            try:
                # Place market order
                order = self.client.futures_create_order(
                    symbol=self.symbol,
                    side="BUY",
                    type="MARKET",
                    quantity=quantity
                )

                # Set stop-loss
                if stop_loss != 0.0:
                    sl_price = current_price * (1 + stop_loss)
                    sl_order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side="SELL",
                        type="STOP_MARKET",
                        stopPrice=sl_price,
                        quantity=quantity
                    )

                # Set take-profit
                if take_profit != 0.0:
                    tp_price = current_price * (1 + take_profit)
                    tp_order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side="SELL",
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=tp_price,
                        quantity=quantity
                    )

                return {
                    "status": "executed",
                    "action": "buy",
                    "order": order,
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": timestamp,
                }

            except Exception as e:
                print(f"[ERROR] Failed to execute BUY order: {e}")
                return {"status": "error", "error": str(e)}

        elif action == "sell" or action == "close":
            # Get current position
            position = self.get_position()
            if position > 0:
                try:
                    # Close long
                    order = self.client.futures_create_order(
                        symbol=self.symbol,
                        side="SELL",
                        type="MARKET",
                        quantity=position
                    )

                    return {
                        "status": "executed",
                        "action": "sell",
                        "order": order,
                        "quantity": position,
                        "timestamp": timestamp,
                    }

                except Exception as e:
                    print(f"[ERROR] Failed to execute SELL order: {e}")
                    return {"status": "error", "error": str(e)}

            else:
                return {"status": "no_position_to_close"}

        return {"status": "no_action"}

    def get_balance(self) -> float:
        """Get USDT balance."""
        try:
            account = self.client.futures_account()
            for asset in account["assets"]:
                if asset["asset"] == "USDT":
                    return float(asset["availableBalance"])
            return 0.0
        except Exception as e:
            print(f"[ERROR] Failed to get balance: {e}")
            return 0.0

    def get_position(self) -> float:
        """Get current BTC position."""
        try:
            positions = self.client.futures_position_information(symbol=self.symbol)
            for pos in positions:
                if pos["symbol"] == self.symbol:
                    return abs(float(pos["positionAmt"]))
            return 0.0
        except Exception as e:
            print(f"[ERROR] Failed to get position: {e}")
            return 0.0

    def _round_quantity(self, quantity: float) -> float:
        """Round quantity to Binance's precision (0.001 BTC)."""
        return round(quantity, 3)
