# ai_binance/train/reinforce/portfolio.py

from dataclasses import dataclass
import numpy as np

@dataclass
class TradeCosts:
    fee_rate: float = 0.0004
    slip_bp: float = 2.0  # 1bp = 0.0001 → 2bp = 0.0002
    turn_cost: float = 0.0

    def fee(self, price: float, size: float) -> float:
        return abs(price * size) * self.fee_rate

    def slippage(self, price: float, size: float) -> float:
        return abs(price * size) * self.slip_bp * 1e-4

class Portfolio:
    def __init__(self, initial_equity: float, costs: TradeCosts):
        self.initial_equity = initial_equity
        self.costs = costs
        self.reset()

    def reset(self):
        self.equity = self.initial_equity
        self.cash = self.initial_equity
        self.position = 0  # 0: flat, 1: long, -1: short
        self.position_size = 0.0
        self.entry_price = np.nan
        self.holding = 0

    def _apply_costs(self, price: float, size: float) -> float:
        return self.costs.fee(price, size) + self.costs.slippage(price, size)

    def open_position(self, price: float, direction: int):
        assert self.position == 0
        size = self.cash / price
        cost = self._apply_costs(price, size)
        self.cash -= size * price
        self.equity -= cost
        self.position = direction
        self.position_size = size
        self.entry_price = price
        self.holding = 0

    def close_position(self, price: float):
        if self.position == 0:
            return
        pnl = self.position * self.position_size * price
        cost = self._apply_costs(price, self.position_size)
        self.cash += pnl
        self.equity = self.cash - cost
        self.position = 0
        self.position_size = 0.0
        self.entry_price = np.nan
        self.holding = 0

    def step(self, price: float, funding: float = 0.0):
        if self.position != 0:
            pos_val = self.position * self.position_size * price
            self.equity = self.cash + pos_val - funding
            self.holding += 1
        else:
            self.equity = self.cash

    def get_reward(self):
        return (self.equity - self.initial_equity) / self.initial_equity

    def info(self, price: float):
        pos_val = self.position * self.position_size * price
        return {
            "equity": self.equity,
            "cash": self.cash,
            "position": self.position,
            "size": self.position_size,
            "entry_price": self.entry_price if not np.isnan(self.entry_price) else None,
            "position_value": pos_val,
            "holding": self.holding
        }
