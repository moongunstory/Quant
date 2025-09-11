# ai_binance/train/reinforce/portfolio.py

from dataclasses import dataclass
import numpy as np

@dataclass
class TradeCosts:
    fee_rate: float = 0.0004
    slip_bp: float = 2.0  # basis points
    turn_cost: float = 0.0

    def fee(self, price: float, size: float) -> float:
        return abs(price * size) * self.fee_rate

    def slippage(self, price: float, size: float) -> float:
        return abs(price * size) * (self.slip_bp * 1e-4)

class Portfolio:
    def __init__(self, initial_equity: float, costs: TradeCosts):
        self.initial_equity = initial_equity
        self.costs = costs
        self.reset()

    def reset(self):
        self.equity = self.initial_equity
        self.cash = self.initial_equity
        self.position = 0          # +1 long, -1 short, 0 flat
        self.position_size = 0.0
        self.entry_price = np.nan
        self.holding = 0

    def _apply_costs(self, price: float, size: float) -> float:
        return self.costs.fee(price, size) + self.costs.slippage(price, size)

    def open_position(self, price: float, direction: int):
        size = abs(self.cash / price)
        cost = self._apply_costs(price, size)
        self.cash -= cost
        
        # LONG
        if direction == 1:
            self.cash -= size * price
        # SHORT
        elif direction == -1:
            self.cash += size * price

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
        self.cash -= cost
        self.position = 0
        self.position_size = 0.0
        self.entry_price = np.nan
        self.holding = 0

    def step(self, price: float, funding: float = 0.0):
        if self.position != 0:
            self.cash -= funding
            pos_val = self.position * self.position_size * price
            self.equity = self.cash + pos_val
            self.holding = int(self.holding) + 1
        else:
            self.equity = self.cash
            self.holding = 0

    def get_reward(self, prev_equity: float) -> float:
        if prev_equity <= 0:
            return 0.0
        return (self.equity - prev_equity) / prev_equity

    def info(self, price: float):
        pos_val = self.position * self.position_size * price
        return {
            "equity": float(self.equity),
            "cash": float(self.cash),
            "position": int(self.position),
            "size": float(self.position_size),
            "entry_price": float(self.entry_price) if not np.isnan(self.entry_price) else None,
            "position_value": float(pos_val),
            "holding": int(self.holding),
        }
