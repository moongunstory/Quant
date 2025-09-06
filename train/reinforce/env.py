# ai_binance/train/reinforce/env.py

from __future__ import annotations
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import List, Optional, Tuple
from .portfolio import Portfolio, TradeCosts

WAIT, LONG, SHORT, CLOSE = 0, 1, 2, 3

class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        fee_rate: float = 0.0004,
        slip_bp: float = 2.0,
        turn_cost: float = 0.0,
        max_position_bars: int | None = None,
        random_start: bool = False,
        obs_cols: Optional[List[str]] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        super().__init__()
        assert df.index.is_monotonic_increasing

        self._full_df = df.copy()
        if start_idx is None: start_idx = 0
        if end_idx is None: end_idx = len(self._full_df)
        self._window = (start_idx, end_idx)
        self.df = self._full_df.iloc[start_idx:end_idx].copy()

        self._set_obs_cols(obs_cols)

        self.price_col = "price_close" if "price_close" in self.df.columns else "Close"
        self.funding_col = "funding_per_bar" if "funding_per_bar" in self.df.columns else None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.obs_cols),), dtype=np.float32)

        self.idx = self.df.index.to_numpy()
        self.random_start = random_start
        self.t = 0
        self.max_position_bars = max_position_bars

        costs = TradeCosts(fee_rate, slip_bp, turn_cost)
        self.portfolio = Portfolio(initial_equity=10_000.0, costs=costs)

    def _set_obs_cols(self, obs_cols: Optional[List[str]]):
        if obs_cols is None:
            cols = [c for c in self._full_df.columns if c.startswith("f_")]
        else:
            cols = obs_cols
        assert cols, "Observation columns must not be empty."
        self.obs_cols = cols

    def _price(self, t: int) -> float:
        return float(self.df.iloc[t][self.price_col])

    def _obs(self, t: int) -> np.ndarray:
        return self.df.iloc[t][self.obs_cols].to_numpy(dtype=np.float32)

    def _funding(self, t: int) -> float:
        if self.funding_col is None or self.portfolio.position == 0:
            return 0.0
        return float(self.df.iloc[t][self.funding_col])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = np.random.randint(0, len(self.df) - 2) if self.random_start else 0
        self.portfolio.reset()
        obs = self._obs(self.t)
        info = self._info()
        return obs, info

    def step(self, action: int):
        terminated = False
        truncated = False
        cur_price = self._price(self.t)
        next_t = self.t + 1
        if next_t >= len(self.df):
            terminated = True
            next_t = self.t
        next_price = self._price(next_t)

        # === 액션 처리 ===
        if action == LONG and self.portfolio.position <= 0:
            self.portfolio.close_position(cur_price)
            self.portfolio.open_position(cur_price, direction=+1)

        elif action == SHORT and self.portfolio.position >= 0:
            self.portfolio.close_position(cur_price)
            self.portfolio.open_position(cur_price, direction=-1)

        elif action == CLOSE and self.portfolio.position != 0:
            self.portfolio.close_position(cur_price)

        # === 펀딩 및 평가 ===
        funding = self._funding(next_t)
        self.portfolio.step(next_price, funding)

        reward = self.portfolio.get_reward()

        if abs(reward) > 0.5:
            print(f"[anomaly] t={self.t} action={action} reward={reward:.6f}")
            print(self.portfolio.info(next_price))

        self.t = next_t

        # === 보유 최대 바 초과 시 강제 청산 ===
        if self.max_position_bars and self.portfolio.position != 0 and self.portfolio.holding >= self.max_position_bars:
            self.portfolio.close_position(next_price)

        obs = self._obs(self.t)
        info = self._info(extra=dict(funding=funding))
        return obs, reward, terminated, truncated, info

    def _info(self, extra: dict | None = None):
        price = self._price(self.t)
        base_info = self.portfolio.info(price)
        base_info.update({
            "t": int(self.t),
            "price": float(price),
            "obs_dim": len(self.obs_cols),
            "window_start": int(self._window[0]),
            "window_end": int(self._window[1]),
            "initial_equity": float(self.portfolio.initial_equity),
        })
        if extra:
            base_info.update(extra)
        return base_info
