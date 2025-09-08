# ai_binance/train/reinforce/env.py

from __future__ import annotations
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import List, Optional, Tuple
from collections import deque
from ai_binance.train.reinforce.portfolio import Portfolio, TradeCosts

WAIT, LONG, SHORT, CLOSE = 0, 1, 2, 3

SEQ_LEN_PER_TF = {
    "5m": 24,
    "15m": 16,
    "1h": 12,
    "4h": 6,
}

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
        price_col: Optional[str] = None,
        tf_name="5m",
    ):
        super().__init__()
        assert df.index.is_monotonic_increasing
        self._full_df = df.copy()
        if start_idx is None: start_idx = 0
        if end_idx is None: end_idx = len(self._full_df)
        self._window = (start_idx, end_idx)
        self.df = self._full_df.iloc[start_idx:end_idx].copy()
        self._set_obs_cols(obs_cols)

        self.price_col = price_col or (
            "price_close" if "price_close" in self.df.columns else (
                "Close" if "Close" in self.df.columns else self.df.columns[0]
            )
        )
        self.funding_col = "funding_per_bar" if "funding_per_bar" in self.df.columns else None

        self.tf_name = tf_name
        self.seq_len = SEQ_LEN_PER_TF.get(tf_name, 16)
        self.obs_buffer = deque(maxlen=self.seq_len)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.seq_len, len(self.obs_cols)),
            dtype=np.float32
        )

        self.random_start = random_start
        self.max_position_bars = max_position_bars

        costs = TradeCosts(fee_rate, slip_bp, turn_cost)
        self.portfolio = Portfolio(initial_equity=10_000.0, costs=costs)

        self.t = 0

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
        if self.portfolio.position == 0:
            return 0.0

        timestamp = self.df.index[t]

        # 정산 시각인지 확인 (8시간마다 정각)
        if (timestamp.hour % 8 == 0) and (timestamp.minute == 0):
            # funding_rate 컬럼 이름 유연하게 처리
            for col_name in ["funding_rate", "FundingRate"]:
                if col_name in self.df.columns:
                    rate = float(self.df.iloc[t][col_name])
                    return self.portfolio.position * rate

        return 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        if len(self.df) <= self.seq_len:
            raise ValueError(
                f"DataFrame length ({len(self.df)}) must be greater than sequence length ({self.seq_len})."
            )

        self.t = np.random.randint(0, len(self.df) - self.seq_len + 1) if self.random_start else 0
        self.portfolio.reset()

        self.obs_buffer.clear()
        for i in range(self.seq_len):
            self.obs_buffer.append(self._obs(self.t + i))

        obs = np.stack(self.obs_buffer, axis=0)
        info = self._info()
        return obs, info

    def step(self, action: int):
        cur_price = self._price(self.t)
        next_t = min(self.t + 1, len(self.df) - 1)
        next_price = self._price(next_t)
        prev_equity = self.portfolio.equity

        # Action handling
        if action == LONG and self.portfolio.position <= 0:
            self.portfolio.close_position(cur_price)
            self.portfolio.open_position(cur_price, direction=+1)
        elif action == SHORT and self.portfolio.position >= 0:
            self.portfolio.close_position(cur_price)
            self.portfolio.open_position(cur_price, direction=-1)
        elif action == CLOSE and self.portfolio.position != 0:
            self.portfolio.close_position(cur_price)

        funding = self._funding(next_t)
        self.portfolio.step(next_price, funding)
        reward = self.portfolio.get_reward(prev_equity)

        terminated = (next_t == len(self.df) - 1)
        truncated = False

        if abs(reward) > 0.5:
            print(f"[anomaly] t={self.t} action={action} reward={reward:.6f}")
            print(self.portfolio.info(next_price))

        self.t = next_t

        if self.max_position_bars and self.portfolio.position != 0 and self.portfolio.holding >= self.max_position_bars:
            self.portfolio.close_position(next_price)

        # append next observation to buffer
        self.obs_buffer.append(self._obs(self.t))
        obs = np.stack(self.obs_buffer, axis=0)
        info = self._info(extra=dict(funding=funding))
        return obs, reward, terminated, truncated, info

    def _action_mask(self) -> np.ndarray:
        mask = np.ones(4, dtype=bool)
        if self.portfolio.position > 0:
            mask[LONG] = False
        elif self.portfolio.position < 0:
            mask[SHORT] = False
        else:
            mask[CLOSE] = False
        return mask

    def _info(self, extra: dict | None = None):
        price = self._price(self.t)
        base = self.portfolio.info(price)
        base.update({
            "t": int(self.t),
            "price": float(price),
            "obs_dim": len(self.obs_cols),
            "window_start": int(self._window[0]),
            "window_end": int(self._window[1]),
            "initial_equity": float(self.portfolio.initial_equity),
            "action_mask": self._action_mask(),  # ✅ 추가된 부분
        })
        if extra:
            base.update(extra)
        return base

    @property
    def position(self) -> int:
        return self.portfolio.position