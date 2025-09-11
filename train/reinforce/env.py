import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Optional
from collections import deque
from ai_binance.train.reinforce.portfolio import Portfolio, TradeCosts

WAIT, LONG, SHORT, CLOSE = 0, 1, 2, 3

SEQ_LEN_PER_TF = {
    "5m": 24,
    "15m": 16,
    "1h": 12,
    "4h": 6,
}


class MultiTimeframeTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tf_data: Dict[str, pd.DataFrame],
        fee_rate: float = 0.0004,
        slip_bp: float = 2.0,
        turn_cost: float = 0.0,
        max_position_bars: int | None = None,
        random_start: bool = False,
        obs_cols: Optional[Dict[str, list[str]]] = None,
        price_col: Optional[str] = None,
    ):
        super().__init__()
        self.tf_data = tf_data
        self.tfs = sorted(tf_data.keys(), key=lambda x: list(SEQ_LEN_PER_TF).index(x))
        self.seq_lens = {tf: SEQ_LEN_PER_TF[tf] for tf in self.tfs}
        self.random_start = random_start
        self.max_position_bars = max_position_bars

        # Observation columns per TF
        if obs_cols is None:
            self.obs_cols = {
                tf: [c for c in df.columns if c.startswith("f_") or c.startswith("btc_")]
                for tf, df in tf_data.items()
            }
        else:
            self.obs_cols = obs_cols
            # Verify that all obs_cols exist in the dataframes
            for tf, cols in obs_cols.items():
                missing_cols = set(cols) - set(tf_data[tf].columns)
                if missing_cols:
                    raise ValueError(f"Missing columns in {tf} data: {missing_cols}")

        # Sync index
        all_indices = [df.index for df in tf_data.values()]
        from functools import reduce
        self.common_index = reduce(lambda x, y: x.intersection(y), all_indices)
        assert len(self.common_index) > max(self.seq_lens.values()), "Not enough overlapping data"

        # Clip data to common index
        self.tf_data = {
            tf: df.loc[self.common_index].copy() for tf, df in tf_data.items()
        }

        self.price_col = price_col or "Close"
        self.price_df = self.tf_data["5m"]  # base for price and step
        self.obs_buffers = {
            tf: deque(maxlen=self.seq_lens[tf]) for tf in self.tfs
        }

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            tf: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.seq_lens[tf], len(self.obs_cols[tf])),
                dtype=np.float32,
            )
            for tf in self.tfs
        })

        self.portfolio = Portfolio(initial_equity=10_000.0,
                                   costs=TradeCosts(fee_rate, slip_bp, turn_cost))
        self.t = 0

    def _get_obs(self, tf: str, t: int) -> np.ndarray:
        df = self.tf_data[tf]
        cols = self.obs_cols[tf]
        return df.iloc[t][cols].to_numpy(dtype=np.float32)

    def _get_price(self, t: int) -> float:
        return float(self.price_df.iloc[t][self.price_col])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        max_seq_len = max(self.seq_lens.values())
        data_len = len(self.common_index)
        if data_len <= max_seq_len:
            raise ValueError("Data too short for required sequence lengths")

        self.t = np.random.randint(0, data_len - max_seq_len) if self.random_start else 0
        self.portfolio.reset()

        for tf in self.tfs:
            self.obs_buffers[tf].clear()
            for i in range(self.seq_lens[tf]):
                self.obs_buffers[tf].append(self._get_obs(tf, self.t + i))

        obs = {tf: np.stack(self.obs_buffers[tf], axis=0) for tf in self.tfs}
        return obs, self._info()

    def step(self, action: int):
        cur_price = self._get_price(self.t)
        next_t = min(self.t + 1, len(self.common_index) - 1)
        next_price = self._get_price(next_t)
        prev_equity = self.portfolio.equity

        if action == LONG and self.portfolio.position <= 0:
            self.portfolio.close_position(cur_price)
            self.portfolio.open_position(cur_price, +1)
        elif action == SHORT and self.portfolio.position >= 0:
            self.portfolio.close_position(cur_price)
            self.portfolio.open_position(cur_price, -1)
        elif action == CLOSE and self.portfolio.position != 0:
            self.portfolio.close_position(cur_price)

        self.portfolio.step(next_price, funding=0.0)
        reward = self.portfolio.get_reward(prev_equity)

        self.t = next_t
        done = (next_t == len(self.common_index) - 1)
        truncated = False

        if self.max_position_bars and self.portfolio.position != 0 and self.portfolio.holding >= self.max_position_bars:
            self.portfolio.close_position(next_price)

        for tf in self.tfs:
            self.obs_buffers[tf].append(self._get_obs(tf, self.t))

        obs = {tf: np.stack(self.obs_buffers[tf], axis=0) for tf in self.tfs}
        return obs, reward, done, truncated, self._info()

    def _info(self):
        price = self._get_price(self.t)
        info = self.portfolio.info(price)
        info.update({
            "t": self.t,
            "price": price,
            "action_mask": self._action_mask(),
        })
        return info

    def _action_mask(self):
        mask = np.ones(4, dtype=bool)
        if self.portfolio.position > 0:
            mask[LONG] = False
        elif self.portfolio.position < 0:
            mask[SHORT] = False
        else:
            mask[CLOSE] = False
        return mask

    @property
    def position(self) -> int:
        return self.portfolio.position
