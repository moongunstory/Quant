import pandas as pd
import numpy as np
from modules.config import TP_THRESHOLD, SL_THRESHOLD, LABEL_HORIZON

class LivePPOEnv:
    def __init__(self, market_df: pd.DataFrame, seq_len: int = 32):
        """
        market_df: OHLCV 포함 데이터프레임 (타임프레임 자유)
        """
        self.df = market_df.copy()
        self.seq_len = seq_len
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        end = self.current_step + 1
        start = end - self.seq_len
        obs = self.df.iloc[start:end]
        return obs.values  # shape: (seq_len, feature_dim)

    def step(self, action: str):
        start_idx = self.current_step + 1
        end_idx = start_idx + LABEL_HORIZON

        if end_idx >= len(self.df):
            return self._get_observation(), 0, True, {}

        entry_price = self.df['close'].iloc[self.current_step]
        future = self.df.iloc[start_idx:end_idx]
        highs = future['high'].values
        lows = future['low'].values

        tp_hit = False
        sl_hit = False

        if action == 'long':
            for i, (h, l) in enumerate(zip(highs, lows)):
                if (h - entry_price) / entry_price >= TP_THRESHOLD:
                    tp_hit = True
                    break
                if (l - entry_price) / entry_price <= SL_THRESHOLD:
                    sl_hit = True
                    break
            reward = 1 if tp_hit else -1 if sl_hit else 0
            done = tp_hit or sl_hit or (i == LABEL_HORIZON - 1)

        elif action == 'short':
            for i, (l, h) in enumerate(zip(lows, highs)):
                if (entry_price - l) / entry_price >= TP_THRESHOLD:
                    tp_hit = True
                    break
                if (entry_price - h) / entry_price <= SL_THRESHOLD:
                    sl_hit = True
                    break
            reward = 1 if tp_hit else -1 if sl_hit else 0
            done = tp_hit or sl_hit or (i == LABEL_HORIZON - 1)

        elif action == 'hold':
            reward = 0  # HOLD는 중립적 행동으로 보상 없음
            done = True  # HOLD는 즉시 종료

        else:
            raise ValueError(f"Unknown action: {action}")

        self.current_step += 1
        obs = self._get_observation()
        return obs, reward, done, {}
