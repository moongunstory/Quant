# ai_binance/train/reinforce/env.py
from __future__ import annotations
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

WAIT, LONG, SHORT, CLOSE = 0, 1, 2, 3

class TradingEnv(gym.Env):
    """
    단일-모델 PPO용 환경 (타이밍 보상 전용).
    - 관측: fe 병합 결과의 f_* 컬럼을 concat한 1D 벡터
    - 액션: {대기, 진입롱, 진입숏, 청산}
    - 보상: pnl - fee - slip - turn_cost - funding
    """
    metadata = {"render.modes": []}

    def __init__(self, df: pd.DataFrame,
                 fee_rate: float = 0.0004,
                 slip_bp: float = 2.0,            # 1bp = 0.0001
                 turn_cost: float = 0.0,
                 max_position_bars: int | None = None):
        super().__init__()
        assert df.index.is_monotonic_increasing
        self.df = df.copy()
        self.obs_cols = [c for c in df.columns if c.startswith("f_")]
        assert len(self.obs_cols) > 0, "f_* 피처가 필요합니다."
        self.price_col = "price_close"
        self.funding_col = "funding_per_bar" if "funding_per_bar" in df.columns else None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.obs_cols),), dtype=np.float32
        )

        # 상태
        self.t = 0
        self.idx = df.index.to_numpy()
        self.position = 0       # 0 flat, +1 long, -1 short
        self.entry_price = np.nan
        self.holding = 0
        self.equity = 0.0

        # 비용 파라미터
        self.fee_rate = fee_rate
        self.slip_bp = slip_bp * 1e-4
        self.turn_cost = turn_cost
        self.max_position_bars = max_position_bars

    # === 내부 유틸 ===
    def _price(self, t):
        return float(self.df.iloc[t][self.price_col])

    def _obs(self, t):
        return self.df.iloc[t][self.obs_cols].astype(np.float32).to_numpy()

    def _pnl_delta(self, new_price):
        if self.position == 0 or np.isnan(self.entry_price):
            return 0.0
        side = 1.0 if self.position > 0 else -1.0
        return side * (new_price - self.last_price)

    def _fees(self, notional):
        return abs(notional) * self.fee_rate

    def _slip(self, notional):
        return abs(notional) * self.slip_bp

    def _funding(self, t):
        if self.funding_col is None:
            return 0.0
        return float(self.df.iloc[t][self.funding_col]) * (1.0 if self.position != 0 else 0.0)

    # === Gym API ===
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.position = 0
        self.entry_price = np.nan
        self.holding = 0
        self.equity = 0.0
        self.last_price = self._price(self.t)
        obs = self._obs(self.t)
        info = self._info()
        return obs, info

    def step(self, action: int):
        done = False
        truncated = False

        cur_price = self._price(self.t)
        next_t = self.t + 1
        if next_t >= len(self.df):
            done = True
            next_t = self.t  # stay

        next_price = self._price(next_t)

        # 기본 PnL 변화(보유 중 마크투마켓)
        pnl = self._pnl_delta(next_price)

        # 액션 실행
        fee = slip = 0.0
        if action == LONG and self.position <= 0:
            # 포지션 열기/전환
            notional_close = 0.0
            if self.position < 0:  # 숏 청산 비용
                notional_close = abs(self.entry_price - cur_price)
                fee += self._fees(notional_close)
                slip += self._slip(notional_close)
            self.position = +1
            self.entry_price = cur_price
            self.holding = 0
            fee += self._fees(cur_price)
            slip += self._slip(cur_price)

        elif action == SHORT and self.position >= 0:
            notional_close = 0.0
            if self.position > 0:
                notional_close = abs(self.entry_price - cur_price)
                fee += self._fees(notional_close)
                slip += self._slip(notional_close)
            self.position = -1
            self.entry_price = cur_price
            self.holding = 0
            fee += self._fees(cur_price)
            slip += self._slip(cur_price)

        elif action == CLOSE and self.position != 0:
            notional_close = abs(self.entry_price - cur_price)
            fee += self._fees(notional_close)
            slip += self._slip(notional_close)
            # 포지션 정리
            self.position = 0
            self.entry_price = np.nan
            self.holding = 0

        # 유지 시간/펀딩
        self.holding += 1 if self.position != 0 else 0
        funding = self._funding(next_t)

        # 전환 비용(옵션)
        turn_penalty = self.turn_cost if action in (LONG, SHORT) else 0.0

        reward = pnl - fee - slip - funding - turn_penalty
        self.equity += reward

        self.t = next_t
        self.last_price = next_price

        # 최대 보유시간(옵션)
        if self.max_position_bars and self.holding >= self.max_position_bars:
            if self.position != 0:
                # 타임스탑: 강제 청산 비용 근사
                notional_close = abs(self.entry_price - next_price)
                reward -= self._fees(notional_close) + self._slip(notional_close)
                self.position = 0
                self.entry_price = np.nan
                self.holding = 0

        obs = self._obs(self.t)
        info = self._info()
        return obs, float(reward), done, truncated, info

    def _info(self):
        # 보조 라벨(Trend Head용)은 df에 있으면 전달
        lbl = self.df.iloc[self.t]["label_4h_dir"] if "label_4h_dir" in self.df.columns else None
        return {
            "t": int(self.t),
            "equity": float(self.equity),
            "position": int(self.position),
            **({"trend_label": int(lbl)} if lbl is not None and not np.isnan(lbl) else {})
        }
