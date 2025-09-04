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
      (단위 포지션, 가격 기준 PnL/비용; 수량은 1로 가정)
    """
    # Gymnasium 표준 키: render_modes
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        fee_rate: float = 0.0004,
        slip_bp: float = 2.0,            # 1bp = 0.0001 → 2bp = 0.0002
        turn_cost: float = 0.0,
        max_position_bars: int | None = None,
        random_start: bool = False,      # 에피소드 시작을 랜덤화(기본 꺼짐: 기존 동작 유지)
    ):
        super().__init__()
        assert df.index.is_monotonic_increasing, "DataFrame index must be increasing (time-ordered)."
        self.df = df.copy()

        # 관측: f_* 만 사용 (옵션 A 일관성)
        self.obs_cols = [c for c in df.columns if c.startswith("f_")]
        assert len(self.obs_cols) > 0, "f_* 피처가 필요합니다."

        # 가격/펀딩 컬럼
        self.price_col = "price_close" if "price_close" in df.columns else "Close"
        self.funding_col = "funding_per_bar" if "funding_per_bar" in df.columns else None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.obs_cols),), dtype=np.float32
        )

        # 상태
        self.idx = df.index.to_numpy()
        self.random_start = bool(random_start)
        self.t = 0
        self.position = 0       # 0 flat, +1 long, -1 short (단위 수량)
        self.entry_price = np.nan
        self.holding = 0
        self.equity = 0.0
        self.last_price = np.nan

        # 비용 파라미터
        self.fee_rate = float(fee_rate)
        self.slip_bp = float(slip_bp) * 1e-4
        self.turn_cost = float(turn_cost)
        self.max_position_bars = max_position_bars

    # === 내부 유틸 ===
    def _price(self, t: int) -> float:
        return float(self.df.iloc[t][self.price_col])

    def _obs(self, t: int) -> np.ndarray:
        return self.df.iloc[t][self.obs_cols].to_numpy(dtype=np.float32)

    def _pnl_delta(self, new_price: float) -> float:
        # 보유 중 구간[t, t+1)의 마크투마켓 PnL (단위 수량=1)
        if self.position == 0 or np.isnan(self.last_price):
            return 0.0
        side = 1.0 if self.position > 0 else -1.0
        return side * (new_price - self.last_price)

    # ✅ 비용은 "거래금액(= 가격 × 수량)" 기준. 단위 수량 가정 → notional = price
    def _fees_on_trade(self, price: float) -> float:
        return float(abs(price) * self.fee_rate)

    def _slip_on_trade(self, price: float) -> float:
        return float(abs(price) * self.slip_bp)

    def _funding(self, t: int) -> float:
        """
        per-bar 펀딩 비용. 데이터가 +면 지불, -면 수취라고 가정.
        (기존 동작 유지: 단위·부호는 입력 컬럼에 따름)
        """
        if self.funding_col is None or self.position == 0:
            return 0.0
        return float(self.df.iloc[t][self.funding_col])

    def _reset_state(self, start_t: int):
        self.t = int(start_t)
        self.position = 0
        self.entry_price = np.nan
        self.holding = 0
        self.equity = 0.0
        self.last_price = self._price(self.t)

    # === Gym API ===
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.random_start and len(self.df) > 2:
            start_t = np.random.randint(0, len(self.df) - 2)
        else:
            start_t = 0
        self._reset_state(start_t)
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
            next_t = self.t  # stay at last index if terminal

        next_price = self._price(next_t)

        # 1) 보유 중 PnL (구간[t, t+1))
        pnl = self._pnl_delta(next_price)

        # 2) 액션 실행 (현재가 기준 체결 가정)
        fee = 0.0
        slip = 0.0
        turn_penalty = 0.0

        if action == LONG and self.position <= 0:
            # (전환 포함) 기존 숏 청산 + 롱 진입
            if self.position < 0:
                # 숏 청산 거래비용
                fee += self._fees_on_trade(cur_price)
                slip += self._slip_on_trade(cur_price)
            # 롱 진입 거래비용
            fee += self._fees_on_trade(cur_price)
            slip += self._slip_on_trade(cur_price)
            self.position = +1
            self.entry_price = cur_price
            self.holding = 0
            if action in (LONG, SHORT):
                turn_penalty = self.turn_cost

        elif action == SHORT and self.position >= 0:
            # (전환 포함) 기존 롱 청산 + 숏 진입
            if self.position > 0:
                fee += self._fees_on_trade(cur_price)
                slip += self._slip_on_trade(cur_price)
            fee += self._fees_on_trade(cur_price)
            slip += self._slip_on_trade(cur_price)
            self.position = -1
            self.entry_price = cur_price
            self.holding = 0
            if action in (LONG, SHORT):
                turn_penalty = self.turn_cost

        elif action == CLOSE and self.position != 0:
            # 포지션 정리 (단순 1회 거래)
            fee += self._fees_on_trade(cur_price)
            slip += self._slip_on_trade(cur_price)
            self.position = 0
            self.entry_price = np.nan
            self.holding = 0

        # 3) 유지 시간/펀딩(다음 시점 기준으로 계산)
        if self.position != 0:
            self.holding += 1
        funding = self._funding(next_t)

        # 4) 보상/자본
        reward = pnl - fee - slip - funding - turn_penalty
        self.equity += reward

        # 5) 시점 전진
        self.t = next_t
        self.last_price = next_price

        # 6) 최대 보유시간(옵션): 강제 청산 비용 근사
        if (self.max_position_bars is not None) and (self.max_position_bars > 0):
            if self.position != 0 and self.holding >= self.max_position_bars:
                # 다음 틱 가격으로 청산한다고 가정
                fee += self._fees_on_trade(next_price)
                slip += self._slip_on_trade(next_price)
                self.position = 0
                self.entry_price = np.nan
                self.holding = 0

        obs = self._obs(self.t)
        info = self._info(extra=dict(
            pnl=float(pnl), fee=float(fee), slip=float(slip),
            funding=float(funding), turn=float(turn_penalty),
            price=float(next_price),
        ))
        return obs, float(reward), terminated, truncated, info

    def _info(self, extra: dict | None = None):
        # 보조 라벨(Trend Head용)은 df에 있으면 전달
        lbl = self.df.iloc[self.t]["label_4h_dir"] if "label_4h_dir" in self.df.columns else None
        base = {
            "t": int(self.t),
            "equity": float(self.equity),
            "position": int(self.position),
        }
        if extra:
            base.update(extra)
        if lbl is not None and not (isinstance(lbl, float) and np.isnan(lbl)):
            base["trend_label"] = int(lbl)
        return base
