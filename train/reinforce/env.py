# ai_binance/train/reinforce/env.py
from __future__ import annotations
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import List, Optional, Tuple

WAIT, LONG, SHORT, CLOSE = 0, 1, 2, 3

class TradingEnv(gym.Env):
    """
    단일-모델 PPO용 환경 (타이밍 보상 전용).

    관측(Observation):
      - 기본: df의 f_* 접두 컬럼을 사용
      - 선택: obs_cols 인자로 피처 목록을 명시하면 그 순서대로 사용 (가변 차원)

    액션(Action): {WAIT(0), LONG(1), SHORT(2), CLOSE(3)}

    보상(Reward):
      - 포트폴리오 가치 증감 Δequity = (cash + position_value_next) - (기존 equity) - 비용들
      - 비용: fee + slip + funding + turn_cost

    HPO 친화 기능:
      - start_idx / end_idx 로 평가 구간을 손쉽게 슬라이스 (롤링 윈도우)
      - obs_cols 주입으로 피처 subset 변경 시 재사용 용이
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        fee_rate: float = 0.0004,
        slip_bp: float = 2.0,            # 1bp = 0.0001 → 2bp = 0.0002
        turn_cost: float = 0.0,
        max_position_bars: int | None = None,
        random_start: bool = False,      # 에피소드 시작을 랜덤화(기본 꺼짐)
        # HPO/가변 관측 지원
        obs_cols: Optional[List[str]] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ):
        super().__init__()
        assert df.index.is_monotonic_increasing, "DataFrame index must be increasing (time-ordered)."

        # 전체 프레임 보관 + 평가 구간 슬라이스
        self._full_df = df.copy()
        n_total = len(self._full_df)
        if start_idx is None: start_idx = 0
        if end_idx   is None: end_idx   = n_total
        assert 0 <= start_idx < end_idx <= n_total, f"Invalid window: [{start_idx}, {end_idx}) out of [0, {n_total})"
        self._window: Tuple[int,int] = (start_idx, end_idx)
        self.df = self._full_df.iloc[start_idx:end_idx].copy()

        # 관측 컬럼 설정 (없으면 f_* 자동 탐색)
        self._set_obs_cols(obs_cols)

        # 가격/펀딩 컬럼
        self.price_col = "price_close" if "price_close" in self.df.columns else "Close"
        # NOTE: processed에는 'FundingRate'가 일반적이며 per-bar funding은 없을 수 있음
        self.funding_col = "funding_per_bar" if "funding_per_bar" in self.df.columns else None

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.obs_cols),), dtype=np.float32
        )

        # 상태
        self.idx = self.df.index.to_numpy()
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
    def _set_obs_cols(self, obs_cols: Optional[List[str]]):
        """관측 피처 목록을 설정하고 유효성 검사."""
        if obs_cols is None:
            cols = [c for c in self._full_df.columns if c.startswith("f_")]
        else:
            # 제공된 obs_cols 순서를 그대로 유지
            missing = [c for c in obs_cols if c not in self._full_df.columns]
            if missing:
                raise ValueError(f"obs_cols not found in DataFrame: {missing[:5]}{'...' if len(missing)>5 else ''}")
            cols = list(obs_cols)
        if len(cols) == 0:
            raise AssertionError("At least one observation feature (f_*) is required.")
        self.obs_cols: List[str] = cols

    def _price(self, t: int) -> float:
        return float(self.df.iloc[t][self.price_col])

    def _obs(self, t: int) -> np.ndarray:
        # 선택된 obs_cols 순서를 보장
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
        (입력 컬럼 단위/부호는 데이터에 따름)
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

    # === 공개 유틸 (HPO 편의) ===
    def set_window(self, start_idx: int, end_idx: int):
        """평가 구간을 변경 (슬라이스)하고 observation_space를 재설정."""
        n_total = len(self._full_df)
        assert 0 <= start_idx < end_idx <= n_total, f"Invalid window: [{start_idx}, {end_idx}) out of [0, {n_total})"
        self._window = (start_idx, end_idx)
        self.df = self._full_df.iloc[start_idx:end_idx].copy()
        # obs_cols는 동일 집합이 df에도 존재해야 함
        missing = [c for c in self.obs_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Selected obs_cols missing in new window df: {missing[:5]}{'...' if len(missing)>5 else ''}")
        # 관측공간 shape 갱신
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.obs_cols),), dtype=np.float32)
        # 인덱스 & 상태 재설정
        self.idx = self.df.index.to_numpy()
        self.reset()

    def select_features(self, obs_cols: List[str]):
        """관측 피처 subset/순서를 변경하고 observation_space를 갱신."""
        self._set_obs_cols(obs_cols)
        # 관측공간 shape 갱신
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.obs_cols),), dtype=np.float32)
        # 현재 스텝의 관측 차원이 바뀌므로 안전하게 reset
        self.reset()

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

        # 거래 관련 비용
        fee = 0.0
        slip = 0.0
        turn_penalty = 0.0

        # 현재 포지션 가치
        position_value = self.position * cur_price
        cash = self.equity - position_value  # equity = cash + position_value

        if action == LONG and self.position <= 0:
            # (전환 포함) 기존 숏 청산 + 롱 진입
            if self.position < 0:
                fee += self._fees_on_trade(cur_price)
                slip += self._slip_on_trade(cur_price)
            fee += self._fees_on_trade(cur_price)
            slip += self._slip_on_trade(cur_price)
            self.position = +1
            self.entry_price = cur_price
            self.holding = 0
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
            turn_penalty = self.turn_cost

        elif action == CLOSE and self.position != 0:
            # 포지션 정리 (단순 1회 거래)
            fee += self._fees_on_trade(cur_price)
            slip += self._slip_on_trade(cur_price)
            self.position = 0
            self.entry_price = np.nan
            self.holding = 0

        # 유지 시간/펀딩(다음 시점 기준으로 계산)
        if self.position != 0:
            self.holding += 1
        funding = self._funding(next_t)

        # 다음 시점의 자산 평가
        new_position_value = self.position * next_price
        new_equity = cash + new_position_value - fee - slip - funding - turn_penalty
        reward = new_equity - self.equity
        self.equity = new_equity

        # 시점 전진
        self.t = next_t
        self.last_price = next_price

        # 최대 보유시간(옵션): 강제 청산 (비용은 info로만 노출; 보상에는 반영하지 않음 - 기존 동작 유지)
        if (self.max_position_bars is not None) and (self.max_position_bars > 0):
            if self.position != 0 and self.holding >= self.max_position_bars:
                fee += self._fees_on_trade(next_price)
                slip += self._slip_on_trade(next_price)
                self.position = 0
                self.entry_price = np.nan
                self.holding = 0

        obs = self._obs(self.t)
        info = self._info(extra=dict(
            fee=float(fee), slip=float(slip),
            funding=float(funding), turn=float(turn_penalty),
            price=float(next_price),
            obs_dim=int(len(self.obs_cols)),
            window_start=int(self._window[0]),
            window_end=int(self._window[1]),
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
