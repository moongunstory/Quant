# train/reinforce/core/crypto_trading_env.py

from __future__ import annotations

import numpy as np
import gymnasium as gym
from collections import deque

from ai_binance.train.reinforce.config import EnvConfig

# ---- simple rolling buffer over steps ----
class Rolling:
    def __init__(self, n=1000): self.buf = deque(maxlen=n)
    def add(self, x): self.buf.append(float(x))
    def mean(self): return float(np.mean(self.buf)) if self.buf else 0.0
    def sum(self):  return float(np.sum(self.buf)) if self.buf else 0.0
    def count(self): return len(self.buf)

class CryptoTradingEnv:
    """
    행동: {-1, 0, +1} (숏/관망/롱) — 올진입/올청산만 허용
    보상/Equity: per-step '순' 수익률(가격 + 펀딩 ± 수수료/패널티)을 곱셈으로 PV에 즉시 반영
    TP/SL: 바의 H/L 관통 시 즉시 청산(보수적 체결가)
    수수료: 진입/청산/플립에 대해 taker_fee 고정(곱셈 반영)
    텐서보드 지표(Trade/*):
        trades_per_1k, tp_hit_rate_1k, sl_hit_rate_1k, forced_exit_rate_1k,
        avg_R, holding_bars_mean, funding_cost_mean(부호 포함), reward_mean,
        equity(순자산, NET), turnover, action_flat_rate_1k, action_flip_rate_1k
    """
    def __init__(
        self,
        data: dict,
        seq_lens: dict,
        maker_fee=0.0002,
        taker_fee=0.0005,
        take_profit_pct: float | None = None,
        stop_loss_pct: float | None = None,
        # OHLCV 인덱스
        ohlcv_close_idx: int | None = None,
        ohlcv_high_idx: int | None = None,
        ohlcv_low_idx: int | None = None,
        enforce_hl: bool = True,   # True면 HL 미지정 시 에러
        # funding 첫 컬럼 인덱스
        funding_col_idx=0,
        # (옵션) 무거래 방지용 '조건부' 기회 패널티 — 기본 OFF
        use_idle_penalty=False,
        idle_kappa=0.7,
        idle_lambda=0.0010,
        ewma_beta=0.9,
        # (옵션) 채터링 억제
        min_hold_bars: int | None = None,    # 최소 보유 바(0이면 비활성)
        flip_penalty: float | None = None,  # 플립 시 추가 패널티(곱셈 반영, 예: 0.001 = 0.1%)
        *,
        env_config: EnvConfig | None = None,
        action_threshold_open: float | None = None,
        action_threshold_close: float | None = None,
        action_threshold_flip: float | None = None,
        cfg: dict | None = None,
    ):
        self.data = data
        self.seq_lens = seq_lens
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.env_config = env_config or EnvConfig()
        cfg = cfg or {}
        self.take_profit_pct = float(
            take_profit_pct if take_profit_pct is not None else self.env_config.take_profit_pct
        )
        self.stop_loss_pct = float(
            stop_loss_pct if stop_loss_pct is not None else self.env_config.stop_loss_pct
        )
        base_th_open = 0.45 if action_threshold_open is None else float(action_threshold_open)
        base_th_close = 0.20 if action_threshold_close is None else float(action_threshold_close)
        base_th_flip = 0.70 if action_threshold_flip is None else float(action_threshold_flip)

        self.th_open = float(cfg.get("th_open", base_th_open))
        self.th_close = float(cfg.get("th_close", base_th_close))
        self.th_flip = float(cfg.get("th_flip", base_th_flip))

        if not (0.0 <= self.th_close <= self.th_open <= self.th_flip <= 1.0):
            raise ValueError("Invalid action hysteresis thresholds provided.")

        # Backwards compatibility for legacy attribute names
        self.action_threshold_open = self.th_open
        self.action_threshold_close = self.th_close
        self.action_threshold_flip = self.th_flip

        # --- OHLCV idx sanity check ---
        self.ohlcv_close_idx = (
            ohlcv_close_idx if ohlcv_close_idx is not None else self.env_config.ohlcv_close_idx
        )
        self.ohlcv_high_idx = (
            ohlcv_high_idx if ohlcv_high_idx is not None else self.env_config.ohlcv_high_idx
        )
        self.ohlcv_low_idx = (
            ohlcv_low_idx if ohlcv_low_idx is not None else self.env_config.ohlcv_low_idx
        )
        if enforce_hl:
            if self.ohlcv_close_idx is None or self.ohlcv_high_idx is None or self.ohlcv_low_idx is None:
                raise ValueError(
                    "ohlcv_high_idx / ohlcv_low_idx / ohlcv_close_idx를 지정하세요. "
                    "예: close=3, high=1, low=2 ([O,H,L,C] 순서일 때)."
                )

        self.funding_col_idx = funding_col_idx
        self.idle_kappa = idle_kappa
        self.ewma_beta = ewma_beta

        base_idle_penalty = float(idle_lambda if use_idle_penalty else 0.0)
        idle_penalty_cfg = cfg.get("idle_penalty")
        self.idle_penalty = max(0.0, float(idle_penalty_cfg if idle_penalty_cfg is not None else base_idle_penalty))
        self.use_idle_penalty = bool(cfg.get("use_idle_penalty", use_idle_penalty)) and self.idle_penalty > 0
        self.idle_lambda = self.idle_penalty

        hold_bars_default = self.env_config.min_hold_bars
        if min_hold_bars is not None:
            hold_bars_default = min_hold_bars
        self.min_hold_bars = int(cfg.get("min_hold_bars", hold_bars_default)) if hold_bars_default else 0

        flip_penalty_default = self.env_config.flip_penalty if flip_penalty is None else flip_penalty
        self.flip_penalty = max(0.0, float(cfg.get("flip_penalty", flip_penalty_default)))

        self.turnover_penalty = max(0.0, float(cfg.get("turnover_penalty", 0.0)))
        self.reward_scale = float(cfg.get("reward_scale", 300.0))

        self.holding_bars_running = 0

        self.length = len(next(iter(data.values())))
        self.init_cash = 1.0

        # Gym spaces
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = sum(v.shape[1] for v in data.values())
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # --- Rolling (step-based) ---
        # 완료 이벤트(스텝당 0/1, 최근 1k 스텝 기준 비율)
        self._roll_trade_close = Rolling(1000)
        self._roll_tp_close    = Rolling(1000)
        self._roll_sl_close    = Rolling(1000)
        self._roll_forced_close= Rolling(1000)
        # 액션 유발 청산 분해
        self._roll_action_flat = Rolling(1000)
        self._roll_action_flip = Rolling(1000)

        # 이벤트 값 기반 평균(완결 시만 기록)
        self._roll_R           = Rolling(1000)
        self._roll_hold_bars   = Rolling(1000)

        # 스텝 기반 롤링
        self._roll_reward  = Rolling(1000)   # per-step net return
        self._roll_funding = Rolling(1000)   # pos * funding_rate (부호 포함)
        self._roll_turnover= Rolling(1000)   # |pos_t - pos_{t-1}|

        self.reset()

    # ------------- helpers -------------
    def _get_price_triplet(self, idx):
        o = self.data["ohlcv"]
        close = float(o[idx, self.ohlcv_close_idx]) if self.ohlcv_close_idx is not None else float(o[idx, -1])
        high = float(o[idx, self.ohlcv_high_idx])  if self.ohlcv_high_idx  is not None else close
        low  = float(o[idx, self.ohlcv_low_idx])   if self.ohlcv_low_idx   is not None else close
        return close, high, low

    def _quantize_action(self, a: float) -> int:
        """
        Hysteresis thresholds are driven by the configuration object to keep the
        environment and agent in sync.
        """
        th_open = self.th_open
        th_close = self.th_close
        th_flip = self.th_flip

        pos = int(getattr(self, "position", 0))

        if abs(a) < th_close:
            return 0

        if pos > 0 and a < -th_flip:
            return -1
        if pos < 0 and a > th_flip:
            return 1

        if a > th_open:
            return 1
        if a < -th_open:
            return -1

        return pos

    # ------------- env core -------------
    def reset(self):
        self.t = max(self.seq_lens.values())
        self.position = 0
        self.entry_price = None
        self.holding_bars_running = 0
        self.done = False

        self.last_close, _, _ = self._get_price_triplet(self.t - 1)

        self.portfolio_value = self.init_cash
        self.portfolio_history = [self.init_cash]

        self.ewma_abs_r = 0.0

        for r in (
            self._roll_trade_close, self._roll_tp_close, self._roll_sl_close,
            self._roll_forced_close, self._roll_action_flat, self._roll_action_flip,
            self._roll_R, self._roll_hold_bars, self._roll_reward,
            self._roll_funding, self._roll_turnover
        ):
            r.buf.clear()

        return self._get_obs()

    def _get_obs(self):
        return {k: v[self.t - self.seq_lens[k]: self.t] for k, v in self.data.items()}

    # --- 공통: 포지션 청산(수수료 곱셈 반영, 로그 기록) ---
    def _close_trade(self, pos_prev, exit_price, *, tp_hit=False, sl_hit=False, forced=False, reason:str=""):
        pnl = pos_prev * (exit_price / self.entry_price - 1.0)  # 실현 수익률
        # PV에 순 반영: 실현 PnL 후 청산 수수료
        self.portfolio_value *= (1.0 + pnl) * (1.0 - self.taker_fee)

        R = pnl / self.stop_loss_pct if self.stop_loss_pct > 0 else 0.0
        hold_bars = int(self.holding_bars_running)

        # 이벤트 기록(이 스텝에서 1로 기록)
        self._roll_trade_close.add(1.0)
        self._roll_tp_close.add(1.0 if tp_hit else 0.0)
        self._roll_sl_close.add(1.0 if sl_hit else 0.0)
        self._roll_forced_close.add(1.0 if forced else 0.0)
        self._roll_R.add(R)
        self._roll_hold_bars.add(hold_bars)

        # 액션 유발 분해
        if reason == "flat":
            self._roll_action_flat.add(1.0)
            self._roll_action_flip.add(0.0)
        elif reason == "flip":
            self._roll_action_flat.add(0.0)
            self._roll_action_flip.add(1.0)
        else:
            self._roll_action_flat.add(0.0)
            self._roll_action_flip.add(0.0)

        # 상태 리셋
        self.position = 0
        self.entry_price = None
        self.open_since = None
        self.holding_bars_running = 0

    def step(self, action: float, is_forced_exit=False):
        # --- 준비 ---
        act = self._quantize_action(float(action))
        pos_prev = int(self.position)

        close, high, low = self._get_price_triplet(self.t)
        info = {"tp_hit": False, "sl_hit": False}
        closed_this_step = False

        # per-step 수익률 및 EWMA(|r|)
        r_step = (close / self.last_close - 1.0) if self.last_close > 0 else 0.0
        self.ewma_abs_r = self.ewma_abs_r * self.ewma_beta + (1.0 - self.ewma_beta) * abs(r_step)

        # 최소 보유바 강제(마스크)
        if (pos_prev != 0) and (not is_forced_exit) and (self.min_hold_bars > 0) and (self.holding_bars_running < self.min_hold_bars):
            act = pos_prev  # 강제 유지

        # 보유 중이었다면 보유바 카운트 증가
        if pos_prev != 0:
            self.holding_bars_running += 1

        nav_before = self.portfolio_value  # 보상 일관성: step 전 NAV

        # --- TP/SL 관통 체크 ---
        if pos_prev != 0 and self.entry_price is not None and not is_forced_exit:
            tp = self.entry_price * (1 + self.take_profit_pct * np.sign(pos_prev))
            sl = self.entry_price * (1 - self.stop_loss_pct * np.sign(pos_prev))
            if pos_prev > 0:
                tp_hit, sl_hit = (high >= tp), (low <= sl)
            else:
                tp_hit, sl_hit = (low <= tp), (high >= sl)
            if tp_hit or sl_hit:
                exit_price = tp if tp_hit else sl
                self._close_trade(pos_prev, exit_price, tp_hit=tp_hit, sl_hit=sl_hit, forced=False, reason=("tp" if tp_hit else "sl"))
                info["tp_hit"] = bool(tp_hit)
                info["sl_hit"] = bool(sl_hit)
                closed_this_step = True

        # --- 보유 중 가격 수익률 반영 ---
        if not closed_this_step:
            if pos_prev != 0:
                self.portfolio_value *= (1.0 + pos_prev * r_step)

        # --- 펀딩비(부호 포함) ---
        if "funding" in self.data:
            frate = float(self.data["funding"][self.t, self.funding_col_idx])
            # 펀딩비는 보통 8시간마다 적용 - 스텝당 비율로 정규화
            # 1시간봉이라면 8로 나누기, 15분봉이라면 32로 나누기 등
            funding_divisor = 8  # 환경에 맞게 조정 필요
            fund_flow = pos_prev * (frate / funding_divisor)
            self.portfolio_value *= (1.0 - fund_flow)
            self._roll_funding.add(fund_flow)
        else:
            self._roll_funding.add(0.0)

        # --- 조건부 Idle 페널티(곱셈 반영하여 PV/Reward 일치) ---
        if self.use_idle_penalty:
            sigma = self.ewma_abs_r
            if sigma > 0 and abs(r_step) > self.idle_kappa * sigma and pos_prev == 0:
                pen = min(self.idle_penalty, self.taker_fee)
                self.portfolio_value *= (1.0 - pen)

        # --- 액션 적용 ---
        if not closed_this_step:
            if is_forced_exit and pos_prev != 0:
                self._close_trade(pos_prev, close, tp_hit=False, sl_hit=False, forced=True, reason="forced")
                closed_this_step = True
            else:
                if pos_prev == 0 and act != 0:
                    # 신규 진입: 진입 수수료
                    self.portfolio_value *= (1.0 - self.taker_fee)
                    self.position = act
                    self.entry_price = close
                    self.holding_bars_running = 0
                elif pos_prev != 0 and act == 0:
                    # 관망으로 청산
                    self._close_trade(pos_prev, close, tp_hit=False, sl_hit=False, forced=False, reason="flat")
                    closed_this_step = True
                elif pos_prev != 0 and act != 0 and act != pos_prev:
                    # 플립: 이전 거래 종료 + 신규 진입
                    self._close_trade(pos_prev, close, tp_hit=False, sl_hit=False, forced=False, reason="flip")
                    closed_this_step = True
                    # 플립 진입 수수료 + 추가 패널티(옵션)
                    self.portfolio_value *= (1.0 - self.taker_fee)
                    if self.flip_penalty > 0:
                        self.portfolio_value *= (1.0 - self.flip_penalty)
                    self.position = act
                    self.entry_price = close
                    self.holding_bars_running = 0

                else:
                    pass

        # --- 거래 종료가 없었다면 0을 기록(분모 일관성) ---
        if not closed_this_step:
            self._roll_trade_close.add(0.0)
            self._roll_tp_close.add(0.0)
            self._roll_sl_close.add(0.0)
            self._roll_forced_close.add(0.0)
            # action_* are per-closure stats; no need to record zeros here

        # turnover 기록 및 빈도 비용 적용
        pos_now = int(self.position)
        turn_units = abs(pos_now - pos_prev)
        if self.turnover_penalty > 0 and turn_units > 0:
            self.portfolio_value *= max(1.0 - self.turnover_penalty * turn_units, 1e-12)
        self._roll_turnover.add(turn_units)

        # --- 보상 계산: step 순수익률 (PV 일치) ---
        reward = np.log(self.portfolio_value / nav_before)
        reward *= self.reward_scale
        self._roll_reward.add(reward)

        # --- 시간 전진 & 기록 ---
        self.last_close = close
        self.t += 1
        self.done = self.t >= self.length

        # step당 1회만 PV 기록(일관성)
        self.portfolio_history.append(float(self.portfolio_value))

        return self._get_obs(), reward, self.done, {
            **info,
            "portfolio_value": float(self.portfolio_value)
        }

    # ---- TensorBoard metrics snapshot ----
    def tb_metrics(self):
        step_cnt = max(1, self._roll_reward.count())  # 최근 1k 스텝 수
        completed = max(1.0, self._roll_trade_close.sum())  # 최근 1k 내 완결 거래 수

        trades_per_1k = 1000.0 * (self._roll_trade_close.sum() / step_cnt)
        tp_rate = self._roll_tp_close.sum() / completed
        sl_rate = self._roll_sl_close.sum() / completed
        forced_rate = self._roll_forced_close.sum() / completed

        return {
            "trades_per_1k": trades_per_1k,
            "tp_hit_rate_1k": tp_rate,
            "sl_hit_rate_1k": sl_rate,
            "avg_R": self._roll_R.mean(),
            "holding_bars_mean": self._roll_hold_bars.mean(),
            "forced_exit_rate_1k": forced_rate,
            "funding_cost_mean": self._roll_funding.mean(),  # 부호 포함
            "reward_mean": self._roll_reward.mean(),
            "equity": float(self.portfolio_history[-1]) if self.portfolio_history else 0.0,
            "turnover": self._roll_turnover.mean(),
            "action_flat_rate_1k": self._roll_action_flat.sum() / completed,
            "action_flip_rate_1k": self._roll_action_flip.sum() / completed,
        }

    def render(self):
        print(f"t={self.t}, position={self.position}")
        if self.position != 0 and self.entry_price is not None:
            tp = self.entry_price * (1 + self.take_profit_pct * np.sign(self.position))
            sl = self.entry_price * (1 - self.stop_loss_pct * np.sign(self.position))
            print(f"  Entry: {self.entry_price:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")
