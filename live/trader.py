# ai_binance/live/trader.py
"""
Trader for ETHUSDT Futures (PPO) — Entry/Exit 정책(같은 틱 전환 금지)
- 학습 환경과 동일한 게이트: |logret| vs (FEE_BUFFER + kσ·vol), 청산은 히스테리시스 완화
- 펀딩 비용: UTC 00:00/08:00/16:00 정각에만 부과(현재 pos 기준)
- 관측: 윈도우 피처 + [pos, time_in_pos_norm, upnl_log]
- 모드: "live" | "paper"
- (옵션) 온라인 학습 롤아웃 큐 전송
- live 모드: 파일 저장 비활성(콘솔만)

사용:
    from queue import Queue
    from ai_binance.live.realtime_ingest import RealtimeIngest
    from ai_binance.live.trader import Trader
    q = Queue(maxsize=2)
    ing = RealtimeIngest(q)
    tr = Trader(mode="paper", q=q, api_key=None, secret_key=None)
    # 스레드로 ing.run() 실행 후 tr.run()
"""

from __future__ import annotations

import os
import math
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from queue import Queue, Empty
from stable_baselines3 import PPO

from ai_binance.live.reporting import update_trade_log, generate_report
from ai_binance.live.execution import BinanceExecutor  # 실주문 어댑터

# =====================
# 경로/설정
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/ai_binance
MODEL_DIR = os.path.join(BASE_DIR, "data", "model")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")

SYMBOL = "ETHUSDT"
WINDOW = 48                # 4h
COMMISSION_SIDE = 0.0005   # 0.05%/side
SLIPPAGE = 0.0001          # 0.01%

# === 학습 환경과 완전 일치하는 게이트/펀딩 상수 ===
VOL_WIN = 24                        # ~2h 표준편차
HYSTERESIS_RATIO = 0.5              # 청산 문턱 완화
FEE_BUFFER = 2 * COMMISSION_SIDE    # 왕복 수수료 0.1%

# Phase 스케줄 (global_steps 기준)
def get_phase_config(global_steps: int) -> Dict[str, float]:
    if global_steps < 75_000:      # Phase 1
        return {"k_sigma": 0.8, "phase": 1}
    elif global_steps < 150_000:   # Phase 2
        return {"k_sigma": 0.5, "phase": 2}
    elif global_steps < 225_000:   # Phase 3
        return {"k_sigma": 0.2, "phase": 3}
    else:                          # Phase 4
        return {"k_sigma": 0.1, "phase": 4}

# 로깅/저장
PRINT_EVERY_BARS = 1
INITIAL_CAPITAL = 100_000.0

# 온라인 학습 전송 단위
ROLLOUT_STEPS = 4096

# 라이브 주문 사이징(환경변수로 오버라이드 가능)
DEFAULT_FIXED_USDT = float(os.getenv("TRADER_FIXED_USDT", "0") or 0)
DEFAULT_RISK_PCT   = float(os.getenv("TRADER_RISK_PCT", "0") or 0)   # 0.005 = 0.5%
DEFAULT_LEVERAGE   = int(os.getenv("TRADER_LEVERAGE", "0") or 0) or None


class Trader:
    def __init__(self, mode: str, q: Queue, api_key: Optional[str] = None, secret_key: Optional[str] = None, learn_q: Optional[Queue] = None):
        assert mode in ("live", "paper")
        self.mode = mode
        self.q = q
        self.learn_q = learn_q
        self.api_key = api_key
        self.secret_key = secret_key
        if self.mode == "live" and (not api_key or not secret_key):
            print(f"[트레이더] 경고: 'live' 모드지만 API 키가 없어 'paper' 모드로 동작합니다.")

        # 모델 로드
        best = os.path.join(MODEL_DIR, "best_model.zip")
        final = os.path.join(MODEL_DIR, "ppo_final_model.zip")
        path = best if os.path.exists(best) else final
        self.model: PPO = PPO.load(path, device="cpu")
        self.model.eval_mode = True

        # 계좌/포지션 상태
        self.eq = INITIAL_CAPITAL
        self.pos = 0
        self.entry_price = None
        self.entry_time: Optional[pd.Timestamp] = None
        self.last_price = None

        # 스텝/통계
        self.global_steps = 0
        self._bars = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.long_trades = 0
        self.long_wins = 0
        self.short_trades = 0
        self.short_wins = 0
        self.hold_trades = 0

        # 세션/리포트
        os.makedirs(REPORT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        self.trade_log_path = os.path.join(LOG_DIR, "run_log.csv")
        self.report_path = os.path.join(REPORT_DIR, "trading_report.md")
        self.is_new_session = True
        self.start_time = datetime.now(timezone.utc)
        self.session_start_time_str = self.start_time.strftime('%Y-%m-%d %H:%M:%S')

        print(f"[트레이더] 모드={self.mode} | 모델={os.path.basename(path)}")
        print(f"[트레이더] 시작 시각 (UTC): {self.session_start_time_str}")
        if self.mode != "live":
            self._write_report()
            print(f"[트레이더] 리포트 파일: {self.report_path}")
            print(f"[트레이더] 매매 기록 파일: {self.trade_log_path}")
        else:
            print(f"[트레이더] live 모드: 파일 리포트/CSV 비활성화 (콘솔만 출력)")

        # 온라인 학습용 롤아웃 버퍼
        self._roll_obs: List[np.ndarray] = []
        self._roll_actions: List[int] = []
        self._roll_rewards: List[float] = []
        self._roll_dones: List[bool] = []
        self._roll_values: List[float] = []
        self._roll_logps: List[float] = []
        self._roll_step = 0

        # 실행기
        self.exec: Optional[BinanceExecutor] = None
        self.fixed_usdt = DEFAULT_FIXED_USDT
        self.risk_pct = DEFAULT_RISK_PCT
        self.leverage = DEFAULT_LEVERAGE
        if self.mode == "live" and self.api_key and self.secret_key:
            try:
                self.exec = BinanceExecutor(self.api_key, self.secret_key)
                print("[트레이더] 실행 어댑터 준비 완료 (BinanceExecutor)")
            except Exception as e:
                print(f"[트레이더] 실행 어댑터 초기화 실패 → paper 경로로 진행: {e}")
                self.exec = None

    # -------------------------
    # 리포트/통계
    # -------------------------
    def _get_stats(self) -> Dict:
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        long_win_rate = (self.long_wins / self.long_trades * 100) if self.long_trades > 0 else 0.0
        short_win_rate = (self.short_wins / self.short_trades * 100) if self.short_trades > 0 else 0.0
        return {
            "win_rate": win_rate,
            "long_win_rate": long_win_rate,
            "short_win_rate": short_win_rate,
        }

    def _write_report(self):
        if self.mode == "live":
            return
        stats = self._get_stats()
        price = self.last_price if self.last_price is not None else 0.0
        unrealized_amount, unrealized_pct = self._get_unrealized_pnl(price)
        pos_str = "LONG" if self.pos == 1 else ("SHORT" if self.pos == -1 else "STANDBY")
        report_data = {
            'session_start_time': self.session_start_time_str,
            'initial_capital': INITIAL_CAPITAL,
            'position': pos_str,
            'total_equity': self.eq,
            'unrealized_pnl_amount': unrealized_amount,
            'unrealized_pnl_percent': unrealized_pct,
            'total_trades': self.total_trades,
            'win_rate': stats['win_rate'],
            'long_trades': self.long_trades,
            'long_win_rate': stats['long_win_rate'],
            'short_trades': self.short_trades,
            'short_win_rate': stats['short_win_rate'],
            'hold_trades': self.hold_trades
        }
        generate_report(self.report_path, report_data, self.is_new_session)
        if self.is_new_session:
            self.is_new_session = False

    # -------------------------
    # 관측 구성 (윈도우 + 상태피처3)
    # -------------------------
    def _state_vec(self, price_series: pd.Series, t: int) -> np.ndarray:
        time_in_pos = 0 if self.entry_time is None else int((price_series.index[t] - self.entry_time).total_seconds() // 300)
        time_in_pos_norm = min(time_in_pos, 1000) / 1000.0
        if self.entry_price is None or self.pos == 0:
            upnl_log = 0.0
        else:
            upnl_log = float(np.log(float(price_series.iloc[t]) / float(self.entry_price))) * float(self.pos)
        return np.array([float(self.pos), float(time_in_pos_norm), float(upnl_log)], dtype=np.float32)

    def _build_obs(self, X: pd.DataFrame, close: pd.Series, t: int) -> np.ndarray:
        # X: 정규화된 피처(5m 확정바 인덱스, UTC), WINDOW 길이 보장
        if t >= WINDOW - 1:
            w = X.iloc[t - (WINDOW - 1): t + 1].values
        else:
            pad = np.tile(X.iloc[[0]].values, (WINDOW - (t + 1), 1))
            w = np.vstack([pad, X.iloc[:t + 1].values])
        return np.concatenate([w.reshape(-1).astype(np.float32), self._state_vec(close, t)], axis=0)

    @torch.no_grad()
    def _action_logp_value(self, obs: np.ndarray, deterministic: bool = True) -> tuple[int, float, float]:
        obs_t = torch.as_tensor(obs).float().unsqueeze(0)
        action, _ = self.model.predict(obs, deterministic=deterministic)

        dist = self.model.policy.get_distribution(obs_t)
        act_t = torch.tensor(int(action), dtype=torch.long, device=obs_t.device)
        logp = dist.log_prob(act_t)
        if logp.ndim > 1:
            logp = logp.sum(-1)
        value = self.model.policy.predict_values(obs_t)
        return int(action), float(logp.cpu().item()), float(value.cpu().item())

    # -------------------------
    # 액션 해석(상태별) + 게이트(환경과 동일)
    # -------------------------
    @staticmethod
    def _interpret_action(raw_action: int, pos: int) -> int:
        # 반환: 목표 pos ∈ {-1,0,1}. 전환 금지.
        if pos == 0:  # 무포지션: 0=패스, 1=진입롱, 2=진입숏
            if raw_action == 1: return +1
            if raw_action == 2: return -1
            return 0
        else:         # 보유중: 0=홀드, 1=청산, 2=홀드
            if raw_action == 1: return 0
            return pos

    @staticmethod
    def _is_funding_event(ts: pd.Timestamp) -> bool:
        # 5분봉 정각 & 8시간 배수(UTC 00/08/16)
        return (getattr(ts, "minute", 0) == 0) and (getattr(ts, "hour", 0) % 8 == 0)

    def _gate(self, desired_target: int, logret_t: float, vol_t: float) -> int:
        if desired_target == self.pos:
            return desired_target
        k_sigma = get_phase_config(self.global_steps)["k_sigma"]
        thr_enter = FEE_BUFFER + k_sigma * float(vol_t)
        thr_exit  = HYSTERESIS_RATIO * thr_enter
        z = abs(float(logret_t))  # |logret|
        if self.pos == 0:  # 진입
            return desired_target if z >= thr_enter else self.pos
        else:              # 청산
            return 0 if (desired_target == 0 and z >= thr_exit) else self.pos

    # -------------------------
    # 주문/정산
    # -------------------------
    def _get_unrealized_pnl(self, current_price: float) -> tuple[float, float]:
        if self.pos == 0 or self.entry_price is None or current_price == 0:
            return 0.0, 0.0
        pnl_pct = (current_price / self.entry_price - 1.0) if self.pos == 1 else (self.entry_price / current_price - 1.0)
        pnl_amount = self.eq * pnl_pct
        return pnl_amount, pnl_pct * 100.0

    def _calc_order_qty(self, price: float) -> Optional[float]:
        if price <= 0:
            return None
        if self.fixed_usdt and self.fixed_usdt > 0:
            nominal = float(self.fixed_usdt)
        elif self.risk_pct and self.risk_pct > 0:
            lev = self.leverage if self.leverage else 1
            nominal = float(self.eq) * float(self.risk_pct) * float(lev)
        else:
            return None
        qty = nominal / price
        return max(qty, 0.0)

    def _open(self, side: int, price: float, ts: pd.Timestamp):
        px = price * (1 + SLIPPAGE if side == 1 else 1 - SLIPPAGE)
        fee = self.eq * COMMISSION_SIDE
        self.eq -= fee
        self.pos = side
        self.entry_price = px
        self.entry_time = ts
        side_str = "LONG" if side == 1 else "SHORT"

        if self.mode == "live" and self.exec is not None:
            try:
                qty = self._calc_order_qty(px)
                if qty is None or qty <= 0:
                    print("[트레이더] 경고: 주문 수량이 설정되지 않아 실주문을 건너뜁니다. (TRADER_FIXED_USDT 또는 TRADER_RISK_PCT 설정 필요)")
                else:
                    side_str_api = "BUY" if side == 1 else "SELL"
                    resp = self.exec.entry_with_stop(
                        symbol=SYMBOL,
                        side=side_str_api,
                        quantity=qty,
                        last_price=px,
                        sl_rate=0.035
                    )
                    print(f"[트레이더] LIVE 주문 실행: cancel→entry→SL 완료: {resp.get('entry', {}).get('orderId', 'n/a')}")
            except Exception as e:
                print(f"[트레이더] LIVE 주문 실패: {e}")

        if self.mode != "live":
            trade_info = {
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'Entry',
                'position': side_str,
                'price': f"{px:,.4f}"
            }
            update_trade_log(self.trade_log_path, trade_info)

    def _close(self, price: float, ts: pd.Timestamp):
        if self.pos == 0 or self.entry_price is None:
            return
        px = price * (1 - SLIPPAGE if self.pos == 1 else 1 + SLIPPAGE)
        pnl_pct = (px / self.entry_price - 1.0) if self.pos == 1 else (self.entry_price / px - 1.0)
        pnl_amount = self.eq * pnl_pct
        fee = self.eq * COMMISSION_SIDE
        net_pnl = pnl_amount - fee
        self.eq += net_pnl

        self.total_trades += 1
        if net_pnl > 0:
            self.winning_trades += 1

        side_str = "LONG" if self.pos == 1 else "SHORT"
        if self.pos == 1:
            self.long_trades += 1
            if net_pnl > 0: self.long_wins += 1
        else:
            self.short_trades += 1
            if net_pnl > 0: self.short_wins += 1

        holding_time = ts - self.entry_time
        duration_str = f"{int(holding_time.total_seconds() // 60)}m {int(holding_time.total_seconds() % 60)}s"

        if self.mode != "live":
            trade_info = {
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'Exit',
                'position': side_str,
                'price': f"{px:,.4f}",
                'profit': f"{net_pnl:,.2f}",
                'duration': duration_str
            }
            update_trade_log(self.trade_log_path, trade_info)

        self.pos = 0
        self.entry_price = None
        self.entry_time = None

    # -------------------------
    # 메인 루프
    # -------------------------
    def run(self):
        print("[트레이더] 실행 중... (큐에서 데이터 읽기 대기)")
        while True:
            try:
                pkt = self.q.get(timeout=300)
            except Empty:
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] 데이터 수신 대기 중... 리포트 갱신.")
                self._write_report()
                continue

            # 인제스트 패킷
            X: pd.DataFrame = pkt["X"]           # 정규화 피처, DatetimeIndex(UTC)
            close: pd.Series = pkt["close"]      # 종가 시리즈(동일 인덱스)
            ts: pd.Timestamp = pkt["ts"]         # 현재 바 타임스탬프(UTC)
            funding_rate = float(pkt.get("funding_rate", 0.0))  # 이벤트 시각이면 유효

            if len(X) < WINDOW:
                continue

            t = len(X) - 1
            obs = self._build_obs(X, close, t)

            # 정책 결정
            action, logp, value = self._action_logp_value(obs, deterministic=True)

            # 로그수익/변동성 (환경 동일)
            lr = float(np.log(float(close.iloc[t]) / float(close.iloc[t - 1]))) if t > 0 else 0.0
            vol_t = float(np.log(close / close.shift(1)).rolling(VOL_WIN, min_periods=1).std().iloc[t])

            # 상태별 해석 → 게이트
            desired = self._interpret_action(int(action), self.pos)
            target = self._gate(desired, lr, vol_t)

            # 표시용 마크투마켓
            p1 = float(close.iloc[t])
            if self.pos != 0 and self.last_price is not None:
                lr_mark = math.log(p1 / self.last_price)
                self.eq *= math.exp((+1 if self.pos == 1 else -1) * lr_mark)
            self.last_price = p1

            # 체결/수수료(동형)
            tx_cost = 0.0
            if target != self.pos:
                if self.pos != 0: tx_cost += COMMISSION_SIDE
                if target != 0:   tx_cost += COMMISSION_SIDE
                if self.pos != 0: self._close(p1, ts)
                if target != 0:   self._open(target, p1, ts)

            # RL 보상(환경 동일): 보유수익 - 수수료 - 펀딩(이벤트 시에만)
            holding_reward = (self.pos * lr)
            funding_penalty = (self.pos * funding_rate) if self._is_funding_event(ts) else 0.0
            rl_reward = holding_reward - tx_cost - funding_penalty

            # 롤아웃 적재
            self._roll_step += 1
            done = (self._roll_step % ROLLOUT_STEPS == 0)
            self._push_rollout(obs, action, rl_reward, done, logp, value)

            # 리포트/상태 출력
            self._bars += 1
            if self._bars % PRINT_EVERY_BARS == 0:
                self._write_report()
                pos_str = "LONG" if self.pos == 1 else ("SHORT" if self.pos == -1 else "STANDBY")
                print(f"[{ts.strftime('%Y-%m-%d %H:%M:%S')}] Status: {pos_str} | Equity: ${self.eq:,.2f}")

            # phase 스텝 증가
            self.global_steps += 1

    # -------------------------
    # 롤아웃 처리
    # -------------------------
    def _push_rollout(self, obs: np.ndarray, action: int, reward: float, done: bool, logp: float, value: float):
        self._roll_obs.append(obs.astype(np.float32))
        self._roll_actions.append(int(action))
        self._roll_rewards.append(float(reward))
        self._roll_dones.append(bool(done))
        self._roll_logps.append(float(logp))
        self._roll_values.append(float(value))

        if len(self._roll_obs) >= ROLLOUT_STEPS and self.learn_q is not None:
            try:
                pkt = {
                    "obs":        np.stack(self._roll_obs).astype(np.float32),
                    "actions":    np.array(self._roll_actions, dtype=np.int64),
                    "rewards":    np.array(self._roll_rewards, dtype=np.float32),
                    "dones":      np.array(self._roll_dones, dtype=bool),
                    "values":     np.array(self._roll_values, dtype=np.float32),
                    "log_probs":  np.array(self._roll_logps, dtype=np.float32),
                }
                self.learn_q.put_nowait(pkt)
            except Exception:
                pass
            finally:
                self._roll_obs.clear()
                self._roll_actions.clear()
                self._roll_rewards.clear()
                self._roll_dones.clear()
                self._roll_logps.clear()
                self._roll_values.clear()
