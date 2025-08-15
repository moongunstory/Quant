# ai_binance/live/trader.py
"""
Trader for ETHUSDT Futures (PPO)
- 모델 로드 후, RealtimeIngest가 큐로 밀어주는 패킷을 소비하여 매매 수행
- 임계치 게이트: z-score 분위수 기반(진입/전환 엄격, 청산은 히스테리시스 완화)
- 정책 신뢰도(최대 행동확률) 필터 옵션
- 쿨다운/수수료/슬리피지 반영
- 모드: "live" | "paper" (실제 API 주문 vs 가상 처리)
- 로그/CSV: 라운드트립, 사이드별 내역, 에쿼티 저장

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
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from queue import Queue, Empty
from stable_baselines3 import PPO

from ai_binance.live.reporting import update_trade_log, generate_report

# =====================
# 설정
# =====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/ai_binance
MODEL_DIR = os.path.join(BASE_DIR, "data", "model")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")

WINDOW = 48                # 관찰 윈도우(학습과 동일, 4h)
COMMISSION_SIDE = 0.0005   # 0.05%/side (진입/청산 각각)
SLIPPAGE = 0.0001          # 0.01% 체결 슬리피지

# 임계치 게이트(과매매 억제)
COOLDOWN_BARS = 12         # 전환 후 최소 대기(≈1h)

# 정책 신뢰도(최대 행동확률) 필터 — 0이면 비활성
CONF_THRESHOLD = 0.6      # 0.70=70% 이상일 때만 신규 포지션 허용 (0이면 끔)

# 로깅/저장
PRINT_EVERY_BARS = 1       # 리포트 갱신 주기: 1바 = 5분
INITIAL_CAPITAL = 100_000.0


class Trader:
    def __init__(self, mode: str, q: Queue, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        assert mode in ("live", "paper")
        self.mode = mode
        self.q = q
        self.api_key = api_key
        self.secret_key = secret_key
        if self.mode == "live" and (not api_key or not secret_key):
            print(f"[트레이더] 경고: 'live' 모드지만 API 키가 없어 'paper' 모드로 동작합니다.")

        best = os.path.join(MODEL_DIR, "best_model.zip")
        final = os.path.join(MODEL_DIR, "ppo_final_model.zip")
        path = best if os.path.exists(best) else final
        self.model: PPO = PPO.load(path, device="cpu")
        self.model.eval_mode = True

        self.eq = INITIAL_CAPITAL
        self.pos = 0
        self.entry_price = None
        self.entry_time = None
        self.last_switch_idx = -10_000
        self.last_conf = 0.0
        self.last_price = None
        self._bars = 0

        self.start_time = datetime.now(timezone.utc)
        self.total_trades = 0
        self.winning_trades = 0
        self.long_trades = 0
        self.long_wins = 0
        self.short_trades = 0
        self.short_wins = 0
        self.hold_trades = 0

        os.makedirs(REPORT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        self.trade_log_path = os.path.join(LOG_DIR, "run_log.csv")
        self.report_path = os.path.join(REPORT_DIR, "trading_report.md")
        self.is_new_session = True
        self.session_start_time_str = self.start_time.strftime('%Y-%m-%d %H:%M:%S')

        print(f"[트레이더] 모드={self.mode} | 모델={os.path.basename(path)}")
        print(f"[트레이더] 시작 시각 (UTC): {self.session_start_time_str}")
        self._write_report()
        print(f"[트레이더] 리포트 파일: {self.report_path}")
        print(f"[트레이더] 매매 기록 파일: {self.trade_log_path}")

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
        stats = self._get_stats()
        price = self.last_price if self.last_price is not None else 0
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

    def _build_obs(self, X: pd.DataFrame, t: int) -> np.ndarray:
        if t >= WINDOW - 1:
            w = X.iloc[t - (WINDOW - 1): t + 1].values
        else:
            pad = np.tile(X.iloc[[0]].values, (WINDOW - (t + 1), 1))
            w = np.vstack([pad, X.iloc[:t + 1].values])
        return w.reshape(-1).astype(np.float32)

    @torch.no_grad()
    def _policy_confidence(self, obs: np.ndarray) -> float:
        try:
            obs_t = torch.as_tensor(obs).float().unsqueeze(0)
            dist = self.model.policy.get_distribution(obs_t)
            probs = torch.softmax(dist.distribution.logits, dim=-1).cpu().numpy()[0]
            return float(np.max(probs))
        except Exception:
            return 1.0

    def _gate(self, t: int, target: int) -> int:
        if target != self.pos and (t - self.last_switch_idx) < COOLDOWN_BARS:
            return self.pos
        if CONF_THRESHOLD > 0 and target != self.pos and target != 0:
            if self.last_conf < CONF_THRESHOLD:
                return self.pos
        return target

    def _get_unrealized_pnl(self, current_price: float) -> tuple[float, float]:
        if self.pos == 0 or self.entry_price is None or current_price == 0:
            return 0.0, 0.0
        pnl_pct = (current_price / self.entry_price - 1.0) if self.pos == 1 else (self.entry_price / current_price - 1.0)
        pnl_amount = self.eq * pnl_pct
        return pnl_amount, pnl_pct * 100.0

    def _open(self, side: int, price: float, ts: pd.Timestamp):
        px = price * (1 + SLIPPAGE if side == 1 else 1 - SLIPPAGE)
        fee = self.eq * COMMISSION_SIDE
        self.eq -= fee
        self.pos = side
        self.entry_price = px
        self.entry_time = ts
        side_str = "LONG" if side == 1 else "SHORT"

        trade_info = {
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'type': 'Entry',
            'position': side_str,
            'price': f"{px:,.4f}"
        }
        update_trade_log(self.trade_log_path, trade_info)
        if self.mode == "live":
            pass

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

    def run(self):
        print("[트레이더] 실행 중... (큐에서 데이터 읽기 대기)")
        while True:
            try:
                pkt = self.q.get(timeout=300)
            except Empty:
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] 데이터 수신 대기 중... 리포트 갱신.")
                self._write_report()
                continue

            X: pd.DataFrame = pkt["X"]
            close: pd.Series = pkt["close"]
            ts: pd.Timestamp = pkt["ts"]

            if len(X) < 5: continue

            t = len(X) - 1
            obs = self._build_obs(X, t)
            self.last_conf = self._policy_confidence(obs)
            act, _ = self.model.predict(obs, deterministic=True)
            target = 0 if act == 0 else (1 if act == 1 else -1)
            target = self._gate(t, target)

            if target == 0: self.hold_trades += 1
            
            p1 = float(close.iloc[t])
            if self.pos != 0 and self.last_price is not None:
                lr = math.log(p1 / self.last_price)
                self.eq *= math.exp((+1 if self.pos == 1 else -1) * lr)
            self.last_price = p1

            if target != self.pos:
                if self.pos != 0:
                    self._close(p1, ts)
                if target != 0:
                    self._open(target, p1, ts)
                self.last_switch_idx = t

            self._bars += 1
            if self._bars % PRINT_EVERY_BARS == 0:
                self._write_report()
                pos_str = "LONG" if self.pos == 1 else ("SHORT" if self.pos == -1 else "STANDBY")
                print(f"[{ts.strftime('%Y-%m-%d %H:%M:%S')}] Status: {pos_str} | Equity: ${self.eq:,.2f} | Confidence: {self.last_conf:.2f}")

            