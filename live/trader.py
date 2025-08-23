# ai_binance/live/trader.py
"""
Trader (HRL-Ready) for UM Futures — Manager(방향/확신) + Worker(타이밍)
- 인제스터 패킷(정규화 피처 X, funding per 5m, ts)을 받아 실시간 의사결정.
- X는 DataFrame(병합 MTF) 또는 dict({'5m','15m','1h','4h'}) 모두 지원.
- Worker 관측: 훈련과 동일한 5m 피처만 역정규화하여 사용(분포 정합 유지).
- Manager: 가능하면 모델, 어려우면 안전한 MTF 휴리스틱.
- 게이트: k·σ + 히스테리시스 유지. 펀딩은 rate/96 per 5m 반영.
- 모드: "live" | "paper"
"""

from __future__ import annotations

import os, math, json
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from queue import Queue, Empty

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces, Env

import joblib
from ai_binance.live.reporting import update_trade_log, generate_report
from ai_binance.live.execution import BinanceExecutor  # 실주문 어댑터

# =====================
# 경로/설정
# =====================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/ai_binance
MODEL_DIR  = os.path.join(BASE_DIR, "data", "model")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")
LOG_DIR    = os.path.join(BASE_DIR, "data", "logs")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")

SYMBOL = "ETHUSDT"

# --- TradeEnv 정합 상수 ---
MAX_HOLDING_STEPS = 72      # 5m * 72 = 6h
WINDOW = 48                 # (구 트레이더 윈도우 — worker 관측에는 사용하지 않음)

# 비용/게이트
COMMISSION_SIDE   = 0.0005   # 0.05%/side
SLIPPAGE          = 0.0001   # 0.01%
VOL_WIN           = 24       # ~2h 표준편차
HYSTERESIS_RATIO  = 0.5
FEE_BUFFER        = 2 * COMMISSION_SIDE

# Phase 스케줄 (global_steps 기준)
def get_phase_config(global_steps: int) -> Dict[str, float]:
    if global_steps < 75_000:      return {"k_sigma": 0.8, "phase": 1}
    elif global_steps < 150_000:   return {"k_sigma": 0.5, "phase": 2}
    elif global_steps < 225_000:   return {"k_sigma": 0.2, "phase": 3}
    else:                          return {"k_sigma": 0.1, "phase": 4}

# 로깅/저장
PRINT_EVERY_BARS = 1
INITIAL_CAPITAL  = 100_000.0

# 온라인 학습 전송 단위
ROLLOUT_STEPS = 4096

# 라이브 주문 사이징(환경변수로 오버라이드 가능)
DEFAULT_FIXED_USDT = float(os.getenv("TRADER_FIXED_USDT", "0") or 0)
DEFAULT_RISK_PCT   = float(os.getenv("TRADER_RISK_PCT", "0") or 0)   # 0.005 = 0.5%
DEFAULT_LEVERAGE   = int(os.getenv("TRADER_LEVERAGE", "0") or 0) or None

# 모델 경로(현 버전 HRL)
MANAGER_MODEL_PATH   = os.path.join(MODEL_DIR, "manager_v2.zip")
MANAGER_VECNORM_PATH = os.path.join(MODEL_DIR, "manager_v2_vecnorm.pkl")
WORKER_MODEL_PATH    = os.path.join(MODEL_DIR, "worker_unified_final.zip")
WORKER_VECNORM_PATH  = os.path.join(MODEL_DIR, "worker_unified_vecnorm.pkl")

# -------------------------
# 유틸: Dummy VecEnv for VecNormalize.load
# -------------------------
class _ObsOnlyEnv(Env):
    """VecNormalize.load를 위해 obs/action shape만 제공하는 더미 env."""
    def __init__(self, obs_shape: Tuple[int, ...], action_space: spaces.Space):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = action_space
    def reset(self, *, seed=None, options=None): return np.zeros(self.observation_space.shape, np.float32), {}
    def step(self, action): return self.reset()[0], 0.0, True, False, {}

# =========================
# HRL 트레이더
# =========================
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

        # === Scaler & feature list (훈련 분포 정합) ===
        # 1) 단일 파일 우선
        scaler_path_single = os.path.join(PROC_DIR, "scaler.joblib")
        feats_path_single  = os.path.join(PROC_DIR, "feature_list.json")
        # 2) 없으면 5m 전용으로 대체
        scaler_path_5m = os.path.join(PROC_DIR, "scaler_5m.joblib")
        feats_path_5m  = os.path.join(PROC_DIR, "fe_feature_list_5m.json")

        if os.path.exists(scaler_path_single) and os.path.exists(feats_path_single):
            self.scaler = joblib.load(scaler_path_single)
            with open(feats_path_single, "r") as f:
                self.feature_list = json.load(f)
            print("[트레이더] processed/scaler.joblib + feature_list.json 사용")
        elif os.path.exists(scaler_path_5m) and os.path.exists(feats_path_5m):
            self.scaler = joblib.load(scaler_path_5m)
            with open(feats_path_5m, "r") as f:
                self.feature_list = json.load(f)
            print("[트레이더] processed/scaler_5m.joblib + fe_feature_list_5m.json 사용")
        else:
            raise FileNotFoundError(
                "Processed artifacts missing: "
                f"{scaler_path_single} & {feats_path_single} (or {scaler_path_5m} & {feats_path_5m})"
            )

        # === Manager (선택) ===
        self.manager_model: Optional[PPO] = None
        self.manager_vecnorm: Optional[VecNormalize] = None
        self.W_MGR = 8  # ManagerV2Env.SEQ_WINDOW

        if os.path.exists(MANAGER_MODEL_PATH) and os.path.exists(MANAGER_VECNORM_PATH):
            try:
                # Manager 관측 차원 추정: (W_MGR × MTF(1h/4h) 피처 수)
                mgr_cols = [c for c in self.feature_list if c.endswith("_1h") or c.endswith("_4h")]
                mgr_cols.sort()
                if len(mgr_cols) == 0:
                    raise RuntimeError("manager obs columns(1h/4h) not found in feature_list")

                obs_shape_mgr = (self.W_MGR * len(mgr_cols),)
                # Manager 액션 스페이스는 2차원 실수(Box(2,))로 가정([long_conf, short_conf])
                act_space_mgr = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

                # VecNormalize.load에 필요한 더미 env (shape 정합 중요)
                tmp_env = DummyVecEnv([lambda: _ObsOnlyEnv(obs_shape=obs_shape_mgr, action_space=act_space_mgr)])
                self.manager_vecnorm = VecNormalize.load(MANAGER_VECNORM_PATH, tmp_env)
                self.manager_vecnorm.training = False
                self.manager_vecnorm.norm_reward = False

                self.manager_model = PPO.load(MANAGER_MODEL_PATH, device="cpu")
                print(f"[트레이더] Manager 로드 완료: {os.path.basename(MANAGER_MODEL_PATH)} | obs_shape={obs_shape_mgr}, feats={len(mgr_cols)}")
            except Exception as e:
                print(f"[트레이더] Manager 로드 실패 → 휴리스틱 사용: {e}")
                self.manager_model = None
                self.manager_vecnorm = None
        else:
            print("[트레이더] Manager 모델/VecNorm 미탑재 → 휴리스틱 사용")

        # === Worker (필수) ===
        self.worker_model: MaskablePPO = MaskablePPO.load(WORKER_MODEL_PATH, device="cpu")
        obs_shape = self.worker_model.observation_space.shape
        tmp_w_env = DummyVecEnv([lambda: _ObsOnlyEnv(obs_shape=obs_shape, action_space=self.worker_model.action_space)])
        self.worker_vecnorm: VecNormalize = VecNormalize.load(WORKER_VECNORM_PATH, tmp_w_env)
        self.worker_vecnorm.training = False
        self.worker_vecnorm.norm_reward = False
        print(f"[트레이더] Worker 로드 완료: {os.path.basename(WORKER_MODEL_PATH)} | obs_shape={obs_shape}")

        # 계좌/포지션 상태
        self.eq = INITIAL_CAPITAL
        self.pos = 0
        self.entry_price = None
        self.entry_time: Optional[pd.Timestamp] = None
        self.last_price = None
        self.holding_steps = 0

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

        print(f"[트레이더] 모드={self.mode}")
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

        # Manager obs 시퀀스 버퍼(모델 사용시)
        self._mgr_buf: Optional[np.ndarray] = None  # shape=(W, feat_dim_mgr)

    # -------------------------
    # 리포트/통계
    # -------------------------
    def _get_stats(self) -> Dict:
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0
        long_win_rate = (self.long_wins / self.long_trades * 100) if self.long_trades > 0 else 0.0
        short_win_rate = (self.short_wins / self.short_trades * 100) if self.short_trades > 0 else 0.0
        return {"win_rate": win_rate, "long_win_rate": long_win_rate, "short_win_rate": short_win_rate}

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
    # Manager 신호 생성
    # -------------------------
    def _manager_goal(self, X_row: pd.Series, X_df: pd.DataFrame) -> Tuple[int, float, int]:
        """
        반환: (direction {-1,0,1}, confidence 0..1, is_ambiguous {0,1})
        우선 매니저 모델 사용; 실패 시 휴리스틱.
        """
        # 시도 1: 모델
        if self.manager_model is not None and self.manager_vecnorm is not None and self._mgr_buf is not None:
            try:
                # 현재 스텝의 1h/4h 피처만 추출 (훈련시와 동일한 이름/순서)
                if not set(self.mgr_cols).issubset(X_df.columns):
                    raise RuntimeError("manager feature columns missing in X")

                xh = X_df[self.mgr_cols].iloc[-1].astype(np.float32).values  # (feat_dim_mgr,)
                # 순환 시퀀스 버퍼 갱신
                self._mgr_buf = np.vstack([self._mgr_buf[1:], xh])
                obs_seq = self._mgr_buf.reshape(1, -1).astype(np.float32)   # (1, feat_dim_mgr * W)

                # VecNormalize 적용 후 예측
                norm_obs = self.manager_vecnorm.normalize_obs(obs_seq)
                act, _ = self.manager_model.predict(norm_obs, deterministic=True)  # shape (1,2)
                long_conf, short_conf = float(act[0][0]), float(act[0][1])

                if max(long_conf, short_conf) < 0.1:
                    direction = 0
                else:
                    direction = 1 if long_conf > short_conf else -1
                conf = max(long_conf, short_conf)
                ambiguous = 1 if abs(long_conf - short_conf) < 0.1 else 0
                return direction, conf, ambiguous
            except Exception as e:
                print(f"[트레이더] Manager 예측 실패 → 휴리스틱 대체: {e}")

        # 시도 2: 휴리스틱 (MTF 피처 기반)
        def _safe(col, default=0.0):
            return float(X_row.get(col, default))

        macd_h1  = _safe("macd_hist_1h")
        macd_h4  = _safe("macd_hist_4h")
        rsi_h1   = _safe("rsi14_1h") - 50.0
        rsi_h4   = _safe("rsi14_4h") - 50.0
        ret3_h1  = _safe("ret3_1h")
        ret12_h1 = _safe("ret12_1h")
        score = (1.8*macd_h1 + 1.2*macd_h4 + 0.6*(rsi_h1/50.0) + 1.0*ret3_h1 + 0.7*ret12_h1)
        direction = 1 if score > 0.0 else (-1 if score < 0.0 else 0)
        conf_raw = abs(score)
        conf = float(1 - math.exp(-min(5.0, max(0.0, conf_raw)) * 1.2))  # 0..1
        ambiguous = 1 if conf < 0.15 else 0
        return direction, conf, ambiguous

    def _manager_goal_dict(self, X_dict: Dict[str, pd.DataFrame]) -> Tuple[int, float, int]:
        """
        dict 구조에서 Manager 휴리스틱만 사용(모델 입력 차원 안전성).
        - '1h'/'4h' 프레임의 최신 행에서 지표 사용.
        """
        def _latest(df: Optional[pd.DataFrame], col: str, default: float = 0.0) -> float:
            try:
                if df is None or df.empty or (col not in df.columns): return default
                return float(df[col].iloc[-1])
            except Exception:
                return default

        X1 = X_dict.get("1h")
        X4 = X_dict.get("4h")

        macd_h1  = _latest(X1, "macd_hist", 0.0)
        macd_h4  = _latest(X4, "macd_hist", 0.0)
        rsi_h1   = _latest(X1, "rsi14", 50.0) - 50.0
        rsi_h4   = _latest(X4, "rsi14", 50.0) - 50.0
        ret3_h1  = _latest(X1, "ret3", 0.0)
        ret12_h1 = _latest(X1, "ret12", 0.0)

        score = (1.8*macd_h1 + 1.2*macd_h4 + 0.6*(rsi_h1/50.0) + 1.0*ret3_h1 + 0.7*ret12_h1)
        direction = 1 if score > 0.0 else (-1 if score < 0.0 else 0)
        conf_raw = abs(score)
        conf = float(1 - math.exp(-min(5.0, max(0.0, conf_raw)) * 1.2))
        ambiguous = 1 if conf < 0.15 else 0
        return direction, conf, ambiguous

    # -------------------------
    # Worker 관측 구성 (TradeEnv._obs와 동일 포맷)
    # -------------------------
    def _build_worker_obs_from_df(self, X_df: pd.DataFrame, t: int,
                                  manager_dir: int, manager_conf: float, manager_regime: int) -> np.ndarray:
        # 5m 훈련 피처만 역정규화(정확한 열 순서 보장)
        x_norm = X_df.iloc[t].reindex(self.feature_list).fillna(0.0)
        x_raw = self.scaler.inverse_transform(np.asarray([x_norm.values], dtype=np.float64))[0].astype(np.float32)
        holding_norm = float(min(self.holding_steps, MAX_HOLDING_STEPS) / MAX_HOLDING_STEPS)
        obs = np.concatenate([
            x_raw,
            np.array([
                float(manager_dir),
                float(manager_conf),
                float(manager_regime),
                float(self.pos != 0),
                holding_norm
            ], dtype=np.float32)
        ], axis=0)
        return obs

    def _build_worker_obs_from_dict(self, X_dict: Dict[str, pd.DataFrame], t_idx: pd.Timestamp,
                                    manager_dir: int, manager_conf: float, manager_regime: int) -> Optional[np.ndarray]:
        """
        dict 구조 → 5m 프레임에서 feature_list 열만 추출하여 역정규화.
        t_idx: 기준 시간(5m 최신 인덱스)
        """
        X5 = X_dict.get("5m")
        if X5 is None or X5.empty:
            return None
        if t_idx not in X5.index:
            # 가장 가까운 과거 시점으로 보정
            t_idx = X5.index[X5.index.get_indexer([t_idx], method="pad")[0]]
        row = X5.loc[t_idx]
        x_norm = row.reindex(self.feature_list).fillna(0.0)
        x_raw = self.scaler.inverse_transform(np.asarray([x_norm.values], dtype=np.float64))[0].astype(np.float32)
        holding_norm = float(min(self.holding_steps, MAX_HOLDING_STEPS) / MAX_HOLDING_STEPS)
        obs = np.concatenate([
            x_raw,
            np.array([
                float(manager_dir),
                float(manager_conf),
                float(manager_regime),
                float(self.pos != 0),
                holding_norm
            ], dtype=np.float32)
        ], axis=0)
        return obs

    # -------------------------
    # 액션 마스크 / 해석 / 게이트
    # -------------------------
    @staticmethod
    def _mask(pos: int) -> np.ndarray:
        # worker: 0 Wait, 1 Enter, 2 Exit
        if pos == 0:
            return np.array([True, True, False], dtype=bool)
        else:
            return np.array([True, False, True], dtype=bool)

    def _interpret(self, raw_action: int, manager_dir: int) -> Tuple[int, str]:
        if self.pos == 0:
            if raw_action == 1 and manager_dir != 0:
                return (1 if manager_dir > 0 else -1), "enter"
            else:
                return 0, "wait"
        else:
            if raw_action == 2:
                return 0, "exit"
            else:
                return self.pos, "hold"

    def _gate(self, desired_target: int, logret_t: float, vol_t: float) -> int:
        if desired_target == self.pos:
            return desired_target
        k_sigma = get_phase_config(self.global_steps)["k_sigma"]
        thr_enter = FEE_BUFFER + k_sigma * float(vol_t)
        thr_exit  = HYSTERESIS_RATIO * thr_enter
        z = abs(float(logret_t))
        if self.pos == 0:
            return desired_target if z >= thr_enter else self.pos
        else:
            return 0 if (desired_target == 0 and z >= thr_exit) else self.pos

    # -------------------------
    # 주문/정산(기존 유지)
    # -------------------------
    def _get_unrealized_pnl(self, current_price: float) -> tuple[float, float]:
        if self.pos == 0 or self.entry_price is None or current_price == 0:
            return 0.0, 0.0
        pnl_pct = (current_price / self.entry_price - 1.0) if self.pos == 1 else (self.entry_price / current_price - 1.0)
        pnl_amount = self.eq * pnl_pct
        return pnl_amount, pnl_pct * 100.0

    def _calc_order_qty(self, price: float) -> Optional[float]:
        if price <= 0: return None
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
        self.holding_steps = 0
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
            update_trade_log(self.trade_log_path, {
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'Entry',
                'position': side_str,
                'price': f"{px:,.4f}"
            })

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
        if net_pnl > 0: self.winning_trades += 1

        side_str = "LONG" if self.pos == 1 else "SHORT"
        if self.pos == 1:
            self.long_trades += 1
            if net_pnl > 0: self.long_wins += 1
        else:
            self.short_trades += 1
            if net_pnl > 0: self.short_wins += 1

        if self.mode != "live":
            holding_time = ts - (self.entry_time or ts)
            duration_str = f"{int(holding_time.total_seconds() // 60)}m {int(holding_time.total_seconds() % 60)}s"
            update_trade_log(self.trade_log_path, {
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'Exit',
                'position': side_str,
                'price': f"{px:,.4f}",
                'profit': f"{net_pnl:,.2f}",
                'duration': duration_str
            })

        self.pos = 0
        self.entry_price = None
        self.entry_time = None
        self.holding_steps = 0

    # -------------------------
    # 메인 루프
    # -------------------------
    @torch.no_grad()
    def run(self):
        print("[트레이더] HRL 모드 실행 중... (큐 수신 대기)")
        while True:
            try:
                pkt = self.q.get(timeout=300)
            except Empty:
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] 데이터 대기... 리포트 갱신.")
                self._write_report()
                continue

            # 인제스트 패킷: X는 DataFrame(병합) 또는 dict({'5m','15m','1h','4h'})
            X_any: Union[pd.DataFrame, Dict[str, pd.DataFrame]] = pkt["X"]
            funding_series_any = pkt.get("funding")
            ts: pd.Timestamp = pkt["ts"]

            # 기준 5m close 시리즈
            if isinstance(X_any, dict):
                X5 = X_any.get("5m")
                if X5 is None or X5.empty:  # 안전 가드
                    continue
                close = pkt.get("close")
                if close is None:
                    # 없으면 5m Close로 대체
                    close = X5.get("Close", pd.Series(index=X5.index, dtype=float)).astype(float)
                X_index = X5.index
            else:
                X_df: pd.DataFrame = X_any
                close = pkt["close"]
                X_index = X_df.index

            if len(X_index) < 1:
                continue

            t = len(X_index) - 1
            t_idx = X_index[t]
            p1 = float(close.iloc[t])
            self.last_price = p1 if self.last_price is None else self.last_price

            # === Manager 신호 ===
            if isinstance(X_any, dict):
                manager_dir, manager_conf, manager_regime = self._manager_goal_dict(X_any)
            else:
                manager_dir, manager_conf, manager_regime = self._manager_goal_df(
                    X_any.iloc[t], X_any
                )

            # === Worker 관측 구성 → VecNormalize → 예측 ===
            if isinstance(X_any, dict):
                obs_raw = self._build_worker_obs_from_dict(X_any, t_idx, manager_dir, manager_conf, manager_regime)
                if obs_raw is None:
                    continue
                funding_series = funding_series_any
                if isinstance(funding_series, pd.Series):
                    f_idx = funding_series.index.get_indexer([t_idx], method="pad")[0]
                    funding_t = float(funding_series.iloc[f_idx])
                else:
                    funding_t = 0.0
                lr = math.log(p1 / float(close.iloc[t-1])) if t > 0 else 0.0
            else:
                obs_raw = self._build_worker_obs_from_df(X_any, t, manager_dir, manager_conf, manager_regime)
                funding_series = funding_series_any if isinstance(funding_series_any, pd.Series) else pd.Series(0.0, index=X_index)
                funding_t = float(funding_series.iloc[t]) if len(funding_series) == len(X_index) else 0.0
                lr = math.log(p1 / float(close.iloc[t-1])) if t > 0 else 0.0

            obs_norm = self.worker_vecnorm.normalize_obs(obs_raw.reshape(1, -1))
            masks = self._mask(self.pos)
            action, _ = self.worker_model.predict(obs_norm, deterministic=True, action_masks=masks)

            # === 상태별 해석 + 게이트 ===
            desired, why = self._interpret(int(action), manager_dir)
            vol_t = float(np.log(close / close.shift(1)).rolling(VOL_WIN, min_periods=1).std().iloc[t])
            target = self._gate(desired, lr, vol_t)

            # === 마크투마켓 ===
            if self.pos != 0:
                lr_mark = math.log(p1 / (self.last_price or p1))
                self.eq *= math.exp((+1 if self.pos == 1 else -1) * lr_mark)
                self.holding_steps += 1
            self.last_price = p1

            # === 체결/수수료 ===
            tx_cost = 0.0
            if target != self.pos:
                if self.pos != 0: tx_cost += COMMISSION_SIDE
                if target != 0:   tx_cost += COMMISSION_SIDE
                if self.pos != 0: self._close(p1, ts)
                if target != 0:   self._open(target, p1, ts)

            # === RL 보상(환경 개념): 보유수익 - 수수료 - 펀딩(연속 분배) ===
            funding_penalty = self.pos * funding_t
            rl_reward = (self.pos * lr) - tx_cost - funding_penalty

            # === 롤아웃 적재(옵션) ===
            self._roll_step += 1
            done = (self._roll_step % ROLLOUT_STEPS == 0)
            try:
                obs_t = torch.as_tensor(obs_norm).float()
                dist = self.worker_model.policy.get_distribution(obs_t)
                act_t = torch.tensor(int(action), dtype=torch.long, device=obs_t.device)
                logp_t = dist.log_prob(act_t)
                if logp_t.ndim > 1: logp_t = logp_t.sum(-1)
                value_t = self.worker_model.policy.predict_values(obs_t)
                logp = float(logp_t.cpu().item())
                value = float(value_t.cpu().item())
            except Exception:
                logp, value = float("nan"), float("nan")

            self._push_rollout(obs_raw, int(action), float(rl_reward), bool(done), float(logp), float(value))

            # === 리포트/상태 출력 ===
            self._bars += 1
            if self._bars % PRINT_EVERY_BARS == 0:
                self._write_report()
                pos_str = "LONG" if self.pos == 1 else ("SHORT" if self.pos == -1 else "STANDBY")
                print(f"[{ts.strftime('%Y-%m-%d %H:%M:%S')}] Status: {pos_str} | Equity: ${self.eq:,.2f} | mgr(dir={manager_dir}, conf={manager_conf:.2f}) | act={int(action)}->{why}")

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
