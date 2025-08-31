# ai_binance/live/trader.py
"""
Trader (HRL-Ready, SLIM) — Manager(방향/확신) + Worker(타이밍)
- 불필요 기능 제거: 휴리스틱 매니저, 온라인 학습, 과도한 리포팅.
- 매니저 입력 계약 강제: (W, columns) = meta/VecNorm 기준으로만 구성.
- Worker 관측: 훈련과 동일한 원시 피처 조합 + 추가 관측 5개.
"""

from __future__ import annotations

import os, json
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
from queue import Queue, Empty

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces, Env

from ai_binance.train.reinforce.manager import TransformerFeatureExtractor
from ai_binance.live.reporting import update_trade_log, generate_report
from ai_binance.live.execution import BinanceExecutor

# =====================
# 경로/설정
# =====================
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "data", "model")
REPORT_DIR = os.path.join(BASE_DIR, "data", "logs", "reports")
LOG_DIR    = os.path.join(BASE_DIR, "data", "logs")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")

for _d in (MODEL_DIR, REPORT_DIR, LOG_DIR, PROC_DIR):
    os.makedirs(_d, exist_ok=True)

SYMBOL = "ETHUSDT"
COMMISSION_SIDE   = 0.0005   # 각 사이드 수수료 비율
SLIPPAGE          = 0.0001   # 체결 슬리피지 가정(비율)
LEVERAGE          = 5        # 실전 모드 레버리지

INITIAL_CAPITAL  = 100_000.0

MANAGER_MODEL_PATH   = os.path.join(MODEL_DIR, "manager_v2.zip")
MANAGER_VECNORM_PATH = os.path.join(MODEL_DIR, "manager_v2_vecnorm.pkl")
MANAGER_META_PATH    = os.path.join(MODEL_DIR, "manager_v2_meta.json")

WORKER_MODEL_PATH    = os.path.join(MODEL_DIR, "worker_unified_final.zip")
WORKER_VECNORM_PATH  = os.path.join(MODEL_DIR, "worker_unified_vecnorm.pkl")

# === Slim toggles ===
STRICT_MANAGER = True          # 매니저 세트 불일치 시 즉시 중단
ENABLE_REPORT_ON_TRADE = False # 틱마다 리포트(거래시 중복 방지)
REPORT_EVERY_N = 1             # >0이면 N틱마다 리포트, 1이면 매 틱
ENFORCE_MAX_HOLDING = False    # 최대 보유기간 강제 청산 비활성(기본)
MAX_HOLDING_STEPS = 72
DEBUG = False                  # 디버그 로그 토글

# =====================
# 유틸 Env (VecNormalize 로드용)
# =====================
class _ObsOnlyEnv(Env):
    def __init__(self, obs_shape: Tuple[int, ...], action_space: spaces.Space):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = action_space
    def reset(self, *, seed=None, options=None):
        return np.zeros(self.observation_space.shape, np.float32), {}
    def step(self, action):
        obs, _ = self.reset()
        return obs, 0.0, True, False, {}

# =====================
# Trader
# =====================
class Trader:
    def __init__(
        self,
        mode: str,
        q: Queue,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        learn_q: Optional[Queue] = None,   # 유지(호환)하되 미사용
    ):
        assert mode in ("live", "paper")
        self.mode, self.q, self.api_key, self.secret_key = mode, q, api_key, secret_key

        # 실행 어댑터
        self.exec = None
        if self.mode == "live" and self.api_key and self.secret_key:
            try:
                self.exec = BinanceExecutor(self.api_key, self.secret_key)
                print("[트레이더] 실행 어댑터 초기화 완료.")
            except Exception as e:
                print(f"[트레이더] 실행 어댑터 초기화 실패: {e}")
        elif self.mode == "live":
            print(f"[트레이더] 경고: 'live' 모드지만 API 키가 없어 'paper' 모드로 동작합니다.")

        # Manager용 feature_list 로드(라이브 전체 후보 목록 보관)
        with open(os.path.join(PROC_DIR, "fe_feature_list_1h.json"), "r") as f:
            mgr_feats_1h = json.load(f)
        with open(os.path.join(PROC_DIR, "fe_feature_list_4h.json"), "r") as f:
            mgr_feats_4h = json.load(f)
        self.mgr_cols_live = mgr_feats_1h + mgr_feats_4h   # 라이브가 제공 가능한 전체 후보
        self.mgr_cols: List[str] = self.mgr_cols_live[:]   # 실제 사용 목록(메타에 의해 재설정됨)
        self.mgr_W: int = 8                                # 실제 사용 길이(메타에 의해 재설정됨)
        print(f"[트레이더] Manager 피처 로드 완료(라이브 후보): {len(self.mgr_cols_live)}개")

        # Worker 모델/VecNorm 로드
        self.worker_model: MaskablePPO = MaskablePPO.load(WORKER_MODEL_PATH, device="cpu")
        obs_shape_w = self.worker_model.observation_space.shape
        tmp_w_env = DummyVecEnv([lambda: _ObsOnlyEnv(obs_shape=obs_shape_w, action_space=self.worker_model.action_space)])
        self.worker_vecnorm: VecNormalize = VecNormalize.load(WORKER_VECNORM_PATH, tmp_w_env)
        self.worker_vecnorm.training = False
        self.worker_vecnorm.norm_reward = False
        print(f"[트레이더] Worker 로드 완료: {os.path.basename(WORKER_MODEL_PATH)} | obs_shape={obs_shape_w}")

        # Worker 피처 리스트 재구성 (훈련 순서와 동일)
        print("[트레이더] Worker 피처 리스트 재구성 중...")
        with open(os.path.join(PROC_DIR, "fe_feature_list_5m.json"), "r") as f:
            w_feats_5m = json.load(f)
        with open(os.path.join(PROC_DIR, "fe_feature_list_15m.json"), "r") as f:
            w_feats_15m = json.load(f)
        with open(os.path.join(PROC_DIR, "fe_feature_list_1h.json"), "r") as f:
            w_feats_1h = json.load(f)
        with open(os.path.join(PROC_DIR, "fe_feature_list_4h.json"), "r") as f:
            w_feats_4h = json.load(f)
        btc_features = [
            'ret_1h_btc1h', 'ret_4h_btc1h', 'atr14_btc1h',
            'HA_O_btc1h', 'HA_H_btc1h', 'HA_L_btc1h', 'HA_C_btc1h',
            'HA_TR_btc1h', 'HA_BC_btc1h', 'HA_R_btc1h'
        ]
        self.worker_feature_list = w_feats_5m + w_feats_15m + w_feats_1h + w_feats_4h + btc_features
        print(f"[트레이더] Worker 피처 리스트 재구성 완료: {len(self.worker_feature_list)}개")

        # 관측 차원 정합성 검증 (추가 관측 5개: mgr_dir, mgr_conf, mgr_regime, has_pos, holding_norm)
        vecnorm_dim = int(self.worker_vecnorm.obs_rms.mean.shape[0])
        extra_dim = 5
        expected_dim = len(self.worker_feature_list) + extra_dim
        if expected_dim != vecnorm_dim:
            raise RuntimeError(f"[트레이더] Worker VecNorm 차원 불일치: expected={expected_dim}, vecnorm={vecnorm_dim}")
        flat_model_obs_dim = int(np.prod(obs_shape_w))
        if flat_model_obs_dim != expected_dim:
            raise RuntimeError(f"[트레이더] Worker 모델 관측 차원 불일치: model={flat_model_obs_dim}, built={expected_dim}")
        print(f"[트레이더] Worker 관측 정합 확인 완료: obs_dim={expected_dim}")

        # Manager 세트 로드(메타/VecNorm 기반 정합 강제)
        self.manager_model: Optional[PPO] = None
        self._load_manager_or_fail()

        # 상태 변수
        start_capital = INITIAL_CAPITAL
        if self.exec:
            try:
                balance = self.exec.get_usdt_balance()
                if balance > 0:
                    start_capital = balance
                    print(f"[트레이더] 실제 계좌 잔액 ${balance:,.2f}으로 시작")
                else:
                    print("[트레이더] 경고: 실제 계좌 잔액이 0 또는 조회 실패 → 기본값으로 시작")
            except Exception as e:
                print(f"[트레이더] 경고: 계좌 잔액 조회 실패({e}) → 기본값으로 시작")

        self.initial_capital = start_capital
        self.eq = self.initial_capital
        self.pos = 0
        self.entry_price: Optional[float] = None
        self.entry_time: Optional[pd.Timestamp] = None
        self.entry_notional: float = 0.0
        self.last_price: Optional[float] = None
        self.holding_steps = 0
        self.global_steps = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.long_trades = 0
        self.long_wins = 0
        self.short_trades = 0
        self.short_wins = 0
        self.start_time = datetime.now(timezone.utc)

        # 초기 리포트 1회
        report_path = os.path.join(REPORT_DIR, "trading_report.md")
        generate_report(report_path, {
            'session_start_time': self.start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            'initial_capital': self.initial_capital,
            'total_equity': self.eq,
            'position': "STANDBY",
            'unrealized_pnl_amount': 0,
            'unrealized_pnl_percent': 0,
            'total_trades': 0,
            'win_rate': 0,
            'long_trades': 0,
            'long_win_rate': 0,
            'short_trades': 0,
            'short_win_rate': 0,
            'hold_trades': 0
        }, is_new_session=True)

    # ---------- Manager 세트 로드 ----------
    def _load_manager_or_fail(self):
        if not (os.path.exists(MANAGER_MODEL_PATH) and os.path.exists(MANAGER_VECNORM_PATH)):
            msg = "[트레이더] Manager 세트 미존재(manager_v2.zip / vecnorm.pkl)"
            if STRICT_MANAGER: raise RuntimeError(msg)
            print(msg); return

        try:
            # 0) 메타 먼저 로드해 W,F 확보
            if not os.path.isfile(MANAGER_META_PATH):
                raise RuntimeError("[Manager] metadata 파일(manager_v2_meta.json)이 필요합니다.")
            with open(MANAGER_META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
            W_tr = int(meta.get("seq_window", 8))
            F_tr = int(meta["feat_dim"])
            trained_cols: Optional[List[str]] = meta.get("columns")

            # 1) (W*F,) 관측으로 맞춘 더미 venv 생성 후 VecNorm 로드
            tmp_mgr_env = DummyVecEnv([lambda: _ObsOnlyEnv(
                obs_shape=(W_tr * F_tr,),
                action_space=spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
            )])
            self.manager_vecnorm: VecNormalize = VecNormalize.load(MANAGER_VECNORM_PATH, tmp_mgr_env)
            self.manager_vecnorm.training = False
            self.manager_vecnorm.norm_reward = False
            mgr_vec_dim = int(self.manager_vecnorm.obs_rms.mean.shape[0])
            if mgr_vec_dim != W_tr * F_tr:
                raise RuntimeError(f"[Manager] VecNorm 차원 불일치: vecnorm={mgr_vec_dim}, expected={W_tr*F_tr}")

            # 2) 라이브 피처를 훈련 스펙에 맞춤(순서/개수 강제)
            if trained_cols:
                missing = [c for c in trained_cols if c not in self.mgr_cols_live]
                if missing:
                    raise RuntimeError(f"[Manager] 라이브에 없는 훈련 피처 존재: {missing[:8]} ...")
                self.mgr_cols = trained_cols
            else:
                self.mgr_cols = self.mgr_cols_live[:F_tr]

            self.mgr_W = W_tr

            # 2) 라이브 피처를 훈련 스펙에 맞춤(순서/개수 강제)
            if trained_cols:
                missing = [c for c in trained_cols if c not in self.mgr_cols_live]
                if missing:
                    raise RuntimeError(f"[Manager] 라이브에 없는 훈련 피처 존재: {missing[:8]} ...")
                self.mgr_cols = trained_cols
            else:
                # 메타가 없으면 응급조치: 앞에서 F_tr개만 사용
                self.mgr_cols = self.mgr_cols_live[:F_tr]
                print(f"[Manager/Live] Hotfix: live features truncated {len(self.mgr_cols_live)} → {len(self.mgr_cols)}")

            self.mgr_W = W_tr
            expected_mgr_dim = self.mgr_W * len(self.mgr_cols)
            if expected_mgr_dim != mgr_vec_dim:
                raise RuntimeError(f"[Manager] VecNorm 차원 불일치: vecnorm={mgr_vec_dim}, expected={expected_mgr_dim} (W={self.mgr_W}, F={len(self.mgr_cols)})")

            # 3) 모델 로드(Transformer 경로 해석을 위해 클래스 import 필요)
            policy_kwargs = dict(
                features_extractor_class=TransformerFeatureExtractor,
                features_extractor_kwargs=dict(
                    d_model=128, nhead=4, num_layers=2,
                    seq_len=W_tr, n_features=F_tr,   # ✅ 추가
                ),
                net_arch=dict(pi=[128], vf=[128]),
            )
            self.manager_model = PPO.load(
                MANAGER_MODEL_PATH,
                device="cpu",
                custom_objects={
                    "policy_kwargs": policy_kwargs,   # 저장된 값과 동일/호환
                    "lr_schedule": (lambda _pr: 3e-4),
                    "learning_rate": 3e-4,
                },
            )

            print(f"[트레이더] Manager 로드 완료: {os.path.basename(MANAGER_MODEL_PATH)} | W={self.mgr_W}, F={len(self.mgr_cols)} (vecnorm={mgr_vec_dim})")
        except Exception as e:
            msg = f"[트레이더] Manager 로드 실패: {e}"
            if STRICT_MANAGER: raise RuntimeError(msg)
            print(msg)
            self.manager_model = None

    # ---------- 관측 구성 ----------
    def _build_worker_obs(
        self,
        X_dict: Dict[str, pd.DataFrame],
        t_idx: pd.Timestamp,
        mgr_dir: int,
        mgr_conf: float,
        mgr_regime: int
    ) -> Optional[np.ndarray]:
        ordered_tfs = ["5m", "15m", "1h", "4h", "btc1h"]
        series_list = []
        for tf in ordered_tfs:
            df = X_dict.get(tf)
            if df is None or df.empty:
                print(f"[트레이더] 경고: {tf} 데이터 누락"); return None
            try:
                pos = df.index.get_indexer([t_idx], method="pad")[0]
                if pos == -1:
                    print(f"[트레이더] 경고: {tf}에서 {t_idx} 타임스탬프 없음"); return None
                series_list.append(df.iloc[pos])
            except Exception as e:
                print(f"[트레이더] 경고: {tf} 데이터 처리 오류: {e}"); return None

        x_combined = pd.concat(series_list)
        x_reordered = x_combined.reindex(self.worker_feature_list).fillna(0.0)
        x_raw = x_reordered.to_numpy(dtype=np.float32)

        holding_norm = min(self.holding_steps, MAX_HOLDING_STEPS) / MAX_HOLDING_STEPS if MAX_HOLDING_STEPS > 0 else 0.0
        extra_obs = np.array([mgr_dir, mgr_conf, mgr_regime, float(self.pos != 0), holding_norm], dtype=np.float32)
        return np.concatenate([x_raw, extra_obs])

    def _build_manager_obs_live(self, X_dict: Dict[str, pd.DataFrame], t_idx: pd.Timestamp,
                                cols: List[str], W: int) -> Optional[np.ndarray]:
        base = X_dict.get("1h")
        if base is None or base.empty:
            return None

        pos = base.index.get_indexer([t_idx], method="pad")[0]
        if pos < 0:
            return None

        start = max(0, pos - (W - 1))
        idx_win = base.index[start:pos+1]
        rows = []
        for ts in idx_win:
            vals = []
            for c in cols:
                tf = "1h" if c.endswith("_1h") else "4h"
                df = X_dict.get(tf)
                if df is None or df.empty or c not in df.columns:
                    vals.append(0.0); continue
                j = df.index.get_indexer([ts], method="pad")[0]
                vals.append(0.0 if j < 0 else float(df[c].iloc[j]))
            rows.append(vals)

        seq = np.asarray(rows, dtype=np.float32)
        if seq.shape[0] < W:  # 앞쪽 패딩
            pad = np.repeat(seq[:1, :], W - seq.shape[0], axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        return seq.astype(np.float32, copy=False)

    @staticmethod
    def _mask(pos: int) -> np.ndarray:
        # [wait, enter, exit]
        return np.array([True, pos == 0, pos != 0], dtype=np.bool_)

    def _interpret(self, raw_action: int, mgr_dir: int) -> Tuple[int, str]:
        if self.pos == 0:
            return (int(np.sign(mgr_dir)), "enter") if raw_action == 1 and mgr_dir != 0 else (0, "wait")
        else:
            return (0, "exit") if raw_action == 2 else (self.pos, "hold")

    def _generate_current_report(self):
        report_path = os.path.join(REPORT_DIR, "trading_report.md")

        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        long_win_rate = (self.long_wins / self.long_trades * 100) if self.long_trades > 0 else 0
        short_win_rate = (self.short_wins / self.short_trades * 100) if self.short_trades > 0 else 0

        unrealized_pnl_percent = 0.0
        if self.pos != 0 and self.entry_price and self.last_price:
            leverage = LEVERAGE if self.mode == 'live' else 1.0
            if self.pos == 1:
                unrealized_pnl_percent = (self.last_price / self.entry_price - 1.0) * 100.0 * leverage
            else:
                unrealized_pnl_percent = (self.entry_price / self.last_price - 1.0) * 100.0 * leverage

        current_pos_str = "STANDBY" if self.pos == 0 else ("LONG" if self.pos == 1 else "SHORT")
        generate_report(report_path, {
            'session_start_time': self.start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            'initial_capital': self.initial_capital,
            'total_equity': self.eq,
            'position': current_pos_str,
            'unrealized_pnl_amount': self.eq - self.initial_capital,
            'unrealized_pnl_percent': unrealized_pnl_percent,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'long_trades': self.long_trades,
            'long_win_rate': long_win_rate,
            'short_trades': self.short_trades,
            'short_win_rate': short_win_rate,
            'hold_trades': 0
        }, is_new_session=False)

    @torch.no_grad()
    def run(self):
        print("[트레이더] HRL 모드 실행 중... (큐 수신 대기)")
        while True:
            try:
                pkt = self.q.get(timeout=300)
            except Empty:
                continue

            X_dict, ts = pkt["X"], pkt["ts"]
            close_series = pkt["close"]
            if close_series.empty:
                continue

            t_idx = close_series.index[-1]
            p1 = close_series.iloc[-1]

            if self.last_price is None:
                self.last_price = p1

            # === Manager: 반드시 RL 사용(STRICT) ===
            if self.manager_model is None:
                raise RuntimeError("[트레이더] Manager 미로딩 상태. STRICT_MANAGER=True에서 허용되지 않습니다.")

            obs_mgr = self._build_manager_obs_live(X_dict, t_idx, self.mgr_cols, self.mgr_W)
            if obs_mgr is None:
                self.last_price = p1
                continue

            obs_mgr_flat = obs_mgr.reshape(1, -1).astype(np.float32)   # ✅ (1, W*F)
            obs_mgr_norm = self.manager_vecnorm.normalize_obs(obs_mgr_flat)

            act, _ = self.manager_model.predict(obs_mgr_norm, deterministic=True)

            # SB3는 n_env=1일 때 (1, action_dim)로 나올 수 있음
            aL = float(act[0,0] if act.ndim == 2 else act[0])
            aS = float(act[0,1] if act.ndim == 2 else act[1])
            margin = abs(aL - aS)
            mgr_dir = 0 if margin < 0.15 else (1 if aL > aS else -1)
            mgr_conf = margin
            mgr_regime = 0

            # === Worker ===
            obs_raw = self._build_worker_obs(X_dict, t_idx, mgr_dir, mgr_conf, mgr_regime)
            if obs_raw is None:
                self.last_price = p1
                continue

            obs_norm = self.worker_vecnorm.normalize_obs(obs_raw.reshape(1, -1))
            action_array, _ = self.worker_model.predict(
                obs_norm,
                deterministic=True,
                action_masks=self._mask(self.pos)
            )
            action = int(action_array[0]) if isinstance(action_array, np.ndarray) else int(action_array)

            # 보유 중 PnL 반영
            if self.pos != 0:
                lr = np.log(p1 / self.last_price)
                lr_eff = lr * (LEVERAGE if self.mode == "live" else 1.0)
                self.eq *= np.exp(self.pos * lr_eff)
                self.holding_steps += 1

            desired, why = self._interpret(action, mgr_dir)
            done = (desired != self.pos)

            # 최대 보유 기간 강제 청산(옵션)
            if ENFORCE_MAX_HOLDING and self.pos != 0 and self.holding_steps >= MAX_HOLDING_STEPS:
                print(f"[{ts.strftime('%H:%M:%S')}] WARN: 최대 보유 기간({MAX_HOLDING_STEPS}) 초과로 강제 청산.")
                self._close(p1, ts)
                desired = 0
                done = True

            # 상태 변경
            traded = False
            if desired != self.pos:
                if self.pos != 0:
                    self._close(p1, ts); traded = True
                if desired != 0:
                    self._open(desired, p1, ts); traded = True

            # 리포트(거래 발생 시 또는 주기)
            self.global_steps += 1
            if (ENABLE_REPORT_ON_TRADE and traded) or (REPORT_EVERY_N > 0 and self.global_steps % REPORT_EVERY_N == 0):
                self._generate_current_report()

            self.last_price = p1

            pos_str = "LONG" if self.pos == 1 else ("SHORT" if self.pos == -1 else "STANDBY")
            log_msg = f"[{ts.strftime('%H:%M:%S')}] Pos: {pos_str} | Eq: ${self.eq:,.2f}"
            if self.pos != 0 and self.entry_price is not None:
                log_msg += f" | Entry: ${self.entry_price:,.4f}"
            log_msg += f" | Mgr(model): {mgr_dir},{mgr_conf:.2f} | Act: {action}->{why}"
            print(log_msg)

    # ---------- 체결/회계 ----------
    def _calc_order_qty(self, price: float) -> float:
        """
        주문 수량 계산
        - live 모드: 전체 자산의 99%에 레버리지 적용
        - paper 모드: 레버리지 없이 99%
        """
        leverage_to_use = LEVERAGE if self.mode == "live" else 1.0
        usdt_size = self.eq * 0.99 * leverage_to_use
        return round(usdt_size / price, 6)

    def _open(self, desired: int, price: float, ts: pd.Timestamp):
        if desired not in (-1, 1) or self.pos != 0:
            return

        px = price * (1 + SLIPPAGE) if desired == 1 else price * (1 - SLIPPAGE)

        if self.mode == "live" and self.exec:
            try:
                self.exec.cancel_all_orders(SYMBOL)
                side = "BUY" if desired == 1 else "SELL"
                qty = self._calc_order_qty(px)
                self.exec.place_market(SYMBOL, side, qty, reduce_only=False)
            except Exception as e:
                print(f"[{ts.strftime('%H:%M:%S')}] WARN: live open failed (ignored): {e}")

        self.pos = desired
        self.entry_price = px
        self.entry_time = ts
        self.holding_steps = 0

        notional = self.eq * 0.99 * (LEVERAGE if self.mode == "live" else 1.0)
        self.entry_notional = notional

        # 진입 비용: 수수료 + 슬리피지
        self.eq -= notional * (COMMISSION_SIDE + SLIPPAGE)

        log_path = os.path.join(LOG_DIR, "run_log.csv")
        update_trade_log(log_path, {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "ENTRY",
            "position": "LONG" if desired == 1 else "SHORT",
            "price": f"{px:,.4f}",
            "profit": "-"
        })

    def _close(self, price: float, ts: pd.Timestamp):
        if self.pos == 0:
            return

        px = price * (1 - SLIPPAGE if self.pos == 1 else 1 + SLIPPAGE)
        pnl_pct = (px / self.entry_price - 1) if self.pos == 1 else (self.entry_price / px - 1)

        if self.mode == "live" and self.exec:
            try:
                self.exec.cancel_all_orders(SYMBOL)
                side = "SELL" if self.pos == 1 else "BUY"
                qty  = self._calc_order_qty(px)
                self.exec.place_market(SYMBOL, side, qty, reduce_only=True)
                # (선택) 완전 평단 확인 루프는 제거(슬림)
            except Exception as e:
                print(f"[{ts.strftime('%H:%M:%S')}] WARN: live close failed (ignored): {e}")

        # 비용 차감(보유 중 PnL은 틱마다 반영함)
        exit_notional = float(getattr(self, "entry_notional", 0.0))
        self.eq -= exit_notional * (COMMISSION_SIDE + SLIPPAGE)

        self.total_trades += 1
        if pnl_pct > 0:
            self.winning_trades += 1

        if self.pos == 1:
            self.long_trades += 1
            if pnl_pct > 0: self.long_wins += 1
        else:
            self.short_trades += 1
            if pnl_pct > 0: self.short_wins += 1

        log_path = os.path.join(LOG_DIR, "run_log.csv")
        duration_sec = (ts - self.entry_time).total_seconds() if self.entry_time else 0
        update_trade_log(log_path, {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "EXIT",
            "position": "LONG" if self.pos == 1 else "SHORT",
            "price": f"{px:,.4f}",
            "profit": f"{pnl_pct * 100:.2f}%",
            "duration": f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"
        })

        # 포지션 리셋
        self.pos = 0
        self.entry_price = None
        self.entry_time = None
        self.entry_notional = 0.0
