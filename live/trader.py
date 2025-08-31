# ai_binance/live/trader.py
"""
Trader (HRL-Ready) for UM Futures — Manager(방향/확신) + Worker(타이밍)
- REFACTORED: fe.py와 동일한 피처를 사용하도록 데이터 파이프라인 수정.
- 인제스터 패킷(원시 피처)을 받아 실시간 의사결정.
- X는 dict({'5m','15m','1h','4h','btc1h'}) 구조를 사용.
- Worker 관측: 훈련과 동일하게 모든 TF의 원시 피처를 조합하여 사용.
"""

from __future__ import annotations

import os, math, json
from datetime import datetime, timezone
from typing import Dict, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from queue import Queue, Empty

import sys
import os

# Ensure ai_binance is in sys.path for model loading
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

SYMBOL = "ETHUSDT"
MAX_HOLDING_STEPS = 72
COMMISSION_SIDE   = 0.0005
SLIPPAGE          = 0.0001
VOL_WIN           = 24
HYSTERESIS_RATIO  = 0.5
FEE_BUFFER        = 2 * COMMISSION_SIDE
LEVERAGE = 5  # 실전 모드에서 사용할 레버리지

def get_phase_config(global_steps: int) -> Dict[str, float]:
    if global_steps < 75_000:      return {"k_sigma": 0.8, "phase": 1}
    elif global_steps < 150_000:   return {"k_sigma": 0.5, "phase": 2}
    elif global_steps < 225_000:   return {"k_sigma": 0.2, "phase": 3}
    else:                          return {"k_sigma": 0.1, "phase": 4}

INITIAL_CAPITAL  = 100_000.0
ROLLOUT_STEPS = 288

MANAGER_MODEL_PATH   = os.path.join(MODEL_DIR, "manager_v2.zip")
MANAGER_VECNORM_PATH = os.path.join(MODEL_DIR, "manager_v2_vecnorm.pkl")
WORKER_MODEL_PATH    = os.path.join(MODEL_DIR, "worker_unified_final.zip")
WORKER_VECNORM_PATH  = os.path.join(MODEL_DIR, "worker_unified_vecnorm.pkl")

class _ObsOnlyEnv(Env):
    def __init__(self, obs_shape: Tuple[int, ...], action_space: spaces.Space):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = action_space
    def reset(self, *, seed=None, options=None): return np.zeros(self.observation_space.shape, np.float32), {}
    def step(self, action): return self.reset()[0], 0.0, True, False, {}

class Trader:
    def __init__(self, mode: str, q: Queue, api_key: Optional[str] = None, secret_key: Optional[str] = None, learn_q: Optional[Queue] = None):
        assert mode in ("live", "paper")
        self.mode, self.q, self.learn_q, self.api_key, self.secret_key = mode, q, learn_q, api_key, secret_key
        
        self.exec = None
        if self.mode == "live" and self.api_key and self.secret_key:
            try:
                self.exec = BinanceExecutor(self.api_key, self.secret_key)
                print("[트레이더] 실행 어댑터 초기화 완료.")
            except Exception as e:
                print(f"[트레이더] 실행 어댑터 초기화 실패: {e}")
        elif self.mode == "live":
            print(f"[트레이더] 경고: 'live' 모드지만 API 키가 없어 'paper' 모드로 동작합니다.")

        # Manager용 feature_list 로드 (훈련 순서와 동일하게)
        mgr_feats_1h, mgr_feats_4h = [], []
        with open(os.path.join(PROC_DIR, "fe_feature_list_1h.json"), "r") as f: mgr_feats_1h = json.load(f)
        with open(os.path.join(PROC_DIR, "fe_feature_list_4h.json"), "r") as f: mgr_feats_4h = json.load(f)
        self.mgr_cols = mgr_feats_1h + mgr_feats_4h
        print(f"[트레이더] Manager 피처 로드 완료: {len(self.mgr_cols)}개 (훈련 순서 반영)")

        # Worker 모델/VecNorm 로드
        self.worker_model: MaskablePPO = MaskablePPO.load(WORKER_MODEL_PATH, device="cpu")
        obs_shape_w = self.worker_model.observation_space.shape
        tmp_w_env = DummyVecEnv([lambda: _ObsOnlyEnv(obs_shape=obs_shape_w, action_space=self.worker_model.action_space)])
        self.worker_vecnorm: VecNormalize = VecNormalize.load(WORKER_VECNORM_PATH, tmp_w_env)
        self.worker_vecnorm.training = False
        self.worker_vecnorm.norm_reward = False
        print(f"[트레이더] Worker 로드 완료: {os.path.basename(WORKER_MODEL_PATH)} | obs_shape={obs_shape_w}")

        # Worker 피처 리스트 재구성 (fe.py 기준, 훈련 순서와 동일하게)
        print("[트레이더] Worker 피처 리스트를 훈련 과정에 맞춰 재구성합니다...")
        w_feats_5m, w_feats_15m, w_feats_1h, w_feats_4h = [], [], [], []
        with open(os.path.join(PROC_DIR, "fe_feature_list_5m.json"), "r") as f: w_feats_5m = json.load(f)
        with open(os.path.join(PROC_DIR, "fe_feature_list_15m.json"), "r") as f: w_feats_15m = json.load(f)
        with open(os.path.join(PROC_DIR, "fe_feature_list_1h.json"), "r") as f: w_feats_1h = json.load(f)
        with open(os.path.join(PROC_DIR, "fe_feature_list_4h.json"), "r") as f: w_feats_4h = json.load(f)
        
        btc_features = [
            'ret_1h_btc1h', 'ret_4h_btc1h', 'atr14_btc1h', 'HA_O_btc1h', 'HA_H_btc1h',
            'HA_L_btc1h', 'HA_C_btc1h', 'HA_TR_btc1h', 'HA_BC_btc1h', 'HA_R_btc1h'
        ]
        self.worker_feature_list = w_feats_5m + w_feats_15m + w_feats_1h + w_feats_4h + btc_features
        print(f"[트레이더] Worker 피처 리스트 재구성 완료: {len(self.worker_feature_list)}개 피처")

        # 관측 차원 정합성 검증
        vecnorm_dim = int(self.worker_vecnorm.obs_rms.mean.shape[0])
        extra_dim = 5
        expected_dim = len(self.worker_feature_list) + extra_dim
        if expected_dim != vecnorm_dim:
            raise RuntimeError(f"[트레이더] 관측 차원 불일치: expected={expected_dim}, vecnorm={vecnorm_dim}")
        print(f"[트레이더] Worker 관측 정합 확인 완료: obs_dim={expected_dim}")

        # Manager 모델 로드 (선택)
        self.manager_model: Optional[PPO] = None
        if os.path.exists(MANAGER_MODEL_PATH) and os.path.exists(MANAGER_VECNORM_PATH):
            try:
                                # self.mgr_cols는 이미 위에서 훈련 순서에 맞춰 생성되었으므로 이 라인은 제거합니다.
                obs_shape_mgr = (8, len(self.mgr_cols))
                act_space_mgr = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

                tmp_mgr_env = DummyVecEnv([lambda: _ObsOnlyEnv(obs_shape=obs_shape_mgr, action_space=act_space_mgr)])
                self.manager_vecnorm = VecNormalize.load(MANAGER_VECNORM_PATH, tmp_mgr_env)
                self.manager_vecnorm.training = False
                self.manager_vecnorm.norm_reward = False

                # ▼ 핵심: 학습시 설정을 강제로 주입해 사이즈 미스매치 방지
                policy_kwargs = dict(
                    features_extractor_class=TransformerFeatureExtractor,
                    features_extractor_kwargs=dict(d_model=128, nhead=4, num_layers=2),
                    net_arch=dict(pi=[128], vf=[128]),
                )
                self.manager_model = PPO.load(
                    MANAGER_MODEL_PATH,
                    device="cpu",
                    custom_objects={
                        "policy_kwargs": policy_kwargs,
                        # 커스텀 스케줄 역직렬화 실패 대비 기본값 주입
                        "lr_schedule": (lambda _pr: 3e-4),
                        "learning_rate": 3e-4,
                    },
                )
                print(f"[트레이더] Manager 로드 완료: {os.path.basename(MANAGER_MODEL_PATH)}")
            except Exception as e:
                print(f"[트레이더] Manager 로드 실패 → 휴리스틱 사용: {e}")
                self.manager_model = None

        # 상태 변수 초기화 (실제 잔액 조회)
        start_capital = INITIAL_CAPITAL
        if self.exec:
            try:
                balance = self.exec.get_usdt_balance()
                if balance > 0:
                    start_capital = balance
                    print(f"[트레이더] 실제 계좌 잔액 ${balance:,.2f}을 초기 자본금으로 설정합니다.")
                else:
                    print("[트레이더] 경고: 실제 계좌 잔액이 0이거나 가져올 수 없어, 기본값으로 시작합니다.")
            except Exception as e:
                print(f"[트레이더] 경고: 실제 계좌 잔액 조회 실패({e}), 기본값으로 시작합니다.")
        
        self.initial_capital = start_capital
        self.eq, self.pos, self.entry_price, self.entry_time, self.last_price, self.holding_steps, self.global_steps = self.initial_capital, 0, None, None, None, 0, 0
        self.total_trades, self.winning_trades, self.long_trades, self.long_wins, self.short_trades, self.short_wins = 0,0,0,0,0,0
        self.start_time = datetime.now(timezone.utc)
        self._roll_obs, self._roll_actions, self._roll_rewards, self._roll_dones, self._roll_values, self._roll_logps = [],[],[],[],[],[]

        # 초기 리포트 생성
        report_path = os.path.join(REPORT_DIR, "trading_report.md")
        initial_report_data = {
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
        }
        generate_report(report_path, initial_report_data, is_new_session=True)

    def _manager_goal_dict(self, X_dict: Dict[str, pd.DataFrame]) -> Tuple[int, float, int]:
        def _latest(df: Optional[pd.DataFrame], col: str, default: float = 0.0) -> float:
            try: return float(df[col].iloc[-1]) if df is not None and not df.empty and col in df.columns else default
            except Exception: return default
        X1, X4 = X_dict.get("1h"), X_dict.get("4h")
        macd_h1, macd_h4 = _latest(X1, "macd_hist_1h"), _latest(X4, "macd_hist_4h")
        rsi_h1, rsi_h4 = _latest(X1, "rsi14_1h", 50.0) - 50.0, _latest(X4, "rsi14_4h", 50.0) - 50.0
        ret3_h1, ret12_h1 = _latest(X1, "ret3_1h"), _latest(X1, "ret12_1h")
        score = (1.8*macd_h1 + 1.2*macd_h4 + 0.6*(rsi_h1/50.0) + 1.0*ret3_h1 + 0.7*ret12_h1)
        direction = np.sign(score).astype(int)
        conf = float(1 - math.exp(-min(5.0, abs(score)) * 0.5))
        return direction, conf, 1 if conf < 0.15 else 0

    def _build_worker_obs(self, X_dict: Dict[str, pd.DataFrame], t_idx: pd.Timestamp, mgr_dir: int, mgr_conf: float, mgr_regime: int) -> Optional[np.ndarray]:
        ordered_tfs = ["5m", "15m", "1h", "4h", "btc1h"]
        series_list = []
        for tf in ordered_tfs:
            df = X_dict.get(tf)
            if df is None or df.empty: print(f"[트레이더] 경고: {tf} 데이터 누락"); return None
            try:
                pos = df.index.get_indexer([t_idx], method="pad")[0]
                if pos == -1: print(f"[트레이더] 경고: {tf}에서 {t_idx} 타임스탬프 없음"); return None
                series_list.append(df.iloc[pos])
            except Exception as e: print(f"[트레이더] 경고: {tf} 데이터 처리 오류: {e}"); return None
        
        x_combined = pd.concat(series_list)
        x_reordered = x_combined.reindex(self.worker_feature_list).fillna(0.0)
        x_raw = x_reordered.to_numpy(dtype=np.float32)

        holding_norm = min(self.holding_steps, MAX_HOLDING_STEPS) / MAX_HOLDING_STEPS
        extra_obs = np.array([mgr_dir, mgr_conf, mgr_regime, float(self.pos != 0), holding_norm], dtype=np.float32)
        return np.concatenate([x_raw, extra_obs])

    def _build_manager_obs_live(self, X_dict, t_idx) -> Optional[np.ndarray]:
        import numpy as np
        import pandas as pd
        base = X_dict.get("1h")
        if base is None or base.empty: return None

        pos = base.index.get_indexer([t_idx], method="pad")[0]
        if pos < 0: return None

        start = max(0, pos - 7)
        idx_win = base.index[start:pos+1]
        rows = []
        for ts in idx_win:
            vals = []
            for c in self.mgr_cols:
                tf = "1h" if c.endswith("_1h") else "4h"
                df = X_dict.get(tf)
                if df is None or df.empty or c not in df.columns:
                    vals.append(0.0); continue
                j = df.index.get_indexer([ts], method="pad")[0]
                vals.append(float(df[c].iloc[0 if j < 0 else j]))
            rows.append(vals)

        seq = np.asarray(rows, dtype=np.float32)
        if seq.shape[0] < 8:  # 앞쪽 패딩
            pad = np.repeat(seq[:1, :], 8 - seq.shape[0], axis=0)
            seq = np.concatenate([pad, seq], axis=0)
        return seq.astype(np.float32, copy=False)

    @staticmethod
    def _mask(pos: int) -> np.ndarray: return np.array([True, pos==0, pos!=0], dtype=bool)

    def _interpret(self, raw_action: int, mgr_dir: int) -> Tuple[int, str]:
        if self.pos == 0: return (np.sign(mgr_dir).astype(int), "enter") if raw_action == 1 and mgr_dir != 0 else (0, "wait")
        else: return (0, "exit") if raw_action == 2 else (self.pos, "hold")

    def _generate_current_report(self):
        report_path = os.path.join(REPORT_DIR, "trading_report.md")
        
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        long_win_rate = (self.long_wins / self.long_trades * 100) if self.long_trades > 0 else 0
        short_win_rate = (self.short_wins / self.short_trades * 100) if self.short_trades > 0 else 0

        unrealized_pnl_amount = 0.0
        unrealized_pnl_percent = 0.0
        
        if self.pos != 0 and self.entry_price and self.last_price:
            leverage = LEVERAGE if self.mode == 'live' else 1.0
            if self.pos == 1:
                unrealized_pnl_percent = (self.last_price / self.entry_price - 1.0) * 100.0 * leverage
            else:
                unrealized_pnl_percent = (self.entry_price / self.last_price - 1.0) * 100.0 * leverage

        current_pos_str = "STANDBY"
        if self.pos == 1: current_pos_str = "LONG"
        if self.pos == -1: current_pos_str = "SHORT"

        report_data = {
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
        }
        
        generate_report(report_path, report_data, is_new_session=False)

    @torch.no_grad()
    def run(self):
        print("[트레이더] HRL 모드 실행 중... (큐 수신 대기)")
        while True:
            try: pkt = self.q.get(timeout=300)
            except Empty: continue

            X_dict, ts = pkt["X"], pkt["ts"]
            close_series = pkt["close"]
            if close_series.empty: continue
            
            t_idx = close_series.index[-1]
            p1 = close_series.iloc[-1]
            
            if self.last_price is None:
                self.last_price = p1

            # === Manager: RL 모델 우선, 실패 시 휴리스틱 ===
            mgr_src = "heur"
            use_mgr = False
            if self.manager_model is not None:
                obs_mgr = self._build_manager_obs_live(X_dict, t_idx)
                if obs_mgr is not None:
                    try:
                        # (8, F) 2D 데이터 정규화 후 (1, 8, F) 3D 배치로 변환
                        obs_mgr_2d_norm = self.manager_vecnorm.normalize_obs(obs_mgr)
                        obs_mgr_3d_for_predict = obs_mgr_2d_norm.reshape(1, *obs_mgr_2d_norm.shape)
                        act, _ = self.manager_model.predict(obs_mgr_3d_for_predict, deterministic=True)
                        act = np.asarray(act)

                        # SB3는 n_envs=1일 때 (1, action_dim)로 나올 수 있음
                        if act.ndim == 2:
                            action_vec = act[0]
                        else:
                            action_vec = act.reshape(-1)

                        if action_vec.shape[0] < 2:
                            raise RuntimeError(f"manager action shape unexpected: {act.shape}")

                        conf_long, conf_short = float(action_vec[0]), float(action_vec[1])

                        # 데드존(학습값과 동일한 0.1) 처리
                        if max(conf_long, conf_short) < 0.1:
                            mgr_dir = 0
                        else:
                            mgr_dir = 1 if conf_long > conf_short else -1
                        mgr_conf = max(conf_long, conf_short)
                        mgr_regime = 0  # 필요하면 레짐 로직 별도 도입
                        use_mgr = True
                        mgr_src = "model"
                    except Exception as e:
                        print(f"[매니저] predict 실패 → 휴리스틱 대체: {e}")

            if not use_mgr:
                mgr_dir, mgr_conf, mgr_regime = self._manager_goal_dict(X_dict)

            obs_raw = self._build_worker_obs(X_dict, t_idx, mgr_dir, mgr_conf, mgr_regime)
            if obs_raw is None: continue

            obs_norm = self.worker_vecnorm.normalize_obs(obs_raw.reshape(1, -1))
            action_array, _ = self.worker_model.predict(obs_norm, deterministic=True, action_masks=self._mask(self.pos))
            action = int(action_array[0]) if isinstance(action_array, np.ndarray) else int(action_array)

            lr = np.log(p1 / self.last_price) if self.last_price else 0.0
            desired, why = self._interpret(action, mgr_dir)
            
            if self.pos != 0:
                self.eq *= np.exp(self.pos * lr)
                self.holding_steps += 1

            # --- Handle trade execution & done signal ---
            done = (desired != self.pos)

            if self.pos != 0 and self.holding_steps >= MAX_HOLDING_STEPS:
                print(f"[{ts.strftime('%H:%M:%S')}] WARN: 최대 보유 기간({MAX_HOLDING_STEPS} 스텝) 초과로 강제 청산.")
                self._close(p1, ts)
                desired = 0 # 이미 청산했으므로 추가 행동 방지
                done = True

            if desired != self.pos:
                if self.pos != 0: self._close(p1, ts)
                if desired != 0: self._open(desired, p1, ts)

            # --- Online Learning: Append to rollout buffer ---
            if self.learn_q is not None:
                reward = float(self.pos * lr)
                
                self._roll_obs.append(obs_raw)
                self._roll_actions.append(action_array) # Send the array
                self._roll_rewards.append(reward)
                self._roll_dones.append(done)
                self._roll_values.append(0.0) # Placeholder
                self._roll_logps.append(0.0)  # Placeholder

                if len(self._roll_obs) >= ROLLOUT_STEPS:
                    try:
                        rollout_data = {
                            "obs": np.array(self._roll_obs, dtype=np.float32),
                            "actions": np.array(self._roll_actions),
                            "rewards": np.array(self._roll_rewards, dtype=np.float32),
                            "dones": np.array(self._roll_dones, dtype=np.bool_),
                            "values": np.array(self._roll_values, dtype=np.float32),
                            "log_probs": np.array(self._roll_logps, dtype=np.float32)
                        }
                        self.learn_q.put_nowait(rollout_data)
                        print(f"[{ts.strftime('%H:%M:%S')}] INFO: Rollout buffer sent to learner ({len(self._roll_obs)} steps).")
                        self._roll_obs, self._roll_actions, self._roll_rewards, self._roll_dones, self._roll_values, self._roll_logps = [],[],[],[],[],[]
                    except Exception as e:
                        print(f"[{ts.strftime('%H:%M:%S')}] WARN: Failed to send rollout to learner: {e}")
                        self._roll_obs, self._roll_actions, self._roll_rewards, self._roll_dones, self._roll_values, self._roll_logps = [],[],[],[],[],[]

            self.last_price = p1
            self.global_steps += 1
            
            pos_str = "LONG" if self.pos == 1 else ("SHORT" if self.pos == -1 else "STANDBY")
            print(f"[{ts.strftime('%H:%M:%S')}] Pos: {pos_str} | Eq: ${self.eq:,.2f} | Mgr({mgr_src}): {mgr_dir},{mgr_conf:.2f} | Act: {action}->{why}")

            self._generate_current_report()

    def _calc_order_qty(self, price: float) -> float:
        '''
        주문 수량을 계산합니다.
        - live 모드: 전체 자산에 5배 레버리지를 적용하여 진입합니다.
        - paper 모드: 레버리지 없이 전체 자산으로 진입합니다.
        '''
        leverage_to_use = 1.0
        if self.mode == "live":
            leverage_to_use = LEVERAGE

        # 현재 총 자산(equity)을 기반으로 주문 USDT 크기를 결정
        usdt_size = self.eq * 0.99 * leverage_to_use
        
        # USDT 크기를 현재 가격으로 나누어 주문할 코인 수량을 계산
        return round(usdt_size / price, 6)

    def _open(self, desired: int, price: float, ts: pd.Timestamp):
        """
        포지션 진입 처리.
        desired:  1=LONG, -1=SHORT
        """
        if desired not in (-1, 1):
            return
        if self.pos != 0:
            # 이미 포지션이 있으면 무시
            return

        # 체결가에 슬리피지 반영
        px = price * (1 + SLIPPAGE) if desired == 1 else price * (1 - SLIPPAGE)

        # === 실거래 진입 (live 전용) ===
        if self.mode == "live" and self.exec:
            try:
                # 안전하게 기존 주문 삭제
                self.exec.cancel_all_orders(SYMBOL)
                side = "BUY" if desired == 1 else "SELL"
                qty = self._calc_order_qty(px)
                self.exec.place_market(SYMBOL, side, qty, reduce_only=False)
            except Exception as e:
                print(f"[{ts.strftime('%H:%M:%S')}] WARN: live open failed (ignored): {e}")

        # === 로컬 상태 갱신 ===
        self.pos = desired
        self.entry_price = px
        self.entry_time = ts
        self.holding_steps = 0

        # 진입 수수료 차감
        self.eq -= self.eq * COMMISSION_SIDE

        # 로그
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

        # === 실제 청산 추가 (live 전용) ===
        if self.mode == "live" and self.exec:
            try:
                # 1) 남은 주문 전량 취소 (잔여 SL/TP 제거)
                self.exec.cancel_all_orders(SYMBOL)
                # 2) reduceOnly 시장가로 반대 방향 청산
                side = "SELL" if self.pos == 1 else "BUY"
                qty  = self._calc_order_qty(px)  # reduceOnly라 포지션 초과 수량이어도 초과분은 무시됨
                self.exec.place_market(SYMBOL, side, qty, reduce_only=True)

                # (옵션) 완전 평단 확인을 1~2초만 대기
                import time as _t
                deadline = _t.time() + 2.0
                while _t.time() < deadline:
                    pr = self.exec._send_request("GET", "/fapi/v2/positionRisk", [("symbol", SYMBOL)])
                    rows = pr if isinstance(pr, list) else [pr]
                    amt = 0.0
                    for r in rows:
                        if r.get("symbol") == SYMBOL:
                            amt = float(r.get("positionAmt", "0"))
                            break
                    if abs(amt) < 1e-12:
                        break
                    _t.sleep(0.1)
            except Exception as e:
                print(f"[{ts.strftime('%H:%M:%S')}] WARN: live close failed (ignored): {e}")

        # === 로컬 회계/로그 (기존 그대로) ===
        self.eq += self.eq * pnl_pct - self.eq * COMMISSION_SIDE
        self.total_trades += 1
        self.winning_trades += 1 if pnl_pct > 0 else 0

        duration_sec = (ts - self.entry_time).total_seconds() if self.entry_time else 0
        log_path = os.path.join(LOG_DIR, "run_log.csv")
        update_trade_log(log_path, {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "type": "EXIT",
            "position": "LONG" if self.pos == 1 else "SHORT",
            "price": f"{px:,.4f}",
            "profit": f"{pnl_pct * 100:.2f}%",
            "duration": f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"
        })

        if self.pos == 1: self.long_trades += 1; self.long_wins += 1 if pnl_pct > 0 else 0
        else:              self.short_trades += 1; self.short_wins += 1 if pnl_pct > 0 else 0
        self.pos = 0

