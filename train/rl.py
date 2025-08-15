"""
RL Training — PPO (Enhanced, no early stop, global phase fix)

핵심
- 변동성 게이트 + 히스테리시스 (시간 강제 없음, 과매매 억제)
- 게이트 점진 완화(Phase별 k_sigma 자동 조절; 전역 스텝 기준으로 정상 동작)
- 학습률 스케줄(SB3 규약: progress_remaining 1→0)
- 탐색(엔트로피) 동적 스케줄(콜백에서 실시간 적용)
- 조기 종료 제거: 최고 성능(VAL) 자동 저장만 유지
- 평가 안정화: TimeLimit + n_eval_episodes=3
- 단순 보상: reward = pos * log_return − (전환 시 수수료)
- 에피소드 길이 제한(10k)로 피드백 주기 단축
- 콘솔 간결, CSV 진단 기록
"""

from __future__ import annotations

import os
import csv
import time
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure

# -----------------
# 경로/설정
# -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
RAW_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "model"))
LOG_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "logs"))

WINDOW = 48                  # 48×5m ≈ 4h
MAX_EPISODE_STEPS = 10_000   # 에피소드 길이 제한(학습/평가 공통)

# 수수료(실거래 값 유지: 테이커 0.05%/사이드)
COMMISSION_SIDE = 0.0005     # 진입 0.05%, 청산 0.05%

# 변동성 게이트(점진적 완화 대상)
VOL_WIN = 24                 # 최근 24바(≈2시간) 표준편차
HYSTERESIS_RATIO = 0.5       # 청산 문턱 = 진입 문턱 × 0.5
FEE_BUFFER = 2 * COMMISSION_SIDE   # 왕복 수수료 = 0.1% = 0.001

SEED = 72
DEVICE = "cpu"
TOTAL_TIMESTEPS = 300_000
EVAL_FREQ = 10_000
LOG_FREQ = 2_000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------
# 유틸: 래퍼 언랩( DummyVecEnv → Monitor → TimeLimit → TradingEnv )
# -----------------
def unwrap_to_trading_env(obj):
    """Monitor/TimeLimit 등 래퍼를 끝까지 벗겨 TradingEnv 반환"""
    env = obj
    if hasattr(env, "envs"):  # DummyVecEnv
        env = env.envs[0]
    while hasattr(env, "env"):  # Monitor/TimeLimit 등 래퍼 제거
        env = env.env
    return env  # TradingEnv

# -----------------
# Phase 스케줄(게이트 완화 + 엔트로피)
# -----------------
def get_phase_config(global_steps: int) -> Dict[str, float]:
    if global_steps < 75_000:      # Phase 1: 강한 제약
        return {"k_sigma": 0.8, "ent_coef": 0.05, "phase": 1}
    elif global_steps < 150_000:   # Phase 2: 중간 제약
        return {"k_sigma": 0.5, "ent_coef": 0.04, "phase": 2}
    elif global_steps < 225_000:   # Phase 3: 약한 제약
        return {"k_sigma": 0.2, "ent_coef": 0.03, "phase": 3}
    else:                          # Phase 4: 최소 제약
        return {"k_sigma": 0.1, "ent_coef": 0.03, "phase": 4}

# -----------------
# 학습률 스케줄(SB3 규약: progress_remaining 1→0)
# -----------------
def learning_rate_schedule(progress_remaining: float) -> float:
    initial_lr, final_lr = 1e-4, 2e-5
    return float(final_lr + (initial_lr - final_lr) * float(progress_remaining))

# -----------------
# PPO 하이퍼(고정 부분)
# -----------------
BASE_PPO_KW = dict(
    learning_rate=learning_rate_schedule,  # 스케줄 함수
    n_steps=8192,
    batch_size=2048,
    n_epochs=10,
    vf_coef=0.5,
    clip_range=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256], ortho_init=False),
)

# -----------------
# 데이터 로더
# -----------------
def _load_norm(split: str) -> pd.DataFrame:
    X = pd.read_parquet(os.path.join(PROC_DIR, f"{split}_normalized.parquet"))
    with open(os.path.join(PROC_DIR, "feature_list.json"), "r") as f:
        feats: List[str] = json.load(f)
    return X.reindex(columns=feats).dropna()

def _load_price(split: str) -> pd.Series:
    df = pd.read_parquet(os.path.join(RAW_DIR, f"fut_{split}_data_5m.parquet"))
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()["Close"].astype(float)

# -----------------
# 콜백: 엔트로피(탐색) 스케줄
# -----------------
class EntropyCoefScheduler(BaseCallback):
    def _on_step(self) -> bool:
        cfg = get_phase_config(self.model.num_timesteps)
        # PPO는 self.ent_coef를 업데이트마다 참조 → 동적 변경 가능
        self.model.ent_coef = cfg["ent_coef"]
        return True

# -----------------
# 환경 (게이트 + 히스테리시스 + 단순 보상)
# -----------------
class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, X: pd.DataFrame, close: pd.Series):
        super().__init__()
        self.X = X
        self.close = close.reindex(X.index).ffill().bfill().astype(float)
        self.features = X.columns.tolist()
        self.n_feat = len(self.features)

        self.action_space = spaces.Discrete(3)  # 0: 홀드, 1: 롱(+1), 2: 숏(-1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(WINDOW * self.n_feat,), dtype=np.float32
        )

        # 로그수익 & 최근 변동성(표준편차)
        self.logret = np.log(self.close / self.close.shift(1)).fillna(0.0)
        self.vol = self.logret.rolling(VOL_WIN, min_periods=1).std().fillna(0.0)

        self._episode_len = len(self.X)

        # 전역 누적 스텝(절대 리셋하지 않음: Phase 판단 기준)
        self.total_steps_global = 0

        self._reset_state()

    def _reset_state(self):
        self.t = WINDOW
        self.pos = 0               # -1/0/+1
        self.entry = None
        self.equity = 1.0
        self.entry_t = None          # 포지션 진입 시점(t) 기록
        self.episode_start = self.t
        # self.total_steps_global 는 리셋하지 않음 (Phase 유지/진행)
        self.diag = {
            "n_steps": 0,
            "n_switch": 0,
            "act_hist": {0: 0, 1: 0, 2: 0},
            "r_sum": 0.0,
            "phase": 1,
            "k_sigma": 0.8,
        }

    def _apply_phase(self):
        cfg = get_phase_config(self.total_steps_global)  # 전역 누적 기준
        self.diag["phase"] = cfg["phase"]
        self.diag["k_sigma"] = cfg["k_sigma"]
        return cfg

    def _obs(self) -> np.ndarray:
        w = self.X.iloc[self.t - WINDOW:self.t].values.astype(np.float32)
        return w.reshape(-1)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        # 매 스텝 Phase 적용(게이트 강도 자동 조절)
        cfg = self._apply_phase()
        k_sigma = cfg["k_sigma"]

        # 액션→목표 포지션: 0=홀드(0), 1=롱(+1), 2=숏(-1)
        requested = 0 if action == 0 else (1 if action == 1 else -1)
        target = requested

        # 히스테리시스 + 변동성 게이트
        if requested != self.pos:
            volatility = float(self.vol.iloc[self.t])
            current_return = float(self.logret.iloc[self.t])

            if self.pos == 0:  # 진입
                thr_enter = FEE_BUFFER + k_sigma * volatility
                if abs(current_return) < thr_enter:
                    target = self.pos  # 진입 거부
            else:               # 청산/전환
                thr_exit = (FEE_BUFFER + k_sigma * volatility) * HYSTERESIS_RATIO
                if requested == 0:  # 청산
                    if abs(current_return) < thr_exit:
                        target = self.pos
                else:  # 전환(롱↔숏)
                    thr_switch = FEE_BUFFER + k_sigma * volatility
                    if abs(current_return) < thr_switch:
                        target = self.pos

        p_prev = float(self.close.iloc[self.t - 1])
        p_curr = float(self.close.iloc[self.t])
        log_ret = float(np.log(p_curr / p_prev))

        # 행동(action)에 따라 목표 포지션을 먼저 확정
        original_pos = self.pos
        
        # 전환이 발생했는지 확인하고 비용 계산
        transaction_cost = 0.0
        if target != original_pos:
            self.diag["n_switch"] += 1
            if original_pos != 0:  # 기존 포지션 청산 비용
                transaction_cost += COMMISSION_SIDE
            if target != 0:        # 신규 포지션 진입 비용
                transaction_cost += COMMISSION_SIDE
                self.entry = p_curr
                self.entry_t = self.t      # << 진입 시점(t) 기록
            else:
                self.entry = None
                self.entry_t = None      # << 청산 시 리셋
            self.pos = target  # 포지션 업데이트
        
        # 새 포지션 기준의 수익 계산
        holding_reward = float(self.pos * log_ret)

        # 포지션 보유 기간 페널티 (Time Decay)
        time_decay_penalty = 0.0
        if self.pos != 0 and self.entry_t is not None:
            holding_period = self.t - self.entry_t
            grace_period_steps = 18  # 90분 / 5분/스텝 = 18 스텝
            if holding_period > grace_period_steps:
                time_decay_penalty = 0.00001 # 유예 기간 이후, 보유하는 매 스텝마다 작은 페널티 부과
        
        # 최종 보상 = 수익 - 거래 비용 - 시간 페널티
        reward = holding_reward - transaction_cost - time_decay_penalty

        # 자본 업데이트(보상과 동일한 값으로 일관 반영)
        self.equity *= float(np.exp(reward))

        # 진단 누적
        self.diag["n_steps"] += 1
        self.diag["act_hist"][action] += 1
        self.diag["r_sum"] += reward

        # 스텝 진행
        self.t += 1
        self.total_steps_global += 1  # 전역 누적 증가 (Phase 경계 통과 가능)

        terminated = (
            self.t >= self._episode_len - 1
            or (self.t - self.episode_start) >= MAX_EPISODE_STEPS
        )
        truncated = False

        info = {
            "equity": self.equity,
            "pos": self.pos,
            "phase": self.diag["phase"],
            "k_sigma": self.diag["k_sigma"],
        }
        return (self._obs() if not terminated else np.zeros_like(self._obs())), float(reward), terminated, truncated, info

# -----------------
# 중간 로그 콜백 (Phase 포함, 평가값 표시 개선)
# -----------------
class MidLogger(BaseCallback):
    def __init__(self, log_freq: int, csv_path: Optional[str]):
        super().__init__()
        self.log_freq = log_freq
        self.csv_path = csv_path
        self._last = 0
        self._t0 = time.time()
        if self.csv_path:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timesteps", "elapsed_sec", "eval_mean", "equity",
                    "act0", "act1", "act2", "switch_rate", "r_mean",
                    "phase", "k_sigma"
                ])
        self._eval_mean = None

    def set_eval(self, val: float | None):
        self._eval_mean = None if val is None else float(val)

    def _on_step(self) -> bool:
        t = int(self.model.num_timesteps)
        if t - self._last >= self.log_freq:
            self._last = t
            elapsed = time.time() - self._t0

            inner = unwrap_to_trading_env(self.model.get_env())
            # (안전) 로그 시점에도 최신 Phase 적용
            inner._apply_phase()

            d = inner.diag
            steps = max(d["n_steps"], 1)
            act0, act1, act2 = d["act_hist"][0], d["act_hist"][1], d["act_hist"][2]
            switch_rate = d["n_switch"] / steps
            r_mean = d["r_sum"] / steps
            eq = float(inner.equity)
            phase = d.get("phase", 1)
            k_sigma = d.get("k_sigma", 0.8)

            disp_eval = "-" if (self._eval_mean is None or not np.isfinite(self._eval_mean)) else round(float(self._eval_mean), 6)
            print(f"[diag] steps={t:,} eq={eq:.6f} eval={disp_eval} act=[{act0},{act1},{act2}] sw={switch_rate:.3f} r_mean={r_mean:.6f} phase={phase} k_sigma={k_sigma}")

            if self.csv_path:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        t, round(elapsed, 2), ("" if disp_eval == "-" else disp_eval), eq,
                        act0, act1, act2, round(switch_rate, 6), round(r_mean, 8),
                        phase, k_sigma
                    ])
            # 누적 리셋 (phase/k_sigma는 유지)
            inner.diag = {
                "n_steps": 0,
                "n_switch": 0,
                "act_hist": {0: 0, 1: 0, 2: 0},
                "r_sum": 0.0,
                "phase": phase,
                "k_sigma": k_sigma,
            }
        return True

# -----------------
# 학습
# -----------------
def main():
    print("[info] Enhanced RL Training Started")
    print("🔧 Gate+Hysteresis, LR schedule, Entropy schedule, (no early stopping)")

    # SB3 및 텐서보드 로그 관련 디렉토리 생성 로직을 비활성화했습니다.
    # log_dir = os.path.join(MODEL_DIR, "sb3_logs")
    # os.makedirs(log_dir, exist_ok=True)

    # 데이터 로드
    X_train = _load_norm("train")
    X_val = _load_norm("val")
    close_train = _load_price("train")
    close_val = _load_price("val")

    def make_train():
        return Monitor(TimeLimit(TradingEnv(X_train, close_train), max_episode_steps=MAX_EPISODE_STEPS))
    def make_val():
        return Monitor(TimeLimit(TradingEnv(X_val, close_val), max_episode_steps=MAX_EPISODE_STEPS))

    train_env = DummyVecEnv([make_train])
    val_env = DummyVecEnv([make_val])

    # 초기 ent_coef는 Phase 1 값으로 설정
    init_cfg = get_phase_config(0)
    ppo_kwargs = dict(BASE_PPO_KW, ent_coef=init_cfg["ent_coef"])

    model = PPO(
        "MlpPolicy",
        train_env,
        device=DEVICE,
        seed=SEED,
        verbose=0,
        tensorboard_log=None,  # 텐서보드 로그 비활성화
        **ppo_kwargs,
    )
    # model.set_logger(configure(log_dir, ["csv"])) # SB3 자체 로그 비활성화

    # 콜백 구성
    # csv_log = os.path.join(LOG_DIR, "train_log.csv") # train_log.csv 생성 비활성화
    mid_logger = MidLogger(LOG_FREQ, None) # csv_path를 None으로 전달
    ent_sched = EntropyCoefScheduler()

    class EnhancedEvalCallback(EvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_eval_count = 0  # 완료된 평가 라운드 수 추적

        def _on_step(self) -> bool:
            # 평가 직전/중 Phase 동기화(트레인/밸리데이션 양쪽)
            try:
                unwrap_to_trading_env(self.model.get_env())._apply_phase()
                unwrap_to_trading_env(self.eval_env)._apply_phase()
            except Exception:
                pass

            cont = super()._on_step()

            # 새 평가 라운드가 끝났으면 그때만 콘솔/CSV 갱신
            if hasattr(self, "evaluations_timesteps"):
                cur_cnt = len(self.evaluations_timesteps)
                if cur_cnt > self._last_eval_count and self.last_mean_reward is not None and np.isfinite(self.last_mean_reward):
                    self._last_eval_count = cur_cnt
                    mid_logger.set_eval(float(self.last_mean_reward))
            return cont

    eval_cb = EnhancedEvalCallback(
        val_env,
        best_model_save_path=MODEL_DIR,  # 최고 성능 모델 자동 저장
        log_path=None,  # SB3 평가 로그 비활성화
        eval_freq=EVAL_FREQ,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # 주기적 체크포인트 저장을 비활성화하고 EvalCallback의 best model 저장 기능만 사용합니다.
    cb = CallbackList([mid_logger, ent_sched, eval_cb])

    # 학습 시작
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=False)
        print("[OK] Training completed")
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user")

if __name__ == "__main__":
    main()
