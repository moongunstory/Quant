# rl_entry_exit.py
# PPO + Lagrangian Cost Constraint (λ 자동) + 4-액션(청산 포함)
# - 액션: 0=no-trade, 1=long, 2=short, 3=flat(청산)
# - 전환 금지: 보유중 long/short 주문은 "홀드", 전환은 flat→다음 틱 재진입만
# - 학습 보상: REWARD_SCALE*(pnl - fee - funding) − λ*(cost − budget)
#              + (트렌드 구간 flat 페널티) + (진입 직후 n틱 방향 보너스)
# - 자산 업데이트: equity *= exp(pnl - fee - funding)
# - λ: EMA(cost) 기반 듀얼 업데이트 (매 스텝)

from __future__ import annotations

import os
import csv
import time
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList

# -----------------
# 경로/설정
# -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
RAW_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "model"))
LOG_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "logs"))

WINDOW = 48                  # 48×5m ≈ 4h
MAX_EPISODE_STEPS = 10_000   # 리셋 주기(학습 내부 에피소드)

# 거래 비용
COMMISSION_SIDE = 0.0005     # 테이커 0.05%/side

# ---- 라그랑주(자동 λ) 파라미터 ----
COST_BUDGET = 1.2e-4         # 현실화(권장 1e-4~1.5e-4 중간값)
LAMBDA_STEP = 100.0          # η: 과도 반응 완화(권장 50~200)
LAMBDA_MAX  = 8.0            # 보상 스케일에 맞춘 상한(권장 5~10)

# ---- 보상 스케일 & 트렌드 보상 파라미터 ----
REWARD_SCALE      = 200.0     # 50~200 권장
TREND_THRESH      = 8e-4      # |1-스텝 로그수익| 기준 트렌드 임계
ENTRY_BONUS_H     = 3         # n틱 후 방향 보너스 평가(예: 3→15분)
ENTRY_BONUS_ALPHA = 50.0      # 진입 방향 일치 보너스 계수
HOLD_PENALTY      = 0.02      # 트렌드에 flat(무포지션)일 때 페널티

SEED = 72
DEVICE = "cpu"
TOTAL_TIMESTEPS = 300_000
EVAL_FREQ = 10_000
LOG_FREQ = 2_000

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -----------------
# 유틸
# -----------------
def unwrap_to_trading_env(obj):
    env = obj
    if hasattr(env, "envs"):  # DummyVecEnv
        env = env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env  # TradingEnv

def learning_rate_schedule(progress_remaining: float) -> float:
    initial_lr, final_lr = 5e-5, 1e-5
    return float(final_lr + (initial_lr - final_lr) * float(progress_remaining))

BASE_PPO_KW = dict(
    learning_rate=learning_rate_schedule,
    n_steps=16384,
    batch_size=4096,
    n_epochs=10,
    vf_coef=0.5,
    clip_range=0.10,
    gamma=0.995,
    gae_lambda=0.95,
    max_grad_norm=0.5,
    ent_coef=0.01,          # no-trade 탐색 약하게
    target_kl=0.02,
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

def _load_aux_data(split: str) -> Tuple[pd.Series, pd.Series]:
    path = os.path.join(RAW_DIR, f"fut_{split}_data_5m.parquet")
    if not os.path.exists(path):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    close = df["Close"].astype(float)
    funding_rate = df.get("FundingRate", pd.Series(0.0, index=df.index)).astype(float)
    return close, funding_rate

# -----------------
# 환경
# -----------------
class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, X: pd.DataFrame, close: pd.Series, funding_rate: pd.Series):
        super().__init__()
        self.X = X
        self.close = close.reindex(X.index).ffill().bfill().astype(float)
        self.funding_rate = funding_rate.reindex(X.index).ffill().bfill().astype(float)
        self.features = X.columns.tolist()
        self.n_feat = len(self.features)

        # 상태 피처 3개: pos, time_in_pos_norm, unrealized_pnl_log
        self.extra_dim = 3

        # 4-액션: 0=no-trade, 1=long, 2=short, 3=flat(청산)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(WINDOW * self.n_feat + self.extra_dim,), dtype=np.float32
        )

        self.logret = np.log(self.close / self.close.shift(1)).fillna(0.0)
        self._episode_len = len(self.X)

        # 라그랑주 듀얼 변수
        self.lam = 1.0                    # warm start
        self.eta = float(LAMBDA_STEP)
        self.cost_budget = float(COST_BUDGET)
        self.lam_max = float(LAMBDA_MAX)

        # 비용 EMA
        self.cost_ema = 0.0
        self.alpha_cost_ema = 0.05        # 빠른 반응

        self._reset_state()

    # ---------- 내부 상태 ----------
    def _reset_state(self):
        self.t = WINDOW
        self.pos = 0               # -1/0/+1
        self.entry = None
        self.entry_t = None
        self.equity = 1.0
        self.episode_start = self.t
        self.diag = {
            "n_steps": 0,
            "n_entry": 0,
            "n_exit": 0,
            "act_hist": {0: 0, 1: 0, 2: 0, 3: 0},
            "r_sum": 0.0,
            "lam": float(self.lam),
            "cost_ema": 0.0,
        }

    def _state_vec(self) -> np.ndarray:
        time_in_pos = 0 if self.entry_t is None else (self.t - self.entry_t)
        time_in_pos_norm = min(time_in_pos, 1000) / 1000.0
        if self.entry is None or self.pos == 0:
            upnl_log = 0.0
        else:
            upnl_log = float(np.log(self.close.iloc[self.t] / self.entry)) * float(self.pos)
        return np.array([float(self.pos), float(time_in_pos_norm), float(upnl_log)], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        w = self.X.iloc[self.t - WINDOW:self.t].values.astype(np.float32).reshape(-1)
        return np.concatenate([w, self._state_vec()], axis=0)

    # ---------- 액션 해석 ----------
    @staticmethod
    def _map_action(action: int, pos: int) -> int:
        """
        반환: target pos ∈ {-1,0,+1}
        - pos==0: {0:0, 1:+1, 2:-1, 3:0}
        - pos!=0: {3:0, 그 외: pos(홀드)}  → 같은 틱 전환 금지
        """
        a = int(action)
        if pos == 0:
            return {0: 0, 1: +1, 2: -1, 3: 0}.get(a, 0)
        else:
            return 0 if a == 3 else pos

    # ---------- Gym API ----------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def step(self, action: int):
        # 시계열
        p_prev = float(self.close.iloc[self.t - 1])
        p_curr = float(self.close.iloc[self.t])
        lr = float(np.log(p_curr / p_prev))
        ts = self.X.index[self.t]

        prev_pos = self.pos
        target = self._map_action(action, prev_pos)

        # --- 체결 비용(전환 금지라 pos 변경 시 1사이드)
        transaction_cost = 0.0
        if prev_pos != target:
            transaction_cost += COMMISSION_SIDE
            if prev_pos == 0 and target != 0:
                # 진입
                self.entry = p_curr
                self.entry_t = self.t
                self.diag["n_entry"] += 1
            elif prev_pos != 0 and target == 0:
                # 청산
                self.entry = None
                self.entry_t = None
                self.diag["n_exit"] += 1

        # --- 펀딩(정각 & 8시간 배수; prev_pos 기준)
        fr = float(self.funding_rate.iloc[self.t])
        is_funding_event = (getattr(ts, "minute", 0) == 0) and (getattr(ts, "hour", 0) % 8 == 0)
        funding_fee_cost = (prev_pos * fr) if is_funding_event else 0.0

        # --- 보상/자산 분리 + 스케일/형태 보강
        # 학습용 보상: r_train = REWARD_SCALE*(pnl - fee - funding) - λ*(cost - budget)
        #  + (트렌드 flat 페널티) + (진입 시 n틱 방향 보너스)
        # 자산반영:    equity *= exp(pnl - fee - funding)
        inst_cost = transaction_cost
        equity_reward = float(prev_pos * lr) - (transaction_cost + funding_fee_cost)
        reward = REWARD_SCALE * equity_reward - (self.lam * (inst_cost - self.cost_budget))

        # (A) 트렌드 구간에서 flat(무포지션 유지) 페널티
        if prev_pos == 0 and target == 0:
            if abs(lr) > TREND_THRESH:
                reward -= HOLD_PENALTY

        # (B) 진입 시, n틱 후 방향 일치 보너스
        if prev_pos == 0 and target != 0:
            if self.t + ENTRY_BONUS_H < self._episode_len:
                p_future = float(self.close.iloc[self.t + ENTRY_BONUS_H])
                fut_lr = float(np.log(p_future / p_curr))  # n틱 후 로그수익
                reward += ENTRY_BONUS_ALPHA * (target * fut_lr)

        # ----- λ 업데이트 (EMA 기반) -----
        self.cost_ema = (1.0 - self.alpha_cost_ema) * self.cost_ema + self.alpha_cost_ema * inst_cost
        lam_grad = self.cost_ema - self.cost_budget
        self.lam = float(np.clip(self.lam + self.eta * lam_grad, 0.0, self.lam_max))

        # 포지션 갱신(다음 틱부터 유효)
        self.pos = target

        # 자산 업데이트(λ 제외)
        self.equity *= float(np.exp(equity_reward))

        # 지표/진단
        self.diag["n_steps"] += 1
        self.diag["act_hist"][int(action)] = self.diag["act_hist"].get(int(action), 0) + 1
        self.diag["r_sum"] += reward
        self.diag["lam"] = float(self.lam)
        self.diag["cost_ema"] = float(self.cost_ema)

        # 진행
        self.t += 1

        terminated = (
            self.t >= self._episode_len - 1
            or (self.t - self.episode_start) >= MAX_EPISODE_STEPS
        )
        truncated = False

        obs = (self._obs() if not terminated else np.zeros_like(self._obs()))
        info = {
            "equity": float(self.equity),
            "pos": int(self.pos),
            "lam": float(self.lam),
            "cost_ema": float(self.cost_ema),
        }
        return obs, float(reward), terminated, truncated, info

# -----------------
# 중간 로그 콜백
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
                    "act0", "act1", "act2", "act3", "entry_cnt", "exit_cnt",
                    "r_mean", "lambda", "cost_ema"
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

            d = inner.diag
            steps = max(d["n_steps"], 1)
            a0 = d["act_hist"].get(0, 0); a1 = d["act_hist"].get(1, 0)
            a2 = d["act_hist"].get(2, 0); a3 = d["act_hist"].get(3, 0)
            r_mean = d["r_sum"] / steps
            eq = float(inner.equity)
            n_entry = d.get("n_entry", 0)
            n_exit = d.get("n_exit", 0)
            lam = d.get("lam", 0.0)
            cost_ema = d.get("cost_ema", 0.0)

            disp_eval = "-" if (self._eval_mean is None or not np.isfinite(self._eval_mean)) else round(float(self._eval_mean), 6)
            print(f"[diag] steps={t:,} eq={eq:.6f} eval={disp_eval} act=[{a0},{a1},{a2},{a3}] "
                  f"entry={n_entry} exit={n_exit} r_mean={r_mean:.6f} λ={lam:.4f} cost_ema={cost_ema:.6f}")

            if self.csv_path:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        t, round(elapsed, 2), ("" if disp_eval == "-" else disp_eval), eq,
                        a0, a1, a2, a3, n_entry, n_exit, round(r_mean, 8),
                        round(lam, 6), round(cost_ema, 8)
                    ])
            # 윈도우 진단 초기화
            inner.diag["n_steps"] = 0
            inner.diag["act_hist"] = {0: 0, 1: 0, 2: 0, 3: 0}
            inner.diag["r_sum"] = 0.0
            inner.diag["n_entry"] = 0
            inner.diag["n_exit"] = 0
        return True

# -----------------
# 학습
# -----------------
def main():
    print("[info] PPO Training (Lagrangian cost constraint + 4-action flat)")
    print(
        f"✅ reward = {REWARD_SCALE}*(pnl−fee−funding)"
        f" − λ*(cost−budget) + trend/hold penalty + entry bonus | "
        f"budget={COST_BUDGET}, eta={LAMBDA_STEP}, λ_max={LAMBDA_MAX}, "
        f"trend_thresh={TREND_THRESH}, bonus_h={ENTRY_BONUS_H}"
    )

    # 데이터 로드
    X_train = _load_norm("train")
    X_val = _load_norm("val")
    close_train, funding_rate_train = _load_aux_data("train")
    close_val, funding_rate_val = _load_aux_data("val")

    def make_train():
        return Monitor(TimeLimit(TradingEnv(X_train, close_train, funding_rate_train), max_episode_steps=MAX_EPISODE_STEPS))
    def make_val():
        return Monitor(TimeLimit(TradingEnv(X_val, close_val, funding_rate_val), max_episode_steps=MAX_EPISODE_STEPS))

    train_env = DummyVecEnv([make_train])
    val_env = DummyVecEnv([make_val])

    model = PPO(
        "MlpPolicy",
        train_env,
        device=DEVICE,
        seed=SEED,
        verbose=0,
        tensorboard_log=None,
        **BASE_PPO_KW,
    )

    mid_logger = MidLogger(LOG_FREQ, None)

    class EnhancedEvalCallback(EvalCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_eval_count = 0

        def _on_step(self) -> bool:
            cont = super()._on_step()
            if hasattr(self, "evaluations_timesteps"):
                cur_cnt = len(self.evaluations_timesteps)
                if cur_cnt > self._last_eval_count and self.last_mean_reward is not None and np.isfinite(self.last_mean_reward):
                    self._last_eval_count = cur_cnt
                    mid_logger.set_eval(float(self.last_mean_reward))
            return cont

    eval_cb = EnhancedEvalCallback(
        val_env,
        best_model_save_path=MODEL_DIR,
        log_path=None,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        verbose=0,
    )

    cb = CallbackList([mid_logger, eval_cb])

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=cb, progress_bar=False)
        print("[OK] Training completed")
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user")

if __name__ == "__main__":
    main()
