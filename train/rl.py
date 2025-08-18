# rl.py (SubprocVecEnv + VecNormalize + LR schedule + CPU + column compatibility + FIXED REWARD)
from __future__ import annotations
import os
# ---- Force CPU ----
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import json
from enum import Enum
from typing import Tuple, List, Callable

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# ===== Settings =====
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RAW_DIR       = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw"))
PROC_DIR      = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
MODEL_DIR     = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))  # ./models/
SPLITS        = ("train", "val", "test")
INTERVAL      = "5m"
WINDOW        = 48            # 5m 기준 4시간
INTERVAL_MIN  = 5
FEE_PER_SIDE  = 0.0005
SLIP_PER_SIDE = 0.0001
REWARD_HORIZON= 12            # 12 * 5m = 1 hour
SEED          = 42
TIMESTEPS     = 2_000_000

# PPO/Env
N_ENVS        = 8                    # 병렬 환경 개수 (CPU 코어에 맞춰 조정)
N_STEPS       = 2048                 # per-env rollout → 총 rollout = N_ENVS * N_STEPS
BATCH_SIZE    = 4096
N_EPOCHS      = 10
CLIP_RANGE    = 0.30
ENTROPY_COEF  = 0.01
VF_COEF       = 0.70
NET_ARCH      = [256, 256]

# ===== Reward Mode =====
class RewardMode(str, Enum):
    PNL_STEP = "pnl_step"          # 1바 실현 PnL 보상
    HORIZON_ON_TRADE = "horizon"   # 진입/전환 시 1회 horizon 보상(보유 중 0)

# ===== ENV (returns-only) =====
class SimpleTradingEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, close, funding_rate=None, window=48,
                 fee_per_side=0.0005, slip_per_side=0.0001,
                 interval_min=5, random_start=True, seed: int = 42,
                 reward_mode: RewardMode = RewardMode.PNL_STEP,
                 reward_horizon: int = REWARD_HORIZON,
                 min_hold_bars: int = 0):
        close = np.asarray(close, dtype=np.float64)
        assert len(close) >= window + 2, "데이터가 너무 짧음"
        self.close = close
        self.funding = np.zeros_like(close, dtype=np.float64) if funding_rate is None else np.asarray(funding_rate, dtype=np.float64)
        self.window = int(window)
        self.cost_per_side = float(fee_per_side + slip_per_side)
        self.ret = np.diff(np.log(self.close))
        self.random_start = bool(random_start)
        self.interval_min = int(interval_min)
        # 8시간(=480분) / interval
        self.fund_div = max(1, int(round(480 / max(1, self.interval_min))))
        self._rng = np.random.default_rng(seed)
        self.pos = 0
        self.bars_in_pos = 0
        self.t = self.window

        self.reward_mode = reward_mode
        self.H = int(reward_horizon)
        self.min_hold = int(min_hold_bars)
        self.last_trade_bar = -10**9

        obs_dim = self.window + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def _obs(self, t):
        rwin = self.ret[t-self.window:t]
        return np.asarray(list(rwin) + [float(self.pos), float(self.bars_in_pos)/100.0], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0
        self.bars_in_pos = 0
        last_safe = len(self.close) - max(self.H, 1) - 2
        start_hi = max(self.window, last_safe)
        self.t = self._rng.integers(self.window, start_hi) if self.random_start else self.window
        return self._obs(self.t), {}

    def step(self, action: int):
        sides = 0
        new_pos = self.pos

        # --- 액션 처리(일관 규칙) ---
        if action == 0:   # hold
            new_pos = self.pos
        elif action == 1: # long
            if   self.pos == 0:  new_pos, sides = +1, 1
            elif self.pos == -1: new_pos, sides = +1, 2
        elif action == 2: # short
            if   self.pos == 0:  new_pos, sides = -1, 1
            elif self.pos == +1: new_pos, sides = -1, 2
        elif action == 3: # flat
            if self.pos != 0:    new_pos, sides = 0, 1

        # --- 최소 보유 강제(옵션) ---
        if self.pos != 0 and self.bars_in_pos < self.min_hold and new_pos != self.pos:
            new_pos, sides = self.pos, 0  # 액션 무시(hold)

        # --- 비용 계산 ---
        fee_step = sides * self.cost_per_side

        # --- 펀딩: 정확히 8시간 주기에만 적용 ---
        funding_penalty = 0.0
        if self.fund_div > 0 and self.pos != 0:
            if ((self.t - 1) % self.fund_div) == 0:
                # long이 +rate 지불(음), short은 -rate 지불(양)이 될 수 있음
                funding_penalty = self.pos * float(self.funding[self.t - 1])

        # --- 보상 계산 ---
        if self.reward_mode == RewardMode.PNL_STEP:
            # 1바 실현 수익
            step_ret = (self.close[self.t] / self.close[self.t - 1]) - 1.0
            reward = (self.pos * step_ret) - fee_step - funding_penalty
        else:  # RewardMode.HORIZON_ON_TRADE
            if sides > 0:
                entry_price = self.close[self.t - 1]
                future_idx  = min(self.t - 1 + self.H, len(self.close) - 1)
                horizon_ret = (self.close[future_idx] / entry_price) - 1.0 if entry_price > 0 else 0.0
                reward = (new_pos * horizon_ret) - fee_step - funding_penalty
            else:
                reward = -funding_penalty  # 보유/홀드 구간엔 horizon 보상 없음

        # --- 상태 업데이트 ---
        if sides > 0:
            self.last_trade_bar = self.t
        self.pos = new_pos
        self.bars_in_pos = (self.bars_in_pos + 1) if self.pos != 0 else 0
        self.t += 1

        # 종료 판정(보상 모드별 안전 인덱스)
        if self.reward_mode == RewardMode.PNL_STEP:
            last_safe_t = len(self.close) - 1
        else:
            last_safe_t = len(self.close) - self.H - 1
        terminated = (self.t >= last_safe_t)

        obs = self._obs(self.t) if not terminated else np.zeros_like(self._obs(self.t-1), dtype=np.float32)

        simple_ret = float(np.exp(self.ret[min(self.t-1, len(self.ret)-1)]) - 1.0)
        info = {"ret": simple_ret, "sides": int(sides), "fee_step": float(fee_step),
                "funding": float(funding_penalty), "pos": int(self.pos)}
        return obs, float(reward), terminated, False, info

# ===== ENV (features-stacked) =====
class FeatureStackedEnv(gym.Env):
    """피처 행렬 X를 window 길이로 스택해 관측 제공. reward는 close/funding 기반."""
    metadata = {"render_modes": []}
    def __init__(self, X: np.ndarray, close: np.ndarray, funding_rate: np.ndarray,
                 window=48, fee_per_side=0.0005, slip_per_side=0.0001,
                 interval_min=5, random_start=True, seed: int = 42,
                 reward_mode: RewardMode = RewardMode.PNL_STEP,
                 reward_horizon: int = REWARD_HORIZON,
                 min_hold_bars: int = 0):
        assert X.ndim == 2, "X must be 2D [T, F]"
        T, F = X.shape
        assert len(close) == T and len(funding_rate) == T, "X/close/funding length mismatch"
        assert T >= window + 2, "데이터가 너무 짧음"
        self.X = X.astype(np.float32)
        self.F = F
        self.close = close.astype(np.float64)
        self.funding = funding_rate.astype(np.float64)
        self.window = int(window)
        self.cost_per_side = float(fee_per_side + slip_per_side)
        self.ret = np.diff(np.log(self.close))
        self.random_start = bool(random_start)
        self.interval_min = int(interval_min)
        self.fund_div = max(1, int(round(480 / max(1, self.interval_min))))
        self._rng = np.random.default_rng(seed)
        self.pos = 0
        self.bars_in_pos = 0
        self.t = self.window

        self.reward_mode = reward_mode
        self.H = int(reward_horizon)
        self.min_hold = int(min_hold_bars)
        self.last_trade_bar = -10**9

        obs_dim = self.window * self.F + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def _obs(self, t):
        xw = self.X[t-self.window:t].reshape(-1)
        return np.concatenate([xw, np.array([float(self.pos), float(self.bars_in_pos)/100.0], dtype=np.float32)], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0
        self.bars_in_pos = 0
        last_safe = len(self.close) - max(self.H, 1) - 2
        start_hi = max(self.window, last_safe)
        self.t = self._rng.integers(self.window, start_hi) if self.random_start else self.window
        return self._obs(self.t), {}

    def step(self, action: int):
        sides = 0
        new_pos = self.pos

        if action == 0:  # hold
            new_pos = self.pos
        elif action == 1:  # long
            if   self.pos == 0:  new_pos, sides = +1, 1
            elif self.pos == -1: new_pos, sides = +1, 2
        elif action == 2:  # short
            if   self.pos == 0:  new_pos, sides = -1, 1
            elif self.pos == +1: new_pos, sides = -1, 2
        elif action == 3:  # flat
            if self.pos != 0:    new_pos, sides = 0, 1

        if self.pos != 0 and self.bars_in_pos < self.min_hold and new_pos != self.pos:
            new_pos, sides = self.pos, 0

        fee_step = sides * self.cost_per_side

        funding_penalty = 0.0
        if self.fund_div > 0 and self.pos != 0:
            if ((self.t - 1) % self.fund_div) == 0:
                funding_penalty = self.pos * float(self.funding[self.t - 1])

        if self.reward_mode == RewardMode.PNL_STEP:
            step_ret = (self.close[self.t] / self.close[self.t - 1]) - 1.0
            reward = (self.pos * step_ret) - fee_step - funding_penalty
        else:
            if sides > 0:
                entry_price = self.close[self.t - 1]
                future_idx  = min(self.t - 1 + self.H, len(self.close) - 1)
                horizon_ret = (self.close[future_idx] / entry_price) - 1.0 if entry_price > 0 else 0.0
                reward = (new_pos * horizon_ret) - fee_step - funding_penalty
            else:
                reward = -funding_penalty

        if sides > 0:
            self.last_trade_bar = self.t
        self.pos = new_pos
        self.bars_in_pos = (self.bars_in_pos + 1) if self.pos != 0 else 0
        self.t += 1

        if self.reward_mode == RewardMode.PNL_STEP:
            last_safe_t = len(self.close) - 1
        else:
            last_safe_t = len(self.close) - self.H - 1
        terminated = (self.t >= last_safe_t)

        obs = self._obs(self.t) if not terminated else np.zeros_like(self._obs(self.t-1), dtype=np.float32)

        simple_ret = float(np.exp(self.ret[min(self.t-1, len(self.ret)-1)]) - 1.0)
        info = {"ret": simple_ret, "sides": int(sides), "fee_step": float(fee_step),
                "funding": float(funding_penalty), "pos": int(self.pos)}
        return obs, float(reward), terminated, False, info

# ===== Data loaders =====
def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Column harmonization for both RAW and PROCESSED files."""
    if isinstance(df.index, pd.DatetimeIndex) and (df.index.name or "").lower() in ("open_time", "time"):
        df = df.reset_index()

    low = {c.lower(): c for c in df.columns}
    ren = {}
    if "open_time" in low: ren[low["open_time"]] = "time"
    if "time" in low: ren[low["time"]] = "time"

    if "close" in low:
        ren[low["close"]] = "close"
    elif "close_ref" in low:
        ren[low["close_ref"]] = "close"
    elif "close" not in low and "Close" in df.columns:
        ren["Close"] = "close"

    if "funding_rate" in low:
        ren[low["funding_rate"]] = "funding_rate"
    elif "fundingrate" in low:
        ren[low["fundingrate"]] = "funding_rate"
    elif "FundingRate" in df.columns:
        ren["FundingRate"] = "funding_rate"

    df = df.rename(columns=ren)
    if "close" not in df.columns:
        raise KeyError("processed/raw 데이터에 close(또는 close_ref/Close)가 필요합니다.")
    if "funding_rate" not in df.columns:
        df["funding_rate"] = 0.0
    return df

def load_raw(split: str) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(RAW_DIR, f"fut_{split}_data_{INTERVAL}.parquet")
    df = pd.read_parquet(path)
    df = _normalize_df(df)
    close = df["close"].to_numpy(dtype=np.float64)
    funding = df["funding_rate"].to_numpy(dtype=np.float64)
    return close, funding

def load_processed(split: str):
    path = os.path.join(PROC_DIR, f"fe_{split}_{INTERVAL}.parquet")
    df = pd.read_parquet(path)
    df = _normalize_df(df)

    feat_path = os.path.join(PROC_DIR, f"fe_feature_list_{INTERVAL}.json")
    if os.path.exists(feat_path):
        with open(feat_path, "r", encoding="utf-8") as f:
            feature_cols: List[str] = json.load(f)
        exclude_low = {"time","open","high","low","close","volume","funding_rate","close_ref",
                       "FundingRate","Open","High","Low","Close","Volume"}
        feature_cols = [c for c in feature_cols if c not in exclude_low and c in df.columns]
        if not feature_cols:
            feature_cols = [c for c in df.columns if c not in exclude_low]
    else:
        exclude_low = {"time","open","high","low","close","volume","funding_rate","close_ref",
                       "FundingRate","Open","High","Low","Close","Volume"}
        feature_cols = [c for c in df.columns if c not in exclude_low]

    X = df[feature_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float64)
    funding = df["funding_rate"].to_numpy(dtype=np.float64)
    return X, close, funding, feature_cols

# ===== Scaler (features only) =====
class ZScaler:
    def __init__(self):
        self.mean = None
        self.std = None
    def fit(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
    def transform(self, X: np.ndarray):
        return (X - self.mean) / self.std

# ===== VecNormalize 통계 동기화 헬퍼 =====
def _sync_vecnorm_stats(src: VecNormalize, dst: VecNormalize):
    assert type(src).__name__ == "VecNormalize" and type(dst).__name__ == "VecNormalize"
    if hasattr(src, "obs_rms") and src.obs_rms is not None:
        dst.obs_rms = src.obs_rms
    if hasattr(src, "ret_rms") and src.ret_rms is not None:
        dst.ret_rms = src.ret_rms
    for k in ("clip_obs", "clip_reward", "gamma", "epsilon"):
        if hasattr(src, k):
            setattr(dst, k, getattr(src, k))
    dst.training = False

# ===== 학습률 스케줄러 (progress_remaining → lr) =====
def linear_schedule(initial_lr: float, final_lr: float) -> Callable[[float], float]:
    def _lr(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * float(progress_remaining)
    return _lr

# ===== Train & Eval =====
def run_all():
    os.makedirs(MODEL_DIR, exist_ok=True)
    proc_ok = all(os.path.exists(os.path.join(PROC_DIR, f"fe_{sp}_{INTERVAL}.parquet")) for sp in SPLITS)

    if proc_ok:
        print("[RL] Using processed features (MTF) [CPU + VecNormalize + SubprocVecEnv]")
        X_tr, c_tr, f_tr, cols = load_processed("train")

        scaler = ZScaler(); scaler.fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        # 스케일러 저장(실거래/평가 재현성용)
        np.savez(os.path.join(MODEL_DIR, "scaler_mtf.npz"), mean=scaler.mean, std=scaler.std)

        def make_train_env(i):
            def _t():
                return FeatureStackedEnv(
                    X_tr_s, c_tr, f_tr, window=WINDOW,
                    fee_per_side=FEE_PER_SIDE, slip_per_side=SLIP_PER_SIDE,
                    interval_min=INTERVAL_MIN, random_start=True, seed=SEED+i,
                    reward_mode=RewardMode.PNL_STEP, reward_horizon=REWARD_HORIZON,
                    min_hold_bars=3
                )
            return _t
        env_tr = SubprocVecEnv([make_train_env(i) for i in range(N_ENVS)])
        env_tr = VecNormalize(env_tr, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

        policy_kwargs = dict(net_arch=NET_ARCH)
        lr_sched = linear_schedule(initial_lr=1e-3, final_lr=3e-4)

        model = PPO(
            "MlpPolicy", env_tr,
            n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
            gamma=0.99, gae_lambda=0.95, clip_range=CLIP_RANGE,
            learning_rate=lr_sched,
            ent_coef=ENTROPY_COEF,
            vf_coef=VF_COEF,
            policy_kwargs=policy_kwargs,
            device="cpu", verbose=1, seed=SEED,
        )
        model.learn(total_timesteps=TIMESTEPS)

        model_path = os.path.join(MODEL_DIR, "ppo_mtf_features.zip")
        model.save(model_path)
        env_stats_path = os.path.join(MODEL_DIR, "ppo_mtf_vecnorm.pkl")
        try:
            env_tr.save(env_stats_path)
            print(f"[RL] Saved VecNormalize stats: {env_stats_path}")
        except Exception as e:
            print("[warn] VecNormalize save failed:", e)
        print(f"[RL] Saved: {model_path}")

    else:
        print("[RL] Using raw (returns-only) [CPU + VecNormalize + SubprocVecEnv]")
        c_tr, f_tr = load_raw("train")

        def make_train_env(i):
            def _t():
                return SimpleTradingEnv(
                    c_tr, f_tr, window=WINDOW,
                    fee_per_side=FEE_PER_SIDE, slip_per_side=SLIP_PER_SIDE,
                    interval_min=INTERVAL_MIN, random_start=True, seed=SEED+i,
                    reward_mode=RewardMode.PNL_STEP, reward_horizon=REWARD_HORIZON,
                    min_hold_bars=3
                )
            return _t
        env_tr = SubprocVecEnv([make_train_env(i) for i in range(N_ENVS)])
        env_tr = VecNormalize(env_tr, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

        policy_kwargs = dict(net_arch=NET_ARCH)
        lr_sched = linear_schedule(initial_lr=1e-3, final_lr=3e-4)

        model = PPO(
            "MlpPolicy", env_tr,
            n_steps=N_STEPS, batch_size=BATCH_SIZE, n_epochs=N_EPOCHS,
            gamma=0.99, gae_lambda=0.95, clip_range=CLIP_RANGE,
            learning_rate=lr_sched,
            ent_coef=ENTROPY_COEF,
            vf_coef=VF_COEF,
            policy_kwargs=policy_kwargs,
            device="cpu", verbose=1, seed=SEED,
        )
        model.learn(total_timesteps=TIMESTEPS)

        model_path = os.path.join(MODEL_DIR, "ppo_simple.zip")
        model.save(model_path)
        env_stats_path = os.path.join(MODEL_DIR, "ppo_simple_vecnorm.pkl")
        try:
            env_tr.save(env_stats_path)
            print(f"[RL] Saved VecNormalize stats: {env_stats_path}")
        except Exception as e:
            print("[warn] VecNormalize save failed:", e)
        print(f"[RL] Saved: {model_path}")

if __name__ == "__main__":
    run_all()
