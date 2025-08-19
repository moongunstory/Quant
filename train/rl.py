# rl_practitioner.py — PPO for Crypto Futures (practitioner-style)
# - Env: 5m bar, next-bar execution (1-bar latency), Box action a∈[-1,1] = target position
# - Reward: pos * return  − (fee + slippage) on turnover − funding_per_step − λ·excess_turnover
# - Costs only when position changes (NO double-count)
# - Funding: 8h rate / 96 per 5m step (uses Funding8h or FundingRate)
# - Stabilizers: min Δpos threshold, cooldown, action smoothing, daily turnover budget(Lagrange)
# - Data: uses processed outputs from fe.py (fe_{train,val,test}_5m.parquet + feature_list JSON)

from __future__ import annotations
import os, json, math, warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

# ===== Paths =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "model"))
REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "reports"))
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(REPORT_DIR, exist_ok=True)

# ===== Default Hyperparams (reasonable starting points) =====
SEED = 42
WINDOW = 48                 # 4h of 5m bars
FEE_BPS = 0.0006            # taker fee per notional traded (e.g., 6 bps)
SLIP_BPS = 0.0003           # baseline slippage per turnover (can be made ATR-linked)
MIN_DPOS = 0.10             # ignore Δpos smaller than this (mask small fiddling)
COOLDOWN = 2                # bars after a change where further changes are blocked
TURN_BUDGET_DAILY = 1.5     # daily turnover budget (×equity notional)
LAMBDA_INIT = 0.0           # Lagrange multiplier start
LAMBDA_STEP = 1e-4          # λ += LAMBDA_STEP * (excess > 0)
LAMBDA_MAX = 25.0
LEVERAGE = 1.0              # scale of exposure; keep 1.0 to start
SMOOTH_ALPHA = 0.25         # action smoothing toward target (EMA)
EVAL_EVERY = 10_000
TOTAL_STEPS = 1_000_000

# ===== Data loading =====
def _load_fe(split: str) -> Tuple[pd.DataFrame, List[str]]:
    X = pd.read_parquet(os.path.join(PROC_DIR, f"fe_{split}_5m.parquet"))
    with open(os.path.join(PROC_DIR, "fe_feature_list_5m.json"), "r", encoding="utf-8") as f:
        feat_cols = json.load(f)
    # Sanity
    req = ["close_ref", "FundingRate"]
    for c in req:
        if c not in X.columns:
            X[c] = 0.0
    # Optional columns
    if "Funding8h" not in X.columns:
        X["Funding8h"] = X["FundingRate"]
    if "FundingSettle" not in X.columns:
        idx = X.index if isinstance(X.index, pd.DatetimeIndex) else pd.to_datetime(X.index, utc=True)
        X["FundingSettle"] = (((idx.hour % 8 == 0) & (idx.minute == 0))).astype("int8")
    return X.sort_index(), feat_cols

def _to_numpy_windows(df: pd.DataFrame, feat_cols: List[str], window: int) -> Dict[str, np.ndarray]:
    idx = df.index
    F = len(feat_cols)
    N = len(df)
    if N <= window + 1:
        raise ValueError("Not enough rows for windowing.")
    # features already scaled by fe.py
    X = df[feat_cols].values.astype(np.float32)
    close = df["close_ref"].astype("float64").values
    fund8h = df.get("Funding8h", df["FundingRate"]).astype("float64").values
    # build rolling windows (flattened)
    T = N - window
    obs = np.empty((T, window * F), dtype=np.float32)
    for t in range(T):
        obs[t] = X[t:t+window].reshape(-1)
    # returns for step t+1 (next-bar; obs at t corresponds to price move from t+window-1 -> t+window)
    ret = (close[window:] - close[window-1:-1]) / np.maximum(close[window-1:-1], 1e-12)
    # per-step funding rate (8h/96)
    fund_step = fund8h[window:] / 96.0
    # timestamps (align to step target bar)
    ts = pd.to_datetime(idx[window:], utc=True)
    return dict(obs=obs, ret=ret.astype(np.float64), fund=fund_step.astype(np.float64), ts=ts)

# ===== Env =====
@dataclass
class CostConfig:
    fee_bps: float = FEE_BPS
    slip_bps: float = SLIP_BPS
    min_dpos: float = MIN_DPOS
    cooldown: int = COOLDOWN
    budget_daily: float = TURN_BUDGET_DAILY
    leverage: float = LEVERAGE
    smooth_alpha: float = SMOOTH_ALPHA

class CryptoFuturesEnv(gym.Env):
    metadata = {"render.modes": []}
    def __init__(self, data: Dict[str, np.ndarray], cost: CostConfig, lambda_init=LAMBDA_INIT, lambda_step=LAMBDA_STEP, lambda_max=LAMBDA_MAX):
        super().__init__()
        self.obs_mat = data["obs"]           # (T, W*F)
        self.rets = data["ret"]              # price return for this step
        self.fund = data["fund"]             # funding per step (sign matters)
        self.ts = data["ts"]                 # timestamps
        self.T = len(self.rets)

        self.cost = cost
        self.lambda_ = lambda_init
        self.lambda_step = lambda_step
        self.lambda_max = lambda_max

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_mat.shape[1:], dtype=np.float32)

        # state
        self.t = 0
        self.pos = 0.0           # executed position for current step
        self.pos_target = 0.0    # desired target; execution at this step
        self.pos_last_change = -10**9
        self.turnover_roll = 0.0 # last 24h turnover rolling sum
        self.turn_hist = []      # store last 288 steps turnovers for rolling sum

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.pos = 0.0
        self.pos_target = 0.0
        self.pos_last_change = -10**9
        self.turnover_roll = 0.0
        self.turn_hist.clear()
        self.lambda_ = min(self.lambda_, self.lambda_max)  # keep current λ (or reset to init if preferred)
        return self.obs_mat[self.t], {}

    def _apply_action(self, a_raw: float) -> Tuple[float, float]:
        # cooldown
        if (self.t - self.pos_last_change) < self.cost.cooldown:
            a = self.pos_target  # locked
        else:
            # smoothing toward raw action
            a = (1 - self.cost.smooth_alpha) * self.pos_target + self.cost.smooth_alpha * float(np.clip(a_raw, -1.0, 1.0))
            # small change masking
            if abs(a - self.pos_target) < self.cost.min_dpos:
                a = self.pos_target
            else:
                self.pos_last_change = self.t
        # turnover to go from current executed pos -> new target (exec now)
        dpos = a - self.pos
        return a, dpos

    def step(self, action: np.ndarray):
        assert self.t < self.T, "Episode already done"
        a_raw = float(action[0])
        # Execute at start of step (next-bar execution)
        a_target, dpos = self._apply_action(a_raw)

        # Costs on turnover (single execution)
        fee_cost = self.cost.fee_bps * abs(dpos) * self.cost.leverage
        slip_cost = self.cost.slip_bps * abs(dpos) * self.cost.leverage

        # Update executed position AFTER paying costs
        self.pos = a_target
        self.pos_target = a_target

        # Price PnL over this bar using executed position (pos held through the bar)
        pnl_ret = (self.pos * self.cost.leverage) * self.rets[self.t]

        # Funding per step (sign: +rate costs longs, −rate costs shorts)
        fund_cost = self.pos * (self.fund[self.t])

        # Rolling daily turnover budget (288 steps ≈ 1 day)
        self.turn_hist.append(abs(dpos))
        if len(self.turn_hist) > 288:
            self.turn_hist.pop(0)
        self.turnover_roll = sum(self.turn_hist)
        excess = max(0.0, self.turnover_roll - self.cost.budget_daily)

        # Lagrange update: push λ up when exceeding budget
        if excess > 0.0:
            self.lambda_ = min(self.lambda_max, self.lambda_ + self.lambda_step)

        reward = pnl_ret - fee_cost - slip_cost - fund_cost - self.lambda_ * excess

        info = dict(
            t=int(self.t),
            ts=str(self.ts[self.t].to_pydatetime()),
            pos=float(self.pos),
            dpos=float(dpos),
            pnl_ret=float(pnl_ret),
            fee=float(fee_cost),
            slip=float(slip_cost),
            fund=float(fund_cost),
            lambda_=float(self.lambda_),
            excess_turn=float(excess),
            ret=float(self.rets[self.t])
        )

        self.t += 1
        terminated = self.t >= self.T
        truncated = False
        obs = self.obs_mat[self.t-1] if not terminated else self.obs_mat[self.T-1]
        return obs, float(reward), terminated, truncated, info

# ===== Callbacks =====
class PrintDiagCallback(BaseCallback):
    def __init__(self, freq=5000, verbose=0):
        super().__init__(verbose)
        self.freq = freq
        self._eq = 1.0
        self._pnl = 0.0
        self._costs = 0.0
        self._turn = 0.0
        self._n = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info: continue
            self._eq *= (1.0 + info.get("ret", 0.0) * info.get("pos", 0.0))
            self._pnl += info.get("pnl_ret", 0.0)
            self._costs += (info.get("fee", 0.0) + info.get("slip", 0.0) + info.get("fund", 0.0))
            self._turn = info.get("excess_turn", self._turn)
            self._n += 1
        if self.num_timesteps % self.freq == 0 and self._n:
            print(f"[diag] steps={self.num_timesteps:,} eq={self._eq:.3f} pnl={self._pnl:.5f} "
                  f"costs={self._costs:.5f} excess={self._turn:.3f}")
        return True

# ===== Training/Eval Utilities =====
def make_env(split="train") -> gym.Env:
    df, feat_cols = _load_fe(split)
    data = _to_numpy_windows(df, feat_cols, WINDOW)
    env = CryptoFuturesEnv(data, CostConfig(
        fee_bps=FEE_BPS, slip_bps=SLIP_BPS, min_dpos=MIN_DPOS, cooldown=COOLDOWN,
        budget_daily=TURN_BUDGET_DAILY, leverage=LEVERAGE, smooth_alpha=SMOOTH_ALPHA
    ))
    return env

def evaluate_model(model: PPO, split="val") -> Dict[str, float]:
    env = make_env(split)
    obs, _ = env.reset()
    done = False
    eq = 1.0
    fees = slips = funds = 0.0
    pnl = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        eq *= (1.0 + info["pos"] * info["ret"])
        fees += info["fee"]; slips += info["slip"]; funds += info["fund"]; pnl += info["pnl_ret"]
        if terminated or truncated:
            break
    dd = 0.0
    # naïve drawdown on equity path is omitted for brevity; eq is final equity multiple
    return dict(final_eq=eq, pnl=pnl, fees=fees, slips=slips, funds=funds, lambda_=info["lambda_"])

def train():
    set_random_seed(SEED)
    train_env = DummyVecEnv([lambda: make_env("train")])
    val_env   = DummyVecEnv([lambda: make_env("val")])

    # PPO configs (robust defaults; tune later)
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        seed=SEED,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=os.path.join(MODEL_DIR, "tb")
    )

    diag = PrintDiagCallback(freq=5000)

    # Lightweight eval saving
    class _EvalAndSave(BaseCallback):
        def __init__(self, eval_every=EVAL_EVERY):
            super().__init__()
            self.eval_every = eval_every
            self.best = -1e9
        def _on_step(self) -> bool:
            if self.num_timesteps % self.eval_every == 0:
                m = self.model
                metrics = evaluate_model(m, "val")
                score = metrics["final_eq"] - (metrics["fees"] + metrics["slips"] + abs(metrics["funds"]))
                print(f"[eval] steps={self.num_timesteps:,} "
                      f"eq={metrics['final_eq']:.3f} pnl={metrics['pnl']:.5f} "
                      f"fees={metrics['fees']:.5f} slip={metrics['slips']:.5f} fund={metrics['funds']:.5f}")
                if score > self.best:
                    self.best = score
                    path = os.path.join(MODEL_DIR, "ppo_practitioner_best.zip")
                    m.save(path)
                    print(f"[save] {path}")
            return True

    model.learn(total_timesteps=TOTAL_STEPS, callback=[diag, _EvalAndSave()])
    final_path = os.path.join(MODEL_DIR, "ppo_practitioner_final.zip")
    model.save(final_path)
    print(f"[save] {final_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train()
