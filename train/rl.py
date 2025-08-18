"""
rl.py — PPO training for ETHUSDT (5m) matching fe.py outputs (REV-1)
- Fix: **Slippage no double-count** — slippage is applied ONLY via execution price; costs exclude slippage.
- Add: **min_hold=24, cooldown=6 bars, flip penalty=3bp** to suppress churn.
- Reward: per-step **ΔEquity fraction** (equity_after / equity_before - 1.0), including fees & funding.
- Uses fe.py processed files from ./ai_binance/data/processed.
- Saves models to ./ai_binance/data/model.

Quick start:
    python ai_binance/train/rl.py --timesteps 1_200_000
"""
from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# ===== Paths (aligned with fe.py) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "model"))
os.makedirs(MODEL_DIR, exist_ok=True)

INTERVAL = "5m"
TRAIN_P = os.path.join(PROC_DIR, f"fe_train_{INTERVAL}.parquet")
VAL_P   = os.path.join(PROC_DIR, f"fe_val_{INTERVAL}.parquet")
TEST_P  = os.path.join(PROC_DIR, f"fe_test_{INTERVAL}.parquet")
FEAT_P  = os.path.join(PROC_DIR, f"fe_feature_list_{INTERVAL}.json")

# ===== Trading/Env Parameters =====
@dataclass
class EnvCfg:
    fee_rate: float = 0.0006          # taker per side (6bp)
    slip_bp: float = 0.0002           # slippage (applied to exec price only)
    min_hold: int = 24                # bars (↑ to reduce churn)
    max_hold: int = 96                # bars (~8h)
    stop_pct: float = 0.010           # 1%
    stop_atr_mult: float = 2.0        # 2x ATR stop
    trail_mult: Optional[float] = 1.2 # trailing multiple of stop width (None to disable)
    leverage: float = 1.0
    alpha_flip_bp: float = 0.0003     # extra penalty when changing position (3bp)
    funding_split: int = 96           # 8h / 5m
    random_reset: bool = True
    flat_at_funding: bool = False     # optional (unused here)
    cooldown_bars: int = 6            # wait after any trade

# ===== Helpers =====

def _ewm_mean(arr: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _atr_from_ohlc(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close)
    ])
    alpha = 1.0 / period
    return _ewm_mean(tr, alpha)


class MarketEnvTargetPos(gym.Env):
    """Gym env using fe.py processed files.
    Observation: scaled feature vector at time t.
    Action: 0→-1(short), 1→0(flat), 2→+1(long). Internally mapped to {-1,0,+1}.
    Reward: **ΔEquity fraction** including fees (no slippage double count) & funding.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        cfg: EnvCfg,
        seed: int = 72,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg

        # Split features and refs
        self.feature_cols = feature_cols
        assert all(c in df.columns for c in feature_cols), "Missing feature columns in df"
        # Refs
        for c in ["Open", "High", "Low", "Close", "FundingRate", "close_ref"]:
            if c not in df.columns:
                df[c] = 0.0
        self.ref = df[["Open", "High", "Low", "Close", "FundingRate", "close_ref"]].copy()
        self.X = df[feature_cols].astype("float64").to_numpy()

        # Precompute ATR from raw refs
        self.atr14 = _atr_from_ohlc(
            self.ref["High"].to_numpy(dtype=float),
            self.ref["Low"].to_numpy(dtype=float),
            self.ref["Close"].to_numpy(dtype=float),
            period=14,
        )

        self.n = len(df)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(feature_cols),), dtype=np.float64)

        # State vars
        self.i: int = 0
        self.pos: int = 0  # -1/0/+1
        self.entry_price: Optional[float] = None
        self.hold_bars: int = 0
        self.peak_pnl: float = 0.0
        self.equity: float = 1.0  # normalized equity
        self.cooldown: int = 0

    # --- Internal utilities ---
    @staticmethod
    def _a_to_pos(a: int) -> int:
        return (-1, 0, +1)[a]

    def _min_switch_blocked(self) -> bool:
        return self.pos != 0 and self.hold_bars < self.cfg.min_hold

    def _stop_hit(self, price_t: float) -> bool:
        if self.pos == 0 or self.entry_price is None:
            return False
        side = self.pos
        ret = (price_t / self.entry_price - 1.0) * (1 if side == +1 else -1)
        stop_atr = self.cfg.stop_atr_mult * (self.atr14[self.i] / price_t)
        stop_cut = min(self.cfg.stop_pct, stop_atr)
        return ret <= -stop_cut

    def _trail_hit(self, price_t: float) -> bool:
        if self.cfg.trail_mult is None or self.pos == 0 or self.entry_price is None:
            return False
        side = self.pos
        cur = (price_t / self.entry_price - 1.0) * (1 if side == +1 else -1)
        dd = (self.peak_pnl - cur) if side == +1 else (cur - self.peak_pnl)
        trail_pct = self.cfg.trail_mult * min(
            self.cfg.stop_pct, self.cfg.stop_atr_mult * (self.atr14[self.i] / price_t)
        )
        return dd >= trail_pct

    def _funding_cost_step(self) -> float:
        # Positive FundingRate means longs pay, shorts receive.
        fr = float(self.ref.iloc[self.i]["FundingRate"])  # per 8h
        return self.pos * (fr / self.cfg.funding_split) * self.cfg.leverage  # cost (+) if paying

    # --- Gym API ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        start_min = 1
        start_max = self.n - 200  # leave runway
        self.i = int(self.rng.integers(start_min, start_max)) if self.cfg.random_reset else start_min
        self.pos = 0
        self.entry_price = None
        self.hold_bars = 0
        self.peak_pnl = 0.0
        self.equity = 1.0
        self.cooldown = 0
        obs = self.X[self.i].copy()
        info = {}
        return obs, info

    def step(self, action: int):
        assert 0 <= action <= 2
        done = False
        info = {}

        # Prices t and t+1
        if self.i >= self.n - 2:
            done = True
            return self.X[self.i], 0.0, done, False, info
        price_t = float(self.ref.iloc[self.i]["close_ref"]) or float(self.ref.iloc[self.i]["Close"])  # safe fallback
        price_tp1 = float(self.ref.iloc[self.i + 1]["close_ref"]) or float(self.ref.iloc[self.i + 1]["Close"])  # next bar

        # Cooldown decay
        if self.cooldown > 0:
            self.cooldown -= 1

        eq_before = self.equity

        # Forced exits (risk guards)
        forced_exit = self._stop_hit(price_t) or self._trail_hit(price_t) or (self.hold_bars >= self.cfg.max_hold)

        # Policy desire
        want_pos = self._a_to_pos(action)
        want_flip = (self.pos != 0) and (want_pos == -self.pos)
        want_flat = (self.pos != 0) and (want_pos == 0)

        # Min-hold gate
        if self._min_switch_blocked():
            want_flip = False
            want_flat = False

        # Execute exits
        if (self.pos != 0) and (forced_exit or want_flat or want_flip):
            # Exit at price_t with slippage on price only
            exec_price = price_t * (1 - self.cfg.slip_bp) if self.pos == +1 else price_t * (1 + self.cfg.slip_bp)
            trade_ret = (exec_price / self.entry_price - 1.0) * self.pos * self.cfg.leverage
            exit_fee = self.cfg.fee_rate
            self.equity *= (1.0 + trade_ret) * (1.0 - exit_fee)

            # Flat out
            self.pos = 0
            self.entry_price = None
            self.hold_bars = 0
            self.peak_pnl = 0.0
            self.cooldown = self.cfg.cooldown_bars

        # Execute entries (no immediate re-entry if cooldown>0)
        can_enter = (self.pos == 0) and (want_pos != 0) and (not forced_exit) and (self.cooldown == 0)
        if can_enter:
            self.pos = want_pos
            # Entry exec price with slippage on price only
            self.entry_price = price_t * (1 + self.cfg.slip_bp) if self.pos == +1 else price_t * (1 - self.cfg.slip_bp)
            entry_cost = self.cfg.fee_rate + self.cfg.alpha_flip_bp  # NO slippage here (already in price)
            self.equity *= (1.0 - entry_cost)
            self.hold_bars = 0
            self.peak_pnl = 0.0
            self.cooldown = self.cfg.cooldown_bars

        # Running PnL for the bar (t→t+1) & funding
        ret_tp1 = (price_tp1 / price_t - 1.0) * self.pos * self.cfg.leverage
        funding_cost = self._funding_cost_step()  # (+) if paying
        self.equity *= (1.0 + ret_tp1 - funding_cost)

        # Track peak pnl for trailing
        if self.pos != 0 and self.entry_price is not None:
            cur = (price_t / self.entry_price - 1.0) * (1 if self.pos == +1 else -1)
            if self.pos == +1:
                self.peak_pnl = max(self.peak_pnl, cur)
            else:
                self.peak_pnl = min(self.peak_pnl, cur)
            self.hold_bars += 1

        self.i += 1
        obs = self.X[self.i].copy()
        eq_after = self.equity
        reward = float(eq_after / (eq_before + 1e-12) - 1.0)

        terminated = done or (not np.isfinite(self.equity)) or (self.equity <= 0.05)
        truncated = False
        info.update({"equity": self.equity, "pos": self.pos})
        return obs, reward, bool(terminated), bool(truncated), info

    def render(self):
        print(f"i={self.i} pos={self.pos} equity={self.equity:.4f}")


# ===== Learning rate schedules =====

def linear_schedule(start: float, end: float):
    def _fn(progress_remaining: float):
        return end + (start - end) * progress_remaining
    return _fn


def cosine_schedule(start: float, end: float):
    def _fn(progress_remaining: float):
        # progress_remaining: 1→0
        cos = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return end + (start - end) * cos
    return _fn


# ===== Utilities to load data/envs =====

def _load_split(split_path: str) -> pd.DataFrame:
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Missing split file: {split_path}")
    return pd.read_parquet(split_path)


def make_env_from_split(split: str, feature_cols: List[str], cfg: EnvCfg, seed: int) -> gym.Env:
    pmap = {"train": TRAIN_P, "val": VAL_P, "test": TEST_P}
    df = _load_split(pmap[split])
    env = MarketEnvTargetPos(df, feature_cols, cfg, seed=seed)
    return Monitor(env)


# ===== Simple evaluation on a single env =====

def run_rollout(env: gym.Env, model: PPO, n_steps: int | None = None) -> dict:
    obs, info = env.reset()
    eq_hist = [1.0]
    pos_hist = []
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        eq_hist.append(info.get("equity", eq_hist[-1]))
        pos_hist.append(info.get("pos", 0))
        steps += 1
        if terminated or truncated:
            break
        if n_steps is not None and steps >= n_steps:
            break
    eq_arr = np.array(eq_hist)
    rets = np.diff(eq_arr) / np.clip(eq_arr[:-1], 1e-12, None)
    sharpe = (np.mean(rets) / (np.std(rets) + 1e-8)) * math.sqrt(365*24*12) if len(rets) > 1 else 0.0
    return {
        "final_equity": float(eq_arr[-1]),
        "total_return": float(eq_arr[-1] - 1.0),
        "sharpe": float(sharpe),
        "steps": int(steps),
    }


# ===== Main training =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_200_000)
    parser.add_argument("--lr_start", type=float, default=3e-4)
    parser.add_argument("--lr_end", type=float, default=1e-4)
    parser.add_argument("--lr_sched", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--policy_width", type=int, default=256)
    parser.add_argument("--logdir", type=str, default=os.path.join(MODEL_DIR, "tb"))
    args = parser.parse_args()

    # Load feature list
    if not os.path.exists(FEAT_P):
        raise FileNotFoundError(f"Missing feature list: {FEAT_P}")
    with open(FEAT_P, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    cfg = EnvCfg()

    # Envs
    def _make_train():
        return make_env_from_split("train", feature_cols, cfg, seed=args.seed)
    def _make_val():
        cfg_eval = EnvCfg(**{**cfg.__dict__, "random_reset": False})
        return make_env_from_split("val", feature_cols, cfg_eval, seed=args.seed)

    train_env = DummyVecEnv([_make_train])
    eval_env = DummyVecEnv([_make_val])

    # LR schedule
    if args.lr_sched == "linear":
        lr = linear_schedule(args.lr_start, args.lr_end)
    else:
        lr = cosine_schedule(args.lr_start, args.lr_end)

    policy_kwargs = dict(net_arch=[args.policy_width, args.policy_width])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=lr,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.005,
        clip_range=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=1,
        tensorboard_log=args.logdir,
        policy_kwargs=policy_kwargs,
    )

    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10, min_evals=10, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=args.logdir,
        eval_freq=50_000,
        deterministic=True,
        render=False,
        callback_after_eval=stop_cb,
    )

    print("[RL] Training started…")
    model.learn(total_timesteps=args.timesteps, callback=eval_cb, progress_bar=True)

    best_path = os.path.join(MODEL_DIR, "best_model.zip")
    last_path = os.path.join(MODEL_DIR, "ppo_final_model.zip")
    try:
        model.save(last_path)
    except Exception:
        pass
    print(f"[RL] Saved model → {last_path}")
    if os.path.exists(best_path):
        print(f"[RL] Best model exists → {best_path}")

    # Final quick checks on val & test
    print("[RL] Running quick evaluation…")
    best_model = PPO.load(best_path) if os.path.exists(best_path) else model

    val_env = make_env_from_split("val", feature_cols, EnvCfg(random_reset=False), seed=args.seed)
    test_env = make_env_from_split("test", feature_cols, EnvCfg(random_reset=False), seed=args.seed)

    val_res = run_rollout(val_env, best_model)
    test_res = run_rollout(test_env, best_model)

    print("================ EVAL SUMMARY ================")
    print(f"Val  — steps={val_res['steps']:,} final_eq={val_res['final_equity']:.3f} ret={(val_res['total_return']*100):.2f}% sharpe={val_res['sharpe']:.2f}")
    print(f"Test — steps={test_res['steps']:,} final_eq={test_res['final_equity']:.3f} ret={(test_res['total_return']*100):.2f}% sharpe={test_res['sharpe']:.2f}")
    print("=============================================")


if __name__ == "__main__":
    main()
