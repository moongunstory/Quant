# ai_binance/train/reinforce/worker.py (REV-3: Robust Sharpe + Deadband + Min-Hold + Manager(flat))
from __future__ import annotations

# ===== allow running as a script =====
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from ai_binance.train.reinforce.common import (
    MODEL_DIR,
    build_worker_inputs, build_manager_inputs,
    GoalBridge,
    FEE_RATE, SLIP_BP,
)
from ai_binance.train.reinforce.manager import ManagerV2Env

# =============================================================================
# Unified Trading Worker (Entry/Exit Timing under Manager Guidance)
#  - Act: 0=Wait, 1=Enter, 2=Exit
#  - Obs: Market features + [manager_dir, manager_conf, manager_regime, in_pos, hold_norm]
#  - Reward@Exit: Robust Sharpe (EWMA) + small PnL(deadband) - early-exit penalty
#  - Costs: Fee + Slippage + Funding always included (paper/live 모두 고려)
# =============================================================================

# ===== Knobs =====
MAX_HOLDING_STEPS = 72        # 5m * 72 = 6h
WAIT_PENALTY = 1e-5           # small penalty per wait
FUNDING_STEP_FRAC = 5.0 / 480.0  # funding (8h) to 5m step

# Robust Sharpe (EWMA) params
EWMA_LAMBDA = 0.94            # higher -> smoother
MIN_EWMA_STEPS = 5            # guard small-N Sharpe explosion

# Deadband around trade cost (ignore tiny PnL)
DEADBAND_COST_MULT = 1.2      # tau = 1.2 * estimated round-trip cost

# Exit behavior
MIN_HOLD_TICKS = 3            # forbid exit for first N ticks after entry
EARLY_EXIT_PEN = 1e-3         # tiny penalty if exit within 1 tick

# Reward mixing
W_SHARPE = 0.9
W_PNL_DB = 0.1

# ===== Eval =====
EVAL_EVERY = 50_000
EVAL_EPISODES = 100

# ===== Utils =====
def _deadband(x: float, tau: float) -> float:
    ax = abs(x)
    if ax < tau:
        return 0.0
    return np.sign(x) * (ax - tau)


class TradeEnv(gym.Env):
    """
    Unified worker env handling both entry and exit timing.
    Uses goals from a manager via GoalBridge.
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str, gb: GoalBridge,
                 fee_rate: float = FEE_RATE,
                 slip_bp: float = SLIP_BP,
                 randomize_start: bool = True,
                 conf_deadzone: float = 0.1,
                 ambiguity_threshold: float = 0.1):
        super().__init__()
        self.gb = gb
        self.conf_deadzone = conf_deadzone
        self.ambiguity_threshold = ambiguity_threshold

        # --- Data Loading ---
        data = build_worker_inputs(split)
        self.X = data["X"]
        self.price = data["price"].astype(float)
        self.funding = data.get("funding_rate", pd.Series(0.0, index=self.X.index)).astype(float)
        self.idx = self.X.index
        self.N = len(self.idx)

        self.fee = float(fee_rate)
        self.slip_bp = float(slip_bp)
        self._randomize_start = bool(randomize_start)
        self._cursor = 0

        # --- Trade State ---
        self.in_position: bool = False
        self.entry_time: int = 0
        self.entry_price: float = 0.0
        self.holding_steps: int = 0
        self.current_dir: int = 0
        self.step_returns: list[float] = []   # for optional debugging

        # EWMA state for robust Sharpe
        self.ewma_r: float = 0.0
        self.ewma_r2: float = 0.0
        self.ewma_n: int = 0

        # --- Spaces ---
        feat_dim = self.X.shape[1]
        # features + manager_dir + manager_conf + manager_regime + in_position + holding_norm
        self.observation_space = spaces.Box(low=-10, high=10, shape=(feat_dim + 5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0:Wait, 1:Enter, 2:Exit

    # ---- Goal decode ----
    def _get_derived_goal(self, i: int) -> Tuple[int, float, int]:
        self.gb.tick(self.idx[i])
        long_conf, short_conf = self.gb.vec()

        # 1) Direction
        if max(long_conf, short_conf) < self.conf_deadzone:
            direction = 0
        else:
            direction = 1 if long_conf > short_conf else -1
        # 2) Confidence
        confidence = max(long_conf, short_conf)
        # 3) Ambiguity flag
        is_ambiguous = 1 if abs(long_conf - short_conf) < self.ambiguity_threshold else 0
        return direction, confidence, is_ambiguous

    # ---- Costs ----
    def _trade_cost_est(self) -> float:
        # approx round-trip cost (ratio): fee in/out + slip in/out
        return 2 * self.fee + 2 * self.slip_bp * 1e-4

    # ---- Metrics at exit ----
    def _calculate_metrics(self, exit_time: int) -> Dict[str, float]:
        if not self.in_position or self.entry_time >= exit_time:
            return dict(pnl=0.0, gross=0.0, fee=0.0, fund=0.0, slip=0.0, sharpe=0.0)

        px_e = self.entry_price
        px_raw_e = float(self.price.iloc[self.entry_time])
        px_raw_x = float(self.price.iloc[exit_time])

        # Exit slippage
        if self.current_dir > 0:   # Long
            px_exec_x = px_raw_x * (1 - self.slip_bp * 1e-4)
            slip_out = (px_raw_x - px_exec_x) / px_raw_e
        else:                       # Short
            px_exec_x = px_raw_x * (1 + self.slip_bp * 1e-4)
            slip_out = (px_exec_x - px_raw_x) / px_raw_e

        # Entry slippage (already applied to entry_price)
        if self.current_dir > 0:
            slip_in = (self.entry_price - px_raw_e) / px_raw_e
        else:
            slip_in = (px_raw_e - self.entry_price) / px_raw_e

        slip_total = (slip_in + slip_out) * np.sign(self.current_dir)
        gross = (px_exec_x - px_e) / px_e * self.current_dir

        # Funding accrual (approx)
        fund = 0.0
        for k in range(self.entry_time, exit_time):
            fr = float(self.funding.iloc[k])
            same = (self.current_dir > 0 and fr > 0) or (self.current_dir < 0 and fr < 0)
            delta = abs(fr) * FUNDING_STEP_FRAC
            fund += (-delta if same else +delta)

        fee = 2 * self.fee
        pnl = gross - fee + fund

        # Robust Sharpe using EWMA state
        if self.ewma_n >= MIN_EWMA_STEPS:
            mu = self.ewma_r
            var = max(self.ewma_r2 - mu * mu, 1e-12)
            sharpe = mu / (np.sqrt(var) + 1e-12)
        else:
            sharpe = float(np.clip(pnl, -0.01, 0.01))

        return dict(pnl=float(pnl), gross=float(gross), fee=float(fee),
                    fund=float(fund), slip=float(slip_total), sharpe=float(sharpe))

    # ---- Observation ----
    def _obs(self) -> np.ndarray:
        i = self._cursor
        x = self.X.iloc[i].to_numpy(np.float32, copy=False)
        manager_dir, manager_conf, manager_regime = self._get_derived_goal(i)
        holding_norm = float(self.holding_steps / MAX_HOLDING_STEPS)
        return np.concatenate([
            x,
            np.array([
                float(manager_dir),
                float(manager_conf),
                float(manager_regime),
                float(self.in_position),
                holding_norm
            ], dtype=np.float32)
        ], axis=0)

    # ---- Action mask ----
    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(3, dtype=bool)
        if self.in_position:
            mask[0] = True  # Wait
            # forbid early exits until MIN_HOLD_TICKS
            mask[2] = bool(self.holding_steps >= MIN_HOLD_TICKS)
        else:
            mask[0] = True  # Wait
            mask[1] = True  # Enter
        return mask

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._randomize_start:
            self._cursor = int(self.np_random.integers(0, max(1, self.N - MAX_HOLDING_STEPS - 5)))
        else:
            self._cursor = 0
        self.in_position = False
        self.entry_time = 0
        self.entry_price = 0.0
        self.holding_steps = 0
        self.current_dir = 0
        self.step_returns = []
        self.ewma_r = 0.0
        self.ewma_r2 = 0.0
        self.ewma_n = 0
        return self._obs(), {}

    def step(self, a: int):
        done = False
        reward = 0.0
        info = {}

        prev_price = self.price.iloc[self._cursor - 1] if self._cursor > 0 else self.price.iloc[0]
        action_is_enter = (a == 1)
        action_is_exit = (a == 2)

        if action_is_enter and not self.in_position:
            self.in_position = True
            self.entry_time = self._cursor
            self.current_dir, _, _ = self._get_derived_goal(self._cursor)
            self.step_returns = []
            self.ewma_r = 0.0
            self.ewma_r2 = 0.0
            self.ewma_n = 0

            # block entry if manager neutral
            if self.current_dir == 0:
                self.in_position = False
                reward = -WAIT_PENALTY
                return self._obs(), reward, False, False, {}

            px_e = float(self.price.iloc[self.entry_time])
            if self.current_dir > 0:  # Long
                self.entry_price = px_e * (1 + self.slip_bp * 1e-4)
            else:                      # Short
                self.entry_price = px_e * (1 - self.slip_bp * 1e-4)
            reward = 0.0

        elif action_is_exit and self.in_position:
            metrics = self._calculate_metrics(self._cursor)
            # combine robust Sharpe + deadbanded PnL + small early-exit pen
            tau = DEADBAND_COST_MULT * self._trade_cost_est()
            pnl_db = _deadband(metrics["pnl"], tau)
            reward = (W_SHARPE * metrics["sharpe"]) + (W_PNL_DB * pnl_db)
            if self.holding_steps <= 1:
                reward -= EARLY_EXIT_PEN

            i_start = self.entry_time
            i_end = self._cursor
            info = {
                'pnl': metrics['pnl'],
                'sharpe': float(metrics['sharpe']),
                'holding_steps': self.holding_steps,
                'dir': int(self.current_dir),
                'entry_ts': str(self.idx[i_start]),
                'exit_ts': str(self.idx[min(i_end, self.N-1)]),
                'gross': metrics['gross'],
                'fee': metrics['fee'],
                'fund': metrics['fund'],
                'slip': metrics['slip'],
                'reward': float(reward),
            }
            self.in_position = False
            done = True

        else:  # Wait
            reward = -WAIT_PENALTY

        # advance time
        self._cursor += 1
        if self.in_position:
            self.holding_steps += 1
            # per-step return and EWMA update
            current_price = self.price.iloc[self._cursor - 1]
            step_return = (current_price / prev_price - 1) * self.current_dir
            self.step_returns.append(step_return)
            r = float(step_return)
            self.ewma_r  = EWMA_LAMBDA * self.ewma_r  + (1 - EWMA_LAMBDA) * r
            self.ewma_r2 = EWMA_LAMBDA * self.ewma_r2 + (1 - EWMA_LAMBDA) * (r * r)
            self.ewma_n += 1

        if self._cursor >= self.N - 2:
            done = True

        if self.in_position and self.holding_steps >= MAX_HOLDING_STEPS:
            metrics = self._calculate_metrics(self._cursor)
            tau = DEADBAND_COST_MULT * self._trade_cost_est()
            pnl_db = _deadband(metrics["pnl"], tau)
            reward = (W_SHARPE * metrics["sharpe"]) + (W_PNL_DB * pnl_db)
            info = {
                'pnl': metrics['pnl'],
                'sharpe': float(metrics['sharpe']),
                'holding_steps': self.holding_steps,
                'forced_exit': True,
                'dir': int(self.current_dir),
                'entry_ts': str(self.idx[self.entry_time]),
                'exit_ts': str(self.idx[min(self._cursor, self.N-1)]),
                'gross': metrics['gross'],
                'fee': metrics['fee'],
                'fund': metrics['fund'],
                'slip': metrics['slip'],
                'reward': float(reward),
            }
            self.in_position = False
            done = True

        return self._obs(), reward, done, False, info


# ===== GoalBridge impl =====
class _ModelGB(GoalBridge):
    def __init__(self, split: str, manager_model: PPO, manager_vecnorm: VecNormalize):
        super().__init__()
        self.m = build_manager_inputs(split)
        self.idx = self.m["XH"].index
        self.k = 0
        self.model = manager_model
        self.vecnorm = manager_vecnorm
        self.W = 8  # must match ManagerV2
        feat_dim = self.m["XH"].shape[1]
        self._buf = np.zeros((self.W, feat_dim), dtype=np.float32)

    def tick(self, ts_5m):
        th = pd.Timestamp(ts_5m).floor("1h")
        while self.k + 1 < len(self.idx) and self.idx[self.k + 1] <= th:
            self.k += 1
        s = max(0, self.k - self.W + 1)
        chunk = self.m["XH"].iloc[s:self.k + 1].to_numpy(dtype=np.float32, copy=False)
        if len(chunk) < self.W:
            pad = np.repeat(chunk[:1], self.W - len(chunk), axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        self._buf[...] = chunk

        # Manager vecnorm expects flattened (W*F,)
        normalized_obs = self.vecnorm.normalize_obs(self._buf.reshape(1, -1))
        actions, _ = self.model.predict(normalized_obs, deterministic=True)
        self.set(actions[0])


# ===== Action Mask hook =====
def _mask_fn(env: TradeEnv) -> np.ndarray:
    return env.valid_action_mask()


# ===== Train Function =====
def train_unified_worker(
    manager_path: str,
    manager_vecnorm_path: str,
    split: str = "train",
    steps: int = 500_000,
    seed: int = 42,
    save_path: Optional[str] = None
):
    manager_model = PPO.load(manager_path, device="cpu")
    # Dummy env for manager vecnorm (ManagerV2Env is flattened (W*F,))
    dummy_manager_env = DummyVecEnv([lambda: ManagerV2Env(split=split)])
    manager_vecnorm = VecNormalize.load(manager_vecnorm_path, dummy_manager_env)
    manager_vecnorm.training = False
    manager_vecnorm.norm_reward = False

    goal_bridge = _ModelGB(split, manager_model, manager_vecnorm)

    def make_env_fn() -> gym.Env:
        env = TradeEnv(split=split, gb=goal_bridge, randomize_start=True)
        return ActionMasker(env, _mask_fn)

    env = DummyVecEnv([make_env_fn])
    vec = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)

    model = MaskablePPO(
        "MlpPolicy", vec,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        device="cpu",
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        clip_range=0.2,
        vf_coef=0.5,
        seed=seed,
        verbose=1,
    )

    print(f"[HRL] Starting training for unified worker for {steps:,} steps...")
    model.learn(total_timesteps=steps, callback=Vitals(tag="unified-worker", every=25_000))

    final_save_path = save_path or os.path.join(MODEL_DIR, "worker_unified_final.zip")
    model.save(final_save_path)

    vecnorm_path = os.path.join(MODEL_DIR, "worker_unified_vecnorm.pkl")
    vec.save(vecnorm_path)

    print(f"[HRL] Training complete. Unified worker model saved to {final_save_path}")
    return final_save_path


class Vitals(BaseCallback):
    def __init__(self, tag="worker-entry", every=10_000, verbose=0):
        super().__init__(verbose)
        self.tag=tag; self.every=int(every); self._last=0
    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t - self._last < self.every:
            return True
        self._last = t
        try:
            ent = float(self.model.ent_coef)
        except Exception:
            ent = float('nan')
        print(f"[Vitals/{self.tag}] t={t:,} ent_coef={ent:.4f}")
        return True


# ===== Main (example) =====
def run_unified_training_pipeline():
    import glob
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    model_dir_abs = os.path.join(project_root, 'data', 'model')

    print("[HRL] Finding latest manager model...")
    search_pattern = os.path.join(model_dir_abs, "manager*.zip")
    manager_models = glob.glob(search_pattern)
    if not manager_models:
        print(f"[ERROR] No manager model found in {model_dir_abs}. Cannot proceed.")
        return

    latest_manager_path = max(manager_models, key=os.path.getmtime)
    print(f"[HRL] Using manager: {latest_manager_path}")

    base_path, _ = os.path.splitext(latest_manager_path)
    vecnorm_path = f"{base_path}_vecnorm.pkl"
    if not os.path.exists(vecnorm_path):
        print(f"[ERROR] VecNormalize path not found: {vecnorm_path}")
        return

    train_unified_worker(manager_path=latest_manager_path, manager_vecnorm_path=vecnorm_path)

if __name__ == "__main__":
    run_unified_training_pipeline()
