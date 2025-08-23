# ai_binance/train/reinforce/worker.py
from __future__ import annotations

# ===== allow running as a script =====
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Callable, Optional

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

from ai_binance.train.reinforce.common import (
    MODEL_DIR,
    build_worker_inputs, build_manager_inputs,
    GoalBridge, EntropyDecay,
    FEE_RATE, SLIP_BP,
)
from ai_binance.train.reinforce.manager import ManagerV2Env

# =============================================================================
# Design: Unified Trading Worker
#  - Manages the full trade cycle (entry and exit) under the manager's guidance.
#  - Act: 0=Wait, 1=Enter, 2=Exit
#  - Obs: Market features + position status (in_position, holding_period)
#  - Reward: Realized PnL upon closing a trade (Exit action).
#            Small penalty for waiting to encourage action.
# =============================================================================

# ===== Knobs =====
MAX_HOLDING_STEPS = 72      # Max holding period (5m * 72 = 6h)
WAIT_PENALTY = 1e-5         # Small penalty for each 'Wait' step
FUNDING_STEP_FRAC = 5.0 / 480.0 # 펀딩(8h 주기)을 5m 스텝으로 환산

# ===== Eval =====
EVAL_EVERY = 50_000
EVAL_EPISODES = 100 # Reduced for faster eval of longer episodes

class TradeEnv(gym.Env):
    """
    A unified worker environment that handles both entry and exit timing.
    Receives a high-level goal (long/short conf) from a manager.
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

        # --- Spaces ---
        feat_dim = self.X.shape[1]
        # Obs: features + manager_dir + manager_conf + manager_regime + in_position + holding_period_norm
        self.observation_space = spaces.Box(low=-10, high=10, shape=(feat_dim + 5,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0:Wait, 1:Enter, 2:Exit

    def _get_derived_goal(self, i: int) -> Tuple[int, float, int]:
        """ Interprets the raw [long_conf, short_conf] from the GoalBridge. """
        self.gb.tick(self.idx[i])
        long_conf, short_conf = self.gb.vec()

        # 1. Derive Direction
        if max(long_conf, short_conf) < self.conf_deadzone:
            direction = 0
        else:
            direction = 1 if long_conf > short_conf else -1
        
        # 2. Derive Confidence
        confidence = max(long_conf, short_conf)

        # 3. Derive Regime (Ambiguity)
        is_ambiguous = 1 if abs(long_conf - short_conf) < self.ambiguity_threshold else 0
        
        return direction, confidence, is_ambiguous

    def _calculate_pnl(self, exit_time: int) -> Dict[str, float]:
        if not self.in_position or self.entry_time >= exit_time:
            return dict(pnl=0.0, gross=0.0, fee=0.0, fund=0.0, slip=0.0)

        px_e = self.entry_price                    # 슬립 적용된 진입가(이미)
        px_raw_e = float(self.price.iloc[self.entry_time])  # 원시 진입가
        px_raw_x = float(self.price.iloc[exit_time])        # 원시 청산가

        # 청산 슬립 적용
        if self.current_dir > 0:   # Long
            px_exec_x = px_raw_x * (1 - self.slip_bp * 1e-4)
            slip_out = (px_raw_x - px_exec_x) / px_raw_e
        else:                       # Short
            px_exec_x = px_raw_x * (1 + self.slip_bp * 1e-4)
            slip_out = (px_exec_x - px_raw_x) / px_raw_e

        # 진입 슬립(이미 entry_price에 반영) → 상대 수익률 기준으로 계산
        if self.current_dir > 0:
            slip_in = (self.entry_price - px_raw_e) / px_raw_e
        else:
            slip_in = (px_raw_e - self.entry_price) / px_raw_e

        # 총 슬립(양쪽)
        slip_total = (slip_in + slip_out) * np.sign(self.current_dir)

        gross = (px_exec_x - px_e) / px_e * self.current_dir  # 체결가 기준 수익
        # 펀딩
        fund = 0.0
        for k in range(self.entry_time, exit_time):
            fr = float(self.funding.iloc[k])
            same = (self.current_dir > 0 and fr > 0) or (self.current_dir < 0 and fr < 0)
            delta = abs(fr) * FUNDING_STEP_FRAC
            fund += (-delta if same else +delta)

        fee = 2 * self.fee
        pnl = gross - fee + fund
        return dict(pnl=float(pnl), gross=float(gross), fee=float(fee),
                    fund=float(fund), slip=float(slip_total))

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

    def valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(3, dtype=bool)
        if self.in_position:
            mask[0] = True # Wait
            mask[2] = True # Exit
        else:
            mask[0] = True # Wait
            mask[1] = True # Enter
        return mask

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
        
        return self._obs(), {}

    def step(self, a: int):
        done = False
        reward = 0.0
        info = {}

        action_is_enter = (a == 1)
        action_is_exit = (a == 2)
        
        if action_is_enter and not self.in_position:
            self.in_position = True
            self.entry_time = self._cursor
            self.current_dir, _, _ = self._get_derived_goal(self._cursor)

            # Block entry if manager is neutral
            if self.current_dir == 0:
                self.in_position = False # Revert state
                reward = -WAIT_PENALTY   # Treat as a wait
                return self._obs(), reward, False, False, {}
            
            px_e = float(self.price.iloc[self.entry_time])
            if self.current_dir > 0: # Long
                self.entry_price = px_e * (1 + self.slip_bp * 1e-4)
            else: # Short
                self.entry_price = px_e * (1 - self.slip_bp * 1e-4)
            
            reward = 0.0 # No reward until exit

        elif action_is_exit and self.in_position:
            comps = self._calculate_pnl(self._cursor)
            reward = comps["pnl"]
            # 메타
            i_start = self.entry_time
            i_end = self._cursor
            info = {
                'pnl': reward,
                'holding_steps': self.holding_steps,
                'dir': int(self.current_dir),
                'entry_ts': str(self.idx[i_start]),
                'exit_ts': str(self.idx[min(i_end, self.N-1)]),
                'gross': comps['gross'],
                'fee': comps['fee'],
                'fund': comps['fund'],
                'slip': comps['slip'],
            }
            self.in_position = False
            done = True

        else: # Wait action
            reward = -WAIT_PENALTY

        self._cursor += 1
        if self.in_position:
            self.holding_steps += 1

        if self._cursor >= self.N - 2:
            done = True
        
        if self.in_position and self.holding_steps >= MAX_HOLDING_STEPS:
            comps = self._calculate_pnl(self._cursor)
            reward = comps["pnl"]
            info = {
                'pnl': reward,
                'holding_steps': self.holding_steps,
                'forced_exit': True,
                'dir': int(self.current_dir),
                'entry_ts': str(self.idx[self.entry_time]),
                'exit_ts': str(self.idx[min(self._cursor, self.N-1)]),
                'gross': comps['gross'],
                'fee': comps['fee'],
                'fund': comps['fund'],
                'slip': comps['slip'],
            }
            self.in_position = False
            done = True

        return self._obs(), reward, done, False, info

# ===== GoalBridges (Updated) =====

class _ModelGB(GoalBridge):
    def __init__(self, split: str, manager_model: PPO, manager_vecnorm: VecNormalize):
        super().__init__()
        self.m = build_manager_inputs(split) # For sequencing
        self.idx = self.m["XH"].index
        self.k = 0
        self.model = manager_model
        self.vecnorm = manager_vecnorm
        
        # Buffer to construct the observation sequence for the manager
        self.W = 8  # Must match SEQ_WINDOW in ManagerV2Env
        feat_dim = self.m["XH"].shape[1]
        self._buf = np.zeros((self.W, feat_dim), dtype=np.float32)

    def tick(self, ts_5m):
        th = pd.Timestamp(ts_5m).floor("1h")
        while self.k + 1 < len(self.idx) and self.idx[self.k + 1] <= th:
            self.k += 1
        
        # Fill buffer
        s = max(0, self.k - self.W + 1)
        chunk = self.m["XH"].iloc[s:self.k + 1].to_numpy(dtype=np.float32, copy=False)
        if len(chunk) < self.W:
            pad = np.repeat(chunk[:1], self.W - len(chunk), axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        self._buf[...] = chunk
        
        obs_seq = self._buf.ravel()
        
        # IMPORTANT: Normalize the observation before prediction
        normalized_obs = self.vecnorm.normalize_obs(obs_seq.reshape(1, -1))
        
        # Get [long_conf, short_conf] from the manager and set it in the bridge
        actions, _ = self.model.predict(normalized_obs, deterministic=True)
        self.set(actions[0])

# ===== Action Mask hook =====
def _mask_fn(env: TradeEnv) -> np.ndarray:
    return env.valid_action_mask()

# ===== Train Function (New Unified Version) =====
def train_unified_worker(
    manager_path: str,
    manager_vecnorm_path: str,
    split: str = "train",
    steps: int = 500_000,
    seed: int = 42,
    save_path: Optional[str] = None
):
    """
    Trains the unified TradeEnv worker with a pre-trained manager.
    """
    manager_model = PPO.load(manager_path, device="cpu")
    # Create a dummy env to load the VecNormalize stats
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
        verbose=1
    )

    callbacks = [Vitals(tag="unified-worker", every=25_000)]

    print(f"[HRL] Starting training for unified worker for {steps:,} steps...")
    model.learn(total_timesteps=steps, callback=callbacks)

    # --- Saving ---
    final_save_path = save_path or os.path.join(MODEL_DIR, "worker_unified_final.zip")
    model.save(final_save_path)
    
    vecnorm_path = os.path.join(MODEL_DIR, "worker_unified_vecnorm.pkl")
    vec.save(vecnorm_path)
    
    print(f"[HRL] Training complete. Unified worker model saved to {final_save_path}")
    return final_save_path

class Vitals(BaseCallback):
    def __init__(self, tag="worker-entry", every=10_000, verbose=0):
        super().__init__(verbose); self.tag=tag; self.every=int(every); self._last=0
    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t - self._last < self.every: return True
        self._last = t
        try: ent = float(self.model.ent_coef)
        except Exception: ent = float('nan')
        print(f"[Vitals/{self.tag}] t={t:,} ent_coef={ent:.4f}")
        return True

# ===== Main execution block (Example) =====
def run_unified_training_pipeline():
    """
    Example of how to run the new unified training pipeline.
    """
    import glob
    
    print("[HRL] Finding latest manager model...")
    search_pattern = os.path.join("ai_binance", "**", "manager*.zip")
    manager_models = glob.glob(search_pattern, recursive=True)
    
    if not manager_models:
        print("[ERROR] No manager model found. Cannot proceed.")
        return

    latest_manager_path = max(manager_models, key=os.path.getmtime)
    print(f"[HRL] Using manager: {latest_manager_path}")

    # Derive vecnorm path from manager path
    base_path, _ = os.path.splitext(latest_manager_path)
    vecnorm_path = f"{base_path}_vecnorm.pkl"

    if not os.path.exists(vecnorm_path):
        print(f"[ERROR] VecNormalize path not found: {vecnorm_path}")
        return

    train_unified_worker(manager_path=latest_manager_path, manager_vecnorm_path=vecnorm_path)

if __name__ == "__main__":
    # This will run the full, unified training pipeline automatically when the script is executed.
    run_unified_training_pipeline()