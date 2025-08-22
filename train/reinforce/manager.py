# ai_binance/train/reinforce/manager.py
from __future__ import annotations

# ===== allow running as a script =====
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn

from ai_binance.train.reinforce.common import (
    MODEL_DIR, build_manager_inputs, GoalBridge,
    M_W1, M_W3, FLIP_PENALTY, TURN_PENALTY
)

# ===== Knobs =====
# --- Major ---
REWARD_SCALE = 5.0
BRIER_WEIGHT = 1.5  # Weight for the calibration reward (Brier score)
CONF_DEADZONE = 0.1 # Confidence threshold to determine a neutral direction

# --- Minor ---
ENT_START = 0.01
ENT_END   = 0.001
ENT_DECAY_STEPS = 100_000
MAX_EPISODE_STEPS = 4_096
LOOKAHEAD_H = 3
SEQ_WINDOW  = 8
BP_REF = 0.0015 # Reference for future return magnitude

# === LR schedule ===
LR_START = 1.5e-4
LR_END   = 5e-5


class ManagerV2Env(gym.Env):
    """
    Manager V2: Outputs separate confidences for long and short.
    
    Obs: [X_1h + 4h(ffill) scaled] sequence
    Act: Box(2,) -> [long_confidence, short_confidence]
    Reward:
        - Directional PnL (main driver)
        - Brier score for confidence calibration
        - Penalties for flipping decisions or mismatching regime
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str):
        super().__init__()
        data = build_manager_inputs(split)
        self.XH = data["XH"]
        self.price = data["price"]
        self.regweak = data["reg4h_weak"].astype(bool)
        self.regsign = data["reg4h_sign"].astype(int)
        self.idx = self.XH.index

        self.t = 0
        self.prev_dir = 0
        self.steps_in_ep = 0

        # ===== sequence buffer =====
        self.W = int(SEQ_WINDOW)
        feat_dim = self.XH.shape[1]
        self._buf = np.zeros((self.W, feat_dim), dtype=np.float32)

        self.observation_space = spaces.Box(low=-10, high=10, shape=(feat_dim * self.W,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.max_ep_steps = MAX_EPISODE_STEPS
        self.end_guard = LOOKAHEAD_H
        raw = len(self.XH) - self.max_ep_steps - self.end_guard
        self.max_start = max(self.W + 1, raw)

    def _fill_buf(self):
        s = max(0, self.t - self.W + 1)
        chunk = self.XH.iloc[s:self.t + 1].to_numpy(dtype=np.float32, copy=False)
        if len(chunk) < self.W:
            pad = np.repeat(chunk[:1], self.W - len(chunk), axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        self._buf[...] = chunk

    def _obs(self):
        return self._buf.ravel()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_in_ep = 0
        self.t = int(self.np_random.integers(self.W, self.max_start)) if self.max_start > self.W else self.W
        self.prev_dir = 0
        self._fill_buf()
        return self._obs(), {}

    def step(self, a):
        # --- Action processing ---
        conf_long, conf_short = a[0], a[1]

        # Determine effective direction with a deadzone
        if max(conf_long, conf_short) < CONF_DEADZONE:
            dir_eff = 0
        else:
            dir_eff = 1 if conf_long > conf_short else -1
        
        weak = bool(self.regweak.iloc[self.t])
        if weak: # No trading in weak regimes
            dir_eff = 0

        # --- Future return for reward calculation ---
        cur  = float(self.price.iloc[self.t])
        nxt1 = float(self.price.iloc[min(self.t + 1, len(self.price) - 1)])
        nxt3 = float(self.price.iloc[min(self.t + 3, len(self.price) - 1)])
        r1 = np.log(max(nxt1, 1e-12) / max(cur, 1e-12))
        r3 = np.log(max(nxt3, 1e-12) / max(cur, 1e-12))
        r_w = M_W1 * r1 + M_W3 * r3

        # --- Reward components ---
        # 1. Directional reward
        R_dir  = dir_eff * r_w

        # 2. Calibration reward (Brier Score)
        #    Penalizes the model for being confident in the wrong direction.
        is_up = 1.0 if r_w > 0 else 0.0
        is_down = 1.0 if r_w < 0 else 0.0
        brier_long = (conf_long - is_up)**2
        brier_short = (conf_short - is_down)**2
        R_cal = -BRIER_WEIGHT * (brier_long + brier_short)

        # 3. Transition penalties
        R_flip = -FLIP_PENALTY * int(dir_eff != self.prev_dir)
        R_mis  = -TURN_PENALTY * int((dir_eff != 0) and (np.sign(dir_eff) != int(self.regsign.iloc[self.t])))

        # --- Total Reward ---
        R = (R_dir + R_cal + R_flip + R_mis) * REWARD_SCALE

        # --- State update ---
        self.prev_dir = dir_eff
        self.t += 1
        self.steps_in_ep += 1
        self._fill_buf()

        # --- Termination ---
        time_over = (self.t >= len(self.XH) - self.end_guard)
        horizon_over = (self.steps_in_ep >= MAX_EPISODE_STEPS)
        terminated = bool(time_over)
        truncated = bool(not terminated and horizon_over)

        info = { "dir": dir_eff, "conf_long": conf_long, "conf_short": conf_short, "r_w": r_w, "R_cal": R_cal }
        return self._obs(), float(R), terminated, truncated, info


def lr_schedule(progress_remaining: float) -> float:
    return float(LR_END + (LR_START - LR_END) * progress_remaining)


def train_manager_v2(split: str = "train", steps: int = 600_000, seed: int = 42, save_path: str | None = None):
    def make_env():
        return Monitor(ManagerV2Env(split=split))
    env = DummyVecEnv([make_env])

    vec = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=0.98)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[dict(pi=[128, 128], vf=[256, 256])]
    )

    model = PPO(
        "MlpPolicy", vec,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        device="cpu",
        learning_rate=lr_schedule,
        gamma=0.98,
        gae_lambda=0.95,
        ent_coef=ENT_START,
        clip_range=0.2,
        vf_coef=0.8,
        max_grad_norm=0.5,
        seed=seed,
        verbose=1,
        policy_kwargs=policy_kwargs
    )

    class EntDecay(BaseCallback):
        def _on_step(self) -> bool:
            frac = max(0.0, 1.0 - self.num_timesteps / ENT_DECAY_STEPS)
            self.model.ent_coef = float(ENT_END + (ENT_START - ENT_END) * frac)
            return True

    model.learn(total_timesteps=steps, callback=EntDecay())

    sp = save_path or os.path.join(MODEL_DIR, "manager_v2.zip")
    model.save(sp)
    vec.save(os.path.join(MODEL_DIR, "manager_v2_vecnorm.pkl"))
    print(f"[OK] Manager V2 saved → {sp}")
    return sp


if __name__ == "__main__":
    train_manager_v2()