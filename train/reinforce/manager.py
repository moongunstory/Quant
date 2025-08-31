# ai_binance/train/reinforce/manager.py (REV-2: Flattened Obs W*F)
"""
Manager V2 — Transformer w/ Flattened Observation (W*F,)
- Root-cause fix for VecNormalize dim mismatch in live.
- Train-time observation is explicitly 1D (W*F,),
  extractor reshapes back to (W,F) for the Transformer.
- Guarantees vecnorm.obs_rms.size == W*F.

Key changes vs prior version:
1) Env.observation_space -> Box(shape=(W*F,)) and _obs() returns 1D.
2) TransformerFeatureExtractor takes (seq_len=W, n_features=F) and reshapes input.
3) Training: pass W,F via policy_kwargs; assert vecnorm shape before saving.
4) Save metadata with columns + dims for live-time verification.
"""

from __future__ import annotations

# ===== allow running as a script =====
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ai_binance.train.reinforce.common import (
    MODEL_DIR, build_manager_inputs,
    M_W1, M_W3, FLIP_PENALTY, TURN_PENALTY
)

# ===== Knobs =====
# --- Major ---
REWARD_SCALE   = 5.0
BRIER_WEIGHT   = 1.5
CONF_DEADZONE  = 0.1

# --- Minor ---
ENT_START        = 0.01
ENT_END          = 0.001
ENT_DECAY_STEPS  = 100_000
MAX_EPISODE_STEPS= 4_096
LOOKAHEAD_H      = 3
SEQ_WINDOW       = 8
BP_REF           = 0.0015

# === LR schedule ===
LR_START = 1.5e-4
LR_END   = 5e-5

# ===== Feature Extractor (Transformer) =====
class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """
    Takes flattened observations (W*F,) and reshapes to (W,F) for a Transformer.
    Returns last-token embedding (d_model).
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int | None = None,
        n_features: int | None = None,
    ):
        super().__init__(observation_space, features_dim=d_model)

        obs_dim = int(np.prod(observation_space.shape))
        assert seq_len is not None and n_features is not None, "TransformerFeatureExtractor needs seq_len and n_features"
        assert seq_len * n_features == obs_dim, f"obs_dim={obs_dim} must equal W*F={seq_len*n_features}"

        self.seq_len     = int(seq_len)
        self.n_features  = int(n_features)
        self.input_proj  = nn.Linear(self.n_features, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.1, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, W*F)
        b, d = observations.shape
        x = observations.view(b, self.seq_len, self.n_features)  # (B,W,F)
        x = self.input_proj(x)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        return x[:, -1, :]  # (B, d_model): use last step representation

# ===== Environment =====
class ManagerV2Env(gym.Env):
    """Manager V2 with flattened observation (W*F,)."""
    metadata = {"render_modes": []}

    def __init__(self, split: str):
        super().__init__()
        data = build_manager_inputs(split)
        self.XH      = data["XH"]          # features DataFrame
        self.price   = data["price"]       # price Series
        self.regweak = data["reg4h_weak"].astype(bool)
        self.regsign = data["reg4h_sign"].astype(int)

        self.idx = self.XH.index
        self.t = 0
        self.prev_dir = 0
        self.steps_in_ep = 0

        self.W = int(SEQ_WINDOW)
        self.feat_dim = int(self.XH.shape[1])
        self._buf = np.zeros((self.W, self.feat_dim), dtype=np.float32)

        # Flattened observation: (W*F,)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(self.W * self.feat_dim,), dtype=np.float32
        )
        # Actions: (conf_long, conf_short) in [0,1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.max_ep_steps = MAX_EPISODE_STEPS
        self.end_guard    = LOOKAHEAD_H
        raw = len(self.XH) - self.max_ep_steps - self.end_guard
        self.max_start = max(self.W + 1, raw)

    # ---- internals ----
    def _fill_buf(self):
        s = max(0, self.t - self.W + 1)
        chunk = self.XH.iloc[s:self.t + 1].to_numpy(dtype=np.float32, copy=False)
        if len(chunk) < self.W:
            pad = np.repeat(chunk[:1], self.W - len(chunk), axis=0)
            chunk = np.concatenate([pad, chunk], axis=0)
        self._buf[...] = chunk

    def _obs(self):
        # return flattened (W*F,)
        return self._buf.reshape(-1).astype(np.float32)

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_in_ep = 0
        self.t = int(self.np_random.integers(self.W, self.max_start)) if self.max_start > self.W else self.W
        self.prev_dir = 0
        self._fill_buf()
        return self._obs(), {}

    def step(self, a):
        conf_long, conf_short = float(a[0]), float(a[1])

        # Direction gating via confidence + regime-weak
        if max(conf_long, conf_short) < CONF_DEADZONE:
            dir_eff = 0
        else:
            dir_eff = 1 if conf_long > conf_short else -1
        if bool(self.regweak.iloc[self.t]):
            dir_eff = 0

        cur  = float(self.price.iloc[self.t])
        nxt1 = float(self.price.iloc[min(self.t + 1, len(self.price) - 1)])
        nxt3 = float(self.price.iloc[min(self.t + 3, len(self.price) - 1)])
        r1 = np.log(max(nxt1, 1e-12) / max(cur, 1e-12))
        r3 = np.log(max(nxt3, 1e-12) / max(cur, 1e-12))
        r_w = M_W1 * r1 + M_W3 * r3

        R_dir  = dir_eff * r_w
        is_up   = 1.0 if r_w > 0 else 0.0
        is_down = 1.0 if r_w < 0 else 0.0
        brier_long  = (conf_long  - is_up)  ** 2
        brier_short = (conf_short - is_down) ** 2
        R_cal  = -BRIER_WEIGHT * (brier_long + brier_short)
        R_flip = -FLIP_PENALTY * int(dir_eff != self.prev_dir)
        R_mis  = -TURN_PENALTY * int((dir_eff != 0) and (np.sign(dir_eff) != int(self.regsign.iloc[self.t])))

        R = (R_dir + R_cal + R_flip + R_mis) * REWARD_SCALE

        self.prev_dir = dir_eff
        self.t += 1
        self.steps_in_ep += 1
        self._fill_buf()

        terminated = bool(self.t >= len(self.XH) - self.end_guard)
        truncated  = bool(not terminated and self.steps_in_ep >= MAX_EPISODE_STEPS)
        info = {"dir": dir_eff, "conf_long": conf_long, "conf_short": conf_short, "r_w": r_w, "R_cal": R_cal}
        return self._obs(), float(R), terminated, truncated, info

# ===== Training =====

def lr_schedule(progress_remaining: float) -> float:
    return float(LR_END + (LR_START - LR_END) * progress_remaining)

class _EntDecay(BaseCallback):
    def __init__(self, ent_start: float, ent_end: float, decay_steps: int):
        super().__init__()
        self.ent_start = float(ent_start)
        self.ent_end   = float(ent_end)
        self.decay_steps = int(decay_steps)
    def _on_step(self) -> bool:
        frac = max(0.0, 1.0 - self.num_timesteps / self.decay_steps)
        self.model.ent_coef = float(self.ent_end + (self.ent_start - self.ent_end) * frac)
        return True


def train_manager_v2(split: str = "train", steps: int = 600_000, seed: int = 42, save_path: str | None = None):
    # Optional: make CPU runs deterministic-ish
    try:
        torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
    except Exception:
        pass
    
    def make_env():
        return Monitor(ManagerV2Env(split=split))

    env = DummyVecEnv([make_env])
    vec = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.0, gamma=0.98)

    # Extract dims for policy kwargs
    W = env.envs[0].env.W
    F = env.envs[0].env.feat_dim

    policy_kwargs = dict(
        features_extractor_class=TransformerFeatureExtractor,
        features_extractor_kwargs=dict(d_model=128, nhead=4, num_layers=2, seq_len=W, n_features=F),
        net_arch=dict(pi=[128], vf=[128]),  # heads after extractor
    )

    model = PPO(
        policy="MlpPolicy",  # uses our custom extractor
        env=vec,
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
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=steps, callback=_EntDecay(ENT_START, ENT_END, ENT_DECAY_STEPS))

    # ---- Guard: ensure vecnorm dim matches W*F before saving ----
    rms_shape = np.asarray(vec.obs_rms.mean).shape
    assert int(np.prod(rms_shape)) == W * F, f"VecNorm size {rms_shape} != {W*F}"
    print(f"[CHECK] VecNorm obs_rms shape: {rms_shape} (expected {(W*F,)})")

    # ---- Save ----
    sp = save_path or os.path.join(MODEL_DIR, "manager_v2.zip")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(sp)
    vec.save(os.path.join(MODEL_DIR, "manager_v2_vecnorm.pkl"))
    print(f"[OK] Manager V2 (Transformer) saved → {sp}")

    # ---- Persist metadata for live matching ----
    env_inner = vec.envs[0].env
    meta = {
        "seq_window": int(env_inner.W),
        "feat_dim": int(env_inner.feat_dim),
        "columns": list(env_inner.XH.columns),
        "vecnorm_obs_size": int(np.prod(rms_shape)),
        "note": "Flattened obs (W*F,) + transformer extractor",
    }
    meta_path = os.path.join(MODEL_DIR, "manager_v2_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Manager V2 metadata saved → {meta_path}")

    return sp


if __name__ == "__main__":
    # quick smoke run with defaults
    train_manager_v2()
