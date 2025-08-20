from __future__ import annotations

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ai_binance.train.reinforce.common import (
    MODEL_DIR, build_manager_inputs, Goal, GoalBridge,
    M_W1, M_W3, FLIP_PENALTY, TURN_PENALTY
)

class ManagerEnv(gym.Env):
    """
    Obs: [X_1h + 4h(ffill) scaled]
    Act: MultiDiscrete([3, 11]) → dir ∈ {-1,0,1}, conf ∈ {0..10}/10
    Rule: 4h 약레짐이면 dir=0(거래 금지), 시도시 소액 패널티 (env 내부 보상에 반영)
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str, gb: GoalBridge):
        super().__init__()
        data = build_manager_inputs(split)
        self.XH = data["XH"]
        self.price = data["price"]
        self.regweak = data["reg4h_weak"].astype(bool)
        self.regsign = data["reg4h_sign"].astype(int)
        self.idx = self.XH.index
        self.t = 0
        self.gb = gb
        self.prev_dir = 0

        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(self.XH.shape[1],),
                                            dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 11])

    def _obs(self):
        return self.XH.iloc[self.t].to_numpy(dtype=np.float32, copy=False)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0; self.prev_dir = 0
        self.gb.set(Goal(0, 0.0))
        return self._obs(), {}

    def step(self, a):
        dir_raw = [-1, 0, 1][int(a[0])]
        conf = float(int(a[1]) / 10.0)

        weak = bool(self.regweak.iloc[self.t])
        tried_nonflat = (dir_raw != 0)
        dir_eff = 0 if weak else dir_raw
        pen = -0.02 if (weak and tried_nonflat) else 0.0

        self.gb.set(Goal(dir_eff, conf))

        cur = float(self.price.iloc[self.t])
        nxt1 = float(self.price.iloc[min(self.t + 1, len(self.price) - 1)])
        nxt3 = float(self.price.iloc[min(self.t + 3, len(self.price) - 1)])
        r1 = (nxt1 - cur) / cur
        r3 = (nxt3 - cur) / cur

        R_dir  = M_W1 * dir_eff * r1 + M_W3 * dir_eff * r3
        R_flip = -FLIP_PENALTY * int(dir_eff != self.prev_dir)
        R_mis  = -TURN_PENALTY  * int(np.sign(dir_eff) != int(self.regsign.iloc[self.t]) and dir_eff != 0)
        R = R_dir + R_flip + R_mis + pen

        self.prev_dir = dir_eff
        self.t += 1
        terminated = (self.t >= len(self.XH) - 2)
        info = {"weak4h": weak, "reg4h": int(self.regsign.iloc[self.t - 1]), "dir": dir_eff}
        return self._obs(), float(R), terminated, False, info

# ===== Train =====
def train_manager(split: str = "train", steps: int = 800_000, seed: int = 42, save_path: str | None = None):
    gb = GoalBridge()
    env = DummyVecEnv([lambda: ManagerEnv(split=split, gb=gb)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=256,
        batch_size=256,
        device="cpu",
        learning_rate=1e-4, gamma=0.95,
        ent_coef=0.005, clip_range=0.2, gae_lambda=0.95,
        vf_coef=1.2, clip_range_vf=0.2,
        seed=seed, verbose=1
    )
    model.learn(total_timesteps=steps)

    sp = save_path or os.path.join(MODEL_DIR, "manager_stage1.zip")
    model.save(sp)
    env.save(os.path.join(MODEL_DIR, "manager_stage1_vecnorm.pkl"))
    return sp

def run_manager():
    print("[HRL] Training Manager…")
    mp = train_manager()
    print(f"[OK] Manager saved → {mp}")

if __name__ == "__main__":
    run_manager()
