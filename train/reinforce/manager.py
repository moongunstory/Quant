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

from ai_binance.train.reinforce.common import (
    MODEL_DIR, build_manager_inputs, Goal, GoalBridge,
    M_W1, M_W3, FLIP_PENALTY, TURN_PENALTY
)

# ===== Training knobs =====
ENT_START = 0.03
ENT_END = 0.003
ENT_DECAY_STEPS = 50_000
MAX_EPISODE_STEPS = 4_096
REWARD_SCALE = 5.0


class ManagerEnv(gym.Env):
    """
    Obs: [X_1h + 4h(ffill) scaled]
    Act: MultiDiscrete([3, 11]) → dir ∈ {-1,0,1}, conf ∈ {0..10}/10
    Rule: 4h 약레짐이면 dir=0(거래 금지), 시도시 소액 패널티
    Reward: 방향 정확도(1h,3h) - flip패널티 - 4h상충패널티
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
        self.gb = gb

        self.t = 0
        self.prev_dir = 0
        self.steps_in_ep = 0

        self.observation_space = spaces.Box(
            low=-10, high=10,
            shape=(self.XH.shape[1],), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([3, 11])

        self.max_ep_steps = MAX_EPISODE_STEPS
        self.max_start = max(0, len(self.XH) - self.max_ep_steps - 3)

    def _obs(self):
        return self.XH.iloc[self.t].to_numpy(dtype=np.float32, copy=False)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_in_ep = 0
        if self.max_start > 0:
            self.t = int(self.np_random.integers(0, self.max_start))
        else:
            self.t = 0
        self.prev_dir = 0
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
        R = (R_dir + R_flip + R_mis + pen) * REWARD_SCALE

        self.prev_dir = dir_eff
        self.t += 1
        self.steps_in_ep += 1

        time_over = (self.t >= len(self.XH) - 2)
        horizon_over = (self.steps_in_ep >= MAX_EPISODE_STEPS)
        terminated = bool(time_over)
        truncated = bool(not terminated and horizon_over)

        info = {
            "weak4h": weak,
            "reg4h": int(self.regsign.iloc[self.t - 1]) if self.t > 0 else 0,
            "dir": dir_eff
        }
        return self._obs(), float(R), terminated, truncated, info


# ===== Entropy linear decay callback =====
class EntropyDecay(BaseCallback):
    def __init__(self, start=ENT_START, end=ENT_END, decay_steps=ENT_DECAY_STEPS, verbose=0):
        super().__init__(verbose)
        self.start, self.end, self.decay = float(start), float(end), int(decay_steps)

    def _on_training_start(self):
        self.model.ent_coef = self.start

    def _on_step(self):
        step = self.num_timesteps
        frac = min(1.0, step / self.decay)
        self.model.ent_coef = float(self.start + (self.end - self.start) * frac)
        return True


# ===== Safe vitals one-line logger (SB3 버전 호환) =====
class VitalsProbe(BaseCallback):
    def __init__(self, tag: str = "manager", every: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.tag = tag
        self.every = int(every)
        self._last = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step - self._last < self.every:
            return True
        self._last = step

        try:
            ent_coef = float(self.model.ent_coef)
        except Exception:
            ent_coef = float('nan')
        frac = min(1.0, step / float(ENT_DECAY_STEPS))
        pct = int(frac * 100)
        print(f"[Vitals/{self.tag}] t={step:,}  ent_coef={ent_coef:.4f}  decay={pct}%")
        return True

    # (선택) 롤아웃마다 한 번 더 요약 출력하고 싶으면
    def _on_rollout_end(self) -> None:
        step = self.num_timesteps
        try:
            ent_coef = float(self.model.ent_coef)
        except Exception:
            ent_coef = float('nan')
        print(f"[Vitals/{self.tag}] rollout_end t={step:,} ent_coef={ent_coef:.4f}")


# ===== Train =====
def train_manager(split: str = "train", steps: int = 800_000, seed: int = 42, save_path: str | None = None):
    gb = GoalBridge()
    env = DummyVecEnv([lambda: ManagerEnv(split=split, gb=gb)])

    vec = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=5.0,
        clip_reward=10.0
    )

    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=512,
        batch_size=256,
        device="cpu",
        learning_rate=1e-4,
        gamma=0.95,
        gae_lambda=0.95,
        ent_coef=ENT_START,    # 콜백으로 0.03 → 0.003
        clip_range=0.2,
        vf_coef=0.8,           # 크리틱 비중 ↑
        clip_range_vf=None,    # 가치함수 클리핑 해제
        seed=seed,
        verbose=1
    )

    callbacks = CallbackList([
        EntropyDecay(),
        VitalsProbe(tag="manager", every=5000),
    ])
    model.learn(total_timesteps=steps, callback=callbacks)

    sp = save_path or os.path.join(MODEL_DIR, "manager_stage1.zip")
    model.save(sp)
    vec.save(os.path.join(MODEL_DIR, "manager_stage1_vecnorm.pkl"))
    return sp


def run_manager():
    print("[HRL] Training Manager…")
    mp = train_manager()
    print(f"[OK] Manager saved → {mp}")


if __name__ == "__main__":
    run_manager()
