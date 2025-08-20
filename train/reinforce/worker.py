from __future__ import annotations
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .common import (
    MODEL_DIR, build_worker_inputs, build_manager_inputs, Goal, GoalBridge, EntropyDecay,
    FEE_RATE, SLIP_BP, TIMING_K, TIMING_K_COEF, TURN_PENALTY, FLIP_PENALTY,
    GATE_WARMUP_STEPS, ALIGN_EPS, OPPORTUNITY_COST, GATE_SOFT_PENALTY_MULT
)

class WorkerEnv(gym.Env):
    """
    Obs: [5m/15m/1h/4h/btc1h features, goal_dir, goal_conf]
    Act: 0 Hold, 1 Long, 2 Short, 3 Flat
    Mask: dir=+1 → {Hold,Long,Flat}, dir=-1 → {Hold,Short,Flat}, dir=0 → {Hold,Flat}
    Gate: 15m 불일치 → '소프트 패널티'만 (차단 금지)
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str, gb: GoalBridge,
                 fee_rate=FEE_RATE, slip_bp=SLIP_BP):
        super().__init__()
        self.gb = gb
        data = build_worker_inputs(split)
        self.X = data["X"]; self.price = data["price"]
        self.gate15 = data["gate15_sign"].astype(int)
        self.regweak = data["reg4h_weak"].astype(bool)
        self.regsign = data["reg4h_sign"].astype(int)

        self.idx = self.X.index
        self.t = 0
        self.pos = 0  # -1,0,1
        self.fee_rate = fee_rate
        self.slip_bp = slip_bp

        self.total_steps = 0
        self.gate_warmup_steps = GATE_WARMUP_STEPS
        self.align_eps = ALIGN_EPS
        self.opp_cost = OPPORTUNITY_COST

        self.observation_space = spaces.Box(low=-10, high=10,
                                            shape=(self.X.shape[1] + 2,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def _mask(self, d: int) -> np.ndarray:
        if d > 0: return np.array([1,1,0,1], dtype=np.int8)
        if d < 0: return np.array([1,0,1,1], dtype=np.int8)
        return np.array([1,0,0,1], dtype=np.int8)

    def _obs(self) -> np.ndarray:
        x = self.X.iloc[self.t].to_numpy(dtype=np.float32, copy=False)
        g = self.gb.vec()
        return np.concatenate([x, g], axis=0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0; self.pos = 0
        return self._obs(), {}

    def step(self, a: int):
        px = float(self.price.iloc[self.t])

        dir_, conf = int(self.gb.vec()[0]), float(self.gb.vec()[1])
        mask = self._mask(dir_)
        pen = 0.0
        if mask[a] == 0:
            a = 0; pen -= 0.01

        gate_active = (self.total_steps >= self.gate_warmup_steps)

        # 15m 게이트: 불일치 시 소프트 패널티만
        wants_entry = (a in (1, 2)) and (self.pos == 0)
        if gate_active and wants_entry:
            gsig = int(self.gate15.iloc[self.t])
            mismatch = (dir_ > 0 and gsig <= 0) or (dir_ < 0 and gsig >= 0)
            if mismatch:
                pen -= GATE_SOFT_PENALTY_MULT * self.fee_rate

        target = {0: self.pos, 1: 1, 2: -1, 3: 0}[a]
        changed = int(target != self.pos)
        flip = int(abs(target - self.pos) == 2)

        fee = self.fee_rate * abs(target - self.pos) * px if changed else 0.0
        signed = 0 if a in (0, 3) else (1 if a == 1 else -1)
        exec_px = px * (1 + signed * self.slip_bp * 1e-4) if changed else px

        nxt_px = float(self.price.iloc[min(self.t + 1, len(self.price) - 1)])
        pnl = (nxt_px - exec_px) / exec_px * self.pos

        # 타이밍 품질
        k_idx = min(self.t + TIMING_K, len(self.price) - 1)
        fut_px = float(self.price.iloc[k_idx])
        dret = (fut_px - px) / px
        timing = (dret if a == 1 else -dret) if a in (1, 2) else 0.0

        # reward
        r = pnl - fee + TIMING_K_COEF * timing \
            - TURN_PENALTY * changed - FLIP_PENALTY * flip + pen

        # 소프트 방향 정렬
        align = self.align_eps * dir_ * ((nxt_px - px) / px)
        r += align

        # 기회비용
        if gate_active and dir_ != 0 and a == 0 and self.t > 0:
            gsig_prev = int(self.gate15.iloc[self.t - 1])
            if (dir_ > 0 and gsig_prev > 0) or (dir_ < 0 and gsig_prev < 0):
                r -= self.opp_cost

        # entry bonus
        if a in (1, 2) and self.pos == 0:
            if (a == 1 and dir_ > 0) or (a == 2 and dir_ < 0):
                r += 0.002 + 0.5 * abs(dret)

        # transition
        self.pos = target
        self.t += 1
        self.total_steps += 1
        terminated = (self.t >= len(self.X) - 2)
        info = {
            "fee": fee, "mask": mask, "dir": dir_,
            "gate15": int(self.gate15.iloc[self.t - 1]),
            "trade": int(changed), "flip": int(flip)
        }
        return self._obs(), float(r), terminated, False, info

# ===== Heuristic GoalBridge (워커 예열) =====
class _HeuristicGB(GoalBridge):
    def __init__(self, split):
        super().__init__()
        self.m = build_manager_inputs(split)
        self.idx = self.m["XH"].index
        self.k = 0
    def tick(self, ts_5m):
        th = pd.Timestamp(ts_5m).floor("1h")
        while self.k + 1 < len(self.idx) and self.idx[self.k + 1] <= th:
            self.k += 1
        reg = int(self.m["reg4h_sign"].iloc[self.k])
        self.set(Goal(reg, 0.5))

# ===== Model GoalBridge (교대 학습 시 매니저 추론 주입) =====
class _ModelGB(GoalBridge):
    def __init__(self, split, manager_model):
        super().__init__()
        self.m = build_manager_inputs(split)
        self.idx = self.m["XH"].index
        self.k = 0
        self.model = manager_model
    def tick(self, ts_5m):
        th = pd.Timestamp(ts_5m).floor("1h")
        while self.k + 1 < len(self.idx) and self.idx[self.k + 1] <= th:
            self.k += 1
        obs = self.m["XH"].iloc[self.k].to_numpy(np.float32, copy=False)
        act = self.model.predict(obs, deterministic=False)[0]  # (dir_bin, conf_bin)
        dir_eff = [-1,0,1][int(act[0])]
        conf = float(int(act[1]) / 10.0)
        self.set(Goal(dir_eff, conf))

# ===== Train (워커) =====
def train_worker_warmup(split: str = "train", steps: int = 1_000_000, seed: int = 42, save_path: str | None = None):
    gb = _HeuristicGB(split)
    class _Harness(WorkerEnv):
        def step(self, a: int):
            gb.tick(self.idx[self.t])
            return super().step(a)

    env = DummyVecEnv([lambda: _Harness(split=split, gb=gb)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=4096, batch_size=1024, device="cpu",
        learning_rate=3e-4, gamma=0.99,
        ent_coef=0.10, clip_range=0.2, gae_lambda=0.95,
        vf_coef=1.2, clip_range_vf=None,
        seed=seed, verbose=1
    )
    model.learn(total_timesteps=steps, callback=EntropyDecay())

    sp = save_path or os.path.join(MODEL_DIR, "worker_stage1.zip")
    model.save(sp)
    env.save(os.path.join(MODEL_DIR, "worker_stage1_vecnorm.pkl"))
    return sp

def train_worker_with_manager(manager_path: str, split: str = "train", steps: int = 200_000,
                              seed: int = 42, save_path: str | None = None):
    m_model = PPO.load(manager_path)
    gb = _ModelGB(split, m_model)

    class _Harness(WorkerEnv):
        def step(self, a: int):
            gb.tick(self.idx[self.t])
            return super().step(a)

    env = DummyVecEnv([lambda: _Harness(split=split, gb=gb)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", env,
        n_steps=4096, batch_size=1024, device="cpu",
        learning_rate=3e-4, gamma=0.99,
        ent_coef=0.10, clip_range=0.2, gae_lambda=0.95,
        vf_coef=1.2, clip_range_vf=None,
        seed=seed, verbose=1
    )
    model.learn(total_timesteps=steps, callback=EntropyDecay())

    sp = save_path or os.path.join(MODEL_DIR, "worker_joint.zip")
    model.save(sp)
    env.save(os.path.join(MODEL_DIR, "worker_joint_vecnorm.pkl"))
    return sp

def run_worker_warmup():
    print("[HRL] Training Worker (warmup)…")
    wp = train_worker_warmup()
    print(f"[OK] Worker saved → {wp}")
