# ai_binance/train/reinforce/eval_manager.py
from __future__ import annotations
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ai_binance.train.reinforce.manager import ManagerEnv, GoalBridge
from ai_binance.train.reinforce.common import MODEL_DIR

MODEL_PATH = os.path.join(MODEL_DIR, "manager_stage1.zip")
VNORM_PATH = os.path.join(MODEL_DIR, "manager_stage1_vecnorm.pkl")

def evaluate(split="val", steps=20000, deterministic=True):
    gb = GoalBridge()
    base_env = DummyVecEnv([lambda: ManagerEnv(split=split, gb=gb)])
    vec = VecNormalize.load(VNORM_PATH, base_env)
    vec.training = False           # 통계 freeze
    vec.norm_reward = False        # 보상은 원값으로

    model = PPO.load(MODEL_PATH, env=vec, device="cpu")

    obs = vec.reset()
    env0 = base_env.envs[0]    # 원본 env 접근
    price = env0.price.to_numpy()

    prev_dir = 0
    flip_cnt = 0
    act_hist = { -1:0, 0:0, 1:0 }

    # 히트율 계산용 버퍼
    dir_list = []
    hit1, hit3, cnt = 0, 0, 0
    gate_violate, gate_cnt = 0, 0

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rew, done, info = vec.step(action)

        dir_eff = info[0]["dir"]
        reg4h   = info[0]["reg4h"]
        weak4h  = info[0]["weak4h"]

        # 행동 통계
        act_hist[dir_eff] += 1
        if dir_eff != prev_dir: flip_cnt += 1
        prev_dir = dir_eff

        # 게이트 위반(약레짐인데 비0 방향)
        if weak4h:
            gate_cnt += 1
            if dir_eff != 0: gate_violate += 1

        # 히트율(1스텝/3스텝) — env 시계열 사용
        t = env0.t - 1  # step 후 시점이 1 증가하므로 현재 행동은 t-1에서 발생
        if 0 <= t < len(price) - 4:
            r1 = (price[t+1] - price[t]) / price[t]
            r3 = (price[t+3] - price[t]) / price[t]
            if dir_eff * r1 > 0: hit1 += 1
            if dir_eff * r3 > 0: hit3 += 1
            cnt += 1

        if done[0]:
            obs = vec.reset()
            prev_dir = 0

    flip_rate = flip_cnt / max(1, (steps))
    act_ratio = {k: v / max(1, sum(act_hist.values())) for k, v in act_hist.items()}
    gate_vrate = gate_violate / max(1, gate_cnt)
    hit1_rate = hit1 / max(1, cnt)
    hit3_rate = hit3 / max(1, cnt)

    print("=== MANAGER EVAL ===")
    print(f"split={split}, steps={steps}, deterministic={deterministic}")
    print(f"action ratio {{-1,0,1}}: {act_ratio}")
    print(f"flip_rate: {flip_rate:.3f}")
    print(f"gate_violation_rate(weak4h & dir!=0): {gate_vrate:.3f}")
    print(f"hit@1h: {hit1_rate:.3f}, hit@3h: {hit3_rate:.3f}")

if __name__ == "__main__":
    evaluate(split="val", steps=20000, deterministic=True)
