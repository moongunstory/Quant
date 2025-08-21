# ai_binance/train/reinforce/worker.py
from __future__ import annotations

# ===== allow running as a script =====
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from ai_binance.train.reinforce.common import (
    MODEL_DIR,
    build_worker_inputs, build_manager_inputs,
    Goal, GoalBridge, EntropyDecay,
    # knobs
    FEE_RATE, SLIP_BP, TIMING_K, TIMING_K_COEF,
    TURN_PENALTY, FLIP_PENALTY, HOLDING_BONUS,
    GATE_WARMUP_STEPS, ALIGN_EPS, OPPORTUNITY_COST, GATE_SOFT_PENALTY_MULT
)

# ===== EVAL config =====
EVAL_EVERY  = 50_000
EVAL_STEPS  = 20_000


# ===============================
#        Helpers (trend)
# ===============================
def _trend_strength(dir_: int, conf: float, gate15: int, regweak: bool, regsign: int) -> float:
    """
    0..1의 트렌드 강도 스칼라.
    - conf: 매니저 확신도 (0..1)
    - regweak: 약레짐이면 감쇠
    - regsign: 4h 방향과 goal dir 일치 시 가산
    - gate15: 15m 게이트와 goal dir 일치 시 가산
    """
    if dir_ == 0:
        return 0.0
    base = 0.35 if bool(regweak) else 1.0
    agree_reg  = 1.0 if dir_ == int(regsign) else 0.7
    agree_gate = 1.0 if ((dir_ > 0 and gate15 > 0) or (dir_ < 0 and gate15 < 0)) else 0.7
    s = conf * base * (0.5 * agree_reg + 0.5 * agree_gate)
    return float(np.clip(s, 0.0, 1.0))


# ===============================
#            ENV
# ===============================
class WorkerEnv(gym.Env):
    """
    Obs: [5m/15m/1h/4h/btc1h features, goal_dir, goal_conf]
    Act: 0 Hold, 1 Long, 2 Short, 3 Flat
    Mask: dir=+1 → {Hold,Long,Flat}, dir=-1 → {Hold,Short,Flat}, dir=0 → {Hold,Flat}
    Gate: 15m 불일치 → \'소프트 패널티\'만 (차단 금지)
    Anti-collapse: ε-entry(확률 진입), 웜업 수수료/슬립 완화, 비진입 페널티
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str, gb: GoalBridge,
                 fee_rate=FEE_RATE, slip_bp=SLIP_BP):
        super().__init__()
        self.gb = gb
        data = build_worker_inputs(split)
        self.X = data["X"]
        self.price = data["price"]
        self.gate15 = data["gate15_sign"].astype(int)
        self.regweak = data["reg4h_weak"].astype(bool)
        self.regsign = data["reg4h_sign"].astype(int)

        self.idx = self.X.index
        self.t = 0
        self.pos = 0  # -1,0,1
        self.entry_price = 0.0
        self.fee_rate = float(fee_rate)
        self.slip_bp = float(slip_bp)

        # meta
        self.total_steps = 0
        self.gate_warmup_steps = int(GATE_WARMUP_STEPS)
        self.align_eps = float(ALIGN_EPS)
        self.opp_cost = float(OPPORTUNITY_COST)
        self.holding_bonus = float(HOLDING_BONUS)

        # --- anti-collapse knobs ---
        self.eps0 = 0.20          # 초기 강제 진입 확률(유효 시)
        self.eps_end = 0.02       # 말기에 남길 최소 확률
        self.eps_decay = 200_000  # 선형 감소 스텝
        self.warm_fee_frac = 0.30 # 웜업 동안 수수료/슬립 완화 비율
        self.warm_steps = 150_000
        self.inaction_tol = 12    # 게이트/매니저 OK인데 진입 안 하면 페널티 주기까지 허용 step
        self.inaction_cnt = 0

        # spaces
        self.observation_space = spaces.Box(
            low=-10, high=10,
            shape=(self.X.shape[1] + 2,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

    # ----- helpers -----
    def _mask(self, d: int) -> np.ndarray:
        if d > 0: return np.array([1, 1, 0, 1], dtype=np.int8)
        if d < 0: return np.array([1, 0, 1, 1], dtype=np.int8)
        return np.array([1, 0, 0, 1], dtype=np.int8)

    def _obs(self) -> np.ndarray:
        x = self.X.iloc[self.t].to_numpy(dtype=np.float32, copy=False)
        g = self.gb.vec()
        return np.concatenate([x, g], axis=0)

    # ----- gym api -----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.pos = 0
        self.entry_price = 0.0
        self.inaction_cnt = 0
        return self._obs(), {}

    def step(self, a: int):
        px = float(self.price.iloc[self.t])
        nxt_px = float(self.price.iloc[min(self.t + 1, len(self.price) - 1)])
        old_pos = self.pos

        # goal
        gvec = self.gb.vec()
        dir_, conf = int(gvec[0]), float(gvec[1])
        mask = self._mask(dir_)

        gate_active = (self.total_steps >= self.gate_warmup_steps)
        gsig  = int(self.gate15.iloc[self.t])
        rweak = bool(self.regweak.iloc[self.t])
        rsgn  = int(self.regsign.iloc[self.t])

        # trend strength ∈ [0,1]
        tmult = _trend_strength(dir_, conf, gsig, rweak, rsgn)

        # ----- ε-entry: 게이트 온 + 방향 있고 + 포지션 0이면 확률적으로 진입 유도 -----
        eps_frac = max(self.eps_end, self.eps0 * (1.0 - min(1.0, self.total_steps / float(self.eps_decay))))
        eps_frac *= (0.5 + 0.5 * tmult)
        if (dir_ != 0) and (self.pos == 0) and gate_active:
            if np.random.random() < eps_frac:
                a = 1 if dir_ > 0 else 2

        # 마스크 위반 시 강제 Hold + 소액 페널티
        r_mask = 0.0
        if mask[a] == 0:
            a_eff = 0
            r_mask -= 0.01
        else:
            a_eff = a

        target = {0: self.pos, 1: 1, 2: -1, 3: 0}[a_eff]
        changed = int(target != self.pos)
        flip = int(abs(target - self.pos) == 2)

        # 웜업 수수료/슬립 완화
        fee_rate_eff = self.fee_rate * (self.warm_fee_frac if self.total_steps < self.warm_steps else 1.0)
        slip_bp_eff  = self.slip_bp   * (self.warm_fee_frac if self.total_steps < self.warm_steps else 1.0)

        # 수익률 스케일 수수료 (px 곱하지 않음)
        fee = fee_rate_eff * abs(target - self.pos) if changed else 0.0
        signed = 0 if a_eff in (0, 3) else (1 if a_eff == 1 else -1)
        exec_px = px * (1 + signed * slip_bp_eff * 1e-4) if changed else px

        # ===== PnL & Holding 보상 (누적/실현) =====
        r_pnl_realized = 0.0
        r_pnl_unrealized = 0.0
        r_holding = 0.0

        if changed:
            if old_pos != 0:  # 포지션 종료 또는 플립 시 실현 손익
                r_pnl_realized = (exec_px - self.entry_price) / self.entry_price * old_pos
            if target != 0:  # 신규 포지션 진입
                self.entry_price = exec_px
            else:  # 포지션 청산
                self.entry_price = 0.0
        
        if self.pos != 0: # 포지션 보유 중
            # 1. 미실현 손익 (shaping)
            r_pnl_unrealized = (px - self.entry_price) / self.entry_price * self.pos
            # 2. 홀딩 보너스
            if self.pos == dir_ and dir_ != 0:
                r_holding = self.holding_bonus

        # 타이밍(2시간) → 트렌드 강도 가중
        k_idx = min(self.t + TIMING_K, len(self.price) - 1)
        fut_px = float(self.price.iloc[k_idx])
        dret = (fut_px - px) / px
        raw_timing = ((dret if a_eff == 1 else -dret) if a_eff in (1, 2) else 0.0)
        r_timing = TIMING_K_COEF * (0.5 + 0.5 * tmult) * raw_timing

        # 매니저 정렬 (conf 가중)
        r_align = self.align_eps * conf * dir_ * ((nxt_px - px) / px)
        
        # 15m 게이트: 불일치 소프트 패널티만
        r_gate = 0.0
        wants_entry = (a_eff in (1, 2)) and (old_pos == 0)
        if gate_active and wants_entry:
            mismatch = (dir_ > 0 and gsig <= 0) or (dir_ < 0 and gsig >= 0)
            if mismatch:
                r_gate -= GATE_SOFT_PENALTY_MULT * self.fee_rate

        # 비진입/기회비용 페널티
        r_inaction = 0.0
        if gate_active and (dir_ != 0) and (self.pos == 0) and (a_eff == 0):
            self.inaction_cnt += 1
            if self.inaction_cnt >= self.inaction_tol:
                r_inaction -= 0.5 * self.opp_cost
        else:
            self.inaction_cnt = 0
        
        r_opp = 0.0
        if gate_active and dir_ != 0 and a_eff == 0 and self.t > 0:
            gsig_prev = int(self.gate15.iloc[self.t - 1])
            if (dir_ > 0 and gsig_prev > 0) or (dir_ < 0 and gsig_prev < 0):
                r_opp -= self.opp_cost

        # 진입 보너스 → 트렌드 강도 가중
        r_entry = 0.0
        if wants_entry:
            if (a_eff == 1 and dir_ > 0) or (a_eff == 2 and dir_ < 0):
                r_entry += (0.002 + 0.5 * abs(dret)) * (0.5 + 0.5 * tmult)

        # 트랜잭션/스위치 비용
        r_fee  = -fee
        r_turn = -TURN_PENALTY * changed
        r_flip = -FLIP_PENALTY * flip

        # 합계
        r = (r_pnl_realized + r_pnl_unrealized * 0.1 + r_holding +
             r_fee + r_turn + r_flip + r_gate + r_timing + r_align +
             r_opp + r_entry + r_inaction + r_mask)

        # 상태 전이
        self.pos = target
        self.t += 1
        self.total_steps += 1
        terminated = (self.t >= len(self.X) - 2)

        info = {
            "mask": mask, "dir": dir_, "gate15": int(self.gate15.iloc[self.t - 1]),
            "trade": int(changed), "flip": int(flip),
            "a_eff": int(a_eff), "conf": float(conf), "trend": float(tmult),
            # reward breakdown
            "r_pnl_realized": r_pnl_realized, "r_pnl_unrealized": r_pnl_unrealized,
            "r_holding": r_holding, "r_fee": r_fee, "r_turn": r_turn, "r_flip": r_flip,
            "r_gate": r_gate, "r_timing": r_timing, "r_align": r_align,
            "r_opp": r_opp, "r_entry": r_entry, "r_inaction": r_inaction, "r_mask": r_mask,
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


# ===== Model GoalBridge (매니저 추론 주입) =====
class _ModelGB(GoalBridge):
    def __init__(self, split, manager_model: PPO):
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
        act = self.model.predict(obs, deterministic=True)[0]
        arr = np.array(act).reshape(-1)
        dir_eff = [-1, 0, 1][int(arr[0])]
        conf = float(int(arr[1]) / 10.0) if len(arr) > 1 else 0.5
        self.set(Goal(dir_eff, conf))


# ===============================
#             EVAL
# ===============================
def _eval_worker(make_env_fn: Callable[[], WorkerEnv],
                 model: PPO,
                 steps: int = EVAL_STEPS,
                 deterministic: bool = True) -> None:
    env = make_env_fn()
    # 게이트/ε-entry 활성화를 위해 total_steps 보정
    try:
        cur_steps = int(getattr(model, "num_timesteps", 0))
        env.total_steps = max(env.total_steps, cur_steps, int(getattr(env, "gate_warmup_steps", 0)) + 1)
    except Exception:
        env.total_steps = max(env.total_steps, int(getattr(env, "gate_warmup_steps", 0)) + 1)

    obs, _ = env.reset()
    # reset이 total_steps를 건드리지 않으므로, 다시 보정
    try:
        cur_steps = int(getattr(model, "num_timesteps", 0))
        env.total_steps = max(env.total_steps, cur_steps, int(getattr(env, "gate_warmup_steps", 0)) + 1)
    except Exception:
        pass

    steps_done = 0

    # metrics
    act_counts = { -1: 0, 0: 0, 1: 0 }     # 모델이 낸 방향 (a 기준)
    exec_counts = { -1: 0, 0: 0, 1: 0 }    # 실제 실행된 방향 (a_eff 기준)
    flip_cnt = 0
    gate_cnt = 0
    hit5_cnt = 0
    hit20_cnt = 0
    entry_cnt = 0
    R_sum = 0.0
    conf_sum = 0.0

    # reward breakdown sums
    rb_keys = [
        "r_pnl_realized", "r_pnl_unrealized", "r_holding", "r_fee", "r_turn", "r_flip",
        "r_gate", "r_timing", "r_align", "r_opp", "r_entry", "r_inaction", "r_mask"
    ]
    rb_sum: Dict[str, float] = {k: 0.0 for k in rb_keys}

    while steps_done < steps:
        a_raw = model.predict(obs, deterministic=deterministic)[0]
        a = int(np.array(a_raw).reshape(-1)[0])

        # 모델이 낸 방향 통계(참고용)
        pred_dir = 0
        if a == 1: pred_dir = 1
        elif a == 2: pred_dir = -1
        act_counts[pred_dir] += 1

        obs, r, done, trunc, info = env.step(a)
        R_sum += float(r)

        # 실제 실행된 액션으로 집계
        a_eff = int(info.get("a_eff", a))
        exec_dir = 1 if a_eff == 1 else (-1 if a_eff == 2 else 0)
        exec_counts[exec_dir] += 1

        # flip / gate ratio
        flip_cnt += int(info.get("flip", 0))
        gate_cnt += int(info.get("gate15", 0) > 0)

        # hit@1 step / @TIMING_K (실행된 방향 기준)
        t = env.t - 1
        if t >= 0 and exec_dir != 0:
            cur_px = float(env.price.iloc[t])
            nxt_px = float(env.price.iloc[min(t + 1, len(env.price) - 1)])
            k_idx = min(t + TIMING_K, len(env.price) - 1)
            fut_px = float(env.price.iloc[k_idx])
            ret1 = (nxt_px - cur_px) / cur_px
            retk = (fut_px - cur_px) / cur_px
            hit5_cnt  += int(np.sign(exec_dir * ret1) > 0)
            hit20_cnt += int(np.sign(exec_dir * retk) > 0)
            entry_cnt += 1

        # rb sum
        for k in rb_keys:
            rb_sum[k] += float(info.get(k, 0.0))

        conf_sum += float(info.get("conf", 0.0))

        steps_done += 1
        if done or trunc:
            obs, _ = env.reset()

    # ratios
    act_ratio  = {k: v / max(1, steps_done) for k, v in act_counts.items()}
    exec_ratio = {k: v / max(1, steps_done) for k, v in exec_counts.items()}
    flip_rate  = flip_cnt / max(1, steps_done)
    gate_rate  = gate_cnt / max(1, steps_done)
    hit5  = (hit5_cnt  / max(1, entry_cnt)) if entry_cnt > 0 else 0.0
    hit20 = (hit20_cnt / max(1, entry_cnt)) if entry_cnt > 0 else 0.0
    avgR  = R_sum / max(1, steps_done)
    conf_avg = conf_sum / max(1, steps_done)

    # policy entropy proxy
    probs = np.array(list(act_ratio.values()), dtype=np.float64) + 1e-12
    H = float(-np.sum(probs * np.log(probs)))

    rb_str = ", ".join([f"{k}={rb_sum[k]:.5g}" for k in rb_keys])
    mode = "det" if deterministic else "sample"
    print(
        f"[EVAL] t={env.total_steps:,} steps={steps_done:,} "
        f"hit5m={hit5:.3f} hit20m={hit20:.3f} flip={flip_rate:.3f} gate={gate_rate:.3f} "
        f"act={{-1: {act_ratio[-1]:.3f}, 0: {act_ratio[0]:.3f}, 1: {act_ratio[1]:.3f}}} "
        f"exec={{-1: {exec_ratio[-1]:.3f}, 0: {exec_ratio[0]:.3f}, 1: {exec_ratio[1]:.3f}}} "
        f"avgR={avgR:+.1g} mode={mode} | H={H:.4f} | conf~{conf_avg:.2f} | rb{{{rb_str}}}"
    )


# ===============================
#        CALLBACKS (Vitals/Eval)
# ===============================
class VitalsProbe(BaseCallback):
    def __init__(self, tag: str = "worker", every: int = 5000, verbose: int = 0):
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
        print(f"[Vitals/{self.tag}] t={step:,} ent_coef={ent_coef:.4f}")
        return True


class EvalEvery(BaseCallback):
    def __init__(self, make_env_fn: Callable[[], WorkerEnv], every: int = EVAL_EVERY, verbose: int = 0):
        super().__init__(verbose)
        self.every = int(every)
        self.make_env_fn = make_env_fn
        self._last = 0

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step - self._last < self.every:
            return True
        self._last = step
        # deterministic / sample 둘 다 출력
        _eval_worker(self.make_env_fn, self.model, steps=EVAL_STEPS, deterministic=True)
        _eval_worker(self.make_env_fn, self.model, steps=EVAL_STEPS, deterministic=False)
        return True


# ===============================
#            TRAIN
# ===============================
def train_worker_warmup(
    split: str = "train",
    steps: int = 1_000_000,
    seed: int = 42,
    save_path: str | None = None
):
    """
    Stage1: 휴리스틱 매니저 목표로 워커 예열
    """
    gb = _HeuristicGB(split)

    class _Harness(WorkerEnv):
        def step(self, a: int):
            gb.tick(self.idx[self.t])
            return super().step(a)

    def make_env_fn() -> WorkerEnv:
        return _Harness(split=split, gb=gb)

    # vec env
    env = DummyVecEnv([lambda: make_env_fn()])
    vec = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=5.0,
        gamma=0.99
    )

    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=4096,
        batch_size=1024,
        n_epochs=20,
        device="cpu",
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.10,           # EntropyDecay로 0.02까지 줄임
        clip_range=0.2,
        vf_coef=2.0,             # critic 비중 ↑
        clip_range_vf=None,      # value 클리핑 해제
        seed=seed,
        verbose=1
    )

    callbacks = CallbackList([
        EntropyDecay(start=0.10, end=0.02, decay_steps=200_000),
        VitalsProbe(tag="worker", every=10_000),
        EvalEvery(make_env_fn, every=EVAL_EVERY),
    ])
    model.learn(total_timesteps=steps, callback=callbacks)

    sp = save_path or os.path.join(MODEL_DIR, "worker_stage1.zip")
    model.save(sp)
    vec.save(os.path.join(MODEL_DIR, "worker_stage1_vecnorm.pkl"))
    return sp


def train_worker_with_manager(
    manager_path: str,
    split: str = "train",
    steps: int = 300_000,
    seed: int = 42,
    save_path: str | None = None
):
    """
    Stage2: 학습된 매니저 정책을 Goal로 사용하여 워커 파인튜닝
    """
    m_model = PPO.load(manager_path)
    gb = _ModelGB(split, m_model)

    class _Harness(WorkerEnv):
        def step(self, a: int):
            gb.tick(self.idx[self.t])
            return super().step(a)

    def make_env_fn() -> WorkerEnv:
        return _Harness(split=split, gb=gb)

    env = DummyVecEnv([lambda: make_env_fn()])
    vec = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=5.0,
        gamma=0.99
    )

    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=4096,
        batch_size=1024,
        n_epochs=20,
        device="cpu",
        learning_rate=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.10,
        clip_range=0.2,
        vf_coef=2.0,
        clip_range_vf=None,
        seed=seed,
        verbose=1
    )

    callbacks = CallbackList([
        EntropyDecay(start=0.10, end=0.02, decay_steps=200_000),
        VitalsProbe(tag="worker", every=10_000),
        EvalEvery(make_env_fn, every=EVAL_EVERY),
    ])
    model.learn(total_timesteps=steps, callback=callbacks)

    sp = save_path or os.path.join(MODEL_DIR, "worker_joint.zip")
    model.save(sp)
    vec.save(os.path.join(MODEL_DIR, "worker_joint_vecnorm.pkl"))
    return sp


def run_worker_warmup():
    print("[HRL] Training Worker (warmup)…")
    wp = train_worker_warmup()
    print(f"[OK] Worker saved → {wp}")


if __name__ == "__main__":
    run_worker_warmup()