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

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

from ai_binance.train.reinforce.common import (
    MODEL_DIR,
    build_worker_inputs, build_manager_inputs,
    Goal, GoalBridge, EntropyDecay,
    FEE_RATE, SLIP_BP,
)

# =============================================================================
# Design: Entry-only Worker (Optimal Stopping)
#  - Act: 0=Wait, 1=Enter   (방향은 매니저가 고정: +1 롱, -1 숏)
#  - Episode: 매니저 확신(conf) 임계 이상인 시점에서 "엔트리 창" 오픈 → Enter 시 종료
#             혹은 MAX_DELAY 지나면 강제결정(Enter)로 종료
#  - Reward: 엔트리 품질의 카운터팩추얼 장점(advantage)
#            R = PnL(enter_at τ; 슬립/수수료/펀딩 포함, horizon=EVAL_H) - PnL(enter_now)
#            → "지금 vs 조금 기다림"의 우열을 직접 학습
#  - MaskablePPO: Invalid 확률질량 제거 → NoOp 고착 방지
# =============================================================================

# ===== Window / Reward knobs =====
EVAL_H_STEPS      = 36          # horizon for entry quality eval (5m*36 ≈ 3h)
MAX_DELAY         = 12          # 최대 진입 지연 허용 (5m*12 ≈ 60m)
DECISION_CONF_THR = 0.60        # 매니저 확신 임계 (창 오픈 조건)

# 펀딩(8h 주기)을 5m 스텝으로 환산
FUNDING_STEP_FRAC = 5.0 / 480.0

# ===== Eval =====
EVAL_EVERY  = 50_000
EVAL_EPISODES = 200


class EntryEnv(gym.Env):
    """
    Entry-only Worker Env (Optimal Stopping)

    Act: 0=Wait, 1=Enter
    Obs: [X_5m..., manager_dir, manager_conf, delay_norm]
    Episode:
      - 매니저 dir!=0 & conf>=DECISION_CONF_THR인 최초 시점 i0를 창 시작으로 설정
      - 에이전트가 Enter를 결정하면 종료. 보상은 카운터팩추얼 advantage:
           R = PnL(i_enter → horizon) - PnL(i0 → horizon)
      - MAX_DELAY에 도달하면 Enter만 허용(강제 결정)
    Costs:
      - 슬리피지: 진입/청산 각각 SLIP_BP 적용(체결가)
      - 수수료: 양쪽 합 2*FEE_RATE
      - 펀딩: 구간 합산(진영과 부호에 따라 ±), 5m 환산
    """
    metadata = {"render_modes": []}

    def __init__(self, split: str, gb: GoalBridge,
                 fee_rate: float = FEE_RATE,
                 slip_bp: float = SLIP_BP,
                 randomize_start: bool = True):
        super().__init__()
        self.gb = gb

        data = build_worker_inputs(split)
        self.X        = data["X"]
        self.price    = data["price"].astype(float)
        self.funding  = data.get("funding_rate", pd.Series(0.0, index=self.X.index)).astype(float)

        self.idx = self.X.index
        self.N   = len(self.idx)

        self.fee     = float(fee_rate)
        self.slip_bp = float(slip_bp)

        # Episode state
        self.i0: int = 0             # window start index
        self.hj: int = 0             # horizon index (inclusive)
        self.dir: int = 0            # +1 / -1 (manager-fixed)
        self.conf: float = 0.0
        self.delay: int = 0          # elapsed steps since i0
        self.baseline_pnl: float = 0.0
        self.best_entry_i: int = 0   # 최적 진입점
        self.best_pnl: float = 0.0   # 최적 PnL
        self._cursor: int = 0        # where next window search begins
        self._randomize_start = bool(randomize_start)

        # spaces
        feat_dim = self.X.shape[1]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(feat_dim + 3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # 0 Wait, 1 Enter

    # ---------- Manager goal ----------
    def _tick_goal(self, i: int) -> Tuple[int, float]:
        self.gb.tick(self.idx[i])
        v = self.gb.vec()
        d = int(v[0]) if len(v) >= 1 else 0
        c = float(v[1]) if len(v) >= 2 else 0.0
        return d, c

    # ---------- Window open ----------
    def _open_window(self, start_i: int) -> Tuple[int, int]:
        """
        start_i 이후로 (dir!=0 & conf>=thr)인 최초 시점 i0 반환.
        horizon은 i0 + EVAL_H_STEPS (데이터 끝으로 클램프).
        """
        i = max(0, int(start_i))
        while i < self.N - 1:
            d, c = self._tick_goal(i)
            if d != 0 and c >= DECISION_CONF_THR:
                self.dir = int(np.sign(d))
                self.conf = float(c)
                i0 = i
                hj = min(self.N - 1, i0 + EVAL_H_STEPS)
                return i0, hj
            i += 1
        # 못 찾으면 끝
        self.dir = 0; self.conf = 0.0
        return self.N - 1, self.N - 1

    # ---------- 최적 진입점 찾기 ----------
    def _find_best_entry(self, start_i: int, end_i: int) -> Tuple[int, float]:
        """주어진 구간에서 최적 진입점과 PnL 반환"""
        best_i, best_pnl = start_i, -float('inf')
        
        for i in range(start_i, min(end_i + 1, self.hj)):
            pnl = self._eval_pnl_from(i, self.hj)
            if pnl > best_pnl:
                best_pnl, best_i = pnl, i
                
        return best_i, best_pnl

    # ---------- PnL evaluator ----------
    def _eval_pnl_from(self, enter_i: int, horizon_j: int) -> float:
        """
        enter_i에서 self.dir로 진입 → horizon_j에서 청산한다고 가정한
        실현PnL(체결가, 슬립/수수료/펀딩 포함).
        """
        if enter_i >= horizon_j:
            return -2.0 * self.fee  # 사실상 '기회 상실' 패널티 수준

        px_e = float(self.price.iloc[enter_i])
        px_h = float(self.price.iloc[horizon_j])

        # 체결가(슬리피지)
        if self.dir > 0:
            exec_in  = px_e * (1 + self.slip_bp * 1e-4)
            exec_out = px_h * (1 - self.slip_bp * 1e-4)
        else:
            exec_in  = px_e * (1 - self.slip_bp * 1e-4)
            exec_out = px_h * (1 + self.slip_bp * 1e-4)

        pnl = (exec_out - exec_in) / exec_in * self.dir

        # 펀딩(단순 합산)
        fund = 0.0
        for k in range(enter_i, horizon_j):
            fr = float(self.funding.iloc[k])
            same = (self.dir > 0 and fr > 0) or (self.dir < 0 and fr < 0)
            delta = abs(fr) * FUNDING_STEP_FRAC
            fund += (-delta if same else +delta)

        fees = 2.0 * self.fee
        return pnl - fees + fund

    # ---------- Observations ----------
    def _obs(self) -> np.ndarray:
        i = min(self.N - 1, self.i0 + self.delay)
        x = self.X.iloc[i].to_numpy(np.float32, copy=False)
        dn = float(self.delay / max(1, MAX_DELAY))  # 0..1
        return np.concatenate([x, np.array([float(self.dir), float(self.conf), dn], dtype=np.float32)], axis=0)

    # ---------- Mask for MaskablePPO ----------
    def valid_action_mask(self) -> np.ndarray:
        """
        - delay==MAX_DELAY 또는 horizon에 거의 근접 → Enter만 허용
        - 그 외 → Wait/Enter 둘 다 허용
        """
        mask = np.zeros(2, dtype=bool)
        i = self.i0 + self.delay
        force = (self.delay >= MAX_DELAY) or (i >= self.hj - 1)
        if force:
            mask[1] = True
        else:
            mask[:] = True
        return mask

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 데이터 탐색 시작점 갱신(랜덤 스타트)
        if self._randomize_start:
            # 0..N-1에서 적당히 랜덤 오프셋
            try:
                rng = self.np_random
            except Exception:
                rng = np.random.RandomState(seed if seed is not None else None)
            self._cursor = int(rng.integers(0, max(1, self.N - 1)))
        else:
            self._cursor = int(getattr(self, "_cursor", 0))

        self.i0, self.hj = self._open_window(self._cursor)
        self.delay = 0

        # 창을 못 열면 곧바로 done 유도 (step에서 처리)
        if self.i0 >= self.N - 1 or self.dir == 0:
            return self._obs(), {}

        # baseline(now) 계산
        self.baseline_pnl = self._eval_pnl_from(self.i0, self.hj)
        
        # 최적 진입점 찾기
        max_search_end = min(self.i0 + MAX_DELAY, self.hj - 1)
        self.best_entry_i, self.best_pnl = self._find_best_entry(self.i0, max_search_end)
        
        return self._obs(), {}

    def step(self, a: int):
        # 창 무효면 종료
        if self.i0 >= self.N - 1 or self.dir == 0:
            return self._obs(), 0.0, True, False, {}

        i_enter: Optional[int] = None
        i_now = self.i0 + self.delay

        # 강제결정 조건
        force = (self.delay >= MAX_DELAY) or (i_now >= self.hj - 1)

        if a == 1 or force:
            i_enter = int(i_now)

        if i_enter is not None:
            pnl = self._eval_pnl_from(i_enter, self.hj)
            adv = float(pnl - self.baseline_pnl)

            # [MODIFIED] 보상을 상대적 이득(adv)이 아닌 절대 수익(pnl)으로 변경.
            # 에이전트가 수익성 자체를 학습하도록 유도하여 adv=0 허점을 해결.
            reward = pnl

            info = {
                "entered_at": int(i_enter),
                "delay": int(self.delay),
                "pnl_abs": float(pnl),
                "baseline": float(self.baseline_pnl),
                "adv": float(adv),
                "dir": int(self.dir),
                "best_pnl": float(self.best_pnl),
                "optimality": float(pnl / self.best_pnl) if self.best_pnl > 0 else 0.0,
            }

            # 다음 에피소드 검색 시작점을 horizon 이후로 옮겨준다 (데이터 커버리지↑)
            self._cursor = min(self.N - 1, self.hj + 1)
            return self._obs(), reward, True, False, info

        # Wait - 최적점 대비 현재 품질 기준 보상
        current_pnl = self._eval_pnl_from(i_now, self.hj)

        # 최적점 대비 현재 성능 비율 (0~1)
        if self.best_pnl > 0:
            optimality_ratio = max(0.0, current_pnl / self.best_pnl)
        else:
            optimality_ratio = 0.0

        # 최적에 가까울수록 높은 보상, 멀수록 낮은 보상
        wait_reward = (optimality_ratio - 0.5) * 0.2  # -0.1 ~ +0.1 범위

        self.delay += 1
        # horizon 근접 시 delay clamp
        max_delay_by_h = max(0, min(MAX_DELAY, self.hj - self.i0 - 1))
        if self.delay > max_delay_by_h:
            self.delay = max_delay_by_h
        return self._obs(), wait_reward, False, False, {}


# ===== GoalBridges =====
class _HeuristicGB(GoalBridge):
    """간단 휴리스틱: 4h 방향을 Goal로 제공, conf는 고정 높게 두어 창이 잘 열리게 함."""
    def __init__(self, split):
        super().__init__()
        self.m = build_manager_inputs(split)
        self.idx = self.m["XH"].index
        self.k = 0
    def tick(self, ts_5m):
        th = pd.Timestamp(ts_5m).floor("1h")
        while self.k + 1 < len(self.idx) and self.idx[self.k + 1] <= th:
            self.k += 1
        reg = int(self.m["reg4h_sign"].iloc[self.k])  # -1/0/+1
        conf = 0.7
        self.set(Goal(reg, conf))

class _ModelGB(GoalBridge):
    """학습된 매니저로부터 dir/conf 주입."""
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
        act = self.model.predict(obs, deterministic=True)[0]
        arr = np.array(act).reshape(-1)
        dir_eff = [-1, 0, 1][int(arr[0])]
        conf = float(int(arr[1]) / 10.0) if len(arr) > 1 else 0.5
        self.set(Goal(dir_eff, conf))


# ===== Action Mask hook =====
def _mask_fn(env: EntryEnv) -> np.ndarray:
    return env.valid_action_mask()


# === REPLACE the whole _eval_entry() in worker.py ===
def _eval_entry(make_env_fn: Callable[[], gym.Env],
                model: MaskablePPO,
                episodes: int = EVAL_EPISODES,
                deterministic: bool = True) -> None:
    # make_env_fn() → ActionMasker(EntryEnv)  [단일 env]
    env = make_env_fn()
    obs, _ = env.reset()

    advs, delays, dirs = [], [], []
    ep = 0
    while ep < episodes:
        action = model.predict(obs, deterministic=deterministic)[0]
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            advs.append(float(reward))
            if isinstance(info, (list, tuple)):  # safety for VecEnv-style
                info = info[0]
            delays.append(int(info.get("delay", 0)))
            dirs.append(int(info.get("dir", 0)))
            obs, _ = env.reset()
            ep += 1

    arr = np.array(advs, dtype=np.float64)
    avg = float(arr.mean()) if len(arr) else 0.0
    med = float(np.median(arr)) if len(arr) else 0.0
    p25 = float(np.percentile(arr, 25)) if len(arr) else 0.0
    p75 = float(np.percentile(arr, 75)) if len(arr) else 0.0
    p95 = float(np.percentile(arr, 95)) if len(arr) else 0.0
    pos = float((arr > 0).mean()) if len(arr) else 0.0
    print(f"[EVAL/entry] mode={'det' if deterministic else 'sample'} eps={episodes} "
          f"avg_adv={avg:+.5f} med={med:+.5f} IQR[{p25:+.5f},{p75:+.5f}] p95={p95:+.5f} "
          f"| pos_rate={pos:.3f} | median_delay={np.median(delays) if delays else 0:.1f}")


# ===== Callbacks =====
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

class EvalEvery(BaseCallback):
    def __init__(self, make_env_fn: Callable[[], gym.Env], every: int = EVAL_EVERY, verbose: int = 0):
        super().__init__(verbose); self.every=int(every); self.make_env_fn=make_env_fn; self._last=0
    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t - self._last < self.every: return True
        self._last = t
        _eval_entry(self.make_env_fn, self.model, episodes=EVAL_EPISODES, deterministic=True)
        _eval_entry(self.make_env_fn, self.model, episodes=EVAL_EPISODES, deterministic=False)
        return True


# ===== Train =====
def _make_env(split: str, gb: GoalBridge, randomize_start: bool = True) -> gym.Env:
    base = EntryEnv(split=split, gb=gb, randomize_start=randomize_start)
    return ActionMasker(base, _mask_fn)

def train_worker_warmup(
    split: str = "train",
    steps: int = 400_000,
    seed: int = 42,
    save_path: Optional[str] = None
):
    """
    Stage1: 휴리스틱 Goal로 엔트리 워커 예열
    """
    gb = _HeuristicGB(split)

    def make_env_fn() -> gym.Env:
        return _make_env(split, gb, randomize_start=True)

    env = DummyVecEnv([make_env_fn])
    vec = VecNormalize(env, norm_obs=False, norm_reward=False, gamma=0.995)

    model = MaskablePPO(
        "MlpPolicy", vec,
        n_steps=2048, batch_size=1024, n_epochs=10, device="cpu",
        learning_rate=1e-4, gamma=0.995, gae_lambda=0.95,
        ent_coef=0.05, clip_range=0.2, vf_coef=1.5,
        seed=seed, verbose=1
    )

    callbacks = CallbackList([
        EntropyDecay(start=0.05, end=0.02, decay_steps=200_000),
        Vitals(tag="worker-entry", every=10_000),
        EvalEvery(lambda: _make_env("val", _HeuristicGB("val"), randomize_start=False), every=EVAL_EVERY),
    ])
    model.learn(total_timesteps=steps, callback=callbacks)

    sp = save_path or os.path.join(MODEL_DIR, "worker_entry_stage1.zip")
    model.save(sp)
    vec.save(os.path.join(MODEL_DIR, "worker_entry_vecnorm.pkl"))
    return sp


def train_worker_with_manager(
    manager_path: str,
    split: str = "train",
    steps: int = 300_000,
    seed: int = 42,
    save_path: Optional[str] = None
):
    """
    Stage2: 학습된 매니저 정책(dir/conf)로 엔트리 워커 파인튜닝
    """
    from stable_baselines3 import PPO  # 매니저 로드
    m_model = PPO.load(manager_path)
    gb = _ModelGB(split, m_model)

    def make_env_fn() -> gym.Env:
        return _make_env(split, gb, randomize_start=True)

    env = DummyVecEnv([make_env_fn])
    vec = VecNormalize(env, norm_obs=False, norm_reward=False, gamma=0.995)

    model = MaskablePPO(
        "MlpPolicy", vec,
        n_steps=2048, batch_size=1024, n_epochs=10, device="cpu",
        learning_rate=1e-4, gamma=0.995, gae_lambda=0.95,
        ent_coef=0.04, clip_range=0.2, vf_coef=1.5,
        seed=seed, verbose=1
    )

    callbacks = CallbackList([
        EntropyDecay(start=0.04, end=0.02, decay_steps=200_000),
        Vitals(tag="worker-entry", every=10_000),
        EvalEvery(lambda: _make_env("val", _ModelGB("val", m_model), randomize_start=False), every=EVAL_EVERY),
    ])
    model.learn(total_timesteps=steps, callback=callbacks)

    sp = save_path or os.path.join(MODEL_DIR, "worker_entry_joint.zip")
    model.save(sp)
    vec.save(os.path.join(MODEL_DIR, "worker_entry_vecnorm.pkl"))
    return sp


def run_worker_warmup():
    print("[HRL] Training Entry-Worker (optimal stopping)…")
    wp = train_worker_warmup()
    print(f"[OK] Worker saved → {wp}")


if __name__ == "__main__":
    run_worker_warmup()
