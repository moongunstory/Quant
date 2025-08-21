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
    MODEL_DIR, build_manager_inputs, Goal, GoalBridge,
    M_W1, M_W3, FLIP_PENALTY, TURN_PENALTY
)

# ===== Knobs =====
ENT_START = 0.015
ENT_END   = 0.002
ENT_DECAY_STEPS = 30_000

MAX_EPISODE_STEPS = 4_096
LOOKAHEAD_H = 3
SEQ_WINDOW  = 8

REWARD_SCALE = 6.0

# === Confidence & penalties ===
BP_REF           = 0.0015
CONF_BRIER       = 0.00   # OFF: 평균 음수 벌점 제거
CONF_SMOOTH      = 0.00   # OFF: 평균 음수 벌점 제거
CONF_HIT_BONUS   = 0.03   # 제로센터 히트 보상(+/-): conf·신호크기 가중
CONF_CAL         = 0.00   # Stage1 off
NOISE_ACT_PEN    = 0.00   # Stage1 off
WEAK_PENALTY     = 0.00   # Stage1 off

# === 페널티 가중치 램프(0 → 0.5 → 1.0) ===
FLIP_W_STAGE1 = 0.0
TURN_W_STAGE1 = 0.0
FLIP_W_STAGE2 = 0.5
TURN_W_STAGE2 = 0.5
FLIP_W_STAGE3 = 1.0
TURN_W_STAGE3 = 1.0

CURRICULUM_STAGE2_STEP = 200_000  # 램프 시작
CURRICULUM_STAGE3_STEP = 350_000  # 다음 램프 시작
RAMP2 = 50_000                    # Stage2 램프 길이
RAMP3 = 50_000                    # Stage3 램프 길이

# === LR schedule ===
LR_START = 2e-4
LR_END   = 8e-5


class ManagerEnv(gym.Env):
    """
    Obs: [X_1h + 4h(ffill) scaled] 최근 SEQ_WINDOW개 스택(flatten)
    Act: MultiDiscrete([3,11]) → dir∈{-1,0,1}, conf∈{0..10}/10
    Rule: 약레짐(weak)에서는 거래 금지 권장(초기 위반 벌점 off)
    Reward: 방향(log r1,r3) + (행동시) 히트 보너스(+/-)
            - (가중치적용) flip
            - (약레짐 外, 가중치적용) turn_mismatch
            - (옵션) 저신호/약레짐 위반
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
        self.prev_conf = 0.0
        self.steps_in_ep = 0

        # runtime-configurable (커리큘럼 대상)
        self.conf_brier = float(CONF_BRIER)
        self.conf_smooth = float(CONF_SMOOTH)
        self.conf_hit_bonus = float(CONF_HIT_BONUS)
        self.conf_cal = float(CONF_CAL)
        self.noise_act_pen = float(NOISE_ACT_PEN)
        self.weak_penalty = float(WEAK_PENALTY)
        self.flip_w = float(FLIP_W_STAGE1)
        self.turn_w = float(TURN_W_STAGE1)

        # ===== sequence buffer =====
        self.W = int(SEQ_WINDOW)
        feat_dim = self.XH.shape[1]
        self._buf = np.zeros((self.W, feat_dim), dtype=np.float32)

        self.observation_space = spaces.Box(low=-10, high=10, shape=(feat_dim * self.W,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 11])

        self.max_ep_steps = MAX_EPISODE_STEPS
        self.end_guard = LOOKAHEAD_H
        raw = len(self.XH) - self.max_ep_steps - self.end_guard
        self.max_start = max(self.W + 1, raw)

    # === runtime penalty update (for curriculum) ===
    def set_penalties(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k) and (v is not None):
                setattr(self, k, float(v))

    # === sequence buffer helpers ===
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
        self.prev_conf = 0.0
        self.gb.set(Goal(0, 0.0))
        self._fill_buf()
        return self._obs(), {}

    def step(self, a):
        dir_raw = [-1, 0, 1][int(a[0])]
        conf = float(int(a[1]) / 10.0)

        weak = bool(self.regweak.iloc[self.t])
        tried_nonflat = (dir_raw != 0)
        dir_eff = 0 if weak else dir_raw

        # 약레짐 위반 벌점(커리큘럼으로 점진 On)
        pen = self.weak_penalty if (weak and tried_nonflat) else 0.0

        self.gb.set(Goal(dir_eff, conf))

        # --- log-returns (tail 완화) ---
        cur  = float(self.price.iloc[self.t])
        nxt1 = float(self.price.iloc[min(self.t + 1, len(self.price) - 1)])
        nxt3 = float(self.price.iloc[min(self.t + 3, len(self.price) - 1)])
        r1 = np.log(max(nxt1, 1e-12) / max(cur, 1e-12))
        r3 = np.log(max(nxt3, 1e-12) / max(cur, 1e-12))
        r_w = M_W1 * r1 + M_W3 * r3

        act_on = float(dir_eff != 0)

        # 방향/전이(가중치로 스케일)
        R_dir  = dir_eff * r_w
        R_flip = -self.flip_w * FLIP_PENALTY * int((not weak) and (dir_eff != self.prev_dir))
        R_mis  = -self.turn_w * TURN_PENALTY * int((not weak) and (dir_eff != 0) and (np.sign(dir_eff) != int(self.regsign.iloc[self.t])))

        # ===== 제로센터 히트 보너스 (+/-) =====
        is_correct = int(dir_eff != 0 and np.sign(dir_eff) == np.sign(r_w) and np.sign(r_w) != 0)
        sign_bonus = 1.0 if is_correct else -1.0
        sig = float(np.clip(abs(r_w) / BP_REF, 0.0, 1.0))  # 신호 크기 가중
        R_hit = self.conf_hit_bonus * act_on * sign_bonus * conf * sig

        # (선택) 캘리브레이션 보상은 기본 비활성
        if self.conf_cal > 0.0:
            target = float(np.clip(abs(r_w) / BP_REF, 0.0, 1.0))
            high_signal = float(abs(r_w) >= BP_REF)
            R_cal = -self.conf_cal * act_on * high_signal * ((conf - target) ** 2)
        else:
            R_cal = 0.0

        # Brier/스무딩/저신호 벌점은 기본 Off
        R_amb = -self.noise_act_pen * act_on * float(abs(r_w) < BP_REF) if self.noise_act_pen > 0.0 else 0.0

        R = (R_dir + R_flip + R_mis + pen + R_hit + R_cal + R_amb) * REWARD_SCALE

        self.prev_dir = dir_eff
        self.prev_conf = conf
        self.t += 1
        self.steps_in_ep += 1
        self._fill_buf()

        time_over = (self.t >= len(self.XH) - self.end_guard)
        horizon_over = (self.steps_in_ep >= MAX_EPISODE_STEPS)
        terminated = bool(time_over)
        truncated = bool(not terminated and horizon_over)

        info = {
            "weak4h": weak,
            "reg4h": int(self.regsign.iloc[self.t - 1]) if self.t > 0 else 0,
            "dir": dir_eff,
            "r1": r1, "r3": r3, "Rw": r_w,
            "R_dir": R_dir, "R_hit": R_hit, "R_cal": R_cal,
            "weak_pen": pen, "flip_w": self.flip_w, "turn_w": self.turn_w
        }
        return self._obs(), float(R), terminated, truncated, info


# ===== Entropy decay / pulse / vitals / curriculum =====
class EntropyDecay(BaseCallback):
    def __init__(self, start=ENT_START, end=ENT_END, decay_steps=ENT_DECAY_STEPS, verbose=0):
        super().__init__(verbose)
        self.start, self.end, self.decay = float(start), float(end), int(decay_steps)
        self.min_floor = None  # 외부 펄스가 설정
    def _on_training_start(self):
        self.model.ent_coef = self.start
    def _on_step(self):
        step = self.num_timesteps
        frac = min(1.0, step / self.decay)
        coef = float(self.start + (self.end - self.start) * frac)
        if self.min_floor is not None:
            coef = max(coef, float(self.min_floor))
        self.model.ent_coef = coef
        return True

class EntropyPulse(BaseCallback):
    """Stage2 직후 탐색 재주입"""
    def __init__(self, decay_cb: EntropyDecay, trigger_step: int, hold_steps: int = 30_000, floor: float = 0.008, verbose: int = 0):
        super().__init__(verbose)
        self.decay_cb = decay_cb
        self.trigger_step = int(trigger_step)
        self.hold_steps = int(hold_steps)
        self.floor = float(floor)
        self.active = False
        self.end_step = None
    def _on_step(self) -> bool:
        t = self.num_timesteps
        if (not self.active) and t >= self.trigger_step:
            self.decay_cb.min_floor = self.floor
            self.active = True
            self.end_step = t + self.hold_steps
            print(f"[EntropyPulse] floor={self.floor} activated at t={t:,} until {self.end_step:,}")
        if self.active and t >= self.end_step:
            self.decay_cb.min_floor = None
            self.active = False
            print(f"[EntropyPulse] floor cleared at t={t:,}")
        return True

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
            ent = float(self.model.ent_coef)
        except Exception:
            ent = float('nan')
        print(f"[Vitals/{self.tag}] t={step:,} ent_coef={ent:.4f}")
        return True
    def _on_rollout_end(self) -> None:
        step = self.num_timesteps
        try:
            ent = float(self.model.ent_coef)
        except Exception:
            ent = float('nan')
        print(f"[Vitals/{self.tag}] rollout_end t={step:,} ent_coef={ent:.4f}")

class PenaltyScheduler(BaseCallback):
    """Stage1 → Stage2(0→0.5, 50k 램프) → Stage3(0.5→1.0, 50k 램프)"""
    def __init__(self, s2:int=CURRICULUM_STAGE2_STEP, s3:int=CURRICULUM_STAGE3_STEP, r2:int=RAMP2, r3:int=RAMP3, verbose:int=0):
        super().__init__(verbose)
        self.s2, self.s3, self.r2, self.r3 = int(s2), int(s3), int(r2), int(r3)
        self._announced2 = False
        self._announced3 = False
    def _on_step(self) -> bool:
        t = self.num_timesteps
        try:
            vecenv = self.model.get_env()
            envs = [e.env for e in vecenv.envs]  # unwrap Monitor
        except Exception:
            return True

        # 기본값(Stage1)
        flip_w = 0.0
        turn_w = 0.0
        weak_pen = 0.0
        noise_pen = 0.0

        # Stage2 램프 (0 → 0.5)
        if t >= self.s2:
            if not self._announced2:
                print(f"[Curriculum] Stage2 ramp start @ t={t:,}")
                self._announced2 = True
            alpha = np.clip((t - self.s2) / max(1, self.r2), 0.0, 1.0)
            flip_w = FLIP_W_STAGE2 * alpha
            turn_w = TURN_W_STAGE2 * alpha
            weak_pen = -0.005 * alpha
            noise_pen = 0.003 * alpha

        # Stage3 램프 (0.5 → 1.0)
        if t >= self.s3:
            if not self._announced3:
                print(f"[Curriculum] Stage3 ramp start @ t={t:,}")
                self._announced3 = True
            beta = np.clip((t - self.s3) / max(1, self.r3), 0.0, 1.0)
            flip_w = FLIP_W_STAGE2 + (FLIP_W_STAGE3 - FLIP_W_STAGE2) * beta
            turn_w = TURN_W_STAGE2 + (TURN_W_STAGE3 - TURN_W_STAGE2) * beta
            # weak/noise는 Stage2 수준 유지

        for env in envs:
            env.set_penalties(
                weak_penalty=weak_pen,
                noise_act_pen=noise_pen,
                conf_cal=0.0,            # 필요 시 추후 0.02~0.04
                flip_w=flip_w,
                turn_w=turn_w
            )
        return True

# ===== Periodic deterministic eval on fixed val split =====
class PeriodicEval(BaseCallback):
    """
    매 N스텝마다 val 데이터로 고정 평가(deterministic=True).
    콘솔 출력 예:
    [EVAL] t=200,000 steps=20,000 hit1h=0.612 hit3h=0.628 flip=0.342 gate=0.000 act={-1:0.33,0:0.34,1:0.33} avgR=+85.1
    """
    def __init__(self, make_env_fn, every:int=50_000, n_eval_steps:int=20_000, verbose:int=0):
        super().__init__(verbose)
        self.make_env_fn = make_env_fn
        self.every = int(every)
        self.n_eval_steps = int(n_eval_steps)
        self._last = 0

    def _clone_vecnorm(self, train_vec):
        # eval env + VecNormalize(학습 통계 복제, 보상정규화 off 유지)
        eval_env = DummyVecEnv([self.make_env_fn])
        eval_vec = VecNormalize(
            eval_env, norm_obs=True, norm_reward=False,
            clip_obs=5.0, clip_reward=10.0
        )
        try:
            eval_vec.obs_rms = train_vec.obs_rms  # 동일 스케일
        except Exception:
            pass
        eval_vec.training = False
        eval_vec._update_running_mean = False
        return eval_vec

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t - self._last < self.every:
            return True
        self._last = t

        # --- build eval vecenv with same normalization ---
        train_vec = self.model.get_env()
        eval_vec = self._clone_vecnorm(train_vec)

        # rollout deterministic
        obs = eval_vec.reset()
        n = self.n_eval_steps
        prev_dir = 0
        flip_cnt = 0
        act_cnt = {-1: 0, 0: 0, 1: 0}
        gate_viol = 0
        hits1 = 0; trials1 = 0
        hits3 = 0; trials3 = 0
        Rsum = 0.0

        for _ in range(n):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_vec.step(action)
            info = info[0] if isinstance(info, (list, tuple)) else info

            dir_eff = int(info.get("dir", 0))
            act_cnt[dir_eff] = act_cnt.get(dir_eff, 0) + 1
            if dir_eff != prev_dir and dir_eff != 0:
                flip_cnt += 1
            prev_dir = dir_eff

            if info.get("weak4h", False) and dir_eff != 0:
                gate_viol += 1

            # hit metrics
            r1 = float(info.get("r1", 0.0))
            r3 = float(info.get("r3", 0.0))
            if dir_eff != 0 and np.sign(r1) != 0:
                trials1 += 1
                if np.sign(dir_eff) == np.sign(r1): hits1 += 1
            if dir_eff != 0 and np.sign(r3) != 0:
                trials3 += 1
                if np.sign(dir_eff) == np.sign(r3): hits3 += 1

            Rsum += float(reward[0])  # VecEnv shape=(1,)

            # 에피소드 끝나도 계속 진행 (고정 길이 평가)
            if done[0]:
                obs = eval_vec.reset()

        # aggregate
        ar = {k: round(v / n, 3) for k, v in act_cnt.items()}
        flip_rate = round(flip_cnt / max(1, n - 1), 3)
        gate_rate = round(gate_viol / n, 3)
        hit1 = round(hits1 / max(1, trials1), 3)
        hit3 = round(hits3 / max(1, trials3), 3)
        avgR = Rsum / n

        sign = "+" if avgR >= 0 else "-"
        print(f"[EVAL] t={t:,} steps={n:,} hit1h={hit1:.3f} hit3h={hit3:.3f} "
              f"flip={flip_rate:.3f} gate={gate_rate:.3f} act={ar} avgR={sign}{abs(avgR):.1f}")
        return True


def lr_schedule(progress_remaining: float) -> float:
    return float(LR_END + (LR_START - LR_END) * progress_remaining)


# ===== Train =====
def train_manager(split: str = "train", steps: int = 800_000, seed: int = 42, save_path: str | None = None):
    gb = GoalBridge()
    def make_env():
        return Monitor(ManagerEnv(split=split, gb=gb))
    env = DummyVecEnv([make_env])

    # reward 정규화 Off → 크리틱 타깃 고정
    vec = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=5.0,
        clip_reward=10.0
    )

    # Critic capacity ↑
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[dict(pi=[128, 128], vf=[256, 256])]
    )

    model = PPO(
        "MlpPolicy", vec,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        device="cpu",
        learning_rate=lr_schedule,
        gamma=0.95,
        gae_lambda=0.95,
        ent_coef=ENT_START,
        clip_range=0.15,         # 정책 스텝 보수화
        vf_coef=1.2,             # 가치망 비중
        clip_range_vf=0.4,       # 값 클립 재도입(안정화)
        max_grad_norm=0.4,       # 그라드 폭 제한 강화
        target_kl=0.01,          # 과대 업데이트 자동 억제
        seed=seed,
        verbose=1,
        policy_kwargs=policy_kwargs
    )

    try:
        vec.seed(seed)
    except Exception:
        pass

    ent_decay = EntropyDecay()
    callbacks = CallbackList([
        ent_decay,
        PenaltyScheduler(),                               # 선형 램프
        EntropyPulse(ent_decay, CURRICULUM_STAGE2_STEP, 30_000, 0.008),  # Stage2 직후 탐색 펄스
        VitalsProbe(tag="manager", every=5000),
        PeriodicEval(lambda: ManagerEnv(split="val", gb=GoalBridge()), every=50_000, n_eval_steps=20_000),
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
