# ai_binance/live/online_learner.py
"""
Online Learner for PPO (ETHUSDT Futures, 5m)

목표
- 최근 구간으로 짧은 미세학습(fine-tune)
- 검증(VAL) 점수 비교: 개선되면 채택/저장, 악화면 자동 롤백
- (신규) 실전 전이(rollout)로 즉시 온라인 업데이트 + KL 가드/롤백
- 블로킹 없이 외부(run.py)에서 스레드로 돌리기 쉬운 구조
- CLI 없음. 상수로 조절.

입출력
- 입력: 정규화된 피처 DataFrame X (fe.py와 동일), Close Series (동일 인덱스)
- (옵션) FundingRate Series (동일 인덱스) — 5분 분배 차감
- 출력: 개선 시 모델 저장(기본 best_model_live.zip), 로그는 print로 최소화

주의
- PPO는 on-policy라 버퍼 재사용보다 "최근 구간 미니-환경"으로 재학습 + (신규) 실전 롤아웃으로 소규모 업데이트 조합이 안정.
- 실거래 중엔 별도 스레드로 호출 권장(거래 루프와 분리).
"""

from __future__ import annotations

import os
import math
import json
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer
import torch as th

# -----------------------
# 경로 & 상수 (수정 가능)
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/ai_binance
MODEL_DIR = os.path.join(BASE_DIR, "data", "model")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")

# 거래 비용/관찰창 (학습/백테스트와 정합)
WINDOW = 48                    # 4h
COMMISSION_SIDE = 0.0005       # 0.05% per side
FUNDING_SPLIT = 96             # 8h / 5m = 96 (분배 방식)

# 미세학습 기본값
FT_TOTAL_STEPS = 50_000        # 한 번에 학습 스텝
VAL_TAIL_BARS   = 3_000        # 최근 검증 길이(약 10일@5m)
MIN_GAIN_RATIO  = 0.015        # +1.5% 이상일 때만 승격(보수 프로파일)
SAVE_NAME       = "best_model_live.zip"  # 개선 시 덮어쓸 파일명
CHECKPOINT_KEEP = 5            # 최근 체크포인트 최대 개수 보관

# PPO 설정(기존 학습과 유사하되 보수적)
PPO_KW = dict(
    learning_rate=lambda pr: float(3e-5 * (0.3 + 0.7 * pr)),  # 진행 후반 감속
    n_steps=4096,
    batch_size=1024,
    n_epochs=8,
    ent_coef=0.01,
    vf_coef=0.5,
    clip_range=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[256, 256], ortho_init=False),
    device="cpu",
    verbose=0,
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# -----------------------
# 미니 환경 (보상 동형화: 수수료 + 펀딩 분배)
# -----------------------
class MiniTradingEnv(gym.Env):
    """
    관찰: 최근 WINDOW 개 피처(정규화 완료)
    행동: 0=홀드, 1=롱, 2=숏
    보상: pos*log_return
         - (포지션 변경 시 수수료 두 번 중 해당 사이드만)
         - (옵션) 펀딩비 분배 차감: pos * funding/96
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, X: pd.DataFrame, close: pd.Series, funding: Optional[pd.Series] = None):
        super().__init__()
        assert len(X) == len(close), "X/close length mismatch"
        self.X = X.reset_index(drop=True)
        self.close = close.reset_index(drop=True).astype(float)
        self.funding = None
        if funding is not None:
            funding = funding.reindex(X.index).reset_index(drop=True).astype(float)
            self.funding = funding
        self.n_feat = self.X.shape[1]

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(WINDOW * self.n_feat,), dtype=np.float32
        )
        self.t = WINDOW
        self.pos = 0

    def _obs(self) -> np.ndarray:
        w = self.X.iloc[self.t - WINDOW:self.t].values.astype(np.float32)
        return w.reshape(-1)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = WINDOW
        self.pos = 0
        return self._obs(), {}

    def step(self, a: int):
        target = 0 if a == 0 else (1 if a == 1 else -1)
        p0 = float(self.close.iloc[self.t - 1])
        p1 = float(self.close.iloc[self.t])
        lr = math.log(p1 / p0)
        reward = float(self.pos * lr)

        # 수수료(변경된 사이드만)
        if target != self.pos:
            if self.pos != 0:
                reward -= COMMISSION_SIDE
            if target != 0:
                reward -= COMMISSION_SIDE
            self.pos = target

        # (옵션) 펀딩비 분배 차감
        if self.funding is not None:
            fund = float(self.funding.iloc[self.t])
            reward -= self.pos * (fund / FUNDING_SPLIT)

        self.t += 1
        done = self.t >= len(self.X)
        obs = (self._obs() if not done else np.zeros(self.observation_space.shape, np.float32))
        return obs, reward, done, False, {}

# -----------------------
# 평가 (빠른 시뮬, 펀딩 옵션)
# -----------------------
def quick_eval(model: PPO,
               X: pd.DataFrame,
               close: pd.Series,
               funding: Optional[pd.Series] = None,
               deterministic: bool = True) -> float:
    """
    간이 평가: 에쿼티 배율을 반환(1.0=변화 없음)
    - 수수료·펀딩(옵션)·마크투마켓 반영
    """
    pos = 0
    eq = 1.0
    if funding is not None:
        funding = funding.reindex(X.index).astype(float)
    for t in range(WINDOW, len(X)):
        obs = X.iloc[t - WINDOW:t].values.reshape(-1).astype(np.float32)
        a, _ = model.predict(obs, deterministic=deterministic)
        target = 0 if a == 0 else (1 if a == 1 else -1)

        p0 = float(close.iloc[t - 1])
        p1 = float(close.iloc[t])
        lr = math.log(p1 / p0)
        reward = float(pos * lr)

        if target != pos:
            if pos != 0:
                reward -= COMMISSION_SIDE
            if target != 0:
                reward -= COMMISSION_SIDE
            pos = target

        if funding is not None:
            reward -= pos * (float(funding.iloc[t]) / FUNDING_SPLIT)

        eq *= math.exp(reward)
    return float(eq)

# -----------------------
# 체크포인트 관리
# -----------------------
def _rotate_checkpoints(prefix: str, keep: int = CHECKPOINT_KEEP):
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(prefix) and f.endswith(".zip")]
    files.sort(reverse=True)
    for f in files[keep:]:
        try:
            os.remove(os.path.join(MODEL_DIR, f))
        except Exception:
            pass

# -----------------------
# 온라인 학습기
# -----------------------
class OnlineLearner:
    def __init__(self,
                 base_model_path: Optional[str] = None,
                 out_model_name: str = SAVE_NAME,
                 ft_steps: int = FT_TOTAL_STEPS,
                 val_tail_bars: int = VAL_TAIL_BARS,
                 min_gain_ratio: float = MIN_GAIN_RATIO):
        """
        base_model_path:
            None이면 MODEL_DIR/best_model.zip → 없으면 ppo_final_model.zip 을 자동 로드
        """
        self.out_model_name = out_model_name
        self.ft_steps = int(ft_steps)
        self.val_tail_bars = int(val_tail_bars)
        self.min_gain_ratio = float(min_gain_ratio)

        # 모델 로드
        if base_model_path is None:
            # LIVE_OUT_NAME 기준으로 찾기 시도
            live_out_path = os.path.join(MODEL_DIR, out_model_name)
            best = os.path.join(MODEL_DIR, "best_model.zip")
            final = os.path.join(MODEL_DIR, "ppo_final_model.zip")
            if os.path.exists(live_out_path):
                base_model_path = live_out_path
            elif os.path.exists(best):
                base_model_path = best
            else:
                base_model_path = final

        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"베이스 모델을 찾을 수 없습니다: {base_model_path}")

        self.model_path = base_model_path
        self.model: PPO = PPO.load(self.model_path, device="cpu")

        print(f"[온라인 학습기] 베이스 모델 로드 완료: {os.path.basename(self.model_path)}")

    # ---- 공개 메서드 ----

    def finetune_on_recent(self,
                           X: pd.DataFrame,
                           close: pd.Series,
                           steps: Optional[int] = None,
                           val_bars: Optional[int] = None,
                           save_if_improved: bool = True,
                           funding: Optional[pd.Series] = None) -> bool:
        """
        최근 스냅샷으로 미세학습 후 개선 여부 반환(True/False)
        - steps/val_bars 미지정 시 기본값 사용
        - save_if_improved=True면 MODEL_DIR/out_model_name 으로 저장
        - funding: 펀딩비 시계열(옵션, 없으면 무시)
        """
        assert isinstance(X, pd.DataFrame) and isinstance(close, pd.Series), "X/close 데이터가 유효하지 않습니다"
        assert len(X) >= WINDOW + 100, "데이터가 너무 짧습니다"
        X = X.copy(); close = close.copy()
        X, close = X.align(close, join="inner", axis=0)
        X = X.dropna()
        close = close.reindex(X.index).astype(float)
        if funding is not None:
            funding = funding.reindex(X.index).astype(float)

        steps = int(steps or self.ft_steps)
        val_bars = int(val_bars or self.val_tail_bars)

        # VAL 스플릿(꼬리쪽)
        X_tr = X.iloc[:-val_bars]
        close_tr = close.iloc[:-val_bars]
        X_val = X.iloc[-val_bars:]
        close_val = close.iloc[-val_bars:]
        funding_tr = funding.iloc[:-val_bars] if funding is not None else None
        funding_val = funding.iloc[-val_bars:] if funding is not None else None

        # 기준 점수
        base_score = quick_eval(self.model, X_val, close_val, funding=funding_val)
        print(f"[온라인 학습기] 기준 점수(VAL): {base_score:.6f}")

        # 환경 구성 & 복제 모델로 학습(원본 안전)
        env = DummyVecEnv([lambda: MiniTradingEnv(X_tr, close_tr, funding=funding_tr)])
        ft_model: PPO = PPO(
            "MlpPolicy", env, tensorboard_log=MODEL_DIR, **PPO_KW
        )
        # 초기 파라미터를 기존 모델로부터 가져오기
        ft_model.policy.load_state_dict(self.model.policy.state_dict(), strict=True)

        print(f"[온라인 학습기] {steps:,} 스텝 동안 미세조정 진행 중…")
        ft_model.learn(total_timesteps=steps, progress_bar=False)

        # 새 점수
        new_score = quick_eval(ft_model, X_val, close_val, funding=funding_val)
        print(f"[온라인 학습기] 새로운 점수(VAL): {new_score:.6f}")

        improved = (new_score >= base_score * (1.0 + self.min_gain_ratio))
        if improved:
            out_path = os.path.join(MODEL_DIR, self.out_model_name)
            # 버전 보관 체크포인트
            tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
            ckpt_name = f"ckpt_live_{tag}.zip"
            ckpt_path = os.path.join(MODEL_DIR, ckpt_name)
            ft_model.save(ckpt_path)
            _rotate_checkpoints("ckpt_live_", CHECKPOINT_KEEP)

            # 주 모델 저장
            if save_if_improved:
                ft_model.save(out_path)
            # 메모리의 주 모델도 교체(거래 모듈이 같은 프로세스에서 참조할 경우 대비)
            self.model = ft_model

            print(f"[온라인 학습기] ✅ 성능 향상 → {self.out_model_name} 및 {ckpt_name} 저장 완료")
        else:
            print("[온라인 학습기] ❌ 성능 향상 없음 → 롤백 (저장 안 함)")

        return bool(improved)

    def update_from_rollout(self,
                            rollout: Dict[str, np.ndarray],
                            *,
                            gamma: float = 0.99,
                            gae_lambda: float = 0.95,
                            max_kl: float = 0.02,
                            epochs: int = 1,
                            batch_size: int = 1024,
                            lr: float = 5e-5,
                            X_val: Optional[pd.DataFrame] = None,
                            close_val: Optional[pd.Series] = None,
                            funding_val: Optional[pd.Series] = None,
                            save_if_improved: bool = True) -> bool:
        """
        실전 전이(rollout)로 즉시 온라인 업데이트 (KL 가드 + 선택적 검증/저장)

        required keys in `rollout`:
            - obs:        (T, obs_dim) float32
            - actions:    (T,) int64 (Discrete) 또는 (T, act_dim) for Box
            - rewards:    (T,) float32    # 실전 보상(수수료/펀딩 포함) 권장
            - dones:      (T,) bool       # True일 때 episode_start = next step
            - values:     (T,) float32    # 수집 당시의 V(s)
            - log_probs:  (T,) float32    # 수집 당시의 log π(a|s)
        """
        assert all(k in rollout for k in ["obs", "actions", "rewards", "dones", "values", "log_probs"]), \
            "rollout dict missing required keys"

        obs = np.asarray(rollout["obs"], dtype=np.float32)
        actions = np.asarray(rollout["actions"])
        rewards = np.asarray(rollout["rewards"], dtype=np.float32)
        dones = np.asarray(rollout["dones"], dtype=bool)
        values = np.asarray(rollout["values"], dtype=np.float32)
        log_probs = np.asarray(rollout["log_probs"], dtype=np.float32)

        T = obs.shape[0]
        assert T >= 32, "rollout too short"

        # episode_starts: t=0 True, t>0 는 이전 step이 done이면 True
        episode_starts = np.zeros(T, dtype=bool)
        episode_starts[0] = True
        episode_starts[1:] = dones[:-1]

        # RolloutBuffer 구성
        rb = RolloutBuffer(
            buffer_size=T,
            observation_space=self.model.observation_space,
            action_space=self.model.action_space,
            device=self.model.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_envs=1,
        )

        for t in range(T):
            rb.add(obs[t], actions[t], rewards[t], episode_starts[t], values[t], log_probs[t])

        # 마지막 상태의 value (없다면 0)
        last_val = float(values[-1]) if np.isfinite(values[-1]) else 0.0
        rb.compute_returns_and_advantage(last_values=last_val, dones=dones[-1])

        # 백업(롤백 대비)
        prev_state = {k: v.clone() for k, v in self.model.policy.state_dict().items()}
        prev_optimizer = {
            k: v.clone() for k, v in self.model.policy.optimizer.state_dict().items()
            if isinstance(v, th.Tensor)
        }

        # 훈련 하이퍼 임시 조정
        old_lr_fn = self.model.lr_schedule
        old_epochs = self.model.n_epochs
        old_batch = self.model.batch_size

        self.model.lr_schedule = lambda _: lr
        self.model.n_epochs = int(epochs)
        self.model.batch_size = int(batch_size)

        # 주입한 버퍼로 학습
        self.model.rollout_buffer = rb
        self.model.train()

        # approx KL 측정: old_logp vs new_logp
        with th.no_grad():
            obs_th = th.as_tensor(obs, device=self.model.device)
            act_th = th.as_tensor(actions, device=self.model.device)
            dist = self.model.policy.get_distribution(obs_th)
            new_logp = dist.log_prob(act_th)
            # 일부 action space에서 차원 합 필요할 수 있음 → 평균으로 스칼라화
            if new_logp.ndim > 1:
                new_logp = new_logp.sum(-1)
            approx_kl = float((th.as_tensor(log_probs, device=self.model.device) - new_logp).mean().abs().item())

        # 하이퍼 복원
        self.model.lr_schedule = old_lr_fn
        self.model.n_epochs = old_epochs
        self.model.batch_size = old_batch

        # KL 가드: 초과 시 롤백
        if approx_kl > max_kl:
            self.model.policy.load_state_dict(prev_state, strict=True)
            opt_state = self.model.policy.optimizer.state_dict()
            for k, v in prev_optimizer.items():
                if k in opt_state:
                    opt_state[k] = v
            self.model.policy.optimizer.load_state_dict(opt_state)
            print(f"[온라인 학습기] ❌ KL 초과({approx_kl:.4f} > {max_kl:.4f}) → 롤백")
            return False

        # (옵션) 검증/저장
        out_path = os.path.join(MODEL_DIR, self.out_model_name)
        if X_val is not None and close_val is not None:
            val_score = quick_eval(self.model, X_val, close_val, funding=funding_val)
            print(f"[온라인 학습기] 온라인 업데이트 후 VAL={val_score:.6f} (KL={approx_kl:.4f})")
        if save_if_improved:
            self.model.save(out_path)
            print(f"[온라인 학습기] ✅ 온라인 업데이트 저장: {self.out_model_name} (KL={approx_kl:.4f})")

        return True

    def reload_best(self) -> None:
        """디스크의 out_model_name을 다시 로드(거래 프로세스와 분리 운용 시 사용)"""
        path = os.path.join(MODEL_DIR, self.out_model_name)
        if os.path.exists(path):
            self.model = PPO.load(path, device="cpu")
            print(f"[온라인 학습기] 리로드 완료: {self.out_model_name}")
        else:
            print(f"[온라인 학습기] 모델을 찾을 수 없음: {self.out_model_name} (현재 모델 유지)")
