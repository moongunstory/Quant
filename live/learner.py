# ai_binance/live/online_learner.py
"""
Online Learner for PPO (ETHUSDT Futures, 5m)

목표
- 최근 구간으로 짧은 미세학습(fine-tune)
- 검증(VAL) 점수 비교: 개선되면 채택/저장, 악화면 자동 롤백
- 블로킹 없이 외부(run.py)에서 스레드로 돌리기 쉬운 구조
- CLI 없음. 상수로 조절.

입출력
- 입력: 정규화된 피처 DataFrame X (fe.py와 동일), Close Series (동일 인덱스)
- 출력: 개선 시 모델 저장(기본 best_model_live.zip), 로그는 print로 최소화

주의
- PPO는 on-policy라 버퍼 재사용보다 "최근 구간 미니-환경"으로 재학습하는 방식이 단순·안정.
- 실거래 중엔 별도 스레드로 호출 권장(거래 루프와 분리).

사용 예 (run.py에서):
    from train.rl import TradingEnv
    ol = OnlineLearner()
    improved = ol.finetune_on_recent(X_recent, close_recent)
"""

from __future__ import annotations

import os
import math
import json
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# -----------------------
# 경로 & 상수 (수정 가능)
# -----------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/ai_binance
MODEL_DIR = os.path.join(BASE_DIR, "data", "model")
REPORT_DIR = os.path.join(BASE_DIR, "data", "reports")

# 거래 비용/관찰창 (학습/백테스트와 정합)
WINDOW = 48                    # 4h
COMMISSION_SIDE = 0.0005       # 0.05% per side

# 미세학습 기본값
FT_TOTAL_STEPS = 50_000        # 한 번에 학습 스텝
VAL_TAIL_BARS   = 3_000        # 최근 검증 길이(약 10일@5m)
MIN_GAIN_RATIO  = 0.0          # 새 점수 >= 기준 점수*(1+이 값)일 때 채택
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
# 미니 환경 (단순·일관 보상)
# -----------------------
class MiniTradingEnv(gym.Env):
    """
    관찰: 최근 WINDOW 개 피처(정규화 완료)
    행동: 0=홀드, 1=롱, 2=숏
    보상: pos*log_return - (포지션 변경 시 수수료 두 번 중 해당 사이드만)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, X: pd.DataFrame, close: pd.Series):
        super().__init__()
        assert len(X) == len(close), "X/close length mismatch"
        self.X = X.reset_index(drop=True)
        self.close = close.reset_index(drop=True).astype(float)
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

        if target != self.pos:         # 포지션 변경 수수료
            if self.pos != 0:
                reward -= COMMISSION_SIDE
            if target != 0:
                reward -= COMMISSION_SIDE
            self.pos = target

        self.t += 1
        done = self.t >= len(self.X)
        obs = (self._obs() if not done else np.zeros(self.observation_space.shape, np.float32))
        return obs, reward, done, False, {}

# -----------------------
# 평가 (빠른 시뮬)
# -----------------------
def quick_eval(model: PPO, X: pd.DataFrame, close: pd.Series, deterministic: bool = True) -> float:
    """
    간이 평가: 에쿼티 배율을 반환(1.0=변화 없음)
    - 수수료·마크투마켓 반영
    """
    pos = 0
    eq = 1.0
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
                           save_if_improved: bool = True) -> bool:
        """
        최근 스냅샷으로 미세학습 후 개선 여부 반환(True/False)
        - steps/val_bars 미지정 시 기본값 사용
        - save_if_improved=True면 MODEL_DIR/out_model_name 으로 저장
        """
        assert isinstance(X, pd.DataFrame) and isinstance(close, pd.Series), "X/close 데이터가 유효하지 않습니다"
        assert len(X) >= WINDOW + 100, "데이터가 너무 짧습니다"
        X = X.copy(); close = close.copy()
        X, close = X.align(close, join="inner", axis=0)
        X = X.dropna()
        close = close.reindex(X.index).astype(float)

        steps = int(steps or self.ft_steps)
        val_bars = int(val_bars or self.val_tail_bars)

        # VAL 스플릿(꼬리쪽)
        X_tr = X.iloc[:-val_bars]
        close_tr = close.iloc[:-val_bars]
        X_val = X.iloc[-val_bars:]
        close_val = close.iloc[-val_bars:]

        # 기준 점수
        base_score = quick_eval(self.model, X_val, close_val)
        print(f"[온라인 학습기] 기준 점수(VAL): {base_score:.6f}")

        # 환경 구성 & 복제 모델로 학습(원본 안전)
        env = DummyVecEnv([lambda: MiniTradingEnv(X_tr, close_tr)])
        ft_model: PPO = PPO(
            "MlpPolicy", env, tensorboard_log=MODEL_DIR, **PPO_KW
        )
        # 초기 파라미터를 기존 모델로부터 가져오기
        ft_model.policy.load_state_dict(self.model.policy.state_dict(), strict=True)

        print(f"[온라인 학습기] {steps:,} 스텝 동안 미세조정 진행 중…")
        ft_model.learn(total_timesteps=steps, progress_bar=False)

        # 새 점수
        new_score = quick_eval(ft_model, X_val, close_val)
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

    def reload_best(self) -> None:
        """디스크의 out_model_name을 다시 로드(거래 프로세스와 분리 운용 시 사용)"""
        path = os.path.join(MODEL_DIR, self.out_model_name)
        if os.path.exists(path):
            self.model = PPO.load(path, device="cpu")
            print(f"[온라인 학습기] 리로드 완료: {self.out_model_name}")
        else:
            print(f"[온라인 학습기] 모델을 찾을 수 없음: {self.out_model_name} (현재 모델 유지)")