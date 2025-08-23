# ai_binance/live/online_learner.py
"""
Online Learner for HRL Worker (MaskablePPO + VecNormalize)

목표(철학 유지, 현 버전 호환):
- 짧은 구간에서 **온라인 업데이트** (rollout 기반, KL 가드/롤백)
- (옵션) 최근 데이터로 **간이 파인튜닝**: 환경 없이 최근 구간을 시뮬해 rollout 수집 → 업데이트
- 검증 점수(간단 equity 배율)로 저장 여부 판단 가능
- 트레이더(run.py)와 분리된 스레드에서 블로킹 없이 동작

입출력
- rollout: 트레이더가 push한 실전 전이(dict)
    {"obs": (T, obs_dim) RAW-OBS(TradeEnv와 동일), "actions": (T,), "rewards": (T,),
     "dones": (T,), "values": (T,), "log_probs": (T,)}
  ※ 트레이더 리팩터에서 obs_raw를 넣도록 설계했으므로 여기서 VecNormalize로 정규화
- (옵션) X/close/funding 을 받아 최근 구간에서 간이 rollout을 자체 수집(finetune_on_recent)

주의
- 본 학습기는 **Worker(MaskablePPO)** 전용. (Manager는 오프라인 재학습 권장)
- 보상은 트레이더와 동일 컨벤션(보유수익−수수료−펀딩의 per-5m 분배)을 기본으로 가정.
"""

from __future__ import annotations
import os, math, json, copy
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import torch as th

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer
from gymnasium import spaces, Env
import joblib

# -----------------------
# 경로 & 상수
# -----------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/ai_binance
MODEL_DIR  = os.path.join(BASE_DIR, "data", "model")
PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")

# 파일명 (worker)
WORKER_MODEL_PATH    = os.path.join(MODEL_DIR, "worker_unified_final.zip")
WORKER_VECNORM_PATH  = os.path.join(MODEL_DIR, "worker_unified_vecnorm.pkl")
OUT_NAME_DEFAULT     = "worker_unified_live.zip"   # 개선 시 저장

# 수수료/펀딩/윈도우(트레이더와 일치)
COMMISSION_SIDE = 0.0005  # 0.05% per side
FUNDING_SPLIT   = 96      # 8h / 5m
VOL_WIN         = 24
MAX_HOLD_STEPS  = 72

# 저장 관리
CHECKPOINT_KEEP = 5

# -----------------------
# 더미 env: VecNormalize 로드용
# -----------------------
class _ObsOnlyEnv(Env):
    def __init__(self, obs_shape: Tuple[int, ...], action_space: spaces.Space):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = action_space
    def reset(self, *, seed=None, options=None): return np.zeros(self.observation_space.shape, np.float32), {}
    def step(self, action): return self.reset()[0], 0.0, True, False, {}

# -----------------------
# 체크포인트 로테이션
# -----------------------
def _rotate_checkpoints(prefix: str, keep: int = CHECKPOINT_KEEP):
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(prefix) and f.endswith(".zip")]
    files.sort(reverse=True)
    for f in files[keep:]:
        try: os.remove(os.path.join(MODEL_DIR, f))
        except Exception: pass

# -----------------------
# 간단 검증(에쿼티 배율)
# -----------------------
def _quick_eval_eq(actions: np.ndarray, close: pd.Series,
                   funding: Optional[pd.Series] = None) -> float:
    """
    actions: 시계열 목표 포지션 {-1,0,1} (이미 게이트 적용되어 있다고 가정)
    """
    pos = 0
    eq = 1.0
    funding = funding.reindex(close.index) if funding is not None else None
    for t in range(1, len(close)):
        p0 = float(close.iloc[t - 1]); p1 = float(close.iloc[t])
        lr = math.log(p1 / p0)
        # 진입/청산 수수료
        if actions[t] != pos:
            if pos != 0: eq *= math.exp(-COMMISSION_SIDE)
            if actions[t] != 0: eq *= math.exp(-COMMISSION_SIDE)
            pos = int(actions[t])
        # 보유수익
        eq *= math.exp(pos * lr)
        # 펀딩
        if funding is not None:
            eq *= math.exp(-pos * float(funding.iloc[t]))
    return float(eq)

# -----------------------
# OnlineLearner (Worker 전용)
# -----------------------
class OnlineLearner:
    def __init__(self,
                 worker_model_path: Optional[str] = None,
                 worker_vecnorm_path: Optional[str] = None,
                 out_model_name: str = OUT_NAME_DEFAULT):
        """
        worker_model_path/vecnorm_path 생략 시 기본 경로 사용.
        """
        self.out_model_name = out_model_name

        self.model_path   = worker_model_path or WORKER_MODEL_PATH
        self.vecnorm_path = worker_vecnorm_path or WORKER_VECNORM_PATH
        if not (os.path.exists(self.model_path) and os.path.exists(self.vecnorm_path)):
            raise FileNotFoundError(f"Worker 모델/VecNorm을 찾을 수 없습니다:\n  {self.model_path}\n  {self.vecnorm_path}")

        # 모델 로드
        self.model: MaskablePPO = MaskablePPO.load(self.model_path, device="cpu")
        obs_shape = self.model.observation_space.shape
        dummy = DummyVecEnv([lambda: _ObsOnlyEnv(obs_shape=obs_shape, action_space=self.model.action_space)])
        self.vec: VecNormalize = VecNormalize.load(self.vecnorm_path, dummy)
        self.vec.training = False
        self.vec.norm_reward = False

        # 인버스 스케일러(인제스터 X를 원시로 복원할 때 사용 가능)
        scaler_path = os.path.join(PROC_DIR, "scaler.joblib")
        feats_path  = os.path.join(PROC_DIR, "feature_list.json")
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        self.feature_list = json.load(open(feats_path, "r")) if os.path.exists(feats_path) else None

        print(f"[온라인 학습기] Worker 로드 완료: {os.path.basename(self.model_path)} | obs_shape={obs_shape}")

    # -------------------
    # 내부: obs 정규화
    # -------------------
    def _normalize_obs(self, obs_raw: np.ndarray) -> np.ndarray:
        """
        obs_raw: (T, obs_dim) or (obs_dim,)
        VecNormalize로 정규화
        """
        x = np.asarray(obs_raw, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.vec.normalize_obs(x)

    # -------------------
    # rollout 기반 업데이트 (주 경로)
    # -------------------
    def update_from_rollout(self,
                            rollout: Dict[str, np.ndarray],
                            *,
                            gamma: float = 0.99,
                            gae_lambda: float = 0.95,
                            max_kl: float = 0.02,
                            epochs: int = 1,
                            batch_size: int = 1024,
                            lr: float = 5e-5,
                            save_if_ok: bool = True) -> bool:
        """
        트레이더 전이(rollout)로 즉시 온라인 업데이트.
        - obs: RAW(TradeEnv 포맷). 여기서 VecNormalize로 정규화 후 학습
        - KL 가드 초과 시 롤백
        """
        required = ["obs", "actions", "rewards", "dones", "values", "log_probs"]
        assert all(k in rollout for k in required), f"rollout keys missing; need {required}"

        obs_raw = np.asarray(rollout["obs"], dtype=np.float32)
        actions = np.asarray(rollout["actions"])
        rewards = np.asarray(rollout["rewards"], dtype=np.float32)
        dones = np.asarray(rollout["dones"], dtype=bool)
        values = np.asarray(rollout["values"], dtype=np.float32)
        log_probs = np.asarray(rollout["log_probs"], dtype=np.float32)
        T = obs_raw.shape[0]
        assert T >= 32, "rollout too short"

        # 정규화된 관측
        obs = self._normalize_obs(obs_raw)

        # episode_starts
        episode_starts = np.zeros(T, dtype=bool)
        episode_starts[0] = True
        episode_starts[1:] = dones[:-1]

        # RolloutBuffer 구성
        rb = RolloutBuffer(
            buffer_size=T,
            observation_space=self.model.policy.observation_space,
            action_space=self.model.policy.action_space,
            device=self.model.device,
            gamma=gamma,
            gae_lambda=gae_lambda,
            n_envs=1,
        )

        for t in range(T):
            rb.add(obs[t], actions[t], rewards[t], episode_starts[t], values[t], log_probs[t])
        last_val = float(values[-1]) if np.isfinite(values[-1]) else 0.0
        rb.compute_returns_and_advantage(last_values=last_val, dones=dones[-1])

        # 백업(롤백 대비)
        prev_state = {k: v.clone() for k, v in self.model.policy.state_dict().items()}
        prev_opt = copy.deepcopy(self.model.policy.optimizer.state_dict())

        # 임시 하이퍼 세팅
        old_lr_fn, old_epochs, old_batch = self.model.lr_schedule, self.model.n_epochs, self.model.batch_size
        self.model.lr_schedule = lambda _: lr
        self.model.n_epochs = int(epochs)
        self.model.batch_size = int(batch_size)

        # 주입 학습
        self.model.rollout_buffer = rb
        self.model.train()

        # KL 측정
        with th.no_grad():
            obs_th = th.as_tensor(obs, device=self.model.device)
            act_th = th.as_tensor(actions, device=obs_th.device)
            dist = self.model.policy.get_distribution(obs_th)
            new_logp = dist.log_prob(act_th)
            if new_logp.ndim > 1: new_logp = new_logp.sum(-1)
            approx_kl = float((th.as_tensor(log_probs, device=self.model.device) - new_logp).mean().item())
            if approx_kl < 0: approx_kl = 0.0

        # 복원
        self.model.lr_schedule, self.model.n_epochs, self.model.batch_size = old_lr_fn, old_epochs, old_batch

        # KL 가드
        if approx_kl > max_kl:
            self.model.policy.load_state_dict(prev_state, strict=True)
            self.model.policy.optimizer.load_state_dict(prev_opt)
            print(f"[온라인 학습기] ❌ KL 초과({approx_kl:.4f} > {max_kl:.4f}) → 롤백")
            return False

        # 저장
        if save_if_ok:
            out_path = os.path.join(MODEL_DIR, self.out_model_name)
            tag = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
            ckpt = os.path.join(MODEL_DIR, f"ckpt_worker_live_{tag}.zip")
            self.model.save(out_path)
            self.model.save(ckpt)
            _rotate_checkpoints("ckpt_worker_live_", CHECKPOINT_KEEP)
            print(f"[온라인 학습기] ✅ 저장: {os.path.basename(out_path)} (KL={approx_kl:.4f})")
        else:
            print(f"[온라인 학습기] 업데이트 완료 (KL={approx_kl:.4f})")

        return True

    # -------------------
    # (옵션) 최근 데이터로 간이 파인튜닝
    #  - 환경 없이 rollout 수집 → update_from_rollout
    # -------------------
    def finetune_on_recent(self,
                           X_norm: pd.DataFrame,
                           close: pd.Series,
                           *,
                           funding: Optional[pd.Series] = None,
                           max_steps: int = 8_192,
                           save_if_ok: bool = True) -> bool:
        """
        X_norm: 인제스터가 만든 '정규화' 피처 프레임 (scaler/feature_list 기준)
        close:  동일 인덱스 종가
        funding: per-5m 분배 시리즈(있으면 사용)
        - 최근 데이터에서 현재 정책으로 rollout 수집 → KL-가드 업데이트
        """
        assert isinstance(X_norm, pd.DataFrame) and isinstance(close, pd.Series), "X/close invalid"
        X_norm, close = X_norm.align(close, join="inner", axis=0)
        funding = funding.reindex(X_norm.index) if funding is not None else None
        if len(X_norm) < 200:  # 너무 짧으면 skip
            print("[온라인 학습기] 데이터가 너무 짧아 파인튜닝을 건너뜁니다.")
            return False

        # 원시 피처 복원(학습 분포 정합)
        if self.scaler is not None and self.feature_list is not None and set(self.feature_list).issubset(X_norm.columns):
            Xr = X_norm[self.feature_list].copy()
            X_raw = self.scaler.inverse_transform(Xr.to_numpy(dtype=np.float64)).astype(np.float32)
            X_raw = pd.DataFrame(X_raw, index=X_norm.index, columns=self.feature_list)
        else:
            # 스케일러가 없으면 그대로 사용(최소 동작)
            X_raw = X_norm.astype(np.float32).copy()

        # rollout 수집
        obs_list, act_list, rew_list, done_list, val_list, logp_list = [], [], [], [], [], []
        pos = 0
        holding = 0
        last_price = float(close.iloc[0])
        T = len(X_raw)
        start = max(1, T - max_steps - 1)

        for i in range(start, T):
            # Manager 휴리스틱(간단)
            def _safe(s, col, d=0.0): 
                try: return float(s[col])
                except Exception: return d
            row = X_raw.iloc[i]
            macd_h1  = _safe(row, "macd_hist_1h")
            macd_h4  = _safe(row, "macd_hist_4h")
            rsi_h1   = _safe(row, "rsi14_1h") - 50.0
            rsi_h4   = _safe(row, "rsi14_4h") - 50.0
            ret3_h1  = _safe(row, "ret3_1h")
            ret12_h1 = _safe(row, "ret12_1h")
            score = (1.8*macd_h1 + 1.2*macd_h4 + 0.6*(rsi_h1/50.0) + 1.0*ret3_h1 + 0.7*ret12_h1)
            mgr_dir = 1 if score > 0 else (-1 if score < 0 else 0)
            mgr_conf = float(1 - math.exp(-min(5.0, abs(score)) * 1.2))
            mgr_reg  = int(abs(macd_h1 - macd_h4) < 1e-6)

            # obs_raw 구성(TradeEnv 포맷: X_raw_row + [mgr_dir, mgr_conf, mgr_reg, in_pos, holding_norm])
            in_pos = float(pos != 0)
            hold_norm = float(min(holding, MAX_HOLD_STEPS) / MAX_HOLD_STEPS)
            obs_raw = np.concatenate([
                row.to_numpy(dtype=np.float32, copy=False),
                np.array([float(mgr_dir), float(mgr_conf), float(mgr_reg), in_pos, hold_norm], dtype=np.float32)
            ], axis=0)

            # 정규화 후 행동/로그확률/가치
            obs_norm = self._normalize_obs(obs_raw)
            with th.no_grad():
                dist = self.model.policy.get_distribution(th.as_tensor(obs_norm))
                action, _ = self.model.predict(obs_norm, deterministic=True, action_masks=self._mask(pos))
                act_t = th.tensor(int(action), dtype=th.long)
                logp = dist.log_prob(act_t)
                if logp.ndim > 1: logp = logp.sum(-1)
                value = self.model.policy.predict_values(th.as_tensor(obs_norm))

            # 보상(트레이더와 동일)
            p1 = float(close.iloc[i]); p0 = float(close.iloc[i - 1])
            lr = math.log(p1 / p0)
            reward = pos * lr
            tx_cost = 0.0
            # 상태별 해석: 무포지션에서 1=Enter, 보유에서 2=Exit
            if pos == 0 and int(action) == 1 and mgr_dir != 0:
                # 진입
                tx_cost += COMMISSION_SIDE
                pos = 1 if mgr_dir > 0 else -1
                holding = 0
            elif pos != 0 and int(action) == 2:
                # 청산
                tx_cost += COMMISSION_SIDE
                pos = 0
                holding = 0
                done = True
            else:
                done = False
                holding += 1

            if pos != 0 and holding >= MAX_HOLD_STEPS:
                tx_cost += COMMISSION_SIDE
                pos = 0
                done = True
                holding = 0

            reward -= tx_cost
            if funding is not None:
                reward -= pos * float(funding.iloc[i])

            # 적재
            obs_list.append(obs_raw.astype(np.float32))
            act_list.append(int(action))
            rew_list.append(float(reward))
            done_list.append(bool(done))
            logp_list.append(float(logp.cpu().item()))
            val_list.append(float(value.cpu().item()))
            last_price = p1

        rollout = {
            "obs": np.stack(obs_list, axis=0),
            "actions": np.array(act_list, dtype=np.int64),
            "rewards": np.array(rew_list, dtype=np.float32),
            "dones": np.array(done_list, dtype=bool),
            "values": np.array(val_list, dtype=np.float32),
            "log_probs": np.array(logp_list, dtype=np.float32),
        }
        return self.update_from_rollout(rollout, save_if_ok=save_if_ok)

    # -------------------
    # 액션 마스크 (상태 기반)
    # -------------------
    @staticmethod
    def _mask(pos: int) -> np.ndarray:
        if pos == 0:
            return np.array([True, True, False], dtype=bool)
        else:
            return np.array([True, False, True], dtype=bool)

    # -------------------
    # 모델 리로드
    # -------------------
    def reload_best(self) -> None:
        path = os.path.join(MODEL_DIR, self.out_model_name)
        if os.path.exists(path):
            self.model = MaskablePPO.load(path, device="cpu")
            print(f"[온라인 학습기] 리로드 완료: {self.out_model_name}")
        else:
            print(f"[온라인 학습기] 모델을 찾을 수 없음: {self.out_model_name} (현재 모델 유지)")
