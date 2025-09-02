# ai_binance/train/reinforce/train.py
from __future__ import annotations
import os, random
import numpy as np
import torch as th
import pandas as pd

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from env import TradingEnv
from policy import MultiHeadPolicy

# ===== 재현성 =====
SEED = 42
random.seed(SEED); np.random.seed(SEED); th.manual_seed(SEED)

# ===== 경로 =====
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
PROC_DIR = os.path.join(DATA_DIR, "processed")
CKPT_DIR = os.path.join(DATA_DIR, "model")
os.makedirs(CKPT_DIR, exist_ok=True)
VEC_PATH  = os.path.join(CKPT_DIR, "unified_vecnorm.pkl")  # 런타임 호환 이름
BEST_PATH = os.path.join(CKPT_DIR, "best_model.zip")       # 개선 시에만 덮어씀

# fe가 남긴 원본 REF 접미사 (관측에서 제외)
REF_SUFFIXES = ["_Open", "_High", "_Low", "_Close", "_Volume", "_FundingRate", "_FundingSettle"]

# ===== 데이터 병합 (간결) =====
def _load_split(prefix: str) -> pd.DataFrame:
    paths = [
        os.path.join(PROC_DIR, f"{prefix}_5m.parquet"),
        os.path.join(PROC_DIR, f"{prefix}_15m.parquet"),
        os.path.join(PROC_DIR, f"{prefix}_1h.parquet"),
        os.path.join(PROC_DIR, f"{prefix}_4h.parquet"),
        os.path.join(PROC_DIR, f"{prefix}_btc1h.parquet"),
    ]
    dfs = []
    for p in paths:
        if not os.path.exists(p): 
            continue
        df = pd.read_parquet(p)
        name = os.path.basename(p)
        if name.endswith("_5m.parquet"):      df = df.add_prefix("f_5m_")
        elif name.endswith("_15m.parquet"):   df = df.add_prefix("f_15m_")
        elif name.endswith("_1h.parquet") and "btc" not in name: df = df.add_prefix("f_1h_")
        elif name.endswith("_4h.parquet"):    df = df.add_prefix("f_4h_")
        elif name.endswith("_btc1h.parquet"): df = df.add_prefix("f_btc1h_")
        dfs.append(df)

    base = dfs[0]
    for d in dfs[1:]:
        base = base.join(d, how="outer")
    base = base.sort_index().ffill().copy()

    # 종가/펀딩 일괄 추가
    extra = {"close": pd.to_numeric(base["f_5m_Close"], errors="coerce").astype(float)}
    extra["price_close"] = extra["close"]
    if "funding_per_bar" not in base.columns:
        extra["funding_per_bar"] = pd.Series(0.0, index=base.index)
    base = pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1)

    # 관측 피처 선택
    obs_cols = [c for c in base.columns if c.startswith("f_") and not any(c.endswith(s) for s in REF_SUFFIXES)]
    base = base.dropna(subset=obs_cols + ["close"]).copy()
    base[obs_cols] = base[obs_cols].astype("float32")

    # 4H 방향 보조 라벨(없으면 생성)
    if "label_4h_dir" not in base.columns:
        h = 48
        ret = (base["close"].shift(-h) - base["close"]) / base["close"]
        lbl = np.where(ret > 0.001, 2, np.where(ret < -0.001, 0, 1)).astype(np.int8)
        base = pd.concat([base, pd.Series(lbl, index=base.index, name="label_4h_dir")], axis=1)

    base.attrs["obs_cols"] = obs_cols
    return base

# ===== 액션 마스크 =====
# 액션: 0 WAIT, 1 LONG, 2 SHORT, 3 CLOSE
def action_mask_fn(env) -> np.ndarray:
    mask = np.ones(env.action_space.n, dtype=np.int8)
    pos = getattr(env, "position", 0)
    if pos == 0: mask[3] = 0
    if pos > 0:  mask[1] = 0
    if pos < 0:  mask[2] = 0
    return mask

# ===== Aux Loss 콜백 (step 수집 → rollout_end 학습 + 로깅) =====
class TrendAuxLossCallback(BaseCallback):
    def __init__(self, coeff: float = 0.1, verbose: int = 0):
        super().__init__(verbose); self.coeff = coeff; self._labels = []

    def _on_rollout_start(self) -> None:
        self._labels.clear()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if isinstance(info, dict) and "trend_label" in info:
                self._labels.append(int(info["trend_label"]))
        return True

    def _on_rollout_end(self) -> None:
        if not self._labels: return
        buf = self.model.rollout_buffer
        obs = th.as_tensor(buf.observations, device=self.model.device, dtype=th.float32).reshape(-1, buf.observations.shape[-1])
        y = th.as_tensor(np.asarray(self._labels, dtype=np.int64), device=self.model.device)
        n = min(obs.shape[0], y.shape[0])
        if n <= 0: self._labels.clear(); return
        aux = self.model.policy.aux_loss(obs[:n], y[:n])
        self.model.policy.optimizer.zero_grad(set_to_none=True)
        (self.coeff * aux).backward(); self.model.policy.optimizer.step()
        self.model.logger.record("train/aux_trend_loss", (self.coeff * aux).item())
        self._labels.clear()

# ===== 베스트만 저장 + 초기 저장 + 얼리 스톱 =====
class BestSaver(BaseCallback):
    def __init__(self, eval_env, eval_freq: int = 50_000, n_eval_episodes: int = 3,
                 patience_evals: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.patience_evals = int(patience_evals)
        self.best_mean = -np.inf
        self._since_best = 0

    def _sync_eval_stats(self):
        # 학습 env의 VecNormalize 통계를 평가 env로 동기화
        train_vec: VecNormalize = self.model.get_vec_normalize_env()  # type: ignore
        if isinstance(self.eval_env, VecNormalize) and train_vec is not None:
            self.eval_env.obs_rms = train_vec.obs_rms
            self.eval_env.returns_rms = train_vec.returns_rms
            self.eval_env.training = False

    def _evaluate_and_maybe_save(self, tag: str):
        self._sync_eval_stats()
        mean_r, _ = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, deterministic=True)
        improved = mean_r > self.best_mean
        if improved:
            self.best_mean = mean_r
            self._since_best = 0
            # 모델 + VecNormalize 동시 스냅샷 (오직 개선 시)
            self.model.save(BEST_PATH)
            vec: VecNormalize = self.model.get_vec_normalize_env()  # type: ignore
            if vec is not None:
                vec.save(VEC_PATH)
            if self.verbose:
                print(f"[BEST-{tag}] mean_reward={mean_r:.2f} → saved {os.path.basename(BEST_PATH)} & vecnorm")
        else:
            self._since_best += 1
            if self.verbose:
                print(f"[EVAL-{tag}] mean_reward={mean_r:.2f} (best={self.best_mean:.2f}) no-improve({self._since_best}/{self.patience_evals})")

        # 얼리 스톱
        if self._since_best >= self.patience_evals:
            if self.verbose:
                print("[EARLY-STOP] no improvement, stopping training.")
            return False
        return True

    def _on_training_start(self) -> None:
        # 0-step 초기 평가/저장: 항상 런타임 파일 보장
        self._evaluate_and_maybe_save("init")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            return self._evaluate_and_maybe_save("step")
        return True

# ===== 메인 =====
def main():
    df_train = _load_split("fe_train")
    df_val   = _load_split("fe_val")

    def make_env_train():
        env = TradingEnv(df_train, fee_rate=0.0004, slip_bp=2.0, turn_cost=0.0, max_position_bars=None)
        return ActionMasker(env, action_mask_fn)

    def make_env_eval():
        env = TradingEnv(df_val, fee_rate=0.0004, slip_bp=2.0, turn_cost=0.0, max_position_bars=None)
        env = Monitor(env)
        return ActionMasker(env, action_mask_fn)

    # === VecNormalize (obs/reward 둘 다 정규화) ===
    venv = DummyVecEnv([make_env_train])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_env = DummyVecEnv([make_env_eval])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, training=False)
    eval_env.obs_rms = venv.obs_rms
    eval_env.returns_rms = venv.returns_rms

    # === PPO (안정성 중심 설정) ===
    model = MaskablePPO(
        policy=MultiHeadPolicy,
        env=venv,
        seed=SEED,
        policy_kwargs=dict(
            trend_dim=3,
            aux_coeff=0.1,
            net_arch=dict(pi=[256], vf=[256])
        ),
        n_steps=4096,
        batch_size=512,
        learning_rate=3e-4,
        ent_coef=0.03,
        vf_coef=0.4,
        n_epochs=15,
        max_grad_norm=0.5,
        clip_range=0.2,
        target_kl=0.03,          # 후반 품질 보호
        tensorboard_log=None,
        verbose=1
    )

    # === 콜백: Aux + BestSaver(초기 저장+얼리스톱) ===
    cb = [TrendAuxLossCallback(coeff=0.1),
          BestSaver(eval_env=eval_env, eval_freq=50_000, n_eval_episodes=3, patience_evals=10, verbose=1)]

    model.learn(total_timesteps=1_000_000, callback=cb)

    # 마지막에 추가 저장하지 않음(베스트만 유지)
    print(f"[DONE] best: {BEST_PATH}\n[DONE] vecnorm: {VEC_PATH}")

if __name__ == "__main__":
    main()
