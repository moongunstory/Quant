# ai_binance/train/reinforce/train.py
from __future__ import annotations
import os, random, json
import numpy as np
import pandas as pd
import torch as th

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

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
VEC_PATH = os.path.join(CKPT_DIR, "unified_vecnorm.pkl")
CKPT_PATH = os.path.join(CKPT_DIR, "unified_multihead.zip")  # 최종 스냅샷(참고용)
OBS_ORDER_PATH = os.path.join(CKPT_DIR, "obs_cols.json")     # 실거래 입력 정렬용

# 학습에서 관측 제외할 원본(참조) 접미사
REF_SUFFIXES = ["_Open", "_High", "_Low", "_Close", "_Volume", "_FundingRate", "_FundingSettle"]

# ===== 데이터 병합(옵션 A: btc1h 파일 미사용) =====
def _load_split(prefix: str) -> pd.DataFrame:
    paths = [
        os.path.join(PROC_DIR, f"{prefix}_5m.parquet"),
        os.path.join(PROC_DIR, f"{prefix}_15m.parquet"),
        os.path.join(PROC_DIR, f"{prefix}_1h.parquet"),
        os.path.join(PROC_DIR, f"{prefix}_4h.parquet"),
        # 옵션 A: 별도 btc1h parquet는 관측에서 제외 (ETH 피처 내부 요약 신호만 사용)
        # os.path.join(PROC_DIR, f"{prefix}_btc1h.parquet"),
    ]
    dfs = []

    # fe.py의 REF_COLS_CANON과 일치
    ref_cols_canon = ["Open", "High", "Low", "Close", "Volume", "FundingRate", "FundingSettle"]

    # TF별 feature_list를 사용해 열 정렬(스케일된 피처만 선택)
    tf_map = {
        "_5m.parquet": "5m",
        "_15m.parquet": "15m",
        "_1h.parquet": "1h",
        "_4h.parquet": "4h",
    }

    # 원본 참조 데이터(5m)에서 close/펀딩을 가져오기 위해 보관
    ref_dfs = {}

    for p in paths:
        if not os.path.exists(p):
            continue

        df = pd.read_parquet(p)
        name = os.path.basename(p)

        # ETH 타임프레임만 처리
        for suffix, tf in tf_map.items():
            if name.endswith(suffix) and "btc" not in name:
                list_path = os.path.join(PROC_DIR, f"fe_feature_list_{tf}.json")
                if os.path.exists(list_path):
                    with open(list_path, "r", encoding="utf-8") as f:
                        feature_list = json.load(f)
                    # 참조 컬럼 따로 보관(5m에서 close/funding 사용)
                    ref_df_cols = [c for c in ref_cols_canon if c in df.columns]
                    ref_dfs[tf] = df[ref_df_cols].copy()
                    # 선택된 특성만 유지
                    df = df.reindex(columns=feature_list, fill_value=0.0)
                # 접두사 부여
                if tf == "5m":      df = df.add_prefix("f_5m_")
                elif tf == "15m":   df = df.add_prefix("f_15m_")
                elif tf == "1h":    df = df.add_prefix("f_1h_")
                elif tf == "4h":    df = df.add_prefix("f_4h_")
                dfs.append(df)
                break

    if not dfs:
        raise FileNotFoundError(f"No parquet files for prefix={prefix}")

    base = dfs[0]
    for d in dfs[1:]:
        base = base.join(d, how="outer")

    base = base.sort_index().ffill().copy()  # defrag 1회

    # 종가/펀딩 추가(5m 참조 데이터 필수)
    if "5m" not in ref_dfs:
        raise RuntimeError("5m reference frame not found. Check processed files and fe pipeline.")
    close_series = pd.to_numeric(ref_dfs["5m"]["Close"], errors="coerce").astype(float)
    extra = {"close": close_series, "price_close": close_series}
    if "funding_per_bar" not in base.columns:
        extra["funding_per_bar"] = pd.Series(0.0, index=base.index)
    base = pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1)

    # 관측 피처: f_* & REF 접미사 제외
    obs_cols = [c for c in base.columns if c.startswith("f_") and not any(c.endswith(s) for s in REF_SUFFIXES)]
    base = base.dropna(subset=obs_cols + ["close"]).copy()
    base[obs_cols] = base[obs_cols].astype("float32")

    # 보조 라벨(4H 방향) 없으면 추가(마지막 48개는 NaN → 콜백에서 자동 스킵)
    if "label_4h_dir" not in base.columns:
        h = 48  # 4H = 48 x 5m
        ret = (base["close"].shift(-h) - base["close"]) / base["close"]
        lbl = np.where(ret > 0.001, 2, np.where(ret < -0.001, 0, 1)).astype(np.int8)
        base = pd.concat([base, pd.Series(lbl, index=base.index, name="label_4h_dir")], axis=1)

    base.attrs["obs_cols"] = obs_cols
    return base

# ===== 액션 마스크 =====
# 0 WAIT, 1 LONG, 2 SHORT, 3 CLOSE
def action_mask_fn(env) -> np.ndarray:
    mask = np.ones(env.action_space.n, dtype=np.int8)
    pos = getattr(env, "position", 0)
    if pos == 0: mask[3] = 0
    if pos > 0:  mask[1] = 0
    if pos < 0:  mask[2] = 0
    return mask

# ===== Aux Loss 콜백 (trend head만 별도 업데이트) =====
class TrendAuxLossCallback(BaseCallback):
    def __init__(self, coeff: float = 0.1, verbose: int = 0):
        super().__init__(verbose)
        self.coeff = float(coeff)
        self._labels: list[int] = []

    def _on_rollout_start(self) -> None:
        self._labels.clear()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if isinstance(info, dict) and "trend_label" in info:
                self._labels.append(int(info["trend_label"]))
        return True

    def _on_rollout_end(self) -> None:
        if not self._labels:
            return
        buf = self.model.rollout_buffer  # (n_steps, n_env, obs_dim)
        obs = th.as_tensor(buf.observations, device=self.model.device, dtype=th.float32).reshape(-1, buf.observations.shape[-1])
        y = th.as_tensor(np.asarray(self._labels, dtype=np.int64), device=self.model.device)
        n = min(obs.shape[0], y.shape[0])
        if n <= 0:
            self._labels.clear(); return

        # ✅ trend head만 업데이트 (backbone 고정), PPO optimizer는 건드리지 않음
        aux_logged = self.model.policy.aux_train_step(obs[:n], y[:n], coeff=self.coeff, max_grad_norm=1.0)
        self.model.logger.record("train/aux_trend_loss", aux_logged)
        self._labels.clear()

# ===== 메인 =====
def main():
    df_train = _load_split("fe_train")
    df_val   = _load_split("fe_val")

    # 학습 관측 컬럼 순서 저장 → 실거래에서 동일 정렬/차원 보장
    obs_cols = df_train.attrs.get("obs_cols") or [
        c for c in df_train.columns if c.startswith("f_") and not any(c.endswith(s) for s in REF_SUFFIXES)
    ]
    with open(OBS_ORDER_PATH, "w", encoding="utf-8") as f:
        json.dump(list(obs_cols), f, ensure_ascii=False, indent=2)
    print(f"[obs] saved order -> {OBS_ORDER_PATH} (dim={len(obs_cols)})")

    def make_env_train():
        env = TradingEnv(df_train, fee_rate=0.0004, slip_bp=2.0, turn_cost=0.0, max_position_bars=None)
        return ActionMasker(env, action_mask_fn)

    def make_env_eval():
        env = TradingEnv(df_val,   fee_rate=0.0004, slip_bp=2.0, turn_cost=0.0, max_position_bars=None)
        env = Monitor(env)  # Eval 경고 제거
        return ActionMasker(env, action_mask_fn)

    # ===== VecNormalize: obs/rwd 정규화(안정성)
    venv = DummyVecEnv([make_env_train])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_env = DummyVecEnv([make_env_eval])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, training=False)
    eval_env.obs_rms = venv.obs_rms  # 관측 정규화 통계 공유
    eval_env.norm_reward = False      # ✅ 평가/조기중단은 보상 정규화 OFF

    # ===== PPO 설정(안정한 기본값)
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
        tensorboard_log=None,
        verbose=1
    )

    # ===== 콜백: Aux + Eval(베스트만 저장)
    aux_cb  = TrendAuxLossCallback(coeff=0.1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=CKPT_DIR,
        eval_freq=100_000,
        n_eval_episodes=5,         # ✅ 분산 완화
        deterministic=True
    )
    cb = CallbackList([aux_cb, eval_cb])

    # ===== 학습
    model.learn(total_timesteps=1_000_000, callback=cb)

    # ===== 저장(최종 스냅샷 + VecNormalize + obs_cols)
    venv.save(VEC_PATH)
    model.save(CKPT_PATH)
    print(f"[OK] vecnorm: {VEC_PATH}")
    print(f"[OK] best:    {os.path.join(CKPT_DIR, 'best_model.zip')}")
    print(f"[OK] last:    {CKPT_PATH}")

if __name__ == "__main__":
    main()
