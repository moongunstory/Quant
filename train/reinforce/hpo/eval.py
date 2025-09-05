# -*- coding: utf-8 -*-
"""
HPO 평가 실행기: 선택된 피처셋으로 짧은 PPO 학습 후 검증 롤아웃을 수행하여
Sharpe / IR / MDD / 거래빈도 등을 산출한다.

- policy.py 수정 불필요: total_steps만 짧게 주어 빠르게 측정
- env는 obs_cols로 가변 차원을 받도록 이미 패치된 상태를 가정
- 캐싱 지원: 동일 (feature set, seed, window, steps) 조합 재평가 방지
"""
from __future__ import annotations
import os, sys, json, hashlib, pickle
from typing import List, Dict, Tuple, Optional
import random
import numpy as np
import pandas as pd
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ----- 안전 임포트 (패키지/스크립트 실행 모두 대응) -----
HERE = os.path.dirname(__file__)
TRAIN_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)

try:
    from reinforce.env import TradingEnv
    from reinforce.policy import MultiHeadPolicy
    from fe import apply_feature_mask
except Exception:
    BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
    if BASE not in sys.path:
        sys.path.append(BASE)
    from reinforce.env import TradingEnv
    from reinforce.policy import MultiHeadPolicy
    from fe import apply_feature_mask

# ----- 메트릭 유틸 -----
def _annualize_sharpe(rets: np.ndarray, bars_per_day: int = 288, days_per_year: int = 252) -> float:
    if rets.size == 0:
        return 0.0
    mu = float(np.mean(rets))
    sd = float(np.std(rets)) + 1e-9
    return (mu / sd) * np.sqrt(bars_per_day * days_per_year)

def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-9)
    return float(-dd.min())

def _cache_key(selected: List[str], seed: int, train_steps: int, val_steps: int, window: Optional[Tuple[int,int]]):
    key = "|".join(sorted(selected)) + f"|{seed}|{train_steps}|{val_steps}|{window}"
    return hashlib.md5(key.encode()).hexdigest()

# ----- 마스킹 함수 (전역: pickle 가능) -----
def action_mask_fn(env) -> np.ndarray:
    """
    WAIT(0) 항상 허용
    LONG(1): 현재 포지션이 롱이 아닐 때
    SHORT(2): 현재 포지션이 숏이 아닐 때
    CLOSE(3): 포지션 있을 때만
    """
    pos = getattr(env, "position", 0)
    n = int(env.action_space.n)
    mask = np.ones(n, dtype=bool)
    if n >= 4:
        mask[1] = (pos <= 0)   # LONG
        mask[2] = (pos >= 0)   # SHORT
        mask[3] = (pos != 0)   # CLOSE
    return mask

# ----- 메인 평가 함수 -----
def evaluate_feature_set(
    df_train_5m: pd.DataFrame,
    df_val_5m: pd.DataFrame,
    selected_feats: List[str],
    env_kwargs: Dict,
    seeds: List[int],
    train_steps: int,
    val_steps: int,
    cache_dir: Optional[str] = None,
    window: Optional[Tuple[int,int]] = None,  # (start_idx, end_idx) on validation frame
) -> Dict[str, float]:
    """
    선택된 피처들로 학습/검증하여 평균 성능을 반환.
    반환 예:
      {"sharpe": 1.23, "ir": 1.23, "mdd": 0.18, "trades_per_day": 27.4}
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # 검증 프레임 윈도우 슬라이스
    if window is None:
        va_df = df_val_5m
    else:
        s, e = window
        va_df = df_val_5m.iloc[s:e]

    # 마스킹(관측 컬럼 선택)
    tr_masked = apply_feature_mask(df_train_5m, selected_feats)
    va_masked = apply_feature_mask(va_df,       selected_feats)

    agg = {"sharpe": 0.0, "ir": 0.0, "mdd": 0.0, "trades_per_day": 0.0}
    K = 0

    for seed in seeds:
        ck = _cache_key(selected_feats, seed, train_steps, val_steps, window)
        cache_path = os.path.join(cache_dir, f"{ck}.pkl") if cache_dir else None

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                m = pickle.load(f)
        else:
            # ==== 재현성 시드 ====
            random.seed(seed)
            np.random.seed(seed)
            th.manual_seed(seed)

            # ==== 환경 생성 + 액션 마스킹 ====
            env_tr = TradingEnv(tr_masked, obs_cols=selected_feats, **env_kwargs)
            env_va = TradingEnv(va_masked, obs_cols=selected_feats, **env_kwargs)
            env_tr = ActionMasker(env_tr, action_mask_fn)
            env_va = ActionMasker(env_va, action_mask_fn)
            env_tr.reset(seed=seed)
            env_va.reset(seed=seed)

            # ==== SB3 에이전트(PPO, 마스킹 지원) ====
            model = MaskablePPO(
                policy=MultiHeadPolicy,
                env=env_tr,
                learning_rate=1e-4,
                n_steps=2048,
                batch_size=256,
                n_epochs=5,
                seed=seed,
                verbose=0,
            )
            model.learn(total_timesteps=train_steps)

            # ==== 검증 롤아웃 ====
            obs, info = env_va.reset(seed=seed)
            rets, equity, actions = [], [], []
            for _ in range(val_steps):
                action, _ = model.predict(obs, deterministic=True)
                if isinstance(action, (np.ndarray, list)):
                    try:
                        action = int(np.asarray(action).squeeze().item())
                    except Exception:
                        action = int(action[0])
                obs, reward, terminated, truncated, info = env_va.step(action)
                rets.append(float(reward))
                equity.append(float(info.get("equity", 0.0)))
                actions.append(int(action))
                if terminated or truncated:
                    obs, info = env_va.reset(seed=seed)

            sharpe = _annualize_sharpe(np.asarray(rets, dtype=np.float32))
            ir     = sharpe
            mdd    = _max_drawdown(np.asarray(equity, dtype=np.float32))
            tpd    = float((np.asarray(actions, dtype=np.int32) != 0).mean() * 288.0)  # 5m → 288 bars/day

            m = {"sharpe": float(sharpe), "ir": float(ir), "mdd": float(mdd), "trades_per_day": float(tpd)}
            if cache_path:
                with open(cache_path, "wb") as f:
                    pickle.dump(m, f)

        for k in agg:
            agg[k] += m[k]
        K += 1

    for k in agg:
        agg[k] /= max(K, 1)
    return agg
