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
import numpy as np
import pandas as pd
from gymnasium.spaces import Box, Discrete
# ----- 안전 임포트 (패키지/스크립트 실행 모두 대응) -----
HERE = os.path.dirname(__file__)
TRAIN_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)

try:
    # 패키지 형태로 실행(-m) 시
    from reinforce.env import TradingEnv
    from reinforce.policy import MultiHeadPolicy
    from fe import apply_feature_mask
except Exception:
    # 스크립트 직접 실행 시 대비
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

    # 마스킹(관측 컬럼 선택) — REF 컬럼은 뒤에 붙음
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
            # 환경 생성 (훈련/검증)
            env_tr = TradingEnv(tr_masked, obs_cols=selected_feats, **env_kwargs)
            env_va = TradingEnv(va_masked, obs_cols=selected_feats, **env_kwargs)

            # 정책 초기화 및 짧은 학습
            # 관측 공간 (연속, selected_feats의 길이에 맞춰 설정)
            observation_space = Box(low=-np.inf, high=np.inf, shape=(len(selected_feats),), dtype=np.float32)

            # 행동 공간 (정해진 4개의 행동)
            action_space = Discrete(4)

            # 재현성 설정 ❶ (policy가 아닌 환경/전역 시드로 이동)
            import random, torch as th
            random.seed(seed)
            np.random.seed(seed)
            th.manual_seed(seed)
            env_tr.reset(seed=seed)
            env_va.reset(seed=seed)

            # 학습률 스케줄 (고정 값 사용)
            lr_schedule = lambda _: 1e-4

            # 정책 객체 생성 ❷ (seed 인자 제거)
            policy = MultiHeadPolicy(
                observation_space=observation_space,
                action_space=action_space,
                lr_schedule=lr_schedule,
            )
            policy.train_ppo(env_tr, total_steps=train_steps)  # << 빠른 평가: steps만 짧게

            # 검증 롤아웃
            rets, equity, actions = policy.rollout(env_va, max_steps=val_steps)  # (rets, equity, actions) 가정
            sharpe = _annualize_sharpe(np.asarray(rets, dtype=np.float32))
            ir     = sharpe  # 벤치마크가 없으므로 동일 대체
            mdd    = _max_drawdown(np.asarray(equity, dtype=np.float32))
            tpd    = float((np.asarray(actions) != 0).mean() * 288.0)  # 5m → 288 bars/day

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

# ----- 간단한 수동 테스트 -----
if __name__ == "__main__":
    from fe import load_processed, build_universe_from_processed
    # 데이터 로드(자동: HPO 확장 프레임 있으면 사용)
    df_tr = load_processed("train", "5m", mode="auto")
    df_va = load_processed("val",   "5m", mode="auto")
    feats = build_universe_from_processed("train", "5m", mode="auto")[:80]  # 80개만 샘플
    met = evaluate_feature_set(
        df_tr, df_va, feats,
        env_kwargs={"fee_rate":0.0004, "slip_bp":2.0, "random_start":False},
        seeds=[0], train_steps=10_000, val_steps=8_000,
        cache_dir="./.hpo_cache"
    )
    print("[eval] metrics:", met)
