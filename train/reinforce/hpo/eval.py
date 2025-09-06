# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, hashlib, pickle
from typing import List, Dict, Tuple, Optional
import random
import numpy as np
import pandas as pd
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

HERE = os.path.dirname(__file__)
TRAIN_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
if TRAIN_DIR not in sys.path:
    sys.path.append(TRAIN_DIR)

try:
    from reinforce.env import TradingEnv
    from reinforce.policy import MultiHeadPolicy
    from prepare.utils import apply_feature_mask
except Exception:
    BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
    if BASE not in sys.path:
        sys.path.append(BASE)
    from reinforce.env import TradingEnv
    from reinforce.policy import MultiHeadPolicy
    from prepare.utils import apply_feature_mask

def _annualize_sharpe(rets: np.ndarray, bars_per_day: int = 288, days_per_year: int = 252) -> float:
    if rets.size == 0:
        return 0.0
    mu = np.float32(np.mean(rets))
    sd = np.float32(np.std(rets)) + 1e-9
    return (mu / sd) * np.sqrt(bars_per_day * days_per_year)

def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-9)
    return np.float32(-dd.min())

def _cache_key(selected: List[str], seed: int, train_steps: int, val_steps: int, window: Optional[Tuple[int,int]]):
    key = "|".join(sorted(selected)) + f"|{seed}|{train_steps}|{val_steps}|{window}"
    return hashlib.md5(key.encode()).hexdigest()

def action_mask_fn(env) -> np.ndarray:
    pos = getattr(env.portfolio, "position", 0)
    n = int(env.action_space.n)
    mask = np.ones(n, dtype=bool)
    if n >= 4:
        mask[1] = (pos <= 0)   # LONG
        mask[2] = (pos >= 0)   # SHORT
        mask[3] = (pos != 0)   # CLOSE
    return mask

def apply_feature_mask_safe(df: pd.DataFrame, selected_feats: List[str], fill_value: float = 0.0) -> pd.DataFrame:
    missing = [c for c in selected_feats if c not in df.columns]
    assert not missing, f"Missing columns in input: {missing}"
    return df[selected_feats].fillna(fill_value)

def evaluate_feature_set(
    df_train_5m: pd.DataFrame,
    df_val_5m: pd.DataFrame,
    selected_feats: List[str],
    env_kwargs: Dict,
    seeds: List[int],
    train_steps: int,
    val_steps: int,
    cache_dir: Optional[str] = None,
    window: Optional[Tuple[int,int]] = None,
) -> Dict[str, float]:
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    if window is None:
        va_df = df_val_5m
    else:
        s, e = window
        if not df_val_5m.index.is_monotonic_increasing:
            raise ValueError("Validation index is not time-ordered.")
        if df_val_5m.index.inferred_freq is None:
            raise ValueError("Validation index frequency is not consistent.")
        try:
            start_dt = df_val_5m.index[s]
            end_dt = df_val_5m.index[e - 1]
            va_df = df_val_5m.loc[start_dt:end_dt]
        except Exception as ex:
            raise IndexError(f"Invalid window indices: {s}, {e} -> {ex}")

    tr_masked = apply_feature_mask_safe(df_train_5m, selected_feats)
    va_masked = apply_feature_mask_safe(va_df, selected_feats)

    # HPO 환경에서 price_close가 누락되는 문제 수정
    # apply_feature_mask_safe가 selected_feats에 없는 모든 컬럼을 제거하므로,
    # TradingEnv에 필수적인 price_close를 다시 추가해준다.
    if 'price_close' in df_train_5m.columns:
        tr_masked['price_close'] = df_train_5m['price_close']
    if 'price_close' in va_df.columns:
        va_masked['price_close'] = va_df['price_close']

    agg = {"sharpe": 0.0, "mdd": 0.0, "trades_per_day": 0.0}
    K = 0

    for seed in seeds:
        ck = _cache_key(selected_feats, seed, train_steps, val_steps, window)
        cache_path = os.path.join(cache_dir, f"{ck}.pkl") if cache_dir else None

        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                m = pickle.load(f)
        else:
            try:
                random.seed(seed)
                np.random.seed(seed)
                th.manual_seed(seed)

                env_tr = TradingEnv(tr_masked, obs_cols=selected_feats, **env_kwargs)
                env_va = TradingEnv(va_masked, obs_cols=selected_feats, **env_kwargs)
                env_tr = ActionMasker(env_tr, action_mask_fn)
                env_va = ActionMasker(env_va, action_mask_fn)
                env_tr.reset(seed=seed)
                env_va.reset(seed=seed)

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

                obs, info = env_va.reset(seed=seed)
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

                initial_eq = np.float32(info.get("initial_equity", 10_000.0))
                rets, equity, actions = [], [], []
                total_reward = np.float32(0.0)

                for i in range(val_steps):
                    action, _ = model.predict(obs, deterministic=True)
                    if isinstance(action, (np.ndarray, list)):
                        try:
                            action = int(np.asarray(action).squeeze().item())
                        except Exception:
                            action = int(action[0])

                    obs, reward, terminated, truncated, info = env_va.step(action)
                    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

                    rets.append(np.float32(reward))
                    equity_val = np.float32(info.get("equity", 0.0)) / initial_eq
                    equity.append(equity_val)
                    actions.append(int(action))
                    total_reward += np.float32(reward)

                    if terminated or truncated:
                        obs, info = env_va.reset(seed=seed)
                        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

                sharpe = _annualize_sharpe(np.asarray(rets, dtype=np.float32))
                mdd = _max_drawdown(np.asarray(equity, dtype=np.float32))
                tpd = np.float32((np.asarray(actions, dtype=np.int32) != 0).mean() * 288.0)

                print(f"[summary] seed={seed} sharpe={sharpe:.4f} mdd={mdd:.4f} tpd={tpd:.1f} reward_sum={total_reward:.4f}")

                m = {"sharpe": sharpe, "mdd": mdd, "trades_per_day": tpd}
                if cache_path:
                    with open(cache_path, "wb") as f:
                        pickle.dump(m, f)

            except Exception as e:
                print(f"[eval error] {e} | seed={seed}")
                print(f"[debug] features used: {selected_feats[:5]} ... ({len(selected_feats)} total)")
                m = {"sharpe": 0.0, "mdd": 1.0, "trades_per_day": 999.0}

        for k in agg:
            agg[k] += m[k]
        K += 1

    for k in agg:
        agg[k] /= max(K, 1)
    return agg
