import os
import json
import random
import numpy as np
import torch as th
import pandas as pd
from env import TradingEnv
from policy import MultiHeadLSTMPolicy
from ppo import train_with_config

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
ETH_TFS = ["5m", "15m", "1h", "4h"]
BTC_TF = "btc1h"

def _load_split(split: str) -> pd.DataFrame:
    """
    ETH 4개 타임프레임 + BTC 1h 데이터를 로드하여 병합.
    가장 빈번한 5m 데이터를 기준으로, 낮은 빈도의 데이터를 `merge_asof`로 병합합니다.
    """
    # 기준이 되는 5분봉 데이터 로드
    base_tf = "5m"
    base_path = os.path.join(DATA_DIR, f"feHPO_{split}_{base_tf}.parquet")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base data file not found for {split} at {base_path}")

    df_all = pd.read_parquet(base_path)
    # merge_asof를 위해 인덱스를 datetime으로 변환하고 정렬
    if not pd.api.types.is_datetime64_any_dtype(df_all.index):
        df_all.index = pd.to_datetime(df_all.index)
    df_all.sort_index(inplace=True)

    # 병합할 나머지 타임프레임 목록 (타임프레임, 접두사)
    tfs_to_merge = [
        ("15m", "f_15m_"),
        ("1h", "f_1h_"),
        ("4h", "f_4h_"),
        (BTC_TF, "btc_"),
    ]

    for tf, prefix in tfs_to_merge:
        path = os.path.join(DATA_DIR, f"feHPO_{split}_{tf}.parquet")
        if os.path.exists(path):
            df_other = pd.read_parquet(path)
            if not pd.api.types.is_datetime64_any_dtype(df_other.index):
                df_other.index = pd.to_datetime(df_other.index)
            df_other.sort_index(inplace=True)
            
            df_other = df_other.add_prefix(prefix)

            df_all = pd.merge_asof(
                df_all, df_other, left_index=True, right_index=True, direction="backward"
            )
        else:
            print(f"[warn] Missing data file for {tf} ({split})")

    # merge_asof로 생긴 시작 부분의 NaN 값들을 이전 값으로 채웁니다.
    df_all.fillna(method='ffill', inplace=True)
    # 그래도 맨 처음에 남은 NaN이 있다면 해당 row들은 제거합니다.
    df_all.dropna(inplace=True)


    if df_all.empty:
        print("[warn] DataFrame is empty after merging and filling. The data files likely do not overlap in time.")

    return df_all.copy()



def run_hpo(
    save_path: str = "best_candidate.json",
    train_tag: str = "train",
    val_tag: str = "val",
    train_steps: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    다양한 하이퍼파라미터 설정을 테스트하여 최적 config를 찾는 HPO 실행 함수.
    """
    search_space = [
        {
            "learning_rate": lr,
            "batch_size": bs,
            "ent_coef": ent,
            "net_arch": [h],
        }
        for lr in [3e-4, 1e-4]
        for bs in [256, 512]
        for ent in [0.01, 0.03]
        for h in [128, 256]
    ]

    df_train = _load_split(train_tag)
    df_val = _load_split(val_tag)

    obs_cols = [c for c in df_train.columns if c.startswith("f_") or c.startswith("btc_")]
    print(f"[HPO] feature_dim = {len(obs_cols)}")

    best_score = -np.inf
    best_config = None
    all_results = []

    for i, config in enumerate(search_space):
        print(f"\n[HPO] Trial {i+1}/{len(search_space)} — config: {config}")

        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)

        env = TradingEnv(df_train, obs_cols=obs_cols)
        eval_env = TradingEnv(df_val, obs_cols=obs_cols)

        policy = MultiHeadLSTMPolicy(
            obs_dim=len(obs_cols),
            seq_len=24,
            action_dim=4,
            trend_dim=3,
            aux_coeff=0.1,
            lstm_hidden_dim=128,
            num_lstm_layers=1,
            mlp_hidden_dims=tuple(config["net_arch"] + [64]),  # [128] → (128, 64) 형태로 변환
            device="cuda" if th.cuda.is_available() else "cpu"
        )

        result = train_with_config(
            env=env,
            eval_env=eval_env,
            policy=policy,
            config=config,
            train_steps=train_steps,
        )

        sharpe = result.get("sharpe", 0.0)
        mdd = result.get("mdd", 0.0)
        tpd = result.get("trades_per_day", 0.0)

        print(f"[HPO] Result: Sharpe={sharpe:.4f}, MDD={mdd:.4f}, TPD={tpd:.1f}")

        all_results.append({
            "trial": i + 1,
            "config": config,
            "sharpe": sharpe,
            "mdd": mdd,
            "tpd": tpd
        })

        if sharpe > best_score:
            best_score = sharpe
            best_config = config

    print(f"\n✅ Best Config: {best_config} (Sharpe={best_score:.4f})")

    out = {
        "best_config": best_config,
        "best_score": best_score,
        "results": all_results
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return best_config
