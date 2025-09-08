import os
import json
import random
import numpy as np
import torch as th
import pandas as pd
from typing import Dict

from env import MultiTimeframeTradingEnv
from policy import MultiTimeframeLSTMPolicy
from ppo import train_with_config

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
ETH_TFS = ["5m", "15m", "1h", "4h"]

def _load_all_timeframes(split: str) -> Dict[str, pd.DataFrame]:
    """Loads all processed ETH dataframes for a given split into a dictionary."""
    dfs = {}
    for tf in ETH_TFS:
        path = os.path.join(DATA_DIR, f"feHPO_{split}_{tf}.parquet")
        if os.path.exists(path):
            dfs[tf] = pd.read_parquet(path)
        else:
            print(f"[warn] Data file not found for {tf} in split {split}, skipping: {path}")
    return dfs

def run_hpo(
    save_path: str = "best_candidate.json",
    train_tag: str = "train",
    val_tag: str = "val",
    train_steps: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Runs Hyperparameter Optimization.
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

    tf_train = _load_all_timeframes(train_tag)
    tf_val = _load_all_timeframes(val_tag)

    if not tf_train or not tf_val:
        raise ValueError("Could not load training or validation data. Please run prepare scripts.")

    # The policy needs to know the observation dimension for each timeframe.
    obs_dims = {tf: df.shape[1] for tf, df in tf_train.items()}
    print(f"[HPO] obs_dims = {obs_dims}")

    best_score = -np.inf
    best_config = None
    all_results = []

    for i, config in enumerate(search_space):
        print(f"\n[HPO] Trial {i+1}/{len(search_space)} — config: {config}")

        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)

        # The env now takes a dictionary of dataframes.
        env = MultiTimeframeTradingEnv(tf_train, price_col="Close")
        eval_env = MultiTimeframeTradingEnv(tf_val, price_col="Close")

        # The policy now needs a dictionary of observation dimensions.
        policy = MultiTimeframeLSTMPolicy(
            obs_dims=obs_dims,
            action_dim=env.action_space.n,
            trend_dim=3,
            aux_coeff=0.1,
            lstm_hidden_dim=128,
            num_lstm_layers=1,
            mlp_hidden_dims=tuple(config["net_arch"] + [64]),
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

    # Save features for main training
    best_config["features"] = {tf: list(df.columns) for tf, df in tf_train.items()}

    out = {
        "best_config": best_config,
        "best_score": best_score,
        "results": all_results
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return best_config