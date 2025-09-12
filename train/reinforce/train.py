import os
import sys
import json
import random
import numpy as np
import torch as th
import pandas as pd

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

from ai_binance.train.reinforce.env import MultiTimeframeTradingEnv
from ai_binance.train.reinforce.policy import MultiTimeframeLSTMPolicy
from ai_binance.train.reinforce.hpo import run_hpo, _load_all_timeframes
from ai_binance.train.reinforce.ppo import train_with_config

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def main():
    # 1. 데이터 로드
    tf_train = _load_all_timeframes("train")
    tf_val = _load_all_timeframes("val")

    # 2. config 로드
    hpo_dir = os.path.join(DATA_DIR, "hpo")
    top5_path = os.path.join(hpo_dir, "hpo_top5.json")
    best_config_path = os.path.join(hpo_dir, "best_config.json")
    best_config = None

    if os.path.exists(top5_path):
        print(f"[info] Found {top5_path}. Loading best config from it...")
        with open(top5_path, "r", encoding="utf-8") as f:
            top5_results = json.load(f)
        
        if top5_results:
            best_result = top5_results[0]
            best_config = best_result["config"]
            print(f"[info] Using config from trial {best_result.get('trial', 'N/A')} with sharpe {best_result.get('sharpe', 0.0):.4f}")
            
            print("[info] Injecting feature set information...")
            best_config["features"] = {tf: list(df.columns) for tf, df in tf_train.items()}
    
    if best_config is None:
        if os.path.exists(best_config_path):
            print(f"[info] No top5 file found. Loading best config from {best_config_path}...")
            with open(best_config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            best_config = loaded["best_config"]
        else:
            print("[info] No config file found. Running HPO...")
            best_config = run_hpo(save_path=best_config_path)

    print(f"[config] Using config: {best_config}")

    # 3. obs_dims
    obs_dim = {tf: len(cols) for tf, cols in best_config["features"].items()}

    # 4. 환경 생성
    env = MultiTimeframeTradingEnv(
        tf_train,
        obs_cols=best_config["features"]
    )
    eval_env = MultiTimeframeTradingEnv(
        tf_val,
        obs_cols=best_config["features"]
    )

    # 5. 정책 생성
    net_arch = best_config.get("net_arch", [256])
    if isinstance(net_arch, int):
        net_arch = [net_arch]

    policy = MultiTimeframeLSTMPolicy(
        obs_dims=obs_dim,
        action_dim=4,
        trend_dim=best_config.get("trend_dim", 3),
        aux_coeff=best_config.get("aux_coeff", 0.1),
        lstm_hidden_dim=best_config.get("lstm_hidden_dim", 128),
        num_lstm_layers=best_config.get("num_lstm_layers", 1),
        mlp_hidden_dims=tuple(net_arch + [64]),
        device="cuda" if th.cuda.is_available() else "cpu"
    )

    # 6. 학습
    result = train_with_config(
        env,
        eval_env,
        policy,
        config=best_config,
        train_steps=1_000_000,
        learning_rate=best_config["learning_rate"]
    )
    print(f"[done] result: {result}")


if __name__ == "__main__":
    main()
