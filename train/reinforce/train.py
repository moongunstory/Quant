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
from ai_binance.train.reinforce.hpo import _load_all_timeframes
from ai_binance.train.reinforce.ppo import train_with_config, load_best_from_top5

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

def main():
    # 1. 데이터 로드
    tf_train = _load_all_timeframes("train")
    tf_val = _load_all_timeframes("val")

    # 2. top5에서 최고 config 로딩
    hpo_dir = os.path.join(DATA_DIR, "hpo")
    top5_path = os.path.join(hpo_dir, "hpo_top5.json")

    if not os.path.exists(top5_path):
        raise FileNotFoundError(f"[error] hpo_top5.json not found at {top5_path}. Run HPO first.")

    best_config = load_best_from_top5(top5_path)
    print(f"[info] Loaded best config from top5: {best_config}")

    # 3. obs_dims 계산
    obs_dim = {tf: len(cols) for tf, cols in best_config["features"].items()}

    # 4. 환경 생성
    env = MultiTimeframeTradingEnv(tf_train, obs_cols=best_config["features"])
    eval_env = MultiTimeframeTradingEnv(tf_val, obs_cols=best_config["features"])

    # 5. 정책 생성
    net_arch = best_config.get("net_arch", [256])
    if isinstance(net_arch, int):
        net_arch = [net_arch]

    policy = MultiTimeframeLSTMPolicy(
        obs_dims=obs_dim,
        action_dim=env.action_space.n,
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
