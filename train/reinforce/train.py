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
CONFIG_PATH = os.path.join(DATA_DIR, "hpo", "best_config.json")


def main():
    # 1. 데이터 로드
    tf_train = _load_all_timeframes("train")
    tf_val = _load_all_timeframes("val")

    # 2. config 로드
    if not os.path.exists(CONFIG_PATH):
        print("[info] No best_config.json found. Running HPO...")
        best_config = run_hpo(save_path=CONFIG_PATH)
    else:
        print("[info] Loading best config...")
        with open(CONFIG_PATH, "r") as f:
            best_config = json.load(f)

    print(f"[config] Using config: {best_config}")

    # 3. obs_dims
    obs_dim = {tf: tf_train[tf].shape[1] for tf in ["5m", "15m", "1h", "4h"]}

    # 4. 환경 생성
    env = MultiTimeframeTradingEnv(tf_train)
    eval_env = MultiTimeframeTradingEnv(tf_val)

    # 5. 정책 생성
    policy = MultiTimeframeLSTMPolicy(
        obs_dim=obs_dim["5m"],
        action_dim=4,
        trend_dim=3,
        aux_coeff=0.1,
        lstm_hidden_dim=128,
        num_lstm_layers=1,
        mlp_hidden_dims=tuple(best_config["net_arch"] + [64]),
        device="cuda" if th.cuda.is_available() else "cpu"
    )

    # 6. 학습
    result = train_with_config(env, eval_env, policy, config=best_config, train_steps=1_000_000)
    print(f"[done] result: {result}")


if __name__ == "__main__":
    main()
