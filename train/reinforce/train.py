# train.py

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

from ai_binance.train.reinforce.env import TradingEnv
from ai_binance.train.reinforce.policy import MultiHeadLSTMPolicy
from ai_binance.train.reinforce.hpo import run_hpo
from ai_binance.train.reinforce.ppo import train_with_config

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CONFIG_PATH = os.path.join(DATA_DIR, "hpo", "best_config.json")

def load_dataframe_from_json(filename: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, "r") as f:
        raw_data = json.load(f)
    
    df = pd.DataFrame(raw_data)
    df = df.dropna().copy()
    return df

def main():
    # 데이터 로드 (수정된 부분)
    df_train = load_dataframe_from_json("feHPO_feature_list_1h.json")
    df_val = df_train.copy()  # 검증 데이터를 따로 없으면 복사해서 사용

    # 보조 정보 유지
    ref_cols = ["price_close", "funding_per_bar", "label_4h_dir"]
    for col in ref_cols:
        if col in df_train.columns:
            df_val[col] = df_train[col]

    # 최적 config 확인 or 생성
    if not os.path.exists(CONFIG_PATH):
        print("[info] No best_config.json found. Running HPO...")
        best_config = run_hpo(save_path=CONFIG_PATH)
    else:
        print("[info] Loading best config...")
        with open(CONFIG_PATH, "r") as f:
            best_config = json.load(f)

    print(f"[config] Using config: {best_config}")

    # 학습 환경 준비
    obs_cols = best_config["features"]
    env = TradingEnv(df_train, obs_cols=obs_cols)
    eval_env = TradingEnv(df_val, obs_cols=obs_cols)

    # 정책 초기화
    policy = MultiHeadLSTMPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: best_config["learning_rate"],
        trend_dim=3,
        net_arch=dict(pi=[256], vf=[256]),
        lstm_hidden_dim=128,
        device="cuda" if th.cuda.is_available() else "cpu"
    )

    # 학습 시작
    result = train_with_config(env, eval_env, policy, config=best_config, train_steps=1_000_000)

    print(f"[done] result: {result}")

if __name__ == "__main__":
    main()
