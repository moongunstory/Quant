import os
import json
import random
import numpy as np
import torch as th
import pandas as pd
from typing import Dict, Any
import optuna
import traceback

from ai_binance.train.reinforce.env import MultiTimeframeTradingEnv
from ai_binance.train.reinforce.policy import MultiTimeframeLSTMPolicy
from ai_binance.train.reinforce.ppo import train_with_config

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
ETH_TFS = ["5m", "15m", "1h", "4h"]

def _load_all_timeframes(split: str) -> Dict[str, pd.DataFrame]:
    dfs = {}
    for tf in ETH_TFS:
        path = os.path.join(DATA_DIR, f"feHPO_{split}_{tf}.parquet")
        if os.path.exists(path):
            dfs[tf] = pd.read_parquet(path)
        else:
            print(f"[warn] Data file not found for {tf} in split {split}, skipping: {path}")
    return dfs

def run_hpo(
    save_path: str,
    n_trials: int = 100,
    train_steps: int = 100_000,
    seed: int = 42,
) -> dict:
    tf_train = _load_all_timeframes("train")
    tf_val = _load_all_timeframes("val")

    if not tf_train or not tf_val:
        raise ValueError("Could not load training or validation data. Please prepare data first.")

    all_features = {tf: [c for c in df.columns if c.startswith('f_')] for tf, df in tf_train.items()}

    def objective(trial: optuna.trial.Trial):
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)

        # Feature selection via individual binary masks
        features = {}
        for tf, feature_list in all_features.items():
            selected = []
            for feat in feature_list:
                if trial.suggest_int(f"use_{tf}_{feat}", 0, 1):
                    selected.append(feat)
            if not selected:
                selected = feature_list[:1]  # Ensure at least 1 feature to avoid error
            features[tf] = selected
        trial.set_user_attr("features", features)

        config = {
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "n_epochs": trial.suggest_int("n_epochs", 5, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["cosine", "linear"]),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.05, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.0, 1.0),
            "clip_range": trial.suggest_float("clip_range", 0.05, 0.5),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "net_arch": [trial.suggest_categorical("net_arch", [64, 128, 256, 512])],
            "trend_dim": trial.suggest_categorical("trend_dim", [2, 3, 4]),
            "aux_coeff": trial.suggest_float("aux_coeff", 0.0, 0.5),
            "lstm_hidden_dim": trial.suggest_categorical("lstm_hidden_dim", [64, 128, 256]),
            "num_lstm_layers": trial.suggest_int("num_lstm_layers", 1, 3),
            "features": features,
        }

        obs_dims = {tf: len(cols) for tf, cols in features.items()}
        env = MultiTimeframeTradingEnv(tf_train, obs_cols=features)
        eval_env = MultiTimeframeTradingEnv(tf_val, obs_cols=features)

        policy = MultiTimeframeLSTMPolicy(
            obs_dims=obs_dims,
            action_dim=env.action_space.n,
            trend_dim=config["trend_dim"],
            aux_coeff=config["aux_coeff"],
            lstm_hidden_dim=config["lstm_hidden_dim"],
            num_lstm_layers=config["num_lstm_layers"],
            mlp_hidden_dims=tuple(config["net_arch"] + [64]),
            device="cuda" if th.cuda.is_available() else "cpu"
        )

        try:
            result = train_with_config(
                env=env,
                eval_env=eval_env,
                policy=policy,
                config=config,
                train_steps=train_steps,
                learning_rate=config["learning_rate"],
                trial=trial
            )

            if result is None:
                print("[warn] Received None result from training.")
                return -1.0, -1.0, 1.0

            sharpe = result.get("sharpe", -1.0)
            ret = result.get("return", -1.0)
            mdd = result.get("mdd", 1.0)

            if any(np.isnan([sharpe, ret, mdd])):
                print("[warn] NaN detected in result metrics.")
                return -1.0, -1.0, 1.0

            return sharpe, ret, mdd

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            print("[Error] Trial failed due to exception:")
            traceback.print_exc()
            return -1.0, -1.0, 1.0  # worst-case defaults

    # Multi-objective optimization
    study = optuna.create_study(
        directions=["maximize", "maximize", "minimize"],
        sampler=optuna.samplers.NSGAIISampler()
    )

    try:
        study.optimize(objective, n_trials=n_trials, timeout=3600*6)
    except KeyboardInterrupt:
        print("Interrupted. Saving partial results...")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    top_trials = study.best_trials[:5]
    top_results = []
    for t in top_trials:
        top_results.append({
            "trial": t.number,
            "sharpe": t.values[0],
            "return": t.values[1],
            "mdd": t.values[2],
            "config": t.params
        })
    with open(os.path.join(os.path.dirname(save_path), "hpo_top5.json"), "w") as f:
        json.dump(top_results, f, indent=2)

    best_trial = study.best_trials[0]
    best_config = best_trial.params
    best_config["features"] = best_trial.user_attrs.get("features", {})

    out = {
        "best_config": best_config,
        "best_score": {
            "sharpe": best_trial.values[0],
            "return": best_trial.values[1],
            "mdd": best_trial.values[2]
        },
        "results": top_results
    }
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] Best config saved to {save_path}")

    return best_config
