import os
import json
import random
import numpy as np
import torch as th
import pandas as pd
from typing import Dict, Any
import optuna

from ai_binance.train.reinforce.env import MultiTimeframeTradingEnv
from ai_binance.train.reinforce.policy import MultiTimeframeLSTMPolicy
from ai_binance.train.reinforce.ppo import train_with_config

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
ETH_TFS = ["5m", "15m", "1h", "4h"]
FIXED_LR = 3e-4

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

    all_available_features = {tf: [c for c in df.columns if c.startswith('f_')] for tf, df in tf_train.items()}

    def objective(trial: optuna.trial.Trial) -> float:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)

        features = {}
        for tf, available in all_available_features.items():
            n_features = trial.suggest_int(f"n_features_{tf}", 20, len(available))
            features[tf] = random.sample(available, n_features)
        trial.set_user_attr("features", features)

        # Hyperparameter search space
        config = {
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
            "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
            "n_epochs": trial.suggest_int("n_epochs", 5, 20),
            "ent_coef": trial.suggest_float("ent_coef", 1e-8, 0.05, log=True),
            "vf_coef": trial.suggest_float("vf_coef", 0.3, 0.7),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.9, 0.99),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
            "net_arch": [trial.suggest_categorical("net_arch", [64, 128, 256, 512])],
            "trend_dim": trial.suggest_categorical("trend_dim", [2, 3, 4]),
            "aux_coeff": trial.suggest_float("aux_coeff", 0.0, 0.5),
            "lstm_hidden_dim": trial.suggest_categorical("lstm_hidden_dim", [64, 128, 256]),
            "num_lstm_layers": trial.suggest_int("num_lstm_layers", 1, 3),
            "features": features,
        }

        obs_dims = {tf: len(cols) for tf, cols in config["features"].items()}
        env = MultiTimeframeTradingEnv(tf_train, obs_cols=config["features"])
        eval_env = MultiTimeframeTradingEnv(tf_val, obs_cols=config["features"])

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
                learning_rate=FIXED_LR,
                trial=trial
            )
            return result.get("sharpe", 0.0)
        except optuna.exceptions.TrialPruned as e:
            print(f"Trial pruned: {e}")
            raise
        except Exception as e:
            print(f"[Error] Trial failed with exception: {e}")
            return -1.0

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="maximize", pruner=pruner)

    try:
        study.optimize(objective, n_trials=n_trials, timeout=3600 * 6)
    except KeyboardInterrupt:
        print("HPO interrupted. Saving current results...")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save top 5 trials
    top5_trials = sorted(study.trials, key=lambda t: t.value or -1, reverse=True)[:5]
    top5_results = [{
        "trial": t.number,
        "sharpe": t.value,
        "config": t.params
    } for t in top5_trials]
    with open(os.path.join(os.path.dirname(save_path), "hpo_top5.json"), "w", encoding="utf-8") as f:
        json.dump(top5_results, f, indent=2, ensure_ascii=False)
    print(f"Top 5 trials saved.")

    # Save best trial
    best_trial = study.best_trial
    best_config = best_trial.params
    best_config["features"] = best_trial.user_attrs.get("features", {})

    out = {
        "best_config": best_config,
        "best_score": best_trial.value,
        "results": [{"trial": t.number, "sharpe": t.value, "config": t.params} for t in study.trials]
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Best trial saved to {save_path}")

    return best_config
