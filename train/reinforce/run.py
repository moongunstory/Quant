# train/reinforce/run.py


import sys
import json
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import torch
import optuna

from ai_binance.config.paths import HPO_DIR, get_train_parquet_path, MODELS_DIR, OPTUNA_DB_PATH
from ai_binance.train.hpo.core.feature_builder import build_feature_dfs
from ai_binance.train.hpo.run import _merge_features, _prepare_data_for_env

# RL 관련 모듈 임포트
from ai_binance.train.reinforce.config import EnvConfig, TrainingConfig
from ai_binance.train.reinforce.core.crypto_trading_env import CryptoTradingEnv
from ai_binance.train.reinforce.core.sac_lstm_agent import SACLSTMAgent
from ai_binance.train.reinforce.core.sequence_replay_buffer import SequenceReplayBuffer
from ai_binance.train.reinforce.sac.trainer import Trainer


def get_best_hpo_trial_path(symbol: str) -> Path:
    """지정된 심볼에 대한 Optuna DB에서 최고 성능 trial의 파라미터 파일 경로를 찾습니다."""
    study_name = f"feature_and_rl_hpo_{symbol.lower()}"
    storage_path = f"sqlite:///{OPTUNA_DB_PATH}"

    print(f"[INFO] Loading study '{study_name}' from '{storage_path}'")
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_path)
    except KeyError:
        raise RuntimeError(f"Study '{study_name}' not found in the database. Please run HPO first.")

    best_trial = study.best_trial
    print(f"[INFO] Best trial found: #{best_trial.number} with value: {best_trial.value:.4f}")

    trial_file_name = f"trial_{best_trial.number}_params.json"
    hpo_params_path = HPO_DIR / trial_file_name

    if not hpo_params_path.exists():
        raise FileNotFoundError(
            f"Cannot find HPO trial file for best trial: {hpo_params_path}\n"
            f"Please ensure HPO results are saved correctly."
        )
    return hpo_params_path


def main(args):
    print(f"[INFO] Automatically selecting best HPO trial for symbol: {args.symbol}")
    hpo_params_path = get_best_hpo_trial_path(args.symbol)

    print(f"[INFO] Loading HPO results from {hpo_params_path}")
    with open(hpo_params_path, "r") as f:
        params = json.load(f)

    # feature_config, rl_params 분리
    if "feature_config" in params and "rl_params" in params:
        feature_config = params["feature_config"]
        rl_params = params["rl_params"]
    else:
        feature_config = {
            k: v
            for k, v in params.items()
            if k not in ["hidden_dim", "actor_lr", "critic_lr", "gamma", "tau", "alpha"]
        }
        rl_params = {
            k: v
            for k, v in params.items()
            if k in ["hidden_dim", "actor_lr", "critic_lr", "gamma", "tau", "alpha"]
        }

    # 2. 피처 생성 및 데이터 준비
    print(f"[INFO] Building features for symbol: {args.symbol}")
    processed_df = pd.read_parquet(get_train_parquet_path(args.symbol))

    # 기본 OHLCV 데이터 분리 (컬럼명 불일치 수정)
    ohlcv_cols_prefixed = ['timestamp', 'ohlcv_open', 'ohlcv_high', 'ohlcv_low', 'ohlcv_close', 'ohlcv_volume']
    cols_to_keep = [col for col in ohlcv_cols_prefixed if col in processed_df.columns]
    df_ohlcv_prefixed = processed_df[cols_to_keep]
    
    # RL 환경에서 사용할 수 있도록 원래 컬럼명으로 변경
    rename_map = {p: p.replace('ohlcv_', '') for p in cols_to_keep if p != 'timestamp'}
    df_ohlcv = df_ohlcv_prefixed.rename(columns=rename_map)

    # HPO 설정에 맞는 피처 리스트 생성 (접두사 제거 및 True 값 필터링)
    selected_features = [k.replace('use_feature__', '') for k, v in feature_config.items() if k.startswith('use_feature__') and v]

    # HPO 설정에 맞는 피처 데이터 생성
    feature_dfs = build_feature_dfs(args.symbol, selected_features)

    print("[INFO] Merging feature dataframes...")
    merged_df = _merge_features(feature_dfs, df_ohlcv)

    if merged_df.empty:
        print("[ERROR] No data available after merging features. Exiting.")
        return

    print(f"[INFO] Final data shape: {merged_df.shape}")

    # 3. RL 환경 및 에이전트 생성
    print("[INFO] Initializing environment and agent...")
    data_dict = _prepare_data_for_env(merged_df)

    rl_params.setdefault("hidden_dim", 128)
    rl_params.setdefault("gamma", 0.99)
    rl_params.setdefault("tau", 0.01)          # ← soft update 완화
    rl_params.setdefault("actor_lr", 3e-4)     # ← LR 상향/언더피팅 방지
    rl_params.setdefault("critic_lr", 3e-4)
    rl_params.setdefault("alpha", 0.2)

    # 새 학습 세팅 파라미터 전달 (에이전트 쪽에 추가된 인자)
    rl_params.setdefault("use_scheduler", True)
    rl_params.setdefault("eta_min", 3e-5)

    default_training_config = TrainingConfig()
    training_config_kwargs = {}

    if "log_std_min" in rl_params:
        training_config_kwargs["log_std_min"] = rl_params.pop("log_std_min")
    if "log_std_max" in rl_params:
        training_config_kwargs["log_std_max"] = rl_params.pop("log_std_max")
    if "alpha_min" in rl_params:
        training_config_kwargs["alpha_min"] = rl_params.pop("alpha_min")
    if "alpha_max" in rl_params:
        training_config_kwargs["alpha_max"] = rl_params.pop("alpha_max")
    if "clip_grad" in rl_params:
        training_config_kwargs["grad_clip_norm"] = rl_params.pop("clip_grad")
    if "reward_scale" in rl_params:
        training_config_kwargs["reward_scale"] = rl_params.pop("reward_scale")
    if "initial_alpha" in rl_params:
        training_config_kwargs["initial_alpha"] = rl_params.pop("initial_alpha")
    if "fixed_alpha" in rl_params:
        training_config_kwargs["fixed_alpha"] = rl_params.pop("fixed_alpha")
    if "target_entropy_scale" in rl_params:
        te_scale = rl_params.pop("target_entropy_scale")
        if te_scale > 0:
            training_config_kwargs["target_entropy_scale"] = (
                default_training_config.target_entropy_scale * te_scale
            )
        else:
            training_config_kwargs["target_entropy_scale"] = te_scale

    training_config = TrainingConfig(**training_config_kwargs)
    env_config = EnvConfig()

    rl_params["input_dims"] = {k: v.shape[1] for k, v in data_dict.items()}
    rl_params["training_config"] = training_config
    action_dim = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    default_seq_len = env_config.seq_lens.get("ohlcv", max(env_config.seq_lens.values()))
    group_seq_lens = {
        group: env_config.seq_lens.get(group, default_seq_len)
        for group in data_dict
    }

    env = CryptoTradingEnv(data_dict, seq_lens=group_seq_lens, env_config=env_config)
    agent = SACLSTMAgent(action_dim=action_dim, device=device, **rl_params)
    replay_buffer = SequenceReplayBuffer(
        max_size=args.buffer_size,
        input_dims=rl_params["input_dims"],
        action_dim=action_dim,
        seq_lens=group_seq_lens,
        batch_size=args.batch_size,
        burn_in=32,                            # ← 추후 agent update에서 활용 가능
    )

    # 4. 트레이너 생성 및 학습 시작
    print("[INFO] Starting training...")
    trainer_config = {
        "total_steps": args.total_steps,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "save_path": MODELS_DIR / args.run_name,
    }

    trainer = Trainer(agent, env, replay_buffer, trainer_config)
    trainer.train()


if __name__ == "__main__":
    class Args:
        symbol = "ETHUSDT"
        run_name = "sac_v1"
        total_steps = 1_000_000
        buffer_size = 500_000          # 리플레이 다양성↑ (메모리 허용 시)
        learning_starts = 20_000       # 워밍업 확대
        batch_size = 512               # 시퀀스 학습 밀도↑
        log_interval = 1_000
        save_interval = 50_000

    args = Args()
    main(args)

# tensorboard --logdir ai_binance/data/models/logs --port 6006    
# http://localhost:6006/
