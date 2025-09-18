# train/hpo/run.py

import sys
import json
import shutil
from pathlib import Path

# 프로젝트 루트 (gtpbitcoin) 기준 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import optuna
import pandas as pd
import numpy as np
import torch

from ai_binance.config.paths import (
    get_processed_ohlcv_path,
    OPTUNA_DB_PATH,
    HPO_DIR
)
from ai_binance.train.hpo.core.feature_selector import select_features_from_trial
from ai_binance.train.hpo.core.feature_builder import build_feature_dfs
from ai_binance.train.hpo.core.feature_registry import rl_hparams

# --- RL 모델 관련 임포트 ---
from ai_binance.train.reinforce.core.crypto_trading_env import CryptoTradingEnv
from ai_binance.train.reinforce.core.sac_lstm_agent import SACLSTMAgent
from ai_binance.train.reinforce.core.sequence_replay_buffer import SequenceReplayBuffer


# HPO 결과 상위 N개 파라미터 저장 경로
TOP_N_PARAMS_PATH = HPO_DIR


# --- 헬퍼 함수들 ---

def _merge_features(dfs: dict, ohlcv_df: pd.DataFrame):
    """다른 시간 주기의 피처들을 5분 단위로 병합"""
    if not dfs:
        return pd.DataFrame()

    for name in dfs:
        if "timestamp" in dfs[name].columns:
            dfs[name] = dfs[name].sort_values("timestamp").dropna(subset=["timestamp"])

    df_5min_list = [dfs[k] for k in ['ohlcv', 'index'] if k in dfs and not dfs[k].empty]
    df_8hour = dfs.get('funding')
    df_daily = dfs.get('dune')

    if not df_5min_list:
        return pd.DataFrame()
    
    merged_df = df_5min_list[0]
    for i in range(1, len(df_5min_list)):
        merged_df = pd.merge(merged_df, df_5min_list[i], on="timestamp", how="inner")

    if df_8hour is not None and not df_8hour.empty:
        df_8hour_resampled = df_8hour.set_index('timestamp').resample('8h').ffill().reset_index()
        merged_df = pd.merge_asof(merged_df.sort_values('timestamp'), df_8hour_resampled.sort_values('timestamp'), on="timestamp", direction="backward")

    if df_daily is not None and not df_daily.empty:
        merged_df = pd.merge_asof(merged_df.sort_values('timestamp'), df_daily.sort_values('timestamp'), on="timestamp", direction="backward")

    # 최종적으로 원본 OHLCV와 병합하여 가격 데이터 확보
    final_df = pd.merge(ohlcv_df, merged_df, on="timestamp", how="inner").dropna()
    return final_df

def _prepare_data_for_env(df: pd.DataFrame):
    """데이터프레임을 RL 환경이 요구하는 numpy dict 형태로 변환"""
    data_dict = {}
    # 컬럼 이름으로 그룹핑 (예: ohlcv_sma_10, ohlcv_rsi_14 -> ohlcv 그룹)
    for group in ['ohlcv', 'funding', 'dune', 'index']:
        group_cols = [c for c in df.columns if c.startswith(group)]
        if group_cols:
            data_dict[group] = df[group_cols].to_numpy()
    return data_dict

def _evaluate_agent(agent, env, eval_steps=1000, mdd_penalty=1.0):
    """학습된 에이전트를 평가하고 Sharpe - alpha * MDD 점수를 반환"""
    obs = env.reset()
    done = False
    steps = 0

    while not done and steps < eval_steps:
        action = agent.select_action(obs)
        obs, _, done, _ = env.step(action)
        steps += 1

    # 평가 종료 시 포지션 있으면 강제 청산 (테이커 수수료 포함)
    if env.position != 0:
        env.step(action=0.0, is_forced_exit=True)

    # 포트폴리오 가치 시계열
    values = np.array(env.portfolio_history)
    if len(values) < 2:
        return -np.inf  # 너무 짧아서 평가 불가

    # 수익률 시계열
    returns = np.diff(values) / values[:-1]

    # Sharpe Ratio (무위험 수익률 0 가정)
    sharpe = returns.mean() / (returns.std() + 1e-8)

    # Max Drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / (peak + 1e-8)
    max_dd = np.max(drawdown)

    # 복합 score 계산
    score = sharpe - mdd_penalty * max_dd

    return float(score)

# --- HPO 실행 함수 ---

def run_hpo(
    symbol: str,
    df_ohlcv: pd.DataFrame,
    n_trials: int = 50,
    hpo_train_steps: int = 5000, # HPO trial 당 RL 학습 스텝 수
    hpo_eval_steps: int = 1000, # HPO trial 당 RL 평가 스텝 수
    top_n: int = 5,
    **kwargs
):
    study = optuna.create_study(direction="maximize",load_if_exists=True,**kwargs)

    def objective(trial):
        # 1. 피처 및 RL 하이퍼파라미터 제안
        feature_config = select_features_from_trial(trial)
        params = {"input_dims": {}}
        for name, cfg in rl_hparams.items():
            if cfg["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, cfg["choices"])
            else:
                params[name] = trial.suggest_float(name, cfg["min"], cfg["max"], log=cfg.get("log", False))

        # 2. 피처 생성 및 데이터 준비
        feature_dfs = build_feature_dfs(symbol, feature_config)
        merged_df = _merge_features(feature_dfs, df_ohlcv)
        if merged_df.empty or len(merged_df) < (hpo_train_steps + hpo_eval_steps):
            raise optuna.exceptions.TrialPruned("Not enough data after feature generation.")

        # 3. RL 환경을 위한 데이터셋 준비
        train_df = merged_df.iloc[:-(hpo_eval_steps+100)] # 평가 데이터 이전까지
        eval_df = merged_df.iloc[-(hpo_eval_steps+100):]
        
        train_data_dict = _prepare_data_for_env(train_df)
        eval_data_dict = _prepare_data_for_env(eval_df)

        if not train_data_dict or not eval_data_dict:
            raise optuna.exceptions.TrialPruned("No feature groups selected.")

        # 4. RL 환경 및 에이전트 생성
        params["input_dims"] = {k: v.shape[1] for k, v in train_data_dict.items()}
        action_dim = 1 # -1 ~ 1 사이의 포지션 크기
        device = "cuda" if torch.cuda.is_available() else "cpu"
        group_seq_lens = {
            "ohlcv": 48,
            "index": 48,
            "funding": 7,
            "dune": 7
        }

        train_env = CryptoTradingEnv(train_data_dict, seq_lens=group_seq_lens)
        eval_env = CryptoTradingEnv(eval_data_dict, seq_lens=group_seq_lens)
        agent = SACLSTMAgent(action_dim=action_dim, device=device, **params)
        replay_buffer = SequenceReplayBuffer(
            max_size=20000,
            input_dims=params["input_dims"],
            action_dim=action_dim,
            seq_lens=group_seq_lens,
            batch_size=64 
        )

        # 5. 단기 RL 학습 루프
        obs = train_env.reset()
        for step in range(hpo_train_steps):
            action = agent.select_action(obs)
            next_obs, reward, done, _ = train_env.step(action)
            replay_buffer.add(obs, action, reward, next_obs, done)
            if len(replay_buffer) > replay_buffer.batch_size:
                agent.update(replay_buffer, batch_size=replay_buffer.batch_size)
            obs = next_obs
            if done:
                obs = train_env.reset()

        # 6. 평가 및 점수 반환
        score = _evaluate_agent(agent, eval_env, hpo_eval_steps)

        # 7. 상위 N개 파라미터 저장 로직
        # (이전과 동일, 단 피처 대신 파라미터를 저장)
        top_n_trials = study.user_attrs.get('top_n_trials', {})
        worst_in_top_n = min(top_n_trials.values()) if len(top_n_trials) == top_n else -float('inf')

        if score > worst_in_top_n or len(top_n_trials) < top_n:
            print(f"\n🔥 Trial {trial.number} is in top {top_n}! Score: {float(score):.6f}. Saving params...\n")
            save_path = TOP_N_PARAMS_PATH / f"trial_{trial.number}_params.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(trial.params, f, indent=2)

            top_n_trials[trial.number] = score
            if len(top_n_trials) > top_n:
                worst_trial_num = min(top_n_trials, key=top_n_trials.get)
                del top_n_trials[worst_trial_num]
                (TOP_N_PARAMS_PATH / f"trial_{worst_trial_num}_params.json").unlink(missing_ok=True)
            study.set_user_attr('top_n_trials', top_n_trials)

        return score

    # HPO 실행
    study.optimize(objective, n_trials=n_trials)

    # 최종 결과 출력
    print("\n--- HPO Finished ---")
    # ... (이전과 동일)
    return study


if __name__ == "__main__":
    symbol = "ethusdt"
    df_ohlcv = pd.read_parquet(get_processed_ohlcv_path(symbol))

    study = run_hpo(
        symbol=symbol,
        df_ohlcv=df_ohlcv,
        n_trials=100,
        top_n=5,
        study_name=f"feature_and_rl_hpo_{symbol}",
        storage=f"sqlite:///{OPTUNA_DB_PATH}"
    )