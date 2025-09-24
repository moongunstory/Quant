# train/hpo/run.py

import sys
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

import optuna
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ai_binance.config.paths import (
    get_validation_parquet_path,
    get_train_parquet_path,
    HPO_DIR,  # 베이스만 import, 나머지는 런타임에 버전별로 생성
)

# === 실행마다 버전 자동 분리 토글 ===
SPLIT_RUN = True  # True: 새 버전(1,2,3,...) 생성 / False: 가장 최근 버전에 이어서

def _resolve_hpo_paths(split_run: bool):
    base = Path(HPO_DIR)
    base.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()]
    if split_run or not nums:
        version = (max(nums) + 1) if nums else 1
    else:
        version = max(nums)
    version_dir = base / str(version)
    db_dir      = version_dir / "db"
    logs_dir    = version_dir / "logs"
    params_dir  = version_dir / "params"
    for d in (db_dir, logs_dir, params_dir):
        d.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "optuna_feature_hpo.db"
    return version, version_dir, logs_dir, db_path, params_dir

VERSION, HPO_VERSION_DIR, HPO_LOGS_DIR, OPTUNA_DB_PATH, TOP_N_PARAMS_PATH = _resolve_hpo_paths(SPLIT_RUN)

from ai_binance.train.hpo.core.feature_selector import select_features_from_trial
from ai_binance.train.hpo.core.feature_builder import build_feature_dfs
from ai_binance.train.prepare.process.feature_registry import rl_hparams

from ai_binance.train.reinforce.core.crypto_trading_env import CryptoTradingEnv
from ai_binance.train.reinforce.core.sac_lstm_agent import SACLSTMAgent
from ai_binance.train.reinforce.core.sequence_replay_buffer import SequenceReplayBuffer


# --- OHLCV 인덱스 (env가 참조) ---
CLOSE_IDX = 3  # [open, high, low, close] 순서 기준
HIGH_IDX  = 1
LOW_IDX   = 2

# 항상 포함할 비용(펀딩) 컬럼 선택: trial이 선택 안 해도 env로 전달되게
def _mandatory_cost_cols(df: pd.DataFrame, symbol: str):
    sym_suffix = "_eth" if symbol == "ethusdt" else "_btc"
    return [c for c in df.columns if c.startswith("funding") and c.endswith(sym_suffix)]


def _merge_features(dfs: dict, ohlcv_df: pd.DataFrame):
    if not dfs:
        return pd.DataFrame()

    for name in dfs:
        if "timestamp" in dfs[name].columns:
            dfs[name] = dfs[name].sort_values("timestamp").dropna(subset=["timestamp"])

    df_5min_list = list(dfs.values())
    df_8hour = None
    df_daily = None

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

    final_df = pd.merge(ohlcv_df, merged_df, on="timestamp", how="inner").dropna()
    return final_df


def _pick_first_existing(df: pd.DataFrame, candidates, used):
    """후보 컬럼명 리스트 중 df에 존재하고 아직 사용되지 않은 첫 번째를 고른다."""
    for c in candidates:
        if c in df.columns and c not in used:
            used.add(c)
            return c
    return None


def _build_ohlcv_block(df: pd.DataFrame, symbol_hint: str = None):
    """
    env가 TP/SL 관통을 위해 참조하는 원시 O/H/L/C를 반드시 [O,H,L,C] 순서로 앞에 둔다.
    - 우선순위: ohlcv_open/high/low/close > open/high/low/close > {symbol}_open 등
    - 반환: (columns_list, np.ndarray)
    """
    used = set()
    sym_suffix = "_eth" if symbol_hint == "ethusdt" else "_btc" if symbol_hint == "btcusdt" else ""

    open_cands = [f"ohlcv_open{sym_suffix}", f"open{sym_suffix}", "ohlcv_open", "open"]
    high_cands = [f"ohlcv_high{sym_suffix}", f"high{sym_suffix}", "ohlcv_high", "high"]
    low_cands  = [f"ohlcv_low{sym_suffix}", f"low{sym_suffix}", "ohlcv_low", "low"]
    close_cands= [f"ohlcv_close{sym_suffix}", f"close{sym_suffix}", "ohlcv_close","close"]

    open_col  = _pick_first_existing(df, open_cands, used)
    high_col  = _pick_first_existing(df, high_cands, used)
    low_col   = _pick_first_existing(df, low_cands, used)
    close_col = _pick_first_existing(df, close_cands, used)

    if not all([open_col, high_col, low_col, close_col]):
        raise RuntimeError(
            "원시 OHLC 컬럼을 찾을 수 없습니다. 다음 중 하나의 네이밍으로 준비하세요: "
            "[ohlcv_open/high/low/close] 또는 [open/high/low/close] (심볼 접두 허용)."
        )

    base_cols = [open_col, high_col, low_col, close_col]

    # 추가 ohlcv_* 피처는 뒤에 이어붙임(선두 4개 인덱스는 고정)
    extra_ohlcv_cols = [c for c in df.columns if c.startswith(f"ohlcv_") and c.endswith(sym_suffix) and c not in base_cols]
    cols = base_cols + extra_ohlcv_cols
    return cols, df[cols].to_numpy()


def _prepare_data_for_env(df: pd.DataFrame, symbol_hint: str = None):
    """
    data_dict를 생성. 'ohlcv' 그룹은 반드시 [O,H,L,C]를 선두 4열로 포함.
    나머지 그룹은 접두사별로 수집.
    """
    data_dict = {}

    # --- ohlcv 블록(필수) ---
    ohlcv_cols, ohlcv_np = _build_ohlcv_block(df, symbol_hint=symbol_hint)
    data_dict["ohlcv"] = ohlcv_np

    # --- 나머지 그룹 ---
    used_cols = set(ohlcv_cols)
    sym_suffix = "_eth" if symbol_hint == "ethusdt" else "_btc"
    other_suffix = "_btc" if sym_suffix == "_eth" else "_eth"
    
    main_symbol_features = [c for c in df.columns if c.endswith(sym_suffix) and c not in used_cols]
    other_symbol_features = [c for c in df.columns if c.endswith(other_suffix) and c not in used_cols]
    
    for group in ['funding', 'dune', 'index']:
        group_cols = [c for c in main_symbol_features if c.startswith(group)]
        if group_cols:
            data_dict[group] = df[group_cols].to_numpy()
            
    if other_symbol_features:
        data_dict['other'] = df[other_symbol_features].to_numpy()

    return data_dict


def _evaluate_agent(agent, env, eval_steps=1000, mdd_penalty=1.0):
    obs = env.reset()
    done, steps = False, 0
    rets = []

    while not done and steps < eval_steps:
        try:
            action = agent.select_action(obs, deterministic=True)
        except TypeError:
            action = agent.select_action(obs)
        obs, r, done, _ = env.step(action)
        rets.append(float(r))
        steps += 1

    if env.position != 0:
        _, r, _, _ = env.step(0.0, is_forced_exit=True)
        rets.append(float(r))

    # Build NAV from net step returns (includes fee/funding if env.reward is net)
    rets = np.asarray(rets, dtype=np.float64)
    if rets.size == 0:
        return -np.inf
    values = np.cumprod(1.0 + rets)
    values = np.insert(values, 0, 1.0)  # start NAV=1.0

    returns = np.diff(values) / values[:-1]
    sharpe = returns.mean() / (returns.std() + 1e-8)

    peak = np.maximum.accumulate(values)
    max_dd = np.max((peak - values) / (peak + 1e-8))

    score = sharpe - mdd_penalty * max_dd
    return float(score)


def run_hpo(
    symbol: str,
    df_ohlcv: pd.DataFrame,
    n_trials: int = 50,
    hpo_train_steps: int = 50_000,
    hpo_eval_steps: int = 5_000,
    top_n: int = 5,
    **kwargs
):
    # 스터디 이름에 버전 접미사 부여
    study_name = kwargs.get("study_name", f"feature_and_rl_hpo_{symbol}")
    if not study_name.endswith(f"_v{VERSION}"):
        study_name = f"{study_name}_v{VERSION}"

    # Windows 안전 SQLite URI
    storage_uri = "sqlite:///" + OPTUNA_DB_PATH.as_posix()

    study = optuna.create_study(
        direction="maximize",
        load_if_exists=True,
        storage=storage_uri,
        study_name=study_name,
    )

    print(f"[HPO] version={VERSION} dir={HPO_VERSION_DIR} db={OPTUNA_DB_PATH} study={study.study_name}")

    available_features = [c for c in df_ohlcv.columns if c != "timestamp"]

    def objective(trial):
        # 재현성
        import random
        seed = 1000 + trial.number
        np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
        
        # trial별 텐서보드 로그 (버전/logs/ 아래)
        log_dir = HPO_LOGS_DIR / f"hpo_trial_{trial.number:03d}"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))

        selected_feats = select_features_from_trial(trial, available_features)
        if not selected_feats:
            raise optuna.exceptions.TrialPruned("No features selected.")

        sym_suffix = "_eth" if symbol == "ethusdt" else "_btc"
        other_suffix = "_btc" if sym_suffix == "_eth" else "_eth"
        base_cols = [f"ohlcv_open{sym_suffix}", f"ohlcv_high{sym_suffix}", f"ohlcv_low{sym_suffix}", f"ohlcv_close{sym_suffix}",
                    f"ohlcv_open{other_suffix}", f"ohlcv_high{other_suffix}", f"ohlcv_low{other_suffix}", f"ohlcv_close{other_suffix}"]

        # >>> NEW: always include funding columns (even if trial didn't select them)
        mandatory_cols = _mandatory_cost_cols(df_ohlcv, symbol)
        if len(mandatory_cols) == 0:
            print(f"[HPO][WARN] No funding columns found for {symbol}. Funding cost will be zero during HPO.")

        # Keep order; remove dups
        cols_to_use = ["timestamp"] + base_cols + selected_feats + mandatory_cols
        cols_to_use = list(dict.fromkeys(cols_to_use))
        merged_df = df_ohlcv[cols_to_use]
        if merged_df.empty or len(merged_df) < (hpo_train_steps + hpo_eval_steps):
            raise optuna.exceptions.TrialPruned("Insufficient data.")

        train_df = merged_df.iloc[:-(hpo_eval_steps + 100)]
        eval_df = merged_df.iloc[-(hpo_eval_steps + 100):]

        # ----- 여기서부터: data_dict 구성 시 OHLCV 4열 선두 배치 -----
        train_data_dict = _prepare_data_for_env(train_df, symbol_hint=symbol)
        eval_data_dict  = _prepare_data_for_env(eval_df,  symbol_hint=symbol)

        if "ohlcv" not in train_data_dict or "ohlcv" not in eval_data_dict:
            raise optuna.exceptions.TrialPruned("No ohlcv group found.")

        params = {"input_dims": {k: v.shape[1] for k, v in train_data_dict.items()}}
        for name, cfg in rl_hparams.items():
            if cfg["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, cfg["choices"])
            else:
                params[name] = trial.suggest_float(name, cfg["min"], cfg["max"], log=cfg.get("log", False))
        
        # guard
        if params.get("critic_lr", 1e-3) < 1e-4 or params.get("actor_lr", 1e-3) < 1e-4:
            raise optuna.exceptions.TrialPruned("lr too low")

        action_dim = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        group_seq_lens = {"ohlcv": 48, "index": 48, "funding": 7, "dune": 7}
        if 'other' in train_data_dict:
            group_seq_lens['other'] = 48

        # --- Env: HL 인덱스 반드시 지정 + 채터링 억제 옵션 ---
        env_kwargs = dict(
            seq_lens=group_seq_lens,
            ohlcv_close_idx=CLOSE_IDX,
            ohlcv_high_idx=HIGH_IDX,
            ohlcv_low_idx=LOW_IDX,
            enforce_hl=True,
            min_hold_bars=6,        # ~30min on 5m bars
            flip_penalty=0.0014,    # ≈ 2*(taker 0.0005 + slippage 0.0002)
        )
        train_env = CryptoTradingEnv(train_data_dict, **env_kwargs)
        eval_env  = CryptoTradingEnv(eval_data_dict,  **env_kwargs)

        agent = SACLSTMAgent(
            action_dim=action_dim,
            device=device,
            total_steps=hpo_train_steps,
            use_scheduler=False,           # ← 고정
            **params
        )
        # 정책 분산 범위 고정(네트워크 기본과 동일하지만 명시)
        if hasattr(agent, "set_log_std_bounds"):
            agent.set_log_std_bounds(-1.2, 0.2)

        # HPO(5k) 기준: 업데이트 구간을 확실히 확보
        hpo_batch_size = 256
        learning_starts = max(10_000, int(0.2 * hpo_train_steps))  # 깊은 버퍼 확보

        replay_buffer = SequenceReplayBuffer(
            max_size=50_000,
            input_dims=params["input_dims"],
            action_dim=action_dim,
            seq_lens=group_seq_lens,
            batch_size=hpo_batch_size
        )

        obs = train_env.reset()
        for step in range(hpo_train_steps):
            action = agent.select_action(obs)  # 학습은 확률 정책
            next_obs, reward, done, _ = train_env.step(action)
            replay_buffer.add(obs, action, reward, next_obs, done)

            # 업데이트 & Loss 로깅
            if len(replay_buffer) >= learning_starts:
                losses = agent.update(replay_buffer, batch_size=replay_buffer.batch_size)
                if losses:
                    for k, v in losses.items():
                        writer.add_scalar(f"Loss/{k}", v, step)

            # (A) 거래 지표 텐서보드 기록
            if step % 200 == 0 and hasattr(train_env, "tb_metrics"):
                m = train_env.tb_metrics()
                for k, v in m.items():
                    writer.add_scalar(f"Trade/{k}", v, step)

            # (B) 타깃 엔트로피 스케줄 (1.0 → 0.7 선형; 에이전트에서 [0.6,1.0]으로 클램프)
            if hasattr(agent, "set_target_entropy_scale"):
                scale = 1.0 - 0.3 * (step / hpo_train_steps)
                agent.set_target_entropy_scale(scale)

            # (C) 프룬(워밍업 이후): 무거래 또는 과도한 턴오버 컷
            if step > learning_starts and step % 1000 == 0 and hasattr(train_env, "tb_metrics"):
                m = train_env.tb_metrics()
                trades_1k = m.get("trades_per_1k", 0.0)
                hold_mean = m.get("holding_bars_mean", 0.0)
                if trades_1k < 20 or hold_mean == 0:
                    raise optuna.exceptions.TrialPruned("no trading")
                if trades_1k > 150:
                    raise optuna.exceptions.TrialPruned("too much turnover")

            obs = next_obs if not done else train_env.reset()

        score = _evaluate_agent(agent, eval_env, hpo_eval_steps)

        top_n_trials = study.user_attrs.get('top_n_trials', {})
        worst_score = min(top_n_trials.values()) if len(top_n_trials) == top_n else -float("inf")

        if score > worst_score or len(top_n_trials) < top_n:
            print(f"\n🔥 Trial {trial.number} is in top {top_n}! Score: {float(score):.6f}. Saving params...\n")
            # top-N 파라미터 저장
            save_path = TOP_N_PARAMS_PATH / f"trial_{trial.number}_params.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(trial.params, f, indent=2)

            top_n_trials[str(trial.number)] = score
            if len(top_n_trials) > top_n:
                worst_trial_key = min(top_n_trials, key=top_n_trials.get)
                del top_n_trials[worst_trial_key]
                (TOP_N_PARAMS_PATH / f"trial_{worst_trial_key}_params.json").unlink(missing_ok=True)
            study.set_user_attr('top_n_trials', top_n_trials)

        writer.close()
        return score

    study.optimize(objective, n_trials=n_trials)

    print("\n--- HPO Finished ---")
    return study


if __name__ == "__main__":
    symbol = "ethusdt"

    eth_train = pd.read_parquet(get_train_parquet_path("ethusdt"))
    eth_val = pd.read_parquet(get_validation_parquet_path("ethusdt"))
    eth_df = pd.concat([eth_train, eth_val]).sort_values("timestamp").reset_index(drop=True)

    btc_train = pd.read_parquet(get_train_parquet_path("btcusdt"))
    btc_val = pd.read_parquet(get_validation_parquet_path("btcusdt"))
    btc_df = pd.concat([btc_train, btc_val]).sort_values("timestamp").reset_index(drop=True)

    # btc_df 컬럼명 충돌을 피하기 위해 merge와 suffix 사용
    df_ohlcv = pd.merge(eth_df, btc_df, on="timestamp", suffixes=('_eth', '_btc'))

    study = run_hpo(
        symbol=symbol,
        df_ohlcv=df_ohlcv,
        n_trials=100,
        top_n=5,
        study_name=f"feature_and_rl_hpo_{symbol}",
    )


"""
tensorboard --logdir ai_binance/data/hpo/16/logs

http://localhost:6006/

"""
