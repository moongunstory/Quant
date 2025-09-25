# train/hpo/run.py

import sys
import json
from pathlib import Path
from types import SimpleNamespace

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
from ai_binance.train.reinforce.config import TrainingConfig, EnvConfig

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

TRAINING_CONFIG = TrainingConfig()
ENV_CONFIG = EnvConfig()

from ai_binance.train.hpo.core.feature_selector import select_features_from_trial

from ai_binance.train.reinforce.core.crypto_trading_env import CryptoTradingEnv
from ai_binance.train.reinforce.core.sac_lstm_agent import SACLSTMAgent
from ai_binance.train.reinforce.core.sequence_replay_buffer import SequenceReplayBuffer

HEALTH = SimpleNamespace(
    check_interval=1_000,
    stage1_step=8_000,
    stage2_step=22_000,
    prune_min_step=15_000,
    critic_loss_limit=0.2,
    trades_bounds=(3.0, 120.0),
    log_std_bounds=(-3.2, -0.3),
    drawdown_guard=(-0.06, 0.90),
)

# 항상 포함할 비용(펀딩) 컬럼 선택: trial이 선택 안 해도 env로 전달되게
def _mandatory_cost_cols(df: pd.DataFrame, symbol: str):
    sym_suffix = "_eth" if symbol == "ethusdt" else "_btc"
    return [c for c in df.columns if c.startswith("funding") and c.endswith(sym_suffix)]


def _resolve_seq_lens(data_dict: dict[str, np.ndarray], env_config: EnvConfig) -> dict[str, int]:
    """Derive sequence lengths for the groups present in ``data_dict``."""

    base = env_config.seq_lens
    default_len = base.get("ohlcv", max(base.values()))
    return {group: base.get(group, default_len) for group in data_dict}


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


def _evaluate_agent(
    agent,
    env,
    eval_steps: int = 1000,
    mdd_penalty: float | None = None,
    risk_free_rate: float | None = None,
    training_config: TrainingConfig | None = None,
):
    config = training_config or TRAINING_CONFIG
    penalty = config.evaluation_mdd_penalty if mdd_penalty is None else mdd_penalty
    rf_rate = config.risk_free_rate if risk_free_rate is None else risk_free_rate
    sharpe_weight = config.evaluation_sharpe_weight
    calmar_weight = config.evaluation_calmar_weight
    periods_per_year = config.periods_per_year

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
    if returns.size == 0:
        return -np.inf

    per_period_rf = rf_rate / periods_per_year if periods_per_year else 0.0
    excess_returns = returns - per_period_rf
    vol = returns.std(ddof=0)
    if vol > 0:
        annual_sharpe = (excess_returns.mean() / vol) * np.sqrt(periods_per_year)
    else:
        annual_sharpe = 0.0

    peak = np.maximum.accumulate(values)
    drawdowns = (peak - values) / (peak + 1e-8)
    max_dd = float(drawdowns.max()) if drawdowns.size else 0.0

    periods = returns.size
    if periods > 0 and values[0] > 0 and values[-1] > 0:
        total_return = values[-1] / values[0]
        cagr = total_return ** (periods_per_year / periods) - 1.0 if total_return > 0 else 0.0
    else:
        cagr = 0.0
    calmar_ratio = cagr / max_dd if max_dd > 0 else 0.0

    score = sharpe_weight * annual_sharpe + calmar_weight * calmar_ratio - penalty * max_dd
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
    study_name += "" if study_name.endswith(f"_v{VERSION}") else f"_v{VERSION}"

    # Windows 안전 SQLite URI
    storage_uri = "sqlite:///" + OPTUNA_DB_PATH.as_posix()

    sampler = optuna.samplers.TPESampler(
        consider_endpoints=True,
        multivariate=True,
        group=True,
        seed=42,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=HEALTH.stage1_step,
        max_resource=hpo_train_steps,
        reduction_factor=3,
    )

    study = optuna.create_study(
        direction="maximize",
        load_if_exists=True,
        storage=storage_uri,
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
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

        last_critic_loss: float | None = None
        last_log_std_mean: float | None = None
        def _log(tag: str, message: str, step_idx: int | None = None) -> None:
            msg = f"[HPO][Trial {trial.number}] {message}"
            print(msg)
            if step_idx is not None:
                try:
                    writer.add_text(tag, msg, int(step_idx))
                except Exception:
                    pass

        def _prune(reason: str, step_idx: int) -> None:
            _log("Health/events", f"PRUNE step={step_idx}: {reason}", step_idx)
            raise optuna.exceptions.TrialPruned(reason)

        def _run_health_checks(step_idx: int, metrics: dict[str, float] | None) -> None:
            metrics_local = metrics or {}

            if last_critic_loss is not None:
                critic_val = float(last_critic_loss)
                if (not np.isfinite(critic_val)) or critic_val > HEALTH.critic_loss_limit:
                    _prune(f"critic_loss={critic_val:.4f}", step_idx)

            if step_idx >= HEALTH.stage1_step:
                trades = metrics_local.get("trades_per_1k")
                if trades is not None:
                    low, high = HEALTH.trades_bounds
                    if (not np.isfinite(trades)) or not (low <= trades <= high):
                        _prune(f"trades_per_1k={trades}", step_idx)

                if last_log_std_mean is not None:
                    log_std = float(last_log_std_mean)
                    low, high = HEALTH.log_std_bounds
                    if (not np.isfinite(log_std)) or not (low <= log_std <= high):
                        _prune(f"log_std_mean={log_std:.3f}", step_idx)

            if step_idx >= HEALTH.stage2_step:
                avg_r = metrics_local.get("avg_R")
                equity_val = metrics_local.get("equity")
                if (
                    avg_r is not None
                    and equity_val is not None
                    and np.isfinite(avg_r)
                    and np.isfinite(equity_val)
                ):
                    avg_floor, equity_floor = HEALTH.drawdown_guard
                    if avg_r < avg_floor and equity_val < equity_floor:
                        _prune(f"avg_R={avg_r:.4f}, equity={equity_val:.4f}", step_idx)

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
        if merged_df.empty:
            raise optuna.exceptions.TrialPruned("No data available after feature merge.")

        total_len = len(merged_df)
        max_seq_len = max(ENV_CONFIG.seq_lens.values())
        min_train_len = hpo_train_steps + max_seq_len
        min_eval_len = hpo_eval_steps + max_seq_len

        if total_len < (min_train_len + min_eval_len):
            raise optuna.exceptions.TrialPruned("Insufficient data for requested horizons.")

        proposed_split = int(total_len * TRAINING_CONFIG.train_split_ratio)
        lower_bound = min_train_len
        upper_bound = total_len - min_eval_len
        if upper_bound < lower_bound:
            raise optuna.exceptions.TrialPruned("Unable to allocate disjoint train/eval windows.")

        split_idx = min(max(proposed_split, lower_bound), upper_bound)
        train_df = merged_df.iloc[:split_idx]
        eval_df = merged_df.iloc[split_idx:]

        # ----- 여기서부터: data_dict 구성 시 OHLCV 4열 선두 배치 -----
        train_data_dict = _prepare_data_for_env(train_df, symbol_hint=symbol)
        eval_data_dict  = _prepare_data_for_env(eval_df,  symbol_hint=symbol)

        if "ohlcv" not in train_data_dict or "ohlcv" not in eval_data_dict:
            raise optuna.exceptions.TrialPruned("No ohlcv group found.")

        input_dims = {k: v.shape[1] for k, v in train_data_dict.items()}

        agent_cfg = {
            "target_entropy_scale": trial.suggest_float("target_entropy_scale", 0.5, 1.5),
            "init_log_std": trial.suggest_float("init_log_std", -2.5, -0.8),
            "log_std_min": trial.suggest_float("log_std_min", -3.5, -1.5),
            "log_std_max": trial.suggest_float("log_std_max", -1.2, -0.3),
            "alpha_init": trial.suggest_float("alpha_init", 0.10, 0.30),
            "alpha_lr": trial.suggest_categorical("alpha_lr", [1e-4, 3e-4]),
            "actor_lr": trial.suggest_categorical("actor_lr", [1e-4, 3e-4]),
            "critic_lr": trial.suggest_categorical("critic_lr", [1e-4, 3e-4]),
            "tau": trial.suggest_categorical("tau", [0.005, 0.01, 0.02]),
            "use_scheduler": False,
        }

        margin = 0.02
        th_open = trial.suggest_float("th_open", 0.25, 0.60)
        close_high = min(0.30, th_open - margin)
        if close_high <= 0.10:
            _prune("invalid hysteresis close bounds", 0)
        th_close = trial.suggest_float("th_close", 0.10, close_high)
        flip_low = max(0.45, th_open + margin)
        if flip_low >= 0.85:
            _prune("invalid hysteresis flip bounds", 0)
        th_flip = trial.suggest_float("th_flip", flip_low, 0.85)

        env_cfg = {
            "th_open": th_open,
            "th_close": th_close,
            "th_flip": th_flip,
            "min_hold_bars": trial.suggest_int("min_hold_bars", 6, 14),
            "turnover_penalty": trial.suggest_float("turnover_penalty", 0.0000, 0.0200),
            "flip_penalty": trial.suggest_float("flip_penalty", 0.0000, 0.0050),
            "reward_scale": trial.suggest_categorical("reward_scale", [100, 300, 1000]),
            "idle_penalty": 0.0,
        }

        if agent_cfg["critic_lr"] < 5e-5 or agent_cfg["actor_lr"] < 5e-5:
            raise optuna.exceptions.TrialPruned("lr too low")

        if not (0.0 < env_cfg["th_close"] < env_cfg["th_open"] < env_cfg["th_flip"] < 1.0):
            _prune("invalid hysteresis", 0)
        if not (agent_cfg["log_std_min"] < agent_cfg["log_std_max"]):
            _prune("invalid log_std bounds", 0)

        action_dim = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        group_seq_lens = _resolve_seq_lens(train_data_dict, ENV_CONFIG)
        max_seq_from_groups = max(group_seq_lens.values())
        if len(train_df) - max_seq_from_groups < hpo_train_steps:
            raise optuna.exceptions.TrialPruned("Training window shorter than training horizon.")
        if len(eval_df) - max_seq_from_groups < hpo_eval_steps:
            raise optuna.exceptions.TrialPruned("Evaluation window shorter than evaluation horizon.")

        # --- Env: HL 인덱스 반드시 지정 + 채터링 억제 옵션 ---
        env_kwargs = dict(
            seq_lens=group_seq_lens,
            env_config=ENV_CONFIG,
            enforce_hl=True,
            cfg=env_cfg,
        )
        train_env = CryptoTradingEnv(train_data_dict, **env_kwargs)
        eval_env = CryptoTradingEnv(eval_data_dict, **env_kwargs)

        agent = SACLSTMAgent(
            input_dims=input_dims,
            action_dim=action_dim,
            device=device,
            total_steps=hpo_train_steps,
            use_scheduler=False,           # ← 고정
            training_config=TRAINING_CONFIG,
            cfg={**agent_cfg, "total_steps": hpo_train_steps},
        )

        # HPO(5k) 기준: 업데이트 구간을 확실히 확보
        hpo_batch_size = TRAINING_CONFIG.hpo_batch_size
        learning_starts = max(10_000, int(0.2 * hpo_train_steps))
        learning_starts = max(learning_starts, hpo_batch_size)
        max_available_steps = len(train_df) - max_seq_from_groups
        if learning_starts >= hpo_train_steps:
            learning_starts = max(group_seq_lens.get("ohlcv", 1), hpo_train_steps // 2)
        learning_starts = min(learning_starts, max_available_steps - 1)
        learning_starts = max(0, learning_starts)

        replay_buffer = SequenceReplayBuffer(
            max_size=100_000,
            input_dims=input_dims,
            action_dim=action_dim,
            seq_lens=group_seq_lens,
            batch_size=hpo_batch_size
        )

        obs = train_env.reset()

        try:
            for step in range(hpo_train_steps):
                action = agent.select_action(obs)  # 학습은 확률 정책
                next_obs, reward, done, _ = train_env.step(action)
                replay_buffer.add(obs, action, reward, next_obs, done)

                if len(replay_buffer) >= learning_starts:
                    losses = agent.update(replay_buffer, batch_size=replay_buffer.batch_size)
                    if losses:
                        for k, v in losses.items():
                            writer.add_scalar(f"Loss/{k}", v, step)
                        critic_val = losses.get("critic_loss")
                        if critic_val is not None:
                            last_critic_loss = float(critic_val)
                        if "log_std_mean" in losses:
                            last_log_std_mean = float(losses["log_std_mean"])
                if step and step % HEALTH.check_interval == 0:
                    metrics = train_env.tb_metrics() if hasattr(train_env, "tb_metrics") else {}
                    for k, v in metrics.items():
                        writer.add_scalar(f"Trade/{k}", v, step)

                    _run_health_checks(step, metrics)

                    equity_val = metrics.get("equity") if metrics else None
                    if equity_val is not None and np.isfinite(equity_val):
                        trial.report(float(equity_val), step)
                        if step >= HEALTH.prune_min_step and trial.should_prune():
                            _prune("optuna_pruner", step)

                obs = next_obs if not done else train_env.reset()

            score = _evaluate_agent(
                agent,
                eval_env,
                hpo_eval_steps,
                training_config=TRAINING_CONFIG,
            )

            top_n_trials = study.user_attrs.get('top_n_trials', {})
            worst_score = min(top_n_trials.values()) if len(top_n_trials) == top_n else -float("inf")

            if score > worst_score or len(top_n_trials) < top_n:
                print(f"\n🔥 Trial {trial.number} is in top {top_n}! Score: {float(score):.6f}. Saving params...\n")
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

            return score
        finally:
            writer.flush()
            writer.close()

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
