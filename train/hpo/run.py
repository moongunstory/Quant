# train/hpo/run.py
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import optuna
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from ai_binance.config.paths import (
    HPO_DIR,  # 베이스만 import, 나머지는 런타임에 버전별로 생성
    ensure_hpo_version_artifacts,
    get_train_parquet_path,
    get_validation_parquet_path,
    list_hpo_versions,
    set_latest_hpo_version,
)
from ai_binance.train.hpo.core.feature_selector import select_features_from_trial
from ai_binance.train.reinforce.config import EnvConfig, TrainingConfig
from ai_binance.train.reinforce.core.crypto_trading_env import CryptoTradingEnv
from ai_binance.train.reinforce.core.sac_lstm_agent import SACLSTMAgent
from ai_binance.train.reinforce.core.sequence_replay_buffer import SequenceReplayBuffer

# === 실행마다 버전 자동 분리 토글 ===
SPLIT_RUN = True  # True: 새 버전(1,2,3,...) 생성 / False: 가장 최근 버전에 이어서


def _safe_replace_symlink(link_path: Path, target: Path) -> None:
    """Create or replace a symlink so that external tools can find latest artifacts."""

    try:
        if link_path.exists() or link_path.is_symlink():
            if link_path.is_dir() and not link_path.is_symlink():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        link_path.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        # 환경이 심볼릭 링크를 지원하지 않는 경우에는 패스(필수 동작 아님)
        pass


def _resolve_hpo_paths(split_run: bool):
    base = Path(HPO_DIR)
    base.mkdir(parents=True, exist_ok=True)
    nums = list_hpo_versions()
    if split_run or not nums:
        version = (nums[-1] + 1) if nums else 1
    else:
        version = nums[-1]

    version_dir, db_dir, logs_dir, params_dir = ensure_hpo_version_artifacts(version)
    db_path = db_dir / "optuna_feature_hpo.db"

    # 최신 버전 포인터 갱신
    set_latest_hpo_version(version)
    _safe_replace_symlink(base / "latest_logs", logs_dir)
    _safe_replace_symlink(base / "latest_params", params_dir)
    _safe_replace_symlink(base / "optuna_feature_hpo.db", db_path)

    return version, version_dir, logs_dir, db_path, params_dir


VERSION, HPO_VERSION_DIR, HPO_LOGS_DIR, OPTUNA_DB_PATH, TOP_N_PARAMS_PATH = _resolve_hpo_paths(SPLIT_RUN)

TRAINING_CONFIG = TrainingConfig()
ENV_CONFIG = EnvConfig()

HEALTH_CHECK_INTERVAL = 1_000
HEALTH_STAGE1_STEP = 12_000      # 학습 업데이트 시작(learning_starts=10k) 이후로 프룬 지연
HEALTH_STAGE2_STEP = 22_000
CRITIC_LOSS_LIMIT = 0.35         # 초반 수렴 여지
PRUNER_ENABLE_STEP = 15_000      # Optuna should_prune() 호출 가드

# 항상 포함할 비용(펀딩) 컬럼 선택: trial이 선택 안 해도 env로 전달되게
def _symbol_suffixes(symbol: str | None) -> tuple[str, str]:
    if symbol == "ethusdt":
        return "_eth", "_btc"
    return "_btc", "_eth"


def _order_funding_columns(cols: Sequence[str], sym_suffix: str) -> list[str]:
    """Ensure the raw funding rate is the first column consumed by the env."""

    preferred = [
        f"funding_fundingRate{sym_suffix}",
        f"funding_rate{sym_suffix}",
        f"fundingFundingRate{sym_suffix}",
        "funding_fundingRate",
        "funding_rate",
    ]

    ordered: list[str] = []
    seen: set[str] = set()
    for candidate in preferred:
        if candidate in cols and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)

    ordered.extend([c for c in cols if c not in seen])
    return ordered


def _mandatory_cost_cols(df: pd.DataFrame, symbol: str) -> list[str]:
    sym_suffix, _ = _symbol_suffixes(symbol)
    cols = [c for c in df.columns if c.startswith("funding") and c.endswith(sym_suffix)]
    if not cols:
        return []
    return _order_funding_columns(cols, sym_suffix)


def _resolve_seq_lens(data_dict: dict[str, np.ndarray], env_config: EnvConfig) -> dict[str, int]:
    """Derive sequence lengths for the groups present in ``data_dict``."""

    base = env_config.seq_lens
    default_len = base.get("ohlcv", max(base.values()))
    return {group: base.get(group, default_len) for group in data_dict}


def _merge_features(dfs: dict[str, pd.DataFrame], ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for frame in dfs.values():
        if "timestamp" not in frame.columns:
            continue
        cleaned = frame.sort_values("timestamp").dropna(subset=["timestamp"])
        frames.append(cleaned)

    if not frames:
        return pd.DataFrame()

    merged_df = frames[0]
    for frame in frames[1:]:
        merged_df = pd.merge(merged_df, frame, on="timestamp", how="inner")

    final_df = (
        pd.merge(
            ohlcv_df.sort_values("timestamp"),
            merged_df.sort_values("timestamp"),
            on="timestamp",
            how="inner",
        )
        .dropna()
        .reset_index(drop=True)
    )
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


def _prepare_data_for_env(
    df: pd.DataFrame, symbol_hint: str | None = None
) -> dict[str, np.ndarray]:
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
    sym_suffix, other_suffix = _symbol_suffixes(symbol_hint)

    main_symbol_features = [c for c in df.columns if c.endswith(sym_suffix) and c not in used_cols]
    other_symbol_features = [c for c in df.columns if c.endswith(other_suffix) and c not in used_cols]
    
    for group in ['funding', 'dune', 'index']:
        group_cols = [c for c in main_symbol_features if c.startswith(group)]
        if group_cols:
            if group == 'funding':
                group_cols = _order_funding_columns(group_cols, sym_suffix)
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

    reward_scale = float(getattr(env, "reward_scale", 1.0))
    log_returns = rets if reward_scale == 0 else rets / reward_scale
    step_returns = np.expm1(log_returns)
    values = np.cumprod(1.0 + step_returns)
    values = np.insert(values, 0, 1.0)  # start NAV=1.0

    returns = step_returns
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
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
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

    sampler = optuna.samplers.TPESampler(
        consider_endpoints=True,
        multivariate=True,
        group=True,
        seed=42,
    )
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=HEALTH_STAGE1_STEP,
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

    latest_logs_pointer = HPO_DIR / "latest_logs"
    print(
        f"[HPO] version={VERSION} dir={HPO_VERSION_DIR} db={OPTUNA_DB_PATH} study={study.study_name}"
    )
    print(f"[HPO] TensorBoard logs: tensorboard --logdir {latest_logs_pointer}")

    shared_cols = set(train_df.columns) & set(val_df.columns)
    if "timestamp" not in shared_cols:
        raise RuntimeError("Both train and validation dataframes must include a 'timestamp' column.")

    available_features = sorted(c for c in shared_cols if c != "timestamp")

    def objective(trial):
        # 재현성
        import random
        seed = 1000 + trial.number
        np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
        
        # trial별 텐서보드 로그 (버전/logs/ 아래)
        log_dir = HPO_LOGS_DIR / f"hpo_trial_{trial.number:03d}"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        writer.add_scalar("Meta/trial_number", float(trial.number), 0)

        last_critic_loss: float | None = None
        last_log_std_mean: float | None = None
        last_policy_entropy: float | None = None
        last_metrics: dict[str, float] = {}

        def _log_health_event(message: str, step_idx: int) -> None:
            msg = f"[HPO][Trial {trial.number}] {message}"
            print(msg)
            try:
                writer.add_text("Health/events", msg, int(step_idx))
            except Exception:
                pass

        def _prune(reason: str, step_idx: int) -> None:
            _log_health_event(f"PRUNE step={step_idx}: {reason}", step_idx)
            raise optuna.exceptions.TrialPruned(reason)

        def _run_health_checks(step_idx: int, metrics: dict[str, float] | None) -> None:
            metrics_local = metrics or {}
            if last_critic_loss is not None and step_idx >= HEALTH_STAGE1_STEP:
                if (not np.isfinite(last_critic_loss)) or last_critic_loss > CRITIC_LOSS_LIMIT:
                    _prune(f"critic_loss={last_critic_loss:.4f}", step_idx)
            if step_idx >= HEALTH_STAGE1_STEP:
                trades = metrics_local.get("trades_per_1k")
                if trades is not None:
                    if not np.isfinite(trades):
                        _prune("trades_per_1k not finite", step_idx)
                    if trades < 1.0 or trades > 120.0:
                        _prune(f"trades_per_1k={trades:.2f}", step_idx)
                if last_log_std_mean is not None:
                    if (not np.isfinite(last_log_std_mean)) or not (-3.3 <= last_log_std_mean <= -0.2):
                        _prune(f"log_std_mean={last_log_std_mean:.3f}", step_idx)
            if step_idx >= HEALTH_STAGE2_STEP:
                avg_r = metrics_local.get("avg_R")
                equity_val = metrics_local.get("equity")
                if (
                    avg_r is not None
                    and equity_val is not None
                    and np.isfinite(avg_r)
                    and np.isfinite(equity_val)
                    and avg_r < -0.06
                    and equity_val < 0.90
                ):
                    _prune(f"avg_R={avg_r:.4f}, equity={equity_val:.4f}", step_idx)

        selected_feats = select_features_from_trial(trial, available_features)
        if not selected_feats:
            raise optuna.exceptions.TrialPruned("No features selected.")

        sym_suffix, other_suffix = _symbol_suffixes(symbol)
        base_cols = [
            f"ohlcv_open{sym_suffix}",
            f"ohlcv_high{sym_suffix}",
            f"ohlcv_low{sym_suffix}",
            f"ohlcv_close{sym_suffix}",
            f"ohlcv_open{other_suffix}",
            f"ohlcv_high{other_suffix}",
            f"ohlcv_low{other_suffix}",
            f"ohlcv_close{other_suffix}",
        ]

        # Always include funding columns even if the trial did not select them explicitly.
        mandatory_cols = [c for c in _mandatory_cost_cols(train_df, symbol) if c in shared_cols]
        if len(mandatory_cols) == 0:
            print(f"[HPO][WARN] No funding columns found for {symbol}. Funding cost will be zero during HPO.")

        # Keep order; remove dups
        cols_to_use = ["timestamp"] + base_cols + selected_feats + mandatory_cols
        cols_to_use = [c for c in dict.fromkeys(cols_to_use) if c == "timestamp" or c in shared_cols]

        missing_base = [c for c in base_cols if c not in shared_cols]
        if missing_base:
            raise optuna.exceptions.TrialPruned(
                f"Required OHLCV base columns are missing from datasets: {missing_base}"
            )

        train_merged = train_df.sort_values("timestamp").reset_index(drop=True)[cols_to_use]
        val_merged = val_df.sort_values("timestamp").reset_index(drop=True)[cols_to_use]

        if train_merged.empty or val_merged.empty:
            raise optuna.exceptions.TrialPruned("No data available after feature merge.")

        total_train_len = len(train_merged)
        total_val_len = len(val_merged)
        max_seq_len = max(ENV_CONFIG.seq_lens.values())
        min_train_len = hpo_train_steps + max_seq_len
        min_eval_len = hpo_eval_steps + max_seq_len

        if total_train_len < min_train_len:
            raise optuna.exceptions.TrialPruned("Training window shorter than requested training horizon.")
        if total_val_len < min_eval_len:
            raise optuna.exceptions.TrialPruned("Validation window shorter than requested evaluation horizon.")

        # ----- 여기서부터: data_dict 구성 시 OHLCV 4열 선두 배치 -----
        train_data_dict = _prepare_data_for_env(train_merged, symbol_hint=symbol)
        eval_data_dict  = _prepare_data_for_env(val_merged,  symbol_hint=symbol)

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

        if not (0.0 < th_close < th_open < th_flip < 1.0):
            _prune("invalid hysteresis", 0)

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

        action_dim = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        group_seq_lens = _resolve_seq_lens(train_data_dict, ENV_CONFIG)
        max_seq_from_groups = max(group_seq_lens.values())
        if len(train_merged) - max_seq_from_groups < hpo_train_steps:
            raise optuna.exceptions.TrialPruned("Training window shorter than training horizon.")
        if len(val_merged) - max_seq_from_groups < hpo_eval_steps:
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
        max_available_steps = len(train_merged) - max_seq_from_groups
        if learning_starts >= hpo_train_steps:
            learning_starts = max(group_seq_lens.get("ohlcv", 1), hpo_train_steps // 2)
        learning_starts = min(learning_starts, max_available_steps - 1)
        learning_starts = max(0, learning_starts)
        writer.add_scalar("Meta/learning_starts", float(learning_starts), 0)

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
                metrics_snapshot = None
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
                            last_critic_loss = float(critic_val)  # 프루닝 판단은 헬스체크에서만
                        if "log_std_mean" in losses:
                            last_log_std_mean = float(losses["log_std_mean"])
                        if "policy_entropy" in losses:
                            last_policy_entropy = float(losses["policy_entropy"])

                if step % 200 == 0 and hasattr(train_env, "tb_metrics"):
                    metrics_snapshot = train_env.tb_metrics()
                    last_metrics = metrics_snapshot
                    for k, v in metrics_snapshot.items():
                        writer.add_scalar(f"Trade/{k}", v, step)

                if step and step % HEALTH_CHECK_INTERVAL == 0:
                    if metrics_snapshot is None and hasattr(train_env, "tb_metrics"):
                        metrics_snapshot = train_env.tb_metrics()
                        last_metrics = metrics_snapshot
                    _run_health_checks(step, last_metrics)
                    if last_metrics:
                        equity_for_report = float(last_metrics.get("equity", 0.0))
                        if np.isfinite(equity_for_report):
                            trial.report(equity_for_report, step)
                            if step >= PRUNER_ENABLE_STEP and trial.should_prune():
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

    eth_train = pd.read_parquet(get_train_parquet_path("ethusdt")).sort_values("timestamp")
    eth_val = pd.read_parquet(get_validation_parquet_path("ethusdt")).sort_values("timestamp")

    btc_train = pd.read_parquet(get_train_parquet_path("btcusdt")).sort_values("timestamp")
    btc_val = pd.read_parquet(get_validation_parquet_path("btcusdt")).sort_values("timestamp")

    train_df = pd.merge(eth_train, btc_train, on="timestamp", suffixes=('_eth', '_btc'))
    val_df = pd.merge(eth_val, btc_val, on="timestamp", suffixes=('_eth', '_btc'))

    study = run_hpo(
        symbol=symbol,
        train_df=train_df.reset_index(drop=True),
        val_df=val_df.reset_index(drop=True),
        n_trials=100,
        top_n=5,
        study_name=f"feature_and_rl_hpo_{symbol}",
    )

"""
tensorboard --logdir ai_binance/data/hpo/16/logs

http://localhost:6006/

"""
