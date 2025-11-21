from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict

import pandas as pd

from ...config import DailyConfig
from .search_space import HpoTrialConfig, make_default_trials_for_horizon
from .objective import evaluate_trial
from .analyzer import select_best_config_from_trials, save_best_configs


def _get_hpo_root_dir(cfg: DailyConfig) -> Path:
    """
    cfg.model_dir (예: data/models/daily) 기준으로
    data/hpo/daily 디렉토리를 추론.
    """
    model_dir = Path(cfg.model_dir).resolve()
    # .../data/models/daily -> parent: models, parent.parent: data
    data_root = model_dir.parent.parent
    hpo_root = data_root / "hpo" / "daily"
    return hpo_root


def _get_trials_path(cfg: DailyConfig, horizon_days: int) -> Path:
    hpo_root = _get_hpo_root_dir(cfg)
    trials_dir = hpo_root / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    return trials_dir / f"trials_{horizon_days}d.parquet"


def _append_trials(path: Path, records: List[Dict]) -> pd.DataFrame:
    df_new = pd.DataFrame.from_records(records)
    if path.exists():
        df_old = pd.read_parquet(path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        # 중복 trial 제거 (동일 hyperparam 조합이 여러 번 있으면 마지막만 남김)
        key_cols = [c for c in df_all.columns if c.startswith("lgbm_")] + [
            "horizon_days",
            "window_days",
            "threshold",
            "feature_group",
        ]
        df_all = df_all.drop_duplicates(subset=key_cols, keep="last")
    else:
        df_all = df_new
    df_all.to_parquet(path)
    return df_all


def run_hpo_for_horizon(
    df_master: pd.DataFrame,
    cfg: DailyConfig,
    horizon_days: int,
    trials: Iterable[HpoTrialConfig] | None = None,
    as_of_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    특정 horizon(예: 3d)에 대해 HPO trial들을 순차 실행하고
    trials parquet를 갱신한 뒤, 전체 trials DataFrame을 반환.
    """
    if trials is None:
        trials = make_default_trials_for_horizon(cfg, horizon_days)

    records: List[Dict] = []
    for trial in trials:
        rec, _info = evaluate_trial(
            df_master=df_master,
            base_cfg=cfg,
            trial=trial,
            as_of_ts=as_of_ts,
        )
        records.append(rec)

    trials_path = _get_trials_path(cfg, horizon_days)
    df_all = _append_trials(trials_path, records)
    return df_all


def run_hpo_for_all_horizons(
    df_master: pd.DataFrame,
    cfg: DailyConfig,
    horizons: Iterable[int] | None = None,
    as_of_ts: pd.Timestamp | None = None,
) -> None:
    """
    cfg.horizons_days (또는 horizons 인자)에 대해
    각 horizon별 HPO를 돌리고, best_config json까지 저장.
    """
    if horizons is None:
        horizons = cfg.horizons_days

    hpo_root = _get_hpo_root_dir(cfg)
    best_dir = hpo_root / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    best_map: Dict[int, Dict] = {}

    for d in horizons:
        df_trials = run_hpo_for_horizon(
            df_master=df_master,
            cfg=cfg,
            horizon_days=d,
            trials=None,
            as_of_ts=as_of_ts,
        )
        # val_score를 기준으로 best config 선택 (objective.py에서 val_score = directional_accuracy로 설정됨)
        best_cfg = select_best_config_from_trials(
            df_trials,
            horizon_days=d,
            metric_col="val_score",  # directional_accuracy 우선, 없으면 accuracy
            mode="max"
        )
        if best_cfg is not None:
            best_map[d] = best_cfg
            # horizon별 개별 json도 같이 저장
            save_best_configs(best_dir, {d: best_cfg})

    # 전체 horizon을 묶은 best_config_daily.json 저장
    if best_map:
        save_best_configs(best_dir, best_map, filename="best_config_daily.json")
