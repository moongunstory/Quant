from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score

from ...config import DailyConfig
from ...dataset import ensure_datetime_index, build_supervised_for_horizon
from ...trainer import _make_sample_weight
from .search_space import HpoTrialConfig


def _make_trial_config(base_cfg: DailyConfig, trial: HpoTrialConfig) -> DailyConfig:
    """
    base_cfg를 복사해서,
    - 해당 horizon에 대한 window_days / threshold만 trial 값으로 덮어쓴 cfg 생성.
    나머지 경로(model_dir, pred_log_path 등)는 그대로 유지.
    """
    cfg = dc_replace(base_cfg)

    # 기존 map을 복사해서 덮어쓴다.
    threshold_map = dict(getattr(cfg, "threshold_map", {}) or {})
    threshold_map[trial.horizon_days] = trial.threshold
    cfg.threshold_map = threshold_map

    window_days_map = dict(getattr(cfg, "window_days_map", {}) or {})
    window_days_map[trial.horizon_days] = trial.window_days
    cfg.window_days_map = window_days_map

    return cfg


def evaluate_trial(
    df_master: pd.DataFrame,
    base_cfg: DailyConfig,
    trial: HpoTrialConfig,
    as_of_ts: pd.Timestamp | None = None,
) -> Tuple[Dict, Dict]:
    """
    한 번의 trial을 실제로 학습/평가해서 결과를 반환.

    반환값:
    - record: 1줄짜리 로그 딕셔너리 (parquet에 바로 넣을 수 있는 형태)
    - info:   부가 정보 (예: feature_names 등, 필요하면 나중에 확장)
    """
    # as_of_ts가 없으면 df_master의 마지막 시점을 기준으로 삼는다.
    df_master = ensure_datetime_index(df_master, base_cfg)
    if as_of_ts is None:
        as_of_ts = df_master.index.max()
    as_of_ts = pd.to_datetime(as_of_ts, utc=True)

    horizon_hours = int(trial.horizon_days * 24)

    # trial 세팅이 반영된 cfg 생성
    cfg = _make_trial_config(base_cfg, trial)

    record: Dict = {
        "horizon_days": trial.horizon_days,
        "window_days": trial.window_days,
        "threshold": float(trial.threshold),
        "feature_group": trial.feature_group,
        "as_of_ts": as_of_ts,
        "status": "ok",
        "val_score": np.nan,
        "metric": "accuracy",
        "n_train": 0,
        "n_val": 0,
    }
    # LGBM 파라미터도 기록용 컬럼으로 풀어준다.
    for k, v in trial.lgbm_params.items():
        record[f"lgbm_{k}"] = v

    try:
        # 학습용 X, y 생성
        X, y, feature_names = build_supervised_for_horizon(
            df=df_master,
            as_of_ts=as_of_ts,
            horizon_hours=horizon_hours,
            cfg=cfg,
        )
    except Exception as e:
        # 데이터 부족 / 라벨 하나만 존재 등은 그냥 실패 trial로 기록
        record["status"] = "fail"
        record["error"] = str(e)
        return record, {}

    n_samples = X.shape[0]
    if n_samples != len(y):
        record["status"] = "fail"
        record["error"] = f"X ({n_samples})와 y ({len(y)}) 길이가 다릅니다."
        return record, {}

    # ---- 학습/검증 분할 (trainer.train_horizon_model과 동일한 룰) ----
    val_ratio = getattr(cfg, "val_ratio", 0.2)
    if not (0.0 < val_ratio < 0.5):
        val_ratio = 0.2

    val_size = max(int(n_samples * val_ratio), 1)
    train_size = n_samples - val_size
    if train_size < 1:
        record["status"] = "fail"
        record["error"] = (
            f"검증 셋이 너무 커서 학습 셋이 없습니다. "
            f"n={n_samples}, val_ratio={val_ratio}"
        )
        return record, {}

    split_idx = train_size
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    # ---- sample_weight 계산 (trainer._make_sample_weight 재사용) ----
    sample_weight_all = _make_sample_weight(y, cfg)
    sw_train = sample_weight_all[:split_idx]
    sw_val = sample_weight_all[split_idx:]

    # ---- LightGBM 모델 생성 ----
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "random_state": getattr(cfg, "random_state", 42),
        "n_estimators": trial.lgbm_params.get("n_estimators", 400),
        "learning_rate": trial.lgbm_params.get("learning_rate", 0.05),
        "num_leaves": trial.lgbm_params.get("num_leaves", 31),
        "max_depth": trial.lgbm_params.get("max_depth", -1),
        "subsample": trial.lgbm_params.get("subsample", 0.8),
        "colsample_bytree": trial.lgbm_params.get("colsample_bytree", 0.8),
    }
    model = LGBMClassifier(**params)

    callbacks = []
    es_rounds = getattr(cfg, "early_stopping_rounds", None)
    if es_rounds and es_rounds > 0:
        callbacks.append(
            early_stopping(
                stopping_rounds=es_rounds,
                first_metric_only=True,
            )
        )

    model.fit(
        X_train,
        y_train,
        sample_weight=sw_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[sw_val],
        eval_metric="multi_logloss",
        callbacks=callbacks,
        verbose=False,
    )

    # ---- 검증 점수 계산 ----
    y_pred = model.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))

    record["val_score"] = acc
    record["n_train"] = int(train_size)
    record["n_val"] = int(val_size)

    info: Dict = {
        "feature_names": feature_names,
    }
    return record, info
