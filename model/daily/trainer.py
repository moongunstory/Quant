from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
from lightgbm import early_stopping

from ..lgbm import ModelMetadata, create_lgbm_classifier, save_model
from .config import DailyConfig


def _make_sample_weight(
    y: np.ndarray,
    cfg: DailyConfig,
) -> np.ndarray:
    """
    y(정답 배열) 기준으로
    - 라벨 비율에 따른 가중치
    - 최근 데이터에 더 큰 가중치를 곱해서
    최종 sample_weight를 만든다.
    """
    n = len(y)
    if n == 0:
        return np.array([], dtype=float)

    # --- 1) 최근 데이터 가중치 (앞: 과거, 뒤: 최근) ---
    if cfg.use_recent_weight and n > 1:
        idx = np.arange(n, dtype=float)
        rel_pos = idx / (n - 1)           # 0.0 ~ 1.0
        # 1.0 ~ 2.0 사이에서, 최근일수록 더 큰 가중치
        recent_w = 1.0 + rel_pos
    else:
        recent_w = np.ones(n, dtype=float)

    # --- 2) 라벨 비율 가중치 (적은 라벨에 더 큰 가중치) ---
    if cfg.use_class_weight:
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        total = float(n)
        class_w_map = {
            cls: total / (n_classes * cnt)
            for cls, cnt in zip(classes, counts)
        }
        label_w = np.array([class_w_map[lab] for lab in y], dtype=float)
    else:
        label_w = np.ones(n, dtype=float)

    # 두 가중치를 곱해서 최종 sample_weight 생성
    return recent_w * label_w


def train_horizon_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    cfg: DailyConfig,
    horizon_days: int,
    horizon_hours: int,
    as_of_date: pd.Timestamp,
) -> tuple[object, str]:
    """
    한 horizon에 대해 LGBM 학습 + 모델 저장까지 처리.

    개선점:
    - 윈도우 안 데이터를 시간 순서대로 학습/검증으로 나눔
    - 최근 데이터와 적은 라벨 쪽에 더 큰 가중치를 줌
    - 검증 성능이 더 이상 좋아지지 않으면 자동으로 학습 중단
    """
    n_samples = X.shape[0]
    if n_samples != len(y):
        raise ValueError(f"X ({n_samples})와 y ({len(y)}) 길이가 다릅니다.")

    # ---- 학습/검증 분할 (시간 순서 기준 앞: 학습, 뒤: 검증) ----
    # val_ratio가 0~0.5 범위를 벗어나면 기본값 0.2 사용
    val_ratio = cfg.val_ratio
    if not (0.0 < val_ratio < 0.5):
        val_ratio = 0.2

    val_size = max(int(n_samples * val_ratio), 1)
    train_size = n_samples - val_size
    if train_size < 1:
        raise ValueError(
            f"검증 셋이 너무 커서 학습 셋이 없습니다. "
            f"n={n_samples}, val_ratio={val_ratio}"
        )

    split_idx = train_size

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    # ---- sample_weight 계산 (최근 + 클래스 불균형) ----
    sample_weight_all = _make_sample_weight(y, cfg)
    sw_train = sample_weight_all[:split_idx]
    sw_val = sample_weight_all[split_idx:]

    # ---- 모델 생성 (HPO 파라미터 사용) ----
    lgbm_params = cfg.get_lgbm_params_for(horizon_days)
    model = create_lgbm_classifier(
        num_classes=3,
        n_estimators=lgbm_params.get("n_estimators", 300),
        learning_rate=lgbm_params.get("learning_rate", 0.05),
        num_leaves=int(lgbm_params.get("num_leaves", 31)),
        max_depth=int(lgbm_params.get("max_depth", -1)),
        subsample=lgbm_params.get("subsample", 0.8),
        colsample_bytree=lgbm_params.get("colsample_bytree", 0.8),
    )

    # ---- 성능 안 좋아지면 자동 멈춤 설정 ----
    callbacks = []
    if cfg.early_stopping_rounds and cfg.early_stopping_rounds > 0:
        callbacks.append(
            early_stopping(
                stopping_rounds=cfg.early_stopping_rounds,
                first_metric_only=True,
            )
        )

    # ---- 학습 실행 ----
    model.fit(
        X_train,
        y_train,
        sample_weight=sw_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[sw_val],
        eval_metric="multi_logloss",
        callbacks=callbacks,
    )

    # ---- 모델 저장 ----
    model_id = f"{as_of_date.date()}_{horizon_days}d"

    os.makedirs(cfg.model_dir, exist_ok=True)
    model_path = os.path.join(
        cfg.model_dir, f"lgbm_cls_{horizon_days}d_{as_of_date.date()}.pkl"
    )

    # horizon별 threshold 기록 (없으면 기본값 사용)
    try:
        threshold_value = cfg.get_threshold_for(horizon_days)
    except AttributeError:
        threshold_value = cfg.threshold

    meta = ModelMetadata(
        feature_names=feature_names,
        target=f"label_{horizon_hours}h",
        task_type="classification",
        num_classes=3,
        return_horizon=horizon_hours,
        return_threshold=threshold_value,
        horizon_days=horizon_days,
    )
    save_model(model, meta, model_path)

    return model, model_id
