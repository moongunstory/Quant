from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..lgbm import ModelMetadata, create_lgbm_classifier, save_model
from .config import DailyConfig


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
    """
    model = create_lgbm_classifier(num_classes=3)
    model.fit(X, y)

    model_id = f"{as_of_date.date()}_{horizon_days}d"

    os.makedirs(cfg.model_dir, exist_ok=True)
    model_path = os.path.join(
        cfg.model_dir, f"lgbm_cls_{horizon_days}d_{as_of_date.date()}.pkl"
    )

    meta = ModelMetadata(
        feature_names=feature_names,
        target=f"label_{horizon_hours}h",
        task_type="classification",
        num_classes=3,
        return_horizon=horizon_hours,
        return_threshold=cfg.threshold,
        horizon_days=horizon_days,
    )
    save_model(model, meta, model_path)

    return model, model_id
