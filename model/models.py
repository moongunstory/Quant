# model/models.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import json
import os

import joblib

try:
    import lightgbm as lgb
except ImportError as e:
    raise ImportError(
        "lightgbm이 필요합니다. 가상환경에서 `pip install lightgbm`으로 설치하세요."
    ) from e


@dataclass
class ModelMetadata:
    feature_names: list[str]
    target: str                 # 예: "label_72h"
    task_type: str              # "classification"
    num_classes: Optional[int]  # 3
    return_horizon: int         # 시간 단위 horizon (예: 72h)
    return_threshold: float     # 라벨링에 쓴 threshold
    horizon_days: int           # 3 / 7 / 30 / 90


def create_lgbm_classifier(
    num_classes: int,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_data_in_leaf: int = 50,
) -> lgb.LGBMClassifier:
    """
    Tabular 데이터용 LightGBM 분류기.
    HPO 결과를 파라미터로 받을 수 있도록 개선됨.
    """
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multiclass" if num_classes > 2 else "binary",
        num_class=num_classes if num_classes > 2 else None,
        min_data_in_leaf=min_data_in_leaf,
        verbose=-1,
        n_jobs=-1,
        random_state=42,
    )


def save_model(model: Any, metadata: ModelMetadata, path: str) -> None:
    """모델(pkl) + 메타데이터(JSON) 같이 저장."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

    meta_path = path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metadata), f, ensure_ascii=False, indent=2)
