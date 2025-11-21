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


def create_lgbm_classifier(num_classes: int) -> lgb.LGBMClassifier:
    """탭ular 데이터용 기본 LightGBM 분류기."""
    return lgb.LGBMClassifier(
        n_estimators=300,          # 500 → 300 정도로 줄여도 충분
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multiclass" if num_classes > 2 else "binary",
        num_class=num_classes if num_classes > 2 else None,
        min_data_in_leaf=50,      # 너무 작은 리프 방지 → 쓸데없는 분기 감소
        verbose=-1,               # LightGBM 내부 로그 깔끔하게
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
