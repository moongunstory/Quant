# train/hpo/core/feature_selector.py

from optuna import Trial
from typing import List
import optuna

def select_features_from_trial(trial: Trial, available_features: List[str]) -> List[str]:
    """
    processed 데이터의 컬럼들 중, trial이 학습 기반으로 일부만 선택하도록 함.
    """
    # 최소/최대 피처 개수 설정 (하이퍼파라미터)
    min_feats = trial.suggest_int("min_num_features", 5, max(5, min(20, len(available_features))))
    max_feats = trial.suggest_int("max_num_features", min_feats, min(len(available_features), 50))

    # 개별 피처별 사용 여부
    use_mask = {}
    for f in available_features:
        use_mask[f] = trial.suggest_categorical(f"use_feature__{f}", [False, True])

    selected = [f for f, use in use_mask.items() if use]

    # 개수가 너무 적으면 부족한 피처들 무작위로 더하기
    if len(selected) < min_feats:
        others = [f for f in available_features if not use_mask[f]]
        needed = min_feats - len(selected)
        selected.extend(others[:needed])

    # 많으면 제한
    if len(selected) > max_feats:
        selected = selected[:max_feats]

    return selected
