from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..config import DailyConfig


@dataclass
class HpoTrialConfig:
    """
    HPO에서 한 번의 시도(trial)을 표현하는 설정 묶음.
    - horizon_days: 예측 대상 기간 (3 / 7 / 30 / 90 등)
    - window_days:  학습에 사용할 과거 구간 길이
    - threshold:    상승/하락을 나누는 수익률 기준
    - lgbm_params:  LightGBM 관련 하이퍼파라미터 딕셔너리
    - feature_group: 나중에 피처 그룹을 바꾸고 싶을 때를 위한 태그(당장은 "all" 고정)
    """
    horizon_days: int
    window_days: int
    threshold: float
    lgbm_params: Dict[str, float]
    feature_group: str = "all"


def _default_threshold_candidates(cfg: DailyConfig, horizon_days: int) -> List[float]:
    """
    주어진 horizon에서 시도해 볼 threshold 후보 목록.
    - 기본 threshold 주변으로 0.5배, 1배, 1.5배 세 개 정도만.
    """
    base = cfg.get_threshold_for(horizon_days)
    base = max(float(base), 0.0001)  # 최소 0보다 조금 크게
    cands = sorted({round(base * f, 6) for f in (0.5, 1.0, 1.5)})
    return cands


def _default_window_days_candidates(cfg: DailyConfig, horizon_days: int) -> List[int]:
    """
    주어진 horizon에서 시도해 볼 window_days 후보.
    - 기본 window_days의 0.5배, 1배, 2배 정도.
    - 너무 길어지는 건 720일 정도에서 자른다.
    """
    base = cfg.get_window_days_for(horizon_days)
    base = max(int(base), 30)  # 최소 30일
    half = max(base // 2, 30)
    double = min(base * 2, 720)
    cands = sorted({half, base, double})
    return cands


def _default_lgbm_param_grid() -> List[Dict[str, float]]:
    """
    간단한 그리드만 제공. 필요하면 여기 숫자만 조금씩 바꾸면 됨.
    """
    learning_rates = [0.03, 0.05, 0.1]
    num_leaves_list = [31, 63]
    max_depth_list = [-1, 6, 10]
    n_estimators_list = [200, 400, 800]

    grid: List[Dict[str, float]] = []
    for lr in learning_rates:
        for nl in num_leaves_list:
            for md in max_depth_list:
                for ne in n_estimators_list:
                    grid.append(
                        {
                            "learning_rate": lr,
                            "num_leaves": nl,
                            "max_depth": md,
                            "n_estimators": ne,
                        }
                    )
    return grid


def make_default_trials_for_horizon(
    cfg: DailyConfig,
    horizon_days: int,
    max_trials: int | None = None,
) -> List[HpoTrialConfig]:
    """
    한 horizon(예: 3d)에 대해 기본 HPO 후보(trial) 리스트를 만든다.
    - threshold / window_days / LGBM 파라미터 조합의 곱에서
      앞에서부터 max_trials개만 잘라서 사용.
    """
    thr_cands = _default_threshold_candidates(cfg, horizon_days)
    win_cands = _default_window_days_candidates(cfg, horizon_days)
    lgbm_grid = _default_lgbm_param_grid()

    trials: List[HpoTrialConfig] = []
    for th in thr_cands:
        for wd in win_cands:
            for params in lgbm_grid:
                trials.append(
                    HpoTrialConfig(
                        horizon_days=horizon_days,
                        window_days=wd,
                        threshold=th,
                        lgbm_params=params,
                        feature_group="all",
                    )
                )

    if max_trials is not None and len(trials) > max_trials:
        # 단순히 앞에서부터 잘라 쓰기 (나중에 무작위 샘플링으로 바꿔도 됨)
        trials = trials[:max_trials]

    return trials
