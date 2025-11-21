from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DailyConfig:
    # 마스터 피처 경로
    master_path: str = "data/processed/master_features_1h.parquet"

    # 저장 경로들
    model_dir: str = "data/models/daily"
    pred_log_path: str = "data/predictions/daily_predictions.parquet"
    report_dir: str = "data/reports"

    # 롤링 윈도우 / horizon
    window_days: int = 540
    horizons_days: Tuple[int, ...] = (3, 7, 30, 90)

    # 기본 컬럼 설정
    timestamp_col: str = "timestamp"
    close_col: str = "fut_close"   # 마스터 테이블에서 사용할 종가 컬럼

    # 라벨 기준: 수익률이 threshold 이상/이하이면 방향 인정
    threshold: float = 0.005       # 0.5%
    min_samples: int = 500         # horizon별 최소 학습 샘플 수
    skip_if_exists: bool = True    # 같은 날짜에 이미 학습했으면 스킵

    # 리포트 파일 저장 여부
    save_report: bool = True
