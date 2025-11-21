from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Optional


threshold_map = {
    3: 0.004,
    7: 0.006,
    30: 0.015,
    90: 0.03,
}

window_days_map = {
    3: 180,
    7: 360,
    30: 540,
    90: 720,
}

@dataclass
class DailyConfig:
    # 마스터 피처 경로
    master_path: str = "data/processed/master_features_1h.parquet"

    # 저장 경로들
    model_dir: str = "data/models/daily"
    pred_log_path: str = "data/predictions/daily_predictions.parquet"
    report_dir: str = "data/reports"

    # 롤링 윈도우 / horizon
    # window_days: 기본 윈도우 길이 (일 단위)
    window_days: int = 540
    horizons_days: Tuple[int, ...] = (3, 7, 30, 90)

    # horizon별 개별 윈도우 설정 (없으면 window_days 사용)
    # 예: window_days_map = {3: 360, 7: 360, 30: 540, 90: 720}
    window_days_map: Optional[Dict[int, int]] = None

    # 기본 컬럼 설정
    timestamp_col: str = "timestamp"
    close_col: str = "fut_close"   # 마스터 테이블에서 사용할 종가 컬럼

    # 라벨 기준: 수익률이 threshold 이상/이하이면 방향 인정 (기본값)
    threshold: float = 0.005       # 0.5%

    # horizon별 개별 threshold 설정 (없으면 threshold 사용)
    # 예: threshold_map = {3: 0.004, 7: 0.006, 30: 0.015, 90: 0.03}
    threshold_map: Optional[Dict[int, float]] = None

    min_samples: int = 500         # horizon별 최소 학습 샘플 수
    skip_if_exists: bool = True    # 같은 날짜에 이미 학습했으면 스킵

    # 학습 관련 설정
    val_ratio: float = 0.2         # 윈도우 마지막 20%를 검증용으로 사용
    early_stopping_rounds: int = 100
    use_class_weight: bool = True  # 라벨 불균형 보정
    use_recent_weight: bool = True # 최근 데이터에 더 큰 가중치

    # 리포트 파일 저장 여부
    save_report: bool = True

    # ---- 편의 메서드 ----
    def get_window_days_for(self, horizon_days: int) -> int:
        """해당 horizon(일 기준)에 사용할 window_days 리턴."""
        if self.window_days_map and horizon_days in self.window_days_map:
            return int(self.window_days_map[horizon_days])
        return self.window_days

    def get_threshold_for(self, horizon_days: int) -> float:
        """해당 horizon(일 기준)에 사용할 threshold 리턴."""
        if self.threshold_map and horizon_days in self.threshold_map:
            return float(self.threshold_map[horizon_days])
        return self.threshold
