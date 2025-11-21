from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field
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
class FeatureConfig:
    """Technical indicator parameters - centralized configuration"""
    # RSI
    rsi_period: int = 14

    # Moving averages (in hours for 1h data)
    ma_short: int = 24      # 24 hours (1 day)
    ma_long: int = 168      # 168 hours (7 days)

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # Volatility windows (in hours)
    vol_24h: int = 24
    vol_7d: int = 168

    # OI and LS Ratio z-score windows (in hours)
    oi_zscore_window: int = 720    # 30 days
    ls_zscore_window: int = 720    # 30 days

    # Funding rate windows
    fr_ma_window: int = 24         # 24 hours
    fr_cumsum_window: int = 8      # 8 hours

    # On-chain z-score windows (in days)
    onchain_zscore_window: int = 365

    # Macro z-score windows (in days)
    macro_zscore_window: int = 252  # 1 trading year

    # DVOL z-score windows (in days)
    dvol_zscore_window: int = 180


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

    # HPO integration
    hpo_best_config_path: str = "data/hpo/daily/best/best_config_daily.json"
    use_hpo_params: bool = True  # True이면 HPO 결과를 자동으로 로딩하여 적용
    lgbm_params_map: Optional[Dict[int, Dict]] = None  # Horizon별 LGBM 파라미터

    # Feature engineering configuration
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self):
        """HPO 결과가 있으면 자동으로 로딩하여 파라미터 덮어쓰기"""
        if not self.use_hpo_params:
            return

        best_config_path = Path(self.hpo_best_config_path)
        if not best_config_path.exists():
            print(f"[INFO] HPO 결과 파일 없음: {best_config_path}")
            print(f"[INFO] 기본 파라미터 사용: threshold_map={self.threshold_map}, window_days_map={self.window_days_map}")
            return

        try:
            with open(best_config_path, 'r', encoding='utf-8') as f:
                hpo_configs = json.load(f)

            # Horizon별 threshold/window_days/lgbm_params 맵 업데이트
            self.threshold_map = {}
            self.window_days_map = {}
            self.lgbm_params_map = {}

            for horizon_str, cfg in hpo_configs.items():
                horizon = int(horizon_str)
                self.threshold_map[horizon] = cfg["threshold"]
                self.window_days_map[horizon] = cfg["window_days"]
                self.lgbm_params_map[horizon] = cfg.get("lgbm_params", {})

            print(f"✅ HPO 최적 파라미터 로딩 완료: {best_config_path}")
            print(f"   - Thresholds: {self.threshold_map}")
            print(f"   - Window days: {self.window_days_map}")
            print(f"   - LGBM params loaded for horizons: {list(self.lgbm_params_map.keys())}")

        except Exception as e:
            print(f"⚠️ HPO 결과 로딩 실패: {e}")
            print(f"[INFO] 기본 파라미터 사용")

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

    def get_lgbm_params_for(self, horizon_days: int) -> Dict:
        """해당 horizon(일 기준)에 사용할 LGBM 파라미터 리턴."""
        if self.lgbm_params_map and horizon_days in self.lgbm_params_map:
            return self.lgbm_params_map[horizon_days]
        # 기본 파라미터
        return {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
