import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "raw"))
OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "processed"))
os.makedirs(OUT_DIR, exist_ok=True)

# Timeframes
TIMEFRAMES = ["5m", "15m", "1h", "4h", "btc1h"]
ETH_TIMEFRAMES = ["5m", "15m", "1h", "4h"]
BASE_INTERVAL = "5m"

# HPO 확장 관련 설정
HPO_OUT_PREFIX = "feHPO"
HPO_FEATURE_LIST_FMT = os.path.join(OUT_DIR, "feHPO_feature_list_{tf}.json")
HPO_SCALER_PATH_FMT = os.path.join(OUT_DIR, "scaler_hpo_{tf}.joblib")
HPO_EXPAND_WINDOWS = [12, 24, 48, 96]
HPO_MAX_FEATURES_HINT = 2000

# 기본 결과 경로 포맷
FEATURE_LIST_PATH_FMT = os.path.join(OUT_DIR, "fe_feature_list_{tf}.json")
SCALER_PATH_FMT = os.path.join(OUT_DIR, "scaler_{tf}.joblib")

# 유지할 기본 열
REF_COLS_CANON = ["Open", "High", "Low", "Close", "Volume", "FundingRate"]

# (선택) 기술 지표 기반 피처 종류 명시
TA_FEATURE_TYPES = [
    "ema_5", "ema_10", "ema_20", "ema_60", "ema_120",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "stoch_k", "stoch_d", "cci_20",
    "bb_upper", "bb_lower", "bb_mid", "bb_std",
    "ha_open", "ha_close", "ha_high", "ha_low",
    "tenkan_sen", "kijun_sen", "senkou_a", "senkou_b", "chikou_span"
]
