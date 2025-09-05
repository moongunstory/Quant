# paths.py — 공통 경로 및 상수 정의

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "raw"))
OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "processed"))
os.makedirs(OUT_DIR, exist_ok=True)

# Timeframes
TIMEFRAMES = ["5m", "15m", "1h", "4h", "btc1h"]
ETH_TIMEFRAMES = ["5m", "15m", "1h", "4h"]
BASE_INTERVAL = "5m"

# Feature selection settings
FEATURE_SEARCH = True
RANDOM_STATE = 72
TOP_K_PER_TF = {"5m": 128, "15m": 128, "1h": 96, "4h": 64}
TF_FOR_SEARCH = ["5m", "15m", "1h", "4h"]

# HPO 확장 관련 설정
HPO_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "hpo"))
os.makedirs(HPO_DIR, exist_ok=True)
HPO_OUT_PREFIX = "feHPO"
HPO_FEATURE_LIST_FMT = os.path.join(HPO_DIR, "feHPO_feature_list_{tf}.json")
HPO_SCALER_PATH_FMT = os.path.join(HPO_DIR, "scaler_hpo_{tf}.joblib")
HPO_EXPAND_WINDOWS = [12, 24, 48, 96]
HPO_MAX_FEATURES_HINT = 2000

# 기본 결과 경로 포맷
FEATURE_LIST_PATH_FMT = os.path.join(OUT_DIR, "fe_feature_list_{tf}.json")
SCALER_PATH_FMT = os.path.join(OUT_DIR, "scaler_{tf}.joblib")

# 유지할 기본 열
REF_COLS_CANON = ["Open", "High", "Low", "Close", "Volume", "FundingRate"]
