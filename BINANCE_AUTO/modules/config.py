import os
from dotenv import load_dotenv


load_dotenv()

# API 키 설정
DUNE_API_KEY = os.getenv("DUNE_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# 타임프레임 설정
TIMEFRAMES = ["5m", "15m", "30m", "1h"]

# 학습 데이터 수집 기간 설정
START_DATE = "2021-01-01"
END_DATE = "2025-05-24"

# 데이터 출력 저장 경로
RAW_DATA_PATH = "data/raw/market_raw_data.csv"
TRAIN_LONG_PATH = "data/label/train_long.csv"
TRAIN_SHORT_PATH = "data/label/train_short.csv"

# 손익절 값 4봉 기준
TP_THRESHOLD = 0.008  # +0.8%
SL_THRESHOLD = 0.008  # -0.8%
LABEL_HORIZON = 4

# 지도 학습 전용 확신도
LGBM_THRESHOLD = 0.6

# 타임프레임별 피처 구성
FEATURE_CATEGORIES_BY_TF = {
    "15m": [
        "rsi", "stochastic_k", "cci", "roc", "mom",
        "macd", "macd_signal", "macd_histogram",
        "ema_20", "ema_50", "sma_20", "sma_50",
        "adx", "atr", "obv", "volume_ratio"
    ],
    "5m": [
        "rsi", "stochastic_k", "macd", "macd_signal",
        "rsi_mean_6", "rsi_std_6",
        "macd_slope_6",
        "stochk_range_6"
    ],
    "30m": [
        "rsi", "macd", "ema_20", "adx"
    ],
    "1h": [
        "rsi", "ema_20", "sma_50", "adx"
    ]
}

# DUNE 쿼리 ID 매핑
DUNE_QUERY_PARTS = {
    'A': '5214958',
    'B': '5215003', 
    'C': '5215021',
    'D': '5215040',
    'E': '5215054',
    'F': '5215063',
    'G': '5215077',
    'H': '5281243',
    'I': '5287851',
    'J': '5287867',
    'K': '5287868',
    'L': '5287869',
    'M': '5287870',
    'N': '5287871',
    'O': '5287872',
    'P': '5287873'
}

