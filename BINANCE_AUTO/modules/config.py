import os
from dotenv import load_dotenv


load_dotenv()

# API 키 설정
DUNE_API_KEY = os.getenv("DUNE_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# 모든 스케줄은 UTC 기준으로 처리한다
TZ = 'UTC'

# 타임프레임 설정
TIMEFRAMES = ["5min", "15min", "30min", "1H"]

# 실제 매매에 사용할 심볼 (ETH/USDT 등)
TRADE_SYMBOL = "ETHUSDT"

# 실거래에 사용할 잔고 비율 (예: 99% = 0.99)
TRADE_BALANCE_RATIO = 0.99

# ---------- Futures Settings ----------
# USDT-M Perpetual Futures 심볼 및 레버리지 설정
FUTURES_SYMBOL = "ETHUSDT"
FUTURES_LEVERAGE = 5
FUTURES_MARGIN_TYPE = "ISOLATED"

# 학습 데이터 수집 기간 설정
START_DATE = "2021-01-01"
END_DATE = "2025-05-24"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = "data/raw/market_raw_data.csv"

TRAIN_PICKLE_PATHS = {
    "long": os.path.join(PROJECT_ROOT, "data", "label", "train_long.pkl"),
    "short": os.path.join(PROJECT_ROOT, "data", "label", "train_short.pkl"),
}

LGBM_MODEL_PATHS = {
    "long": os.path.join(PROJECT_ROOT, "data", "models", "lgbm", "lgbm_long.pkl"),
    "short": os.path.join(PROJECT_ROOT, "data", "models", "lgbm", "lgbm_short.pkl"),
}

PPO_IMITATION_MODEL_PATHS = {
    "long": os.path.join(PROJECT_ROOT, "data", "models", "ppo_staging", "long_imitation.pt"),
    "short": os.path.join(PROJECT_ROOT, "data", "models", "ppo_staging", "short_imitation.pt"),
}

VALUE_PRETRAIN_OUTPUT_PATH = {
    "long": os.path.join(PROJECT_ROOT, "data", "models", "ppo_staging", "value_long.pt"),
    "short": os.path.join(PROJECT_ROOT, "data", "models", "ppo_staging", "value_short.pt"),
}

PPO_FINAL_MODEL_PATHS = {
    "long": os.path.join(PROJECT_ROOT, "data", "models", "ppo", "ppo_long.pt"),
    "short": os.path.join(PROJECT_ROOT, "data", "models", "ppo", "ppo_short.pt"),
}

# PPO 실전 강화학습 설정
PPO_BUFFER_PATHS = {
    "long": os.path.join(PROJECT_ROOT, "data", "buffer", "long_rollout.pkl"),
    "short": os.path.join(PROJECT_ROOT, "data", "buffer", "short_rollout.pkl")
}

# 캐시 저장 경로
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")

# 온체인 Dune 결과 저장 디렉토리
ONCHAIN_CACHE_DIR = os.path.join(CACHE_DIR, "onchain")

# 손익절 값 4봉 기준
TP_THRESHOLD = 0.008  # +0.8%
SL_THRESHOLD = -0.008  # -0.8%
LABEL_HORIZON = 4

# 지도 학습 전용 확신도
LGBM_THRESHOLD = 0.6

# 실제 매매 전용 확신도
LONG_THRESHOLD = 0.685
SHORT_THRESHOLD = 0.685

# === PPO 학습 설정 ===
SEQ_LEN = 32               # 시계열 길이 (window size)
HIDDEN_DIM = 128           # LSTM hidden dim
LEARNING_RATE = 3e-4       # PPO optimizer 학습률
PPO_EPOCHS = 5            # 학습 epoch 수 (보통 3~10)
PPO_BATCH_SIZE = 64        # 미니배치 사이즈
PPO_MAX_STEPS = 2048       # 수집할 step 수 (GAE 계산용)
GAMMA = 0.99               # 할인율 (보통 0.99)
LAMBDA = 0.95              # GAE lambda (보통 0.95)
CLIP_EPS = 0.2             # 클리핑 범위 (보통 0.1~0.3)
VALUE_COEF = 0.5           # value loss 가중치
ENTROPY_COEF = 0.01        # entropy 가중치
PPO_BUFFER_SIZE = 256
PPO_INPUT_DIM = 61  # 실전 피처 수 기준

# PPO 모델 공통 설정
WINDOW_SIZE = 32
HIDDEN_DIM = 128         # LSTM hidden state 크기
ACTION_DIM = 2           # 행동 공간 (예: HOLD, ENTER)
LEARNING_RATE = 0.0005   # 학습률
EPOCHS = 10              # 학습 epoch 수
BATCH_SIZE = 64          # 배치 사이즈

# 타임프레임별 피처 구성
FEATURE_CATEGORIES_BY_TF = {
    "15min": [
        "rsi", "stochastic_k", "cci", "roc", "mom",
        "macd", "macd_signal", "macd_histogram",
        "ema_20", "ema_50", "sma_20", "sma_50",
        "adx", "atr", "obv", "volume_ratio"
    ],
    "5min": [
        "rsi", "stochastic_k", "macd", "macd_signal",
        "rsi_mean_6", "rsi_std_6",
        "macd_slope_6",
        "stochk_range_6"
    ],
    "30min": [
        "rsi", "macd", "ema_20", "adx"
    ],
    "1H": [
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

# 실시간 수집용 최소 캔들 확보 수 (지표 계산 안정성 확보 목적)
REQUIRED_CANDLE_COUNTS = {
    "5min": 50,
    "15min": 70,
    "30min": 70,
    "1H": 70
}

# Binance 호환용 인터벌 변환 맵 
BINANCE_INTERVAL_MAP = {
    "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
    "1H": "1h", "2H": "2h", "4H": "4h", "1D": "1d"
}