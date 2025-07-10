import os
from dotenv import load_dotenv

load_dotenv()

# === API Keys ===
DUNE_API_KEY = os.getenv("DUNE_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# === Time & Symbol Settings ===
TZ = 'UTC'
TIMEFRAMES = ["5min", "15min", "30min", "1H"] 
AUX_TIMEFRAMES = ["btc", "dune"] # "dune" < 일단 빼버림
TRADE_SYMBOL = "ETHUSDT"
FUTURES_SYMBOL = "ETHUSDT"
FUTURES_LEVERAGE = 5
FUTURES_MARGIN_TYPE = "ISOLATED"
TRADE_BALANCE_RATIO = 0.99

# === Date Range for Data Collection ===
START_DATE = "2021-01-01"
END_DATE = "2025-05-24"

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "market_raw_data.csv")

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

SCALER_PATH = os.path.join(PROJECT_ROOT, "data", "models", "ppo", "scaler.pkl")

PPO_BUFFER_PATHS = {
    "long": os.path.join(PROJECT_ROOT, "data", "buffer", "long_rollout.pkl"),
    "short": os.path.join(PROJECT_ROOT, "data", "buffer", "short_rollout.pkl"),
}

USE_POLICY_FROM_IMITATION = True

# === Cache Directories ===
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
ONCHAIN_CACHE_DIR = os.path.join(CACHE_DIR, "onchain")

# === Labeling Parameters ===
TP_THRESHOLD = 0.01
SL_THRESHOLD = -0.01
LABEL_HORIZON = 8

# === Thresholds ===
LGBM_THRESHOLD = 0.5
LONG_THRESHOLD = 0.685
SHORT_THRESHOLD = 0.685

# === PPO Hyperparameters ===
PPO_CONFIG = {
    "seq_len": 32,
    "hidden_dim": 128,
    "max_steps": 2048,
    "buffer_size": 2048,
    "learning_rate": 3e-5,
    "epochs": 20,
    "batch_size": 256,
    "entropy_coef": 0.02,
    "value_coef": 0.05,
    "clip_eps": 0.2,
    "gamma": 0.99,
    "lambda": 0.95,
    "action_dim": 2,
    "neutral_band_ratio": 0.1,
}

# === Imitation Learning Config ===
IMITATION_CONFIG = {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "value_loss_coef": 0.2,
    "early_stopping_patience": 5,
}

# === Features Per Timeframe ===
FEATURE_CATEGORIES_BY_TF = {
    "5min": [
        "open", "high", "low", "close", "volume", "returns", "high_low_range", "open_close_range",
        "rsi", "stoch_k", "stoch_d", "macd", "macd_signal", "macd_hist", "cci", "roc",
        "sma_20", "sma_50", "ema_20", "ema_50", "adx", "plus_di", "minus_di",
        "atr", "bb_percent_b", "bb_bandwidth", "obv", "volume_ma_20",
        "smoothed_ha_open", "smoothed_ha_close", "smoothed_ha_high", "smoothed_ha_low"
    ],
    "15min": [
        "open", "high", "low", "close", "volume", "returns", "high_low_range", "open_close_range",
        "rsi", "macd", "macd_signal", "macd_hist", "sma_50", "ema_50", "adx", "atr",
        "bb_percent_b", "bb_bandwidth", "obv", "volume_ma_20",
        "smoothed_ha_open", "smoothed_ha_close", "smoothed_ha_high", "smoothed_ha_low"
    ],
    "30min": [
        "open", "high", "low", "close", "volume", "returns", "high_low_range", "open_close_range",
        "rsi", "macd", "macd_signal", "macd_hist", "sma_50", "ema_50", "adx", "atr",
        "bb_percent_b", "bb_bandwidth", "obv", "volume_ma_20",
        "smoothed_ha_open", "smoothed_ha_close", "smoothed_ha_high", "smoothed_ha_low"
    ],
    "1H": [
        "open", "high", "low", "close", "volume", "returns", "high_low_range", "open_close_range",
        "rsi", "macd", "macd_signal", "macd_hist", "sma_50", "ema_50", "adx", "atr",
        "bb_percent_b", "bb_bandwidth", "obv", "volume_ma_20",
        "smoothed_ha_open", "smoothed_ha_close", "smoothed_ha_high", "smoothed_ha_low"
    ],
    "btc": [ # BTC는 별도 처리 (BINANCE_INTERVAL_MAP 불필요)
        "btc_open", "btc_high", "btc_low", "btc_close", "btc_volume",
        "btc_returns", "btc_high_low_range", "btc_open_close_range",
        "btc_rsi", "btc_macd", "btc_macd_signal", "btc_macd_hist",
        "btc_smoothed_ha_open", "btc_smoothed_ha_close", "btc_smoothed_ha_high", "btc_smoothed_ha_low"
    ]
}

# === Dune Queries ===
DUNE_QUERY_PARTS = {
    'A': '5214958', 'B': '5215003', 'C': '5215021', 'D': '5215040',
    'E': '5215054', 'F': '5215063', 'G': '5215077', 'H': '5281243',
    'I': '5287851', 'J': '5287867', 'K': '5287868', 'L': '5287869',
    'M': '5287870', 'N': '5287871', 'O': '5287872', 'P': '5287873'
}

# === Real-time Requirements ===
REQUIRED_CANDLE_COUNTS = {
    "5min": 50,
    "15min": 70,
    "30min": 70,
    "1H": 70
}

BINANCE_INTERVAL_MAP = {
    "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
    "1H": "1h", "2H": "2h", "4H": "4h", "1D": "1d"
}
