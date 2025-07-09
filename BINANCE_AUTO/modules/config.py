import os
from dotenv import load_dotenv

load_dotenv()

# === API Keys ===
DUNE_API_KEY = os.getenv("DUNE_API_KEY")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# === Dune Analytics Query Parts ===
DUNE_QUERY_PARTS = {
    "onchain_raw_1": "5214958",
    "onchain_raw_2": "5215003",
    "onchain_raw_3": "5215021",
    "onchain_raw_4": "5215040",
    "onchain_raw_5": "5215054",
    "onchain_raw_6": "5215063",
    "onchain_raw_7": "5215077",
    "onchain_raw_8": "5281243",
    "onchain_raw_9": "5287851",
    "onchain_raw_10": "5287867",
    "onchain_raw_11": "5287868",
    "onchain_raw_12": "5287869",
    "onchain_raw_13": "5287870",
    "onchain_raw_14": "5287871",
    "onchain_raw_15": "5287872",
    "onchain_raw_16": "5287873"
}

# === Time & Symbol Settings ===
TZ = 'UTC'

# 📊 ETH 메인 타임프레임들 (Binance interval 매핑 필요)
ETH_TIMEFRAMES = ["1min", "5min", "15min", "1H"]

# 🔧 보조 데이터 활성화 플래그
ENABLE_BTC = True      # BTC 보조 데이터 수집 여부
ENABLE_DUNE = False    # DUNE 온체인 데이터 수집 여부 (일단 비활성화)

# 🔄 하위 호환성을 위한 통합 리스트 (기존 코드 호환용)
TIMEFRAMES = ETH_TIMEFRAMES.copy()
if ENABLE_BTC:
    TIMEFRAMES.append("btc")
if ENABLE_DUNE:
    TIMEFRAMES.append("dune")

AUX_TIMEFRAMES = "btc"
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

TRAIN_PICKLE_PATHS = {
    "long": os.path.join(PROJECT_ROOT, "data", "processed", "train_long.pkl"),
    "short": os.path.join(PROJECT_ROOT, "data", "processed", "train_short.pkl"),
}

VALUE_PRETRAIN_OUTPUT_PATH = {
    "long": os.path.join(PROJECT_ROOT, "data", "models", "ppo_staging", "value_long.pt"),
    "short": os.path.join(PROJECT_ROOT, "data", "models", "ppo_staging", "value_short.pt"),
}

PPO_FINAL_MODEL_PATHS = {
    "long": os.path.join(PROJECT_ROOT, "data", "models", "ppo", "ppo_long.pt"),
    "short": os.path.join(PROJECT_ROOT, "data", "models", "ppo", "ppo_short.pt"),
}

PPO_BUFFER_PATHS = {
    "long": os.path.join(PROJECT_ROOT, "data", "buffer", "long_rollout.pkl"),
    "short": os.path.join(PROJECT_ROOT, "data", "buffer", "short_rollout.pkl"),
}

# === Cache Directories ===
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "cache")
ONCHAIN_CACHE_DIR = os.path.join(CACHE_DIR, "onchain")

# === PPO Hyperparameters ===
PPO_CONFIG = {
    "seq_len": 32,
    "hidden_dim": 128,
    "max_steps": 2048,
    "buffer_size": 2048,
    "learning_rate": 3e-5,
    "epochs": 10,
    "batch_size": 256,
    "entropy_coef": 0.02,
    "value_coef": 1.0,
    "clip_eps": 0.2,
    "gamma": 0.99,
    "lambda": 0.95,
    "action_dim": 1, # Continuous action for liquidation confidence (0.0 to 1.0)
    "min_profit_target": 0.002, # 0.2% profit target for pre-training reward
    "max_loss_tolerance": 0.001, # 0.1% loss tolerance for pre-training reward
    "neutral_band_ratio": 0.1,
    "reward_scaling_factor": 100.0,
}

# === Features Per Timeframe ===
FEATURE_CATEGORIES_BY_TF = {
    "1min": [
        "open", "high", "low", "close", "volume", "returns", "high_low_range", "open_close_range",
        "rsi", "stoch_k", "stoch_d", "macd", "macd_signal", "macd_hist", "cci", "roc",
        "sma_10", "sma_20", "ema_10", "ema_20", "adx", "plus_di", "minus_di",
        "atr", "bb_percent_b", "bb_bandwidth", "obv", "volume_ma_20",
        "smoothed_ha_open", "smoothed_ha_close", "smoothed_ha_high", "smoothed_ha_low"
    ],
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

# === Real-time Requirements ===
REQUIRED_CANDLE_COUNTS = {
    "1min": 70,
    "5min": 70,
    "15min": 70,
    "1H": 70
}

# 📈 ETH 타임프레임만을 위한 Binance interval 매핑
BINANCE_INTERVAL_MAP = {
    "1min": "1m", "3min": "3m", "5min": "5m", "15min": "15m", "30min": "30m",
    "1H": "1h", "2H": "2h", "4H": "4h", "1D": "1d"
    # 주의: "btc", "dune"는 별도 처리되므로 여기에 포함하지 않음
}

# === BTC 설정 ===
BTC_INTERVAL = "1h"  # BTC 데이터 수집용 고정 인터벌