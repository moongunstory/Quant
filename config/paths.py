from pathlib import Path

# ------------------ 프로젝트 루트 ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------ 데이터 디렉토리 ------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ------------------ 원시 데이터 경로 ------------------

def get_raw_dir(symbol: str, category: str) -> Path:
    if category == "dune":
        return RAW_DATA_DIR / "dune"
    return RAW_DATA_DIR / category / symbol.lower()

def get_ohlcv_path(symbol: str) -> Path:
    return get_raw_dir(symbol, "ohlcv") / "ohlcv.csv"

def get_index_price_path(symbol: str) -> Path:
    return get_raw_dir(symbol, "ohlcv") / "index_price.csv"

def get_funding_rate_path(symbol: str) -> Path:
    return get_raw_dir(symbol, "funding") / "funding_rate.csv"

def get_dune_path(symbol: str, name: str) -> Path:
    return get_raw_dir(symbol, "dune") / f"{name}.csv"

# ------------------ 가공된 OHLCV 세트 경로 ------------------

def get_processed_ohlcv_dir(symbol: str) -> Path:
    return PROCESSED_DATA_DIR / symbol.lower()

def get_train_parquet_path(symbol: str) -> Path:
    return get_processed_ohlcv_dir(symbol) / "train_set.parquet"

def get_validation_parquet_path(symbol: str) -> Path:
    return get_processed_ohlcv_dir(symbol) / "validation_set.parquet"

def get_test_parquet_path(symbol: str) -> Path:
    return get_processed_ohlcv_dir(symbol) / "test_set.parquet"

# ------------------ 기타 피처 경로 ------------------

def get_processed_funding_path(symbol: str) -> Path:
    return PROCESSED_DATA_DIR / "funding_index" / f"{symbol.lower()}_funding.parquet"

def get_processed_index_path(symbol: str) -> Path:
    return PROCESSED_DATA_DIR / "funding_index" / f"{symbol.lower()}_index.parquet"

def get_processed_dune_path(symbol: str, name: str) -> Path:
    return PROCESSED_DATA_DIR / "dune" / f"{name}.parquet"

# ------------------ 기타 경로 ------------------

def get_train_feature_data_path() -> Path:
    return PROCESSED_DATA_DIR / "train_features.csv"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = MODELS_DIR / "logs"

# ------------------ HPO 관련 ------------------

HPO_DIR = DATA_DIR / "hpo"
HPO_LOGS_DIR = HPO_DIR / "logs"
HPO_TRIALS_DIR = HPO_DIR / "trials"
HPO_BEST_CONFIG_DIR = HPO_DIR / "best_config"

def get_team_best_config_path(team_name: str) -> Path:
    return HPO_BEST_CONFIG_DIR / f"{team_name}_best.json"

def get_team_trials_path(team_name: str) -> Path:
    return HPO_TRIALS_DIR / f"{team_name}_trials.csv"

def get_stage1_top_combinations_path() -> Path:
    return HPO_BEST_CONFIG_DIR / "stage1_top_combinations.json"

# ------------------ 실시간 운영 ------------------

LIVE_DIR = PROJECT_ROOT / "live"

# ------------------ Optuna DB ------------------

OPTUNA_DB_PATH = HPO_DIR / "optuna_feature_hpo.db"
