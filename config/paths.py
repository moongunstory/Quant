# config/paths.py

from pathlib import Path

# ------------------ 프로젝트 루트 ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ------------------ 데이터 디렉토리 ------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ------------------ 원시 데이터 경로 ------------------

def get_raw_dir(symbol: str, category: str) -> Path:
    return RAW_DATA_DIR / category / symbol.lower()

def get_symbol_dir(symbol: str) -> Path:
    return get_raw_dir(symbol, "ohlcv")

def get_ohlcv_path(symbol: str) -> Path:
    return get_raw_dir(symbol, "ohlcv") / "ohlcv.csv"

def get_index_price_path(symbol: str) -> Path:
    return get_raw_dir(symbol, "ohlcv") / "index_price.csv"

def get_funding_rate_path(symbol: str) -> Path:
    return get_raw_dir(symbol, "funding") / "funding_rate.csv"

def get_dune_path(symbol: str, name: str) -> Path:
    return get_raw_dir(symbol, "dune") / f"{name}.csv"

# ------------------ 가공 데이터 경로 ------------------

def get_processed_feature_path(symbol: str, category: str, name: str = None) -> Path:
    """
    category: ohlcv, funding_index, dune 등
    name: 저장 파일 이름 (기본: symbol.lower()), 예: ethusdt_funding, ethusdt_index
    """
    file_name = f"{name or symbol.lower()}.parquet"
    return PROCESSED_DATA_DIR / category / file_name

# 📌 추천: 자주 쓰는 경로를 위한 헬퍼 함수

def get_processed_ohlcv_path(symbol: str) -> Path:
    return get_processed_feature_path(symbol, "ohlcv")

def get_processed_funding_path(symbol: str) -> Path:
    return get_processed_feature_path(symbol, "funding_index", f"{symbol.lower()}_funding")

def get_processed_index_path(symbol: str) -> Path:
    return get_processed_feature_path(symbol, "funding_index", f"{symbol.lower()}_index")

def get_processed_dune_path(symbol: str) -> Path:
    return get_processed_feature_path(symbol, "dune")

# ------------------ 기타 ------------------

def get_train_feature_data_path() -> Path:
    return PROCESSED_DATA_DIR / "train_features.csv"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = MODELS_DIR / "logs"

# ------------------ HPO 관련 ------------------

HPO_DIR = DATA_DIR / "hpo"
HPO_TRIALS_DIR = HPO_DIR / "trials"
HPO_BEST_CONFIG_DIR = HPO_DIR / "best_config"

def get_team_best_config_path(team_name: str) -> Path:
    return HPO_BEST_CONFIG_DIR / f"{team_name}_best.json"

def get_team_trials_path(team_name: str) -> Path:
    return HPO_TRIALS_DIR / f"{team_name}_trials.csv"

def get_stage1_top_combinations_path() -> Path:
    return HPO_BEST_CONFIG_DIR / "stage1_top_combinations.json"

LIVE_DIR = PROJECT_ROOT / "live"

# ------------------ Optuna DB 경로 ------------------

OPTUNA_DB_PATH = HPO_DIR / "optuna_feature_hpo.db"