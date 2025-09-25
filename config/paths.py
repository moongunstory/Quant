from pathlib import Path
from typing import Iterable, Optional, Tuple

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

LATEST_HPO_VERSION_FILE = HPO_DIR / "latest_version.txt"

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


def _iter_hpo_version_dirs() -> Iterable[Path]:
    """Yield numeric sub-directories that represent HPO run versions."""

    if not HPO_DIR.exists():
        return []

    return (p for p in HPO_DIR.iterdir() if p.is_dir() and p.name.isdigit())


def list_hpo_versions() -> list[int]:
    """Return sorted HPO version numbers that currently exist on disk."""

    versions = [int(p.name) for p in _iter_hpo_version_dirs()]
    return sorted(versions)


def get_latest_hpo_version() -> Optional[int]:
    """Return the latest recorded HPO version if any exist."""

    if LATEST_HPO_VERSION_FILE.exists():
        try:
            return int(LATEST_HPO_VERSION_FILE.read_text().strip())
        except ValueError:
            pass

    versions = list_hpo_versions()
    if versions:
        return versions[-1]
    return None


def set_latest_hpo_version(version: int) -> None:
    """Persist the provided version number as the most recent HPO run."""

    HPO_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_HPO_VERSION_FILE.write_text(str(int(version)))


def get_hpo_version_dir(version: int) -> Path:
    """Return the base directory that stores artifacts for a given version."""

    return HPO_DIR / str(int(version))


def ensure_hpo_version_artifacts(version: int) -> Tuple[Path, Path, Path, Path]:
    """Ensure artifact directories exist for the requested HPO version."""

    version_dir = get_hpo_version_dir(version)
    db_dir = version_dir / "db"
    logs_dir = version_dir / "logs"
    params_dir = version_dir / "params"

    for path in (version_dir, db_dir, logs_dir, params_dir):
        path.mkdir(parents=True, exist_ok=True)

    return version_dir, db_dir, logs_dir, params_dir


def get_hpo_logs_dir(version: Optional[int] = None) -> Path:
    """Return the TensorBoard logs directory for the specified version."""

    if version is None:
        version = get_latest_hpo_version()
        if version is None:
            raise FileNotFoundError("No HPO runs have been recorded yet.")

    _, _, logs_dir, _ = ensure_hpo_version_artifacts(version)
    return logs_dir


def get_optuna_db_path_for_version(version: Optional[int] = None) -> Path:
    """Return the Optuna DB path for the supplied HPO version."""

    if version is None:
        version = get_latest_hpo_version()
        if version is None:
            raise FileNotFoundError("No HPO database found. Run HPO first.")

    _, db_dir, _, _ = ensure_hpo_version_artifacts(version)
    return db_dir / "optuna_feature_hpo.db"


def get_hpo_params_dir(version: Optional[int] = None, *, create: bool = False) -> Path:
    """Return the directory holding saved trial parameter JSON files."""

    if version is None:
        version = get_latest_hpo_version()
        if version is None:
            raise FileNotFoundError("No HPO parameter directory available. Run HPO first.")

    version_dir, _, _, params_dir = ensure_hpo_version_artifacts(version)
    if create:
        params_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not params_dir.exists():
            raise FileNotFoundError(f"HPO params directory is missing for version {version}: {params_dir}")

    return params_dir
