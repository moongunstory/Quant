# train/prepare/process/funding_index_features.py

import pandas as pd
from ai_binance.config.paths import get_funding_rate_path, get_index_price_path


# === Funding 관련 ===
def load_funding_data(symbol: str) -> pd.DataFrame:
    path = get_funding_rate_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"❌ Funding 데이터 없음: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.rename(columns={"fundingRate": "funding_fundingRate"})
    return df.reset_index() 


def compute_funding_features(df: pd.DataFrame, window: int = 96) -> pd.DataFrame:
    df = df.copy()
    df[f"funding_funding_ma_{window}"] = df["funding_fundingRate"].rolling(window=window).mean()
    df[f"funding_funding_std_{window}"] = df["funding_fundingRate"].rolling(window=window).std()
    df[f"funding_funding_z_{window}"] = (
        df["funding_fundingRate"] - df[f"funding_funding_ma_{window}"]
    ) / (df[f"funding_funding_std_{window}"] + 1e-8)
    df["funding_funding_sign"] = (df["funding_fundingRate"] > 0).astype(int) - (df["funding_fundingRate"] < 0).astype(int)
    return df


def get_available_funding_features(windows: list[int]) -> list[str]:
    features = []
    for w in windows:
        features.extend([
            f"funding_funding_ma_{w}",
            f"funding_funding_std_{w}",
            f"funding_funding_z_{w}",
        ])
    features.append("funding_funding_sign")
    return features


# === Index 관련 ===
def load_index_data(symbol: str) -> pd.DataFrame:
    path = get_index_price_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"❌ Index 데이터 없음: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.rename(columns={"index_close": "index_price"})
    df = df.set_index("timestamp").sort_index().resample("5min").mean()

    df = df.rename(columns={
        "open": "index_open",
        "high": "index_high",
        "low": "index_low",
        "index_price": "index_price"
    })
    return df.reset_index()


def compute_index_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df["index_pct_change"] = df["index_price"].pct_change(fill_method=None)

    for w in windows:
        df[f"index_ma_{w}"] = df["index_price"].rolling(window=w).mean()
        df[f"index_std_{w}"] = df["index_price"].rolling(window=w).std()
        df[f"index_zscore_{w}"] = (df["index_price"] - df[f"index_ma_{w}"]) / (df[f"index_std_{w}"] + 1e-8)
        df[f"index_momentum_{w}"] = df["index_price"].diff(periods=w)

    return df


def get_available_index_features(windows: list[int]) -> list[str]:
    features = ["index_pct_change"]
    for w in windows:
        features.extend([
            f"index_ma_{w}",
            f"index_std_{w}",
            f"index_zscore_{w}",
            f"index_momentum_{w}"
        ])
    return features
