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


def compute_funding_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    features = []

    if config.get("funding_ma"):
        for w in config["funding_ma"]:
            ma = df["funding_fundingRate"].rolling(window=w).mean()
            ma.name = f"funding_funding_ma_{w}"
            features.append(ma)

    if config.get("funding_z"):
        for w in config["funding_z"]:
            ma = df["funding_fundingRate"].rolling(window=w).mean()
            std = df["funding_fundingRate"].rolling(window=w).std()
            z_score = (df["funding_fundingRate"] - ma) / (std + 1e-8)
            z_score.name = f"funding_funding_z_{w}"
            features.append(z_score)

    if config.get("funding_sign"):
        sign = (df["funding_fundingRate"] > 0).astype(int) - (df["funding_fundingRate"] < 0).astype(int)
        sign.name = "funding_funding_sign"
        features.append(sign)

    return pd.concat([df] + features, axis=1)


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

    # 컬럼명 변경 (index_close → index_price 만 처리)
    df = df.rename(columns={"index_close": "index_price"})

    # 이미 index_* 컬럼이므로 rename 불필요
    df = df.set_index("timestamp").sort_index().resample("5min").mean()

    return df.reset_index()


def compute_index_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    features = []
    # pct_change는 windows 리스트와 무관하게 항상 계산
    pct_change = df["index_price"].pct_change(fill_method=None)
    pct_change.name = "index_pct_change"
    features.append(pct_change)

    for w in windows:
        ma = df["index_price"].rolling(window=w).mean()
        std = df["index_price"].rolling(window=w).std()
        
        ma.name = f"index_ma_{w}"
        std.name = f"index_std_{w}"
        
        zscore = (df["index_price"] - ma) / (std + 1e-8)
        zscore.name = f"index_zscore_{w}"
        
        momentum = df["index_price"].diff(periods=w)
        momentum.name = f"index_momentum_{w}"
        
        features.extend([ma, std, zscore, momentum])

    # 원본 df에 모든 피처를 한 번에 병합
    return pd.concat([df] + features, axis=1)


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
