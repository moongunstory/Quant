import os
import numpy as np
import pandas as pd
import joblib
import talib

from typing import Dict, List, Optional
from .paths import RAW_DIR, OUT_DIR, REF_COLS_CANON, BASE_INTERVAL, HPO_FEATURE_LIST_FMT, HPO_EXPAND_WINDOWS
from .feature_engineering import generate_feature_combinations

# === 유틸 함수들 통합 ===
def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    std = df.std(numeric_only=True)
    zero_std_cols = std[std == 0].index.tolist()
    if zero_std_cols:
        print(f"[INFO] Zero std columns detected: {zero_std_cols[:5]}...")
        df[zero_std_cols] = 0.0
    return df.fillna(0.0)

def enforce_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        for tcol in ["Open_time", "open_time", "time"]:
            if tcol in df.columns:
                idx = pd.to_datetime(df[tcol], utc=True, errors="coerce")
                if idx.notna().any():
                    df = df.set_index(idx).drop(columns=[tcol])
                    break
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = pd.to_datetime(df.index, utc=True)

    if df.index.tz is None:
        print(f"[WARNING] Timezone info missing, assuming UTC")
    elif str(df.index.tz) != 'UTC':
        print(f"[WARNING] Non-UTC timezone detected: {df.index.tz}")

    df = df.sort_index()
    df.index.name = "time"
    return df

def zscore(s: pd.Series, win: Optional[int] = None) -> pd.Series:
    if win is None:
        mu, sd = s.mean(), s.std()
        return ((s - mu) / (sd or 1e-9)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std().replace(0, np.nan)
    return ((s - mu) / sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)

def pct_slope(x: pd.Series, win: int) -> pd.Series:
    return x.pct_change(fill_method=None).rolling(win, min_periods=win).mean().fillna(0.0)

# === 데이터 로딩 ===
def load_raw(split: str, interval: str) -> pd.DataFrame:
    if interval == "btc1h":
        path = os.path.join(RAW_DIR, "btcusdt", f"fut_{split}_data_1h.parquet")
    else:
        path = os.path.join(RAW_DIR, "ethusdt", f"fut_{split}_data_{interval}.parquet")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found: {path}")

    df = pd.read_parquet(path)

    for col in ["open", "high", "low", "close", "volume"]:
        real = next((c for c in df.columns if c.lower() == col), None)
        if real:
            df[real] = pd.to_numeric(df[real], errors="coerce")

    if "FundingRate" not in df.columns:
        if "funding_rate" in df.columns:
            df.rename(columns={"funding_rate": "FundingRate"}, inplace=True)
        else:
            df["FundingRate"] = 0.0
    df["FundingRate"] = pd.to_numeric(df["FundingRate"], errors="coerce").fillna(0.0)

    for col in ["Volume", "Quote_asset_volume", "Taker_buy_base", "Taker_buy_quote"]:
        if col in df.columns:
            base = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}Raw"] = base
            df[f"{col}Log"] = np.log1p(np.clip(base, 0, None))
    if "VolumeRaw" in df.columns:
        df["Volume"] = df["VolumeRaw"]

    df = enforce_dt_index(df)

    for c in REF_COLS_CANON:
        if c not in df.columns:
            df[c] = 0.0

    if interval == BASE_INTERVAL:
        df["FundingSettle"] = (((df.index.hour % 8 == 0) & (df.index.minute == 0))).astype("int8")

    return df

# === 기술 지표 피처 추가 ===
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close, high, low, open_, volume = df["Close"], df["High"], df["Low"], df["Open"], df.get("Volume", None)

    # === 기존 지표들 ===
    df["bb_mid"] = close.rolling(20).mean()
    df["bb_std"] = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    for period in [5, 10, 20, 60, 120]:
        df[f"ema_{period}"] = talib.EMA(close, timeperiod=period)

    df["rsi_14"] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(close, 12, 26, 9)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd, macd_signal, macd_hist
    k, d = talib.STOCH(high, low, close, 14, 3, 0, 3, 0)
    df["stoch_k"], df["stoch_d"] = k, d
    df["cci_20"] = talib.CCI(high, low, close, 20)

    ha_close = (open_ + high + low + close) / 4
    ha_open = (open_.shift(1) + ha_close.shift(1)) / 2
    df["ha_close"], df["ha_open"] = ha_close, ha_open
    df["ha_high"] = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    df["ha_low"] = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)

    period1 = (high.rolling(9).max() + low.rolling(9).min()) / 2
    period2 = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df["tenkan_sen"], df["kijun_sen"] = period1, period2
    df["senkou_a"] = ((period1 + period2) / 2).shift(26)
    df["senkou_b"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df["chikou_span"] = close.shift(-26)

    # === 추가 기술 지표들 ===
    # 1. 추세 관련
    df["adx_14"] = talib.ADX(high, low, close, timeperiod=14)
    df["aroondown"], df["aroonup"] = talib.AROON(high, low, timeperiod=14)
    df["aroon_osc"] = talib.AROONOSC(high, low, timeperiod=14)

    # 2. 모멘텀 및 오실레이터
    df["apo"] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    df["ppo"] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    df["ultosc"] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df["willr_14"] = talib.WILLR(high, low, close, timeperiod=14)

    # 3. 변동성 기반
    df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
    df["natr_14"] = talib.NATR(high, low, close, timeperiod=14)

    # 4. 거래량 및 자금 흐름 (Volume)
    if volume is not None:
        df["obv"] = talib.OBV(close, volume)
        df["ad"] = talib.AD(high, low, close, volume)
        df["adosc"] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

    # 5. 추세 반전 / Trailing Stop
    df["psar"] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    return df.dropna()

# === 지표 + 기존 방식으로 확장 피처 생성 ===
def add_hpo_candidates(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    기술 지표 + 파생 피처 조합까지 포함된 확장 버전 (HPO 대상).
    """
    df = df.copy()
    df = add_technical_indicators(df)
    df = generate_feature_combinations(df)
    return sanitize(df)

# === 결과 로딩 ===
def load_processed(split: str, tf: str, mode: str = "auto") -> pd.DataFrame:
    base_p = os.path.join(OUT_DIR, f"fe_{split}_{tf}.parquet")
    hpo_p = os.path.join(OUT_DIR, f"feHPO_{split}_{tf}.parquet")
    path = hpo_p if mode == "hpo" or (mode == "auto" and os.path.exists(hpo_p)) else base_p
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found: {path}")
    df = pd.read_parquet(path)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df

def feature_universe(df: pd.DataFrame, prefix: str = "f_") -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]

def build_universe_from_processed(split: str = "train", tf: str = "5m", mode: str = "auto") -> List[str]:
    df = load_processed(split, tf, mode=mode)
    feats = feature_universe(df, prefix="f_")
    if len(feats) < 10:
        raise RuntimeError(f"Feature universe too small: {len(feats)}")
    return feats
