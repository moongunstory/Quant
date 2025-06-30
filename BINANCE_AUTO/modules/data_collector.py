import os
import pandas as pd
import re
import requests
import json
import numpy as np
from datetime import timedelta
from binance.client import Client
from modules.config import (
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
    TIMEFRAMES,
    REQUIRED_CANDLE_COUNTS,
    BINANCE_INTERVAL_MAP,
    DUNE_API_KEY,
    CACHE_DIR,
    ONCHAIN_CACHE_DIR,
    TZ,
    FEATURE_CATEGORIES_BY_TF,
)

from modules.training.data_preparation.collector import fetch_btc_historical_features
from modules.training.data_preparation.processor import create_dune_derived_features


client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)


def fetch_ohlcv_from_binance(symbol, tf, now, count):
    api_tf = BINANCE_INTERVAL_MAP[tf]
    klines = client.futures_klines(
        symbol=symbol,
        interval=api_tf,
        limit=count
    )

    if not klines:
        print(f"[WARNING] {symbol} {tf} OHLCV 수집 실패: 빈 응답")
        return pd.DataFrame()

    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ],
    )

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_convert("UTC")
    df = df.astype(float).sort_index()

    if not df.empty:
        print(f"[DEBUG] {symbol} {tf} OHLCV 수집 완료: {len(df)} rows")
        print(f"[DEBUG] {symbol} {tf} 인덱스 범위: {df.index.min()} ~ {df.index.max()}")

    return df


def update_cache(symbol, tf, new_df, cache_dir, max_len):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_{tf}.pkl")

    if new_df.index.tz is None:
        new_df.index = new_df.index.tz_localize("UTC")

    use_old_cache = True
    if os.path.exists(path):
        old_df = pd.read_pickle(path)
        if old_df.index.tz is None:
            old_df.index = old_df.index.tz_localize("UTC")

        old_max = old_df.index.max()
        new_max = new_df.index.max()

        tf_minutes = int(re.findall(r"\d+", tf)[0]) if "min" in tf else int(re.findall(r"\d+", tf)[0]) * 60
        max_allowed_delay = pd.Timedelta(minutes=tf_minutes)
        
        if old_max < new_max - max_allowed_delay:
            print(f"[INFO] {tf} 캐시가 너무 구식입니다 → 무시하고 새로 덮어씁니다.")
            use_old_cache = False

    if use_old_cache and os.path.exists(path):
        combined = pd.concat([old_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_df

    combined = combined.sort_index().iloc[-max_len:]
    print(f"[DEBUG] {tf} cache combined rows: {len(combined)} / max index: {combined.index.max()}")
    combined.to_pickle(path)
    return combined


def fetch_latest_dune_row():
    url = "https://api.dune.com/api/v1/query/5182378/results"
    headers = {"x-dune-api-key": DUNE_API_KEY}
    response = requests.get(url, headers=headers)
    data = response.json()

    if "result" not in data or "rows" not in data["result"]:
        return pd.DataFrame()

    df = pd.DataFrame(data["result"]["rows"])
    if df.empty:
        return pd.DataFrame()

    df["day"] = pd.to_datetime(df["day"], utc=True).dt.floor("D")
    df.set_index("day", inplace=True)
    df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    latest_index = df.index.max()
    return df.loc[[latest_index]]


def add_indicators_for_live(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
    from ta.volume import OnBalanceVolumeIndicator
    from ta.volatility import AverageTrueRange
    from ta.trend import CCIIndicator
    
    df = df.copy()
    
    if 'close' not in df.columns or df['close'].isna().all():
        return pd.DataFrame()

    # Get features for this timeframe from config
    features = FEATURE_CATEGORIES_BY_TF.get(tf, [])
    
    # Calculate indicators based on required features
    if "rsi" in features:
        df["rsi"] = RSIIndicator(df['close']).rsi()
    
    if "stochastic_k" in features:
        df["stochastic_k"] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    
    if "cci" in features:
        df["cci"] = CCIIndicator(df['high'], df['low'], df['close']).cci()
    
    if any(feat in features for feat in ["roc", "mom"]):
        df["roc"] = df['close'].pct_change(periods=10) * 100
        df["mom"] = df['close'] - df['close'].shift(10)
    
    if any(feat in features for feat in ["macd", "macd_signal", "macd_histogram"]):
        macd = MACD(df['close'])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_histogram"] = macd.macd_diff()
    
    if "ema_20" in features:
        df["ema_20"] = EMAIndicator(df['close'], window=20).ema_indicator()
    
    if "ema_50" in features:
        df["ema_50"] = EMAIndicator(df['close'], window=50).ema_indicator()
    
    if "sma_20" in features:
        df["sma_20"] = SMAIndicator(df['close'], window=20).sma_indicator()
    
    if "sma_50" in features:
        df["sma_50"] = SMAIndicator(df['close'], window=50).sma_indicator()
    
    if "adx" in features:
        df["adx"] = ADXIndicator(df['high'], df['low'], df['close']).adx()
    
    if "atr" in features:
        df["atr"] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    if "obv" in features:
        df["obv"] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    if "volume_ratio" in features:
        df["volume_ratio"] = df['volume'] / df['volume'].rolling(20).mean()
    
    # 5min specific features
    if tf == "5min":
        if "rsi_mean_6" in features:
            df["rsi_mean_6"] = df["rsi"].rolling(6).mean()
        if "rsi_std_6" in features:
            df["rsi_std_6"] = df["rsi"].rolling(6).std()
        if "macd_slope_6" in features:
            df["macd_slope_6"] = df["macd"].diff().rolling(6).mean()
        if "stochk_range_6" in features:
            df["stochk_range_6"] = df["stochastic_k"].rolling(6).max() - df["stochastic_k"].rolling(6).min()
    
    df = df.ffill().bfill()
    return df


class RealTimeDataCollector:
    def __init__(self):
        self.now = pd.Timestamp.utcnow().floor("30min") - pd.Timedelta(minutes=1)
        self.symbol = "ETHUSDT"
        self.btc_symbol = "BTCUSDT"
        self.cache_dir = CACHE_DIR

    def collect_btc_features(self):
        start = (self.now.floor("30min") - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        end = self.now.floor("30min").strftime("%Y-%m-%d %H:%M:%S")
        df = fetch_btc_historical_features(start, end, interval="30m")

        if df.empty:
            raise ValueError("BTC 데이터 없음")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        latest = df.loc[[df.index.max()]]
        latest.index = pd.DatetimeIndex([self.now.floor("5min")])
        return latest

    def collect_dune_features(self):
        dune_raw = fetch_latest_dune_row()
        if dune_raw.empty:
            raise ValueError("Dune 결과 없음")

        dune_df = create_dune_derived_features(dune_raw)
        if dune_df.index.tz is None:
            dune_df.index = dune_df.index.tz_localize("UTC")

        dune_latest = dune_df.loc[[dune_df.index.max()]]
        dune_latest.index = pd.DatetimeIndex([self.now.floor("5min")])

        date_str = dune_latest.index.max().strftime("%m.%d")
        os.makedirs(ONCHAIN_CACHE_DIR, exist_ok=True)
        dune_latest.to_json(os.path.join(ONCHAIN_CACHE_DIR, f"{date_str}.json"), orient="records", date_format="iso")

        files = sorted(os.listdir(ONCHAIN_CACHE_DIR))
        if len(files) > 3:
            for f in files[:-3]:
                os.remove(os.path.join(ONCHAIN_CACHE_DIR, f))

        return dune_latest

    def refresh_caches(self):
        print("[디버그] 🔄 캐시 새로고침 중...")
        for tf in TIMEFRAMES:
            new_df = fetch_ohlcv_from_binance(self.symbol, tf, self.now, REQUIRED_CANDLE_COUNTS[tf])
            if not new_df.empty:
                update_cache(self.symbol, tf, new_df, self.cache_dir, REQUIRED_CANDLE_COUNTS[tf])

    def _load_timeframe_data(self, tf, seq_len):
        """Load and process individual timeframe data without merging"""
        path = os.path.join(self.cache_dir, f"{self.symbol}_{tf}.pkl")
        if not os.path.exists(path):
            print(f"🚨 {tf} 캐시 누락: {path}")
            return None

        df = pd.read_pickle(path).sort_index()
        df = add_indicators_for_live(df, tf)
        
        if df.empty:
            print(f"🚨 {tf} 지표 생성 실패")
            return None

        # Get only the features defined for this timeframe
        features = FEATURE_CATEGORIES_BY_TF.get(tf, [])
        available_features = [col for col in features if col in df.columns]
        
        if not available_features:
            print(f"🚨 {tf} 유효한 피처가 없음")
            return None

        df_features = df[available_features]
        
        # Return last seq_len rows
        return df_features.iloc[-seq_len:] if len(df_features) >= seq_len else df_features

    def _validate_timeframe_data(self, df_dict):
        """Simple validation - check for NaN values only"""
        for tf, df in df_dict.items():
            if df is None:
                print(f"🚨 {tf} 데이터가 None입니다")
                return False
            if df.isna().any().any():
                print(f"🚨 {tf} 결측값 발견")
                return False
        return True

    def run(self, seq_len=32, auto_refresh_cache=False):
        """
        Returns Dict[str, np.ndarray] with independent timeframe sequences
        Keys: "5min", "15min", "30min", "1H", "btc", "dune"
        """
        print(f"[디버그] 🕒 추론 기준 시간: {self.now}")
        
        self.now = pd.Timestamp.utcnow().floor("30min") - pd.Timedelta(minutes=1)

        if auto_refresh_cache:
            self.refresh_caches()

        # Load individual timeframe data
        tf_data = {}
        for tf in TIMEFRAMES:
            tf_data[tf] = self._load_timeframe_data(tf, seq_len)
        
        # Validate timeframe data
        if not self._validate_timeframe_data(tf_data):
            return None

        # Collect external features
        try:
            btc_row = self.collect_btc_features().iloc[0]
            dune_row = self.collect_dune_features().iloc[0]
        except Exception as e:
            print(f"🚨 외부 피처 수집 실패: {e}")
            return None

        print(f"[디버그] ✅ MTF 데이터 수집 완료")
        
        # Return dictionary with independent sequences
        result = {}
        for tf in TIMEFRAMES:
            # Map timeframe names to match expected output format
            output_key = "5m" if tf == "5min" else tf
            result[output_key] = tf_data[tf].values.astype(np.float32)
        
        result["btc"] = btc_row.values.astype(np.float32)
        result["dune"] = dune_row.values.astype(np.float32)

        return result