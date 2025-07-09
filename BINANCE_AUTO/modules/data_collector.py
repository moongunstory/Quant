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
    import pandas_ta as ta
    
    df = df.copy()
    
    if 'close' not in df.columns or df['close'].isna().all():
        return pd.DataFrame()

    # Get features for this timeframe from config
    features = FEATURE_CATEGORIES_BY_TF.get(tf, [])
    
    # Basic price and volume features
    if "returns" in features: df['returns'] = df['close'].pct_change()
    if "high_low_range" in features: df['high_low_range'] = (df['high'] - df['low']) / df['close']
    if "open_close_range" in features: df['open_close_range'] = (df['close'] - df['open']) / df['open']

    # Calculate indicators based on required features using pandas_ta
    # Momentum Indicators
    if "rsi" in features: df["rsi"] = ta.rsi(df['close'])
    if "stoch_k" in features and "stoch_d" in features:
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None and not stoch.empty:
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
    if "cci" in features: df["cci"] = ta.cci(df['high'], df['low'], df['close'])
    if "roc" in features: df["roc"] = ta.roc(df['close'])
    if "mom" in features: df["mom"] = ta.mom(df['close'])

    # Trend Indicators
    if "macd" in features and "macd_signal" in features and "macd_hist" in features:
        macd = ta.macd(df['close'])
        if macd is not None and not macd.empty:
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
    if "sma_10" in features: df['sma_10'] = ta.sma(df['close'], length=10)
    if "sma_20" in features: df['sma_20'] = ta.sma(df['close'], length=20)
    if "sma_50" in features: df['sma_50'] = ta.sma(df['close'], length=50)
    if "ema_10" in features: df['ema_10'] = ta.ema(df['close'], length=10)
    if "ema_20" in features: df['ema_20'] = ta.ema(df['close'], length=20)
    if "ema_50" in features: df['ema_50'] = ta.ema(df['close'], length=50)
    if "adx" in features and "plus_di" in features and "minus_di" in features:
        adx_result = ta.adx(df['high'], df['low'], df['close'])
        if adx_result is not None and not adx_result.empty:
            df['adx'] = adx_result['ADX_14']
            df['plus_di'] = adx_result['DMP_14']
            df['minus_di'] = adx_result['DMN_14']

    # Volatility Indicators
    if "atr" in features: df['atr'] = ta.atr(df['high'], df['low'], df['close'])
    if "bb_upper" in features and "bb_middle" in features and "bb_lower" in features and "bb_percent_b" in features and "bb_bandwidth" in features:
        bbands = ta.bbands(df['close'])
        if bbands is not None and not bbands.empty:
            df['bb_upper'] = bbands['BBL_20_2.0'] # BBL is lower, BBU is upper
            df['bb_middle'] = bbands['BBM_20_2.0']
            df['bb_lower'] = bbands['BBU_20_2.0']
            df['bb_percent_b'] = bbands['BBP_20_2.0']
            df['bb_bandwidth'] = bbands['BBB_20_2.0']

    # Volume Indicators
    if "obv" in features: df['obv'] = ta.obv(df['close'], df['volume'])
    if "volume_ma_20" in features: df['volume_ma_20'] = ta.sma(df['volume'], length=20)

    # volume_ratio는 volume_ma_20이 필요
    if "volume_ratio" in features and "volume_ma_20" in df.columns:
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

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
        btc_df = fetch_ohlcv_from_binance(self.btc_symbol, "1h", self.now, 70) # Fetch enough data for indicators

        if btc_df.empty:
            raise ValueError("BTC 데이터 없음")
        if btc_df.index.tz is None:
            btc_df.index = btc_df.index.tz_localize("UTC")

        # Calculate BTC features based on FEATURE_CATEGORIES_BY_TF["btc"]
        btc_features = pd.DataFrame(index=btc_df.index)
        features = FEATURE_CATEGORIES_BY_TF.get("btc", [])

        if "btc_open" in features: btc_features["btc_open"] = btc_df["open"]
        if "btc_high" in features: btc_features["btc_high"] = btc_df["high"]
        if "btc_low" in features: btc_features["btc_low"] = btc_df["low"]
        if "btc_close" in features: btc_features["btc_close"] = btc_df["close"]
        if "btc_volume" in features: btc_features["btc_volume"] = btc_df["volume"]

        if "btc_returns" in features: btc_features["btc_returns"] = btc_df["close"].pct_change()
        if "btc_high_low_range" in features: btc_features["btc_high_low_range"] = (btc_df["high"] - btc_df["low"]) / btc_df["close"]
        if "btc_open_close_range" in features: btc_features["btc_open_close_range"] = (btc_df["close"] - btc_df["open"]) / btc_df["open"]

        if "btc_rsi" in features: btc_features["btc_rsi"] = ta.rsi(btc_df["close"])
        if "btc_macd" in features and "btc_macd_signal" in features and "btc_macd_hist" in features:
            macd = ta.macd(btc_df["close"])
            if macd is not None and not macd.empty:
                btc_features["btc_macd"] = macd['MACD_12_26_9']
                btc_features["btc_macd_signal"] = macd['MACDs_12_26_9']
                btc_features["btc_macd_hist"] = macd['MACDh_12_26_9']

        btc_features = btc_features.fillna(0)
        latest = btc_features.loc[[btc_features.index.max()]]
        latest.index = pd.DatetimeIndex([self.now.floor("5min")]) # Align index with other data
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

    def get_recent_market_df(self, tf: str = "5min") -> pd.DataFrame:
        """Return latest cached market dataframe for the specified timeframe."""
        path = os.path.join(self.cache_dir, f"{self.symbol}_{tf}.pkl")
        if not os.path.exists(path):
            return pd.DataFrame()
        df = pd.read_pickle(path)
        return df.sort_index()
