import pandas as pd
import requests
import time
from binance.client import Client
import pandas_ta as ta
import os
import sys
from datetime import timedelta

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    DUNE_API_KEY, BINANCE_API_KEY, BINANCE_SECRET_KEY,
    TIMEFRAMES, START_DATE, END_DATE,
    FEATURE_CATEGORIES_BY_TF, DUNE_QUERY_PARTS,
    RAW_DATA_PATH, BINANCE_INTERVAL_MAP, FUTURES_SYMBOL
)

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)

def call_dune_api(query_id: str, query_name: str) -> pd.DataFrame:
    """Dune API 호출"""
    url = f"https://api.dune.com/api/v1/query/{query_id}/results"
    headers = {"x-dune-api-key": DUNE_API_KEY}
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    if 'result' in data and 'rows' in data['result']:
        df = pd.DataFrame(data['result']['rows'])
        if not df.empty and 'day' in df.columns:
            df['day'] = df['day'].astype(str).str.replace(r' Asia/Seoul.*$', '', regex=True)
            df['timestamp'] = pd.to_datetime(df['day'], utc=True)
            df = df.drop(columns=['day']).set_index('timestamp')
        return df
    return pd.DataFrame()

def collect_all_dune_data() -> pd.DataFrame:
    """모든 DUNE 데이터 수집"""
    all_dfs = []
    part_names = list(DUNE_QUERY_PARTS.keys())
    
    for i in range(0, len(part_names), 3):
        batch = part_names[i:i+3]
        for part_name in batch:
            query_id = DUNE_QUERY_PARTS[part_name]
            df = call_dune_api(query_id, f"Part_{part_name}")
            if not df.empty:
                all_dfs.append(df)
        
        if i + 3 < len(part_names):
            time.sleep(5)
    
    if all_dfs:
        dune_df = all_dfs[0]
        for df in all_dfs[1:]:
            dune_df = dune_df.combine_first(df)
        return dune_df
    return pd.DataFrame()

def fetch_ohlcv_with_extended_period(symbol, interval, start_str, end_str):
    """OHLCV 데이터 수집"""
    start_date = pd.to_datetime(start_str, utc=True)
    extended_start = start_date - timedelta(days=5)
    extended_start_str = extended_start.strftime("%Y-%m-%d")
    
    start_ms = int(pd.to_datetime(extended_start_str).timestamp() * 1000)
    end_ms = int(pd.to_datetime(end_str).timestamp() * 1000)
    klines = client.futures_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ms,
        endTime=end_ms,
        limit=1000,
    )
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"])
    
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_convert("UTC")
    return df.astype(float).dropna()

def add_indicators_with_validation(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """기술적 지표 추가 - config 기반 피처 선택"""
    prefix = f"{tf}_"
    features = FEATURE_CATEGORIES_BY_TF.get(tf, [])
    result_df = pd.DataFrame(index=df.index)
    
    # 기본 지표 계산
    if "rsi" in features and len(df) >= 14:
        result_df[prefix + "rsi"] = ta.rsi(df["close"], length=14)
    
    if "stochastic_k" in features and len(df) >= 14:
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14)
        if stoch is not None and not stoch.empty:
            result_df[prefix + "stochastic_k"] = stoch["STOCHk_14_3_3"]
    
    if "cci" in features and len(df) >= 20:
        result_df[prefix + "cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    
    if "roc" in features and len(df) >= 10:
        result_df[prefix + "roc"] = ta.roc(df["close"], length=10)
    
    if "mom" in features and len(df) >= 10:
        result_df[prefix + "mom"] = ta.mom(df["close"], length=10)
    
    if "macd" in features and len(df) >= 26:
        macd = ta.macd(df["close"])
        if macd is not None and not macd.empty:
            result_df[prefix + "macd"] = macd["MACD_12_26_9"]
            if "macd_signal" in features:
                result_df[prefix + "macd_signal"] = macd["MACDs_12_26_9"]
            if "macd_histogram" in features:
                result_df[prefix + "macd_histogram"] = macd["MACDh_12_26_9"]
    
    if "ema_20" in features and len(df) >= 20:
        result_df[prefix + "ema_20"] = ta.ema(df["close"], length=20)
    
    if "ema_50" in features and len(df) >= 50:
        result_df[prefix + "ema_50"] = ta.ema(df["close"], length=50)
    
    if "sma_20" in features and len(df) >= 20:
        result_df[prefix + "sma_20"] = ta.sma(df["close"], length=20)
    
    if "sma_50" in features and len(df) >= 50:
        result_df[prefix + "sma_50"] = ta.sma(df["close"], length=50)
    
    if "adx" in features and len(df) >= 14:
        adx_result = ta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_result is not None and not adx_result.empty:
            result_df[prefix + "adx"] = adx_result["ADX_14"]
    
    if "atr" in features and len(df) >= 14:
        result_df[prefix + "atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    
    if "obv" in features:
        result_df[prefix + "obv"] = ta.obv(df["close"], df["volume"])
    
    if "volume_ratio" in features and len(df) >= 20:
        volume_ma = df["volume"].rolling(window=20).mean()
        result_df[prefix + "volume_ratio"] = df["volume"] / volume_ma
    
    # 5분봉 특화 피처
    if tf == "5min":
        if "rsi_mean_6" in features and prefix + "rsi" in result_df.columns:
            result_df[prefix + "rsi_mean_6"] = result_df[prefix + "rsi"].rolling(6).mean()
        if "rsi_std_6" in features and prefix + "rsi" in result_df.columns:
            result_df[prefix + "rsi_std_6"] = result_df[prefix + "rsi"].rolling(6).std()
        if "macd_slope_6" in features and prefix + "macd" in result_df.columns:
            result_df[prefix + "macd_slope_6"] = result_df[prefix + "macd"].diff().rolling(6).mean()
        if "stochk_range_6" in features and prefix + "stochastic_k" in result_df.columns:
            result_df[prefix + "stochk_range_6"] = result_df[prefix + "stochastic_k"].rolling(6).apply(lambda x: x.max() - x.min(), raw=True)
    
    return result_df

def fetch_btc_historical_features(start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
    """BTC 피처 수집"""
    btc_df = fetch_ohlcv_with_extended_period("BTCUSDT", interval, start_date, end_date)
    
    btc_features = pd.DataFrame(index=btc_df.index)
    
    # BTC 피처 계산
    btc_features["btc_return_1h"] = btc_df["close"].pct_change()
    btc_features["btc_high_low_diff"] = (btc_df["high"] - btc_df["low"]) / btc_df["close"]
    btc_features["btc_close_vs_high"] = (btc_df["close"] - btc_df["high"]) / btc_df["high"]
    btc_features["btc_close_vs_low"] = (btc_df["close"] - btc_df["low"]) / btc_df["low"]
    
    # 캔들 패턴
    body_size = abs(btc_df["close"] - btc_df["open"])
    candle_range = btc_df["high"] - btc_df["low"]
    
    btc_features["btc_bullish"] = (btc_df["close"] > btc_df["open"]).astype(float)
    btc_features["btc_bearish"] = (btc_df["close"] < btc_df["open"]).astype(float)
    btc_features["btc_doji"] = (body_size < 0.1 * candle_range).astype(float)
    
    # doji인 경우 bullish/bearish는 0으로 설정
    doji_mask = btc_features["btc_doji"] == 1.0
    btc_features.loc[doji_mask, "btc_bullish"] = 0.0
    btc_features.loc[doji_mask, "btc_bearish"] = 0.0
    
    # 트렌드 패턴
    btc_features["btc_uptrend"] = ((btc_features["btc_return_1h"] > 0.005) & 
                                  (btc_features["btc_bullish"] == 1.0)).astype(float)
    btc_features["btc_downtrend"] = ((btc_features["btc_return_1h"] < -0.005) & 
                                    (btc_features["btc_bearish"] == 1.0)).astype(float)
    btc_features["btc_recovery"] = ((btc_features["btc_return_1h"] > 0) & 
                                   (btc_features["btc_bearish"] == 1.0)).astype(float)
    btc_features["btc_correction"] = ((btc_features["btc_return_1h"] < 0) & 
                                     (btc_features["btc_bullish"] == 1.0)).astype(float)
    btc_features["btc_sideways"] = ((btc_features["btc_uptrend"] + btc_features["btc_downtrend"] + 
                                    btc_features["btc_recovery"] + btc_features["btc_correction"]) == 0).astype(float)
    
    return btc_features.fillna(0)

def collect_all_market_data() -> dict:
    """전체 마켓 데이터 수집 - MTF 독립 구조"""
    result = {}
    
    # 1. 각 타임프레임별 ETH 데이터 독립 수집
    for tf in TIMEFRAMES:
        print(f"[수집] {tf} 타임프레임 데이터...")
        interval = BINANCE_INTERVAL_MAP[tf]
        eth_df = fetch_ohlcv_with_extended_period(FUTURES_SYMBOL, interval, START_DATE, END_DATE)
        eth_indicators = add_indicators_with_validation(eth_df, tf)
        
        # 시작 날짜부터 필터링
        target_start_dt = pd.to_datetime(START_DATE, utc=True)
        result[tf] = eth_indicators[eth_indicators.index >= target_start_dt].copy()
        
        print(f"[완료] {tf}: {len(result[tf])} rows, {len(result[tf].columns)} features")
    
    # 2. BTC 피처 독립 수집
    print("[수집] BTC 피처...")
    btc_df = fetch_btc_historical_features(START_DATE, END_DATE)
    result["btc"] = btc_df.copy()
    print(f"[완료] BTC: {len(result['btc'])} rows, {len(result['btc'].columns)} features")
    
    # 3. DUNE 온체인 데이터 독립 수집
    print("[수집] DUNE 온체인 데이터...")
    dune_df = collect_all_dune_data()
    if not dune_df.empty:
        result["dune"] = dune_df.copy()
        print(f"[완료] DUNE: {len(result['dune'])} rows, {len(result['dune'].columns)} features")
    else:
        result["dune"] = pd.DataFrame()
        print("[경고] DUNE 데이터 수집 실패")
    
    # 4. 선택적 저장 (딕셔너리 구조 유지)
    save_dir = os.path.dirname(os.path.join(PROJECT_ROOT, RAW_DATA_PATH))
    os.makedirs(save_dir, exist_ok=True)
    
    # 각 타임프레임별로 개별 저장
    for key, df in result.items():
        if not df.empty:
            save_path = os.path.join(save_dir, f"market_data_{key}.pkl")
            df.to_pickle(save_path)
            print(f"[저장] {key}: {save_path}")
    
    total_features = sum(len(df.columns) for df in result.values() if not df.empty)
    print(f"[✅ 수집 완료] 총 {len(result)} 타임프레임, 총 {total_features} 피처")
    
    return result

if __name__ == "__main__":
    result_dict = collect_all_market_data()