import pandas as pd
import requests
import time
from binance.client import Client
import pandas_ta as ta
import os
import sys
from datetime import timedelta
import logging

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    DUNE_API_KEY, BINANCE_API_KEY, BINANCE_SECRET_KEY,
    TIMEFRAMES, START_DATE, END_DATE,
    FEATURE_CATEGORIES_BY_TF, DUNE_QUERY_PARTS,
    RAW_DATA_PATH, BINANCE_INTERVAL_MAP, FUTURES_SYMBOL, AUX_TIMEFRAMES
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def log_ohlcv_nans(df: pd.DataFrame, tf: str, stage: str = "") -> None:
    """Log NaN counts for OHLCV columns."""
    prefix = f"[{tf}]"
    if stage:
        prefix += f" {stage}"
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            logger.info(f"{prefix} {col} NaNs: {df[col].isna().sum()}")

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
    """OHLCV 데이터 수집 (페이지네이션 포함)"""
    start_dt = pd.to_datetime(start_str, utc=True)
    end_dt = pd.to_datetime(end_str, utc=True)
    extended_start_dt = start_dt - timedelta(days=5)

    start_ms = int(extended_start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_klines = []
    max_limit = 1000

    while True:
        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            startTime=start_ms,
            endTime=end_ms,
            limit=max_limit,
        )

        if not klines:
            break

        all_klines.extend(klines)

        last_close_time = klines[-1][6]
        if last_close_time >= end_ms:
            break

        start_ms = last_close_time + 1
        time.sleep(0.2)

    if not all_klines:
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_convert("UTC")

    df = df.astype(float)
    log_ohlcv_nans(df, interval, stage="loaded")
    df = df.dropna()
    return df

# ✅ 수정: 확장된 피처를 모두 계산하도록 로직 업데이트
def add_indicators_with_validation(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """기술적 지표 추가 (확장된 피처 목록 사용)"""
    is_btc = (tf == 'btc')
    prefix = "btc_" if is_btc else ""
    features = FEATURE_CATEGORIES_BY_TF.get(tf, [])
    
    result_df = pd.DataFrame(index=df.index)

    # 기본 OHLCV 및 파생 피처
    if prefix + "open" in features: result_df[prefix + "open"] = df["open"]
    if prefix + "high" in features: result_df[prefix + "high"] = df["high"]
    if prefix + "low" in features: result_df[prefix + "low"] = df["low"]
    if prefix + "close" in features: result_df[prefix + "close"] = df["close"]
    if prefix + "volume" in features: result_df[prefix + "volume"] = df["volume"]
    if prefix + "returns" in features: result_df[prefix + "returns"] = ta.percent_return(df["close"], append=True)
    if prefix + "high_low_range" in features: result_df[prefix + "high_low_range"] = df["high"] - df["low"]
    if prefix + "open_close_range" in features: result_df[prefix + "open_close_range"] = df["open"] - df["close"]

    # 기술적 지표
    if prefix + "rsi" in features: result_df[prefix + "rsi"] = ta.rsi(df["close"], length=14)
    if prefix + "stoch_k" in features or prefix + "stoch_d" in features:
        stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, append=True)
        if stoch is not None and not stoch.empty:
            if prefix + "stoch_k" in features: result_df[prefix + "stoch_k"] = stoch["STOCHk_14_3_3"]
            if prefix + "stoch_d" in features: result_df[prefix + "stoch_d"] = stoch["STOCHd_14_3_3"]
    
    if prefix + "macd" in features or prefix + "macd_signal" in features or prefix + "macd_hist" in features:
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9, append=True)
        if macd is not None and not macd.empty:
            if prefix + "macd" in features: result_df[prefix + "macd"] = macd["MACD_12_26_9"]
            if prefix + "macd_signal" in features: result_df[prefix + "macd_signal"] = macd["MACDs_12_26_9"]
            if prefix + "macd_hist" in features: result_df[prefix + "macd_hist"] = macd["MACDh_12_26_9"]

    if prefix + "cci" in features: result_df[prefix + "cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    if prefix + "roc" in features: result_df[prefix + "roc"] = ta.roc(df["close"], length=10)
    if prefix + "sma_10" in features: result_df[prefix + "sma_10"] = ta.sma(df["close"], length=10)
    if prefix + "sma_20" in features: result_df[prefix + "sma_20"] = ta.sma(df["close"], length=20)
    if prefix + "sma_50" in features: result_df[prefix + "sma_50"] = ta.sma(df["close"], length=50)
    if prefix + "ema_10" in features: result_df[prefix + "ema_10"] = ta.ema(df["close"], length=10)
    if prefix + "ema_20" in features: result_df[prefix + "ema_20"] = ta.ema(df["close"], length=20)
    if prefix + "ema_50" in features: result_df[prefix + "ema_50"] = ta.ema(df["close"], length=50)

    if prefix + "adx" in features or prefix + "plus_di" in features or prefix + "minus_di" in features:
        adx = ta.adx(df["high"], df["low"], df["close"], length=14, append=True)
        if adx is not None and not adx.empty:
            if prefix + "adx" in features: result_df[prefix + "adx"] = adx["ADX_14"]
            if prefix + "plus_di" in features: result_df[prefix + "plus_di"] = adx["DMP_14"]
            if prefix + "minus_di" in features: result_df[prefix + "minus_di"] = adx["DMN_14"]

    if prefix + "atr" in features: result_df[prefix + "atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    
    if prefix + "bb_percent_b" in features or prefix + "bb_bandwidth" in features:
        bbs = ta.bbands(df["close"], length=20, std=2, append=True)
        if bbs is not None and not bbs.empty:
            if prefix + "bb_percent_b" in features: result_df[prefix + "bb_percent_b"] = bbs["BBP_20_2.0"]
            if prefix + "bb_bandwidth" in features: result_df[prefix + "bb_bandwidth"] = bbs["BBB_20_2.0"]

    if prefix + "obv" in features: result_df[prefix + "obv"] = ta.obv(df["close"], df["volume"])
    if prefix + "volume_ma_20" in features: result_df[prefix + "volume_ma_20"] = df["volume"].rolling(window=20).mean()

    # Heikin-Ashi
    if any(f.startswith(prefix + "smoothed_ha") for f in features):
        # ✅ 수정: pandas-ta 라이브러리 문제 우회를 위해 직접 구현
        ha_df = df.copy()
        ha_df['HA_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # 첫 번째 HA_open 계산
        ha_df.iloc[0, ha_df.columns.get_loc('HA_open')] = (df.iloc[0]['open'] + df.iloc[0]['close']) / 2

        # 나머지 HA_open 계산 (루프 사용)
        for i in range(1, len(ha_df)):
            ha_df.iloc[i, ha_df.columns.get_loc('HA_open')] = (ha_df.iloc[i-1]['HA_open'] + ha_df.iloc[i-1]['HA_close']) / 2

        ha_df['HA_high'] = ha_df[['high', 'HA_open', 'HA_close']].max(axis=1)
        ha_df['HA_low'] = ha_df[['low', 'HA_open', 'HA_close']].min(axis=1)

        if ha_df is not None and not ha_df.empty:
            if prefix + "smoothed_ha_open" in features: result_df[prefix + "smoothed_ha_open"] = ha_df["HA_open"].rolling(window=5).mean()
            if prefix + "smoothed_ha_close" in features: result_df[prefix + "smoothed_ha_close"] = ha_df["HA_close"].rolling(window=5).mean()
            if prefix + "smoothed_ha_high" in features: result_df[prefix + "smoothed_ha_high"] = ha_df["HA_high"].rolling(window=5).mean()
            if prefix + "smoothed_ha_low" in features: result_df[prefix + "smoothed_ha_low"] = ha_df["HA_low"].rolling(window=5).mean()

    log_ohlcv_nans(result_df, tf, stage="warmup")
    result_df = result_df.dropna()
    log_ohlcv_nans(result_df, tf, stage="post_drop")

    return result_df

# ✅ 수정: BTC 피처 생성 로직을 add_indicators_with_validation 함수로 통합
def fetch_btc_historical_features(start_date: str, end_date: str, interval: str = "1h") -> pd.DataFrame:
    """BTC 피처 수집"""
    btc_df = fetch_ohlcv_with_extended_period("BTCUSDT", interval, start_date, end_date)
    log_ohlcv_nans(btc_df, "btc", stage="after_load")
    
    # add_indicators_with_validation 함수를 사용하여 BTC 피처 계산
    btc_features = add_indicators_with_validation(btc_df, "btc")
    
    log_ohlcv_nans(btc_features, "btc", stage="post_indicator")
    return btc_features

def collect_all_market_data() -> dict:
    """전체 마켓 데이터 수집 - MTF 독립 구조"""
    result = {}
    
    # 1. 각 타임프레임별 ETH 데이터 독립 수집
    for tf in TIMEFRAMES:
        print(f"[수집] {tf} 타임프레임 데이터...")
        interval = BINANCE_INTERVAL_MAP[tf]
        eth_df = fetch_ohlcv_with_extended_period(FUTURES_SYMBOL, interval, START_DATE, END_DATE)
        log_ohlcv_nans(eth_df, tf, stage="after_load")
        eth_indicators = add_indicators_with_validation(eth_df, tf)
        log_ohlcv_nans(eth_indicators, tf, stage="post_indicator")
        
        target_start_dt = pd.to_datetime(START_DATE, utc=True)
        result[tf] = eth_indicators[eth_indicators.index >= target_start_dt].copy()
        
        print(f"[완료] {tf}: {len(result[tf])} rows, {len(result[tf].columns)} features")
    
    # 2. 보조 자산 및 외부 데이터 수집
    for aux in AUX_TIMEFRAMES:
        if aux == "btc":
            print("[수집] BTC 피처...")
            btc_df = fetch_btc_historical_features(START_DATE, END_DATE)
            result["btc"] = btc_df.copy()
            print(f"[완료] BTC: {len(result['btc'])} rows, {len(result['btc'].columns)} features")

        elif aux == "dune":
            print("[수집] DUNE 온체인 데이터...")
            dune_df = collect_all_dune_data()
            if not dune_df.empty:
                result["dune"] = dune_df.copy()
                print(f"[완료] DUNE: {len(result['dune'])} rows, {len(result['dune'].columns)} features")
            else:
                result["dune"] = pd.DataFrame()
                print("[경고] DUNE 데이터 수집 실패")

    # 3. 데이터 저장
    save_dir = os.path.dirname(os.path.join(PROJECT_ROOT, RAW_DATA_PATH))
    os.makedirs(save_dir, exist_ok=True)
    
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
