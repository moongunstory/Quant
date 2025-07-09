import pandas as pd
import numpy as np
import requests
import time
from binance.client import Client
import pandas_ta as ta
import os
import sys
from datetime import timedelta
import logging
from typing import Dict
import pickle

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    DUNE_API_KEY, BINANCE_API_KEY, BINANCE_SECRET_KEY,
    ETH_TIMEFRAMES, ENABLE_BTC, ENABLE_DUNE, BTC_INTERVAL,
    START_DATE, END_DATE, FEATURE_CATEGORIES_BY_TF, DUNE_QUERY_PARTS,
    BINANCE_INTERVAL_MAP, FUTURES_SYMBOL,
    TRAIN_PICKLE_PATHS
)

client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
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


# =============================================================================
# 데이터 수집 함수들
# =============================================================================

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
    # 키를 숫자 기준으로 정렬하여 순서대로 쿼리 실행
    part_names = sorted(DUNE_QUERY_PARTS.keys(), key=lambda k: int(k.split('_')[-1]))
    
    for i in range(0, len(part_names), 3):
        batch = part_names[i:i+3]
        for part_name in batch:
            query_id = DUNE_QUERY_PARTS[part_name]
            logger.info(f"[Dune] Fetching data for {part_name} (Query ID: {query_id})...")
            df = call_dune_api(query_id, f"Part_{part_name}")
            if not df.empty:
                all_dfs.append(df)
        
        if i + 3 < len(part_names):
            logger.info("[Dune] Waiting for 5 seconds before next batch...")
            time.sleep(5)
    
    if all_dfs:
        # 모든 데이터프레임을 하나로 합치고, 인덱스(시간) 기준으로 정렬
        dune_df = pd.concat(all_dfs)
        dune_df = dune_df.sort_index()
        # 중복된 인덱스가 있을 경우 첫 번째 값만 남김 (기간이 겹칠 경우 대비)
        dune_df = dune_df[~dune_df.index.duplicated(keep='first')]
        logger.info(f"[Dune] Successfully collected and combined data. Total rows: {len(dune_df)}")
        return dune_df
        
    logger.warning("[Dune] No data collected from Dune Analytics.")
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

        # 다음 요청을 위해 마지막 캔들의 close_time + 1ms로 이동
        last_close_time = klines[-1][6]  # close_time
        if last_close_time >= end_ms:
            break

        start_ms = last_close_time + 1

        # Binance API 요청 제한 대응을 위해 약간 대기 (optional)
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


def add_indicators_with_validation(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """기술적 지표 추가 + OHLCV 컬럼 포함"""
    # 원본 OHLCV 컬럼을 포함하여 복사
    result_df = df[["open", "high", "low", "close", "volume"]].copy()
    features = FEATURE_CATEGORIES_BY_TF.get(tf, [])

    # 1. 기본 가격 및 거래량 피처
    if "returns" in features: result_df['returns'] = result_df['close'].pct_change()
    if "high_low_range" in features: result_df['high_low_range'] = (result_df['high'] - result_df['low']) / result_df['close']
    if "open_close_range" in features: result_df['open_close_range'] = (result_df['close'] - result_df['open']) / result_df['open']

    # 2. 기술 지표 계산 (pandas_ta 활용)
    # 모멘텀 지표
    if "rsi" in features: result_df['rsi'] = ta.rsi(result_df['close'])
    if "stoch_k" in features and "stoch_d" in features:
        stoch = ta.stoch(result_df['high'], result_df['low'], result_df['close'])
        if stoch is not None and not stoch.empty:
            result_df['stoch_k'] = stoch['STOCHk_14_3_3']
            result_df['stoch_d'] = stoch['STOCHd_14_3_3']
    if "cci" in features: result_df['cci'] = ta.cci(result_df['high'], result_df['low'], result_df['close'])
    if "roc" in features: result_df['roc'] = ta.roc(result_df['close'])
    if "mom" in features: result_df['mom'] = ta.mom(result_df['close'])

    # 추세 지표
    if "macd" in features and "macd_signal" in features and "macd_hist" in features:
        macd = ta.macd(result_df['close'])
        if macd is not None and not macd.empty:
            result_df['macd'] = macd['MACD_12_26_9']
            result_df['macd_signal'] = macd['MACDs_12_26_9']
            result_df['macd_hist'] = macd['MACDh_12_26_9']
    if "sma_10" in features: result_df['sma_10'] = ta.sma(result_df['close'], length=10)
    if "sma_20" in features: result_df['sma_20'] = ta.sma(result_df['close'], length=20)
    if "sma_50" in features: result_df['sma_50'] = ta.sma(result_df['close'], length=50)
    if "ema_10" in features: result_df['ema_10'] = ta.ema(result_df['close'], length=10)
    if "ema_20" in features: result_df['ema_20'] = ta.ema(result_df['close'], length=20)
    if "ema_50" in features: result_df['ema_50'] = ta.ema(result_df['close'], length=50)
    if "adx" in features and "plus_di" in features and "minus_di" in features:
        adx_result = ta.adx(result_df['high'], result_df['low'], result_df['close'])
        if adx_result is not None and not adx_result.empty:
            result_df['adx'] = adx_result['ADX_14']
            result_df['plus_di'] = adx_result['DMP_14']
            result_df['minus_di'] = adx_result['DMN_14']

    # 변동성 지표
    if "atr" in features: result_df['atr'] = ta.atr(result_df['high'], result_df['low'], result_df['close'])
    if "bb_upper" in features and "bb_middle" in features and "bb_lower" in features and "bb_percent_b" in features and "bb_bandwidth" in features:
        bbands = ta.bbands(result_df['close'])
        if bbands is not None and not bbands.empty:
            result_df['bb_upper'] = bbands['BBL_20_2.0'] # BBL is lower, BBU is upper
            result_df['bb_middle'] = bbands['BBM_20_2.0']
            result_df['bb_lower'] = bbands['BBU_20_2.0']
            result_df['bb_percent_b'] = bbands['BBP_20_2.0']
            result_df['bb_bandwidth'] = bbands['BBB_20_2.0']

    # 거래량 지표
    if "obv" in features: result_df['obv'] = ta.obv(result_df['close'], result_df['volume'])
    if "volume_ma_20" in features: result_df['volume_ma_20'] = ta.sma(result_df['volume'], length=20)

    # volume_ratio는 volume_ma_20이 필요
    if "volume_ratio" in features and "volume_ma_20" in result_df.columns:
        result_df['volume_ratio'] = result_df['volume'] / result_df['volume_ma_20']

    # Heikin Ashi 지표 (chained assignment 경고 방지용 대체 구현)
    def compute_smoothed_heikin_ashi(df: pd.DataFrame, length: int = 3) -> pd.DataFrame:
        ha = pd.DataFrame(index=df.index)
        ha['HA_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha['HA_open'] = 0.0
        ha.iloc[0, ha.columns.get_loc("HA_open")] = (df.iloc[0]["open"] + df.iloc[0]["close"]) / 2
        for i in range(1, len(df)):
            ha.iloc[i, ha.columns.get_loc("HA_open")] = (
                ha.iloc[i - 1]["HA_open"] + ha.iloc[i - 1]["HA_close"]
            ) / 2
        ha['HA_high'] = df[['high', 'open', 'close']].max(axis=1)
        ha['HA_low'] = df[['low', 'open', 'close']].min(axis=1)

        # Smooth with SMA
        ha_smooth = pd.DataFrame(index=df.index)
        ha_smooth['smoothed_ha_open'] = ha['HA_open'].rolling(length).mean()
        ha_smooth['smoothed_ha_close'] = ha['HA_close'].rolling(length).mean()
        ha_smooth['smoothed_ha_high'] = ha['HA_high'].rolling(length).mean()
        ha_smooth['smoothed_ha_low'] = ha['HA_low'].rolling(length).mean()

        return ha_smooth

    if any(f in features for f in ["smoothed_ha_open", "smoothed_ha_close", "smoothed_ha_high", "smoothed_ha_low"]):
        ha_df = compute_smoothed_heikin_ashi(result_df)
        for col in ha_df.columns:
            if col in features:
                result_df[col] = ha_df[col]

    # Log warm-up NaNs before clipping
    log_ohlcv_nans(result_df, tf, stage="warmup")

    # Drop warm-up rows where indicators produced NaNs
    result_df = result_df.dropna()

    # Log after drop to confirm cleanliness
    log_ohlcv_nans(result_df, tf, stage="post_drop")

    return result_df


def fetch_btc_historical_features(start_date: str, end_date: str, interval: str = None) -> pd.DataFrame:
    """BTC 피처 수집"""
    if interval is None:
        interval = BTC_INTERVAL  # config에서 설정된 기본값 사용
    
    btc_df = fetch_ohlcv_with_extended_period("BTCUSDT", interval, start_date, end_date)
    log_ohlcv_nans(btc_df, "btc", stage="after_load")
    
    btc_features = pd.DataFrame(index=btc_df.index)
    
    # BTC 피처 계산 (FEATURE_CATEGORIES_BY_TF에 정의된 피처들)
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

    # BTC Heikin Ashi 지표
    def compute_smoothed_heikin_ashi(df: pd.DataFrame, length: int = 3) -> pd.DataFrame:
        ha = pd.DataFrame(index=df.index)
        ha['HA_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha['HA_open'] = 0.0
        ha.loc[df.index[0], 'HA_open'] = (df.iloc[0]['open'] + df.iloc[0]['close']) / 2
        for i in range(1, len(df)):
            ha.iloc[i, ha.columns.get_loc('HA_open')] = (
                ha.iloc[i - 1]['HA_open'] + ha.iloc[i - 1]['HA_close']
            ) / 2
        ha['HA_high'] = df[['high', 'open', 'close']].max(axis=1)
        ha['HA_low'] = df[['low', 'open', 'close']].min(axis=1)

        ha_smooth = pd.DataFrame(index=df.index)
        ha_smooth['btc_smoothed_ha_open'] = ha['HA_open'].rolling(length).mean()
        ha_smooth['btc_smoothed_ha_close'] = ha['HA_close'].rolling(length).mean()
        ha_smooth['btc_smoothed_ha_high'] = ha['HA_high'].rolling(length).mean()
        ha_smooth['btc_smoothed_ha_low'] = ha['HA_low'].rolling(length).mean()
        return ha_smooth

    # BTC Heikin Ashi 지표 (리팩토링 적용)
    if any(f in features for f in ["btc_smoothed_ha_open", "btc_smoothed_ha_close", "btc_smoothed_ha_high", "btc_smoothed_ha_low"]):
        ha_df = compute_smoothed_heikin_ashi(btc_df)
        for col in ha_df.columns:
            if col in features:
                btc_features[col] = ha_df[col]

    btc_features = btc_features.fillna(0)
    log_ohlcv_nans(btc_df, "btc", stage="post_indicator")
    return btc_features


# =============================================================================
# 데이터 가공 함수들
# =============================================================================

def create_dune_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """DUNE 파생 피처 생성 - DUNE DataFrame만 처리"""
    df_processed = df.copy()
    
    # 수치형 파생 피처
    if 'eth_to_cex' in df.columns and 'eth_from_cex' in df.columns:
        df_processed['cex_netflow'] = df['eth_to_cex'] - df['eth_from_cex']
    
    if 'whale_to_cex' in df.columns and 'cex_to_whale' in df.columns:
        df_processed['whale_netflow'] = df['whale_to_cex'] - df['cex_to_whale']
    
    if 'deposit_amount' in df.columns and 'withdraw_amount' in df.columns:
        df_processed['staking_netflow'] = df['deposit_amount'] - df['withdraw_amount']
    
    # 이벤트 flag 피처 (00:00에만 값 설정, 나머지는 NaN)
    midnight_mask = (df_processed.index.hour == 0) & (df_processed.index.minute == 0)
    
    if 'cex_netflow' in df_processed.columns:
        df_processed['cex_increase_flag'] = np.nan
        df_processed.loc[midnight_mask, 'cex_increase_flag'] = (
            df_processed.loc[midnight_mask, 'cex_netflow'] > 0
        ).astype(int)
    
    if 'whale_netflow' in df_processed.columns:
        df_processed['whale_increase_flag'] = np.nan
        df_processed.loc[midnight_mask, 'whale_increase_flag'] = (
            df_processed.loc[midnight_mask, 'whale_netflow'] > 0
        ).astype(int)
    
    if 'staking_netflow' in df_processed.columns:
        df_processed['staking_increase_flag'] = np.nan
        df_processed.loc[midnight_mask, 'staking_increase_flag'] = (
            df_processed.loc[midnight_mask, 'staking_netflow'] > 0
        ).astype(int)
    
    return df_processed


def apply_feature_processing(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """피처별 처리 정책 적용 - BTC/DUNE DataFrame만 처리"""
    df_processed = df.copy()
    log_ohlcv_nans(df_processed, data_type, stage="before_process")
    
    if data_type == "btc":
        # BTC 피처 .ffill() 적용
        btc_features = [col for col in df.columns if col.startswith('btc_')]
        for col in btc_features:
            df_processed[col] = df_processed[col].ffill()
    
    elif data_type == "dune":
        # DUNE 수치형 피처 .ffill() 적용
        dune_numeric_features = ['cex_netflow', 'whale_netflow', 'staking_netflow']
        for col in dune_numeric_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].ffill()
        # DUNE flag 피처는 .ffill() 금지 (그대로 유지)
    
    log_ohlcv_nans(df_processed, data_type, stage="after_process")
    return df_processed


# =============================================================================
# 메인 파이프라인 함수
# =============================================================================

def run_data_pipeline() -> Dict[str, pd.DataFrame]:
    """전체 데이터 파이프라인 실행"""
    print("="*60)
    print("🚀 데이터 파이프라인 시작")
    print("="*60)
    
    result = {}
    
    # 1. ETH 타임프레임별 데이터 수집 및 처리
    print("\n📊 ETH 데이터 수집 및 처리 중...")
    for tf in ETH_TIMEFRAMES:
        print(f"  ⏰ {tf} 타임프레임 처리 중...")
        interval = BINANCE_INTERVAL_MAP[tf]
        eth_df = fetch_ohlcv_with_extended_period(FUTURES_SYMBOL, interval, START_DATE, END_DATE)
        log_ohlcv_nans(eth_df, tf, stage="after_load")
        eth_indicators = add_indicators_with_validation(eth_df, tf)
        log_ohlcv_nans(eth_indicators, tf, stage="post_indicator")
        
        # 시작 날짜부터 필터링
        target_start_dt = pd.to_datetime(START_DATE, utc=True)
        result[tf] = eth_indicators[eth_indicators.index >= target_start_dt].copy()
        
        print(f"  ✅ {tf}: {len(result[tf])} rows, {len(result[tf].columns)} features")
    
    # 2. BTC 데이터 수집 및 처리 (선택적)
    if ENABLE_BTC:
        print("\n🟡 BTC 데이터 수집 및 처리 중...")
        btc_df = fetch_btc_historical_features(START_DATE, END_DATE)
        btc_processed = apply_feature_processing(btc_df, "btc")
        result["btc"] = btc_processed.copy()
        print(f"  ✅ BTC: {len(result['btc'])} rows, {len(result['btc'].columns)} features")
    else:
        print("\n⏭️  BTC 데이터 수집 스킵 (비활성화됨)")
    
    # 3. DUNE 데이터 수집 및 처리 (선택적)
    if ENABLE_DUNE:
        print("\n🔗 DUNE 온체인 데이터 수집 및 처리 중...")
        try:
            dune_df = collect_all_dune_data()
            if not dune_df.empty:
                dune_with_derived = create_dune_derived_features(dune_df)
                dune_processed = apply_feature_processing(dune_with_derived, "dune")
                result["dune"] = dune_processed.copy()
                print(f"  ✅ DUNE: {len(result['dune'])} rows, {len(result['dune'].columns)} features")
            else:
                result["dune"] = pd.DataFrame()
                print("  ⚠️  DUNE 데이터 수집 실패 - 빈 DataFrame으로 설정")
        except Exception as e:
            print(f"  ⚠️  DUNE 데이터 처리 중 오류 발생: {e}")
            result["dune"] = pd.DataFrame()
    else:
        print("\n⏭️  DUNE 데이터 수집 스킵 (비활성화됨)")
    
    # 4. 최종 학습용 데이터 저장
    print("\n🎯 최종 학습용 데이터 저장 중...")
    long_path = TRAIN_PICKLE_PATHS["long"]
    short_path = TRAIN_PICKLE_PATHS["short"]

    os.makedirs(os.path.dirname(long_path), exist_ok=True)
    os.makedirs(os.path.dirname(short_path), exist_ok=True)

    # Long/Short 모델용 데이터 저장
    with open(long_path, "wb") as f:
        pickle.dump(result, f)
    
    with open(short_path, "wb") as f:
        pickle.dump(result, f)

    print(f"  📁 Long 모델용: {long_path}")
    print(f"  📁 Short 모델용: {short_path}")
    
    # 5. 최종 결과 요약
    print("\n" + "="*60)
    print("🎉 데이터 파이프라인 완료!")
    print("="*60)
        
    total_features = sum(len(df.columns) for df in result.values() if not df.empty)
    non_empty_datasets = len([df for df in result.values() if not df.empty])
    
    print(f"📈 총 {non_empty_datasets} 개 데이터셋")
    print(f"🔢 총 {total_features} 개 피처")
    
    for key, df in result.items():
        if not df.empty:
            print(f"  • {key}: {len(df)} rows × {len(df.columns)} cols")
    
    return result


if __name__ == "__main__":
    mtf_data = run_data_pipeline()