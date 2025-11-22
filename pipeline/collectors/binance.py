# ingest/sources/binance.py
"""
Binance data collection functions.

Pure functions for collecting BTC market data:
- Spot/Futures OHLCV (1h only)
- Funding rate history
- Open Interest
- Long/Short ratio

All functions return pd.DataFrame without side effects.
"""

import os
from typing import Optional, Literal
from datetime import datetime
import pandas as pd
import time
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ==================== Constants ====================

SPOT_BASE = os.getenv("BINANCE_SPOT_BASE_URL", "https://api.binance.com")
FUTURES_BASE = os.getenv("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com")

SYMBOL_SPOT = "BTCUSDT"
SYMBOL_FUTURES = "BTCUSDT"

# OHLCV는 1시간 봉만 수집 (4h, 1d는 가공 단계에서 resample로 생성)
TIMEFRAMES = ['1h']

# Data retention policy
OHLCV_MAX_DAYS = 540
OI_LS_KEEP_ALL = True  # Accumulate forever


# ==================== OHLCV ====================

def collect_ohlcv(
    market: Literal['spot', 'futures'],
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    session,
    logger
) -> pd.DataFrame:
    """
    Collect OHLCV candlestick data.
    
    Args:
        market: 'spot' or 'futures'
        interval: '1h'  # ingestion 단계에서는 1h만 사용
        start_dt: Start datetime
        end_dt: End datetime
        session: requests.Session for HTTP calls
        logger: Logger instance
        
    Returns:
        DataFrame with columns:
            timestamp, open, high, low, close, volume, 
            taker_buy_base, quote_volume, taker_buy_ratio
    """
    if interval not in TIMEFRAMES:
        raise ValueError(f"Invalid interval: {interval}. Must be one of {TIMEFRAMES}")
    
    # Convert to milliseconds
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    # Select endpoint
    if market == 'spot':
        base_url = SPOT_BASE
        endpoint = "/api/v3/klines"
        symbol = SYMBOL_SPOT
    else:
        base_url = FUTURES_BASE
        endpoint = "/fapi/v1/klines"
        symbol = SYMBOL_FUTURES
    
    url = f"{base_url}{endpoint}"
    
    # Collect data in chunks (max 1500 per request)
    all_data = []
    current_start = start_ms
    limit = 1500
    
    logger.info(f"Collecting {market} OHLCV ({interval}): {start_dt.date()} to {end_dt.date()}")
    
    while current_start < end_ms:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': limit
        }
        
        try:
            response = session.get(url, params=params, timeout=10)
            if not response.ok:
                logger.warning(f"Request failed: {response.status_code}")
                break
            
            data = response.json()
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            break
        
        if not data or len(data) == 0:
            break
        
        all_data.extend(data)
        
        # Update start time for next chunk
        last_timestamp = data[-1][0]
        current_start = last_timestamp + 1
        
        # Break if we got less than limit (means we reached the end)
        if len(data) < limit:
            break
        
        time.sleep(0.1)  # Be nice to the API
    
    if not all_data:
        logger.warning(f"No data collected for {market} {interval}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Clean up
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 
             'taker_buy_base', 'quote_volume']].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert to numeric
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base', 'quote_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate taker buy ratio
    df['taker_buy_ratio'] = df['taker_buy_base'] / df['volume']
    
    logger.info(f"Collected {len(df)} candles")
    
    return df


# ==================== Funding Rate ====================

def collect_funding_rate(
    start_dt: datetime,
    end_dt: datetime,
    session,
    logger
) -> pd.DataFrame:
    """
    Collect funding rate history.
    
    Funding rate is charged every 8 hours (00:00, 08:00, 16:00 UTC).
    
    Returns:
        DataFrame with columns: timestamp, funding_rate
    """
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    url = f"{FUTURES_BASE}/fapi/v1/fundingRate"
    
    all_data = []
    current_start = start_ms
    limit = 1000
    
    logger.info(f"Collecting funding rate: {start_dt.date()} to {end_dt.date()}")
    
    while current_start < end_ms:
        params = {
            'symbol': SYMBOL_FUTURES,
            'startTime': current_start,
            'endTime': end_ms,
            'limit': limit
        }
        
        try:
            response = session.get(url, params=params, timeout=10)
            if not response.ok:
                logger.warning(f"Request failed: {response.status_code}")
                break
            
            data = response.json()
        except Exception as e:
            logger.error(f"Error fetching funding rate: {e}")
            break
        
        if not data or len(data) == 0:
            break
        
        all_data.extend(data)
        
        # Update for next chunk
        last_timestamp = data[-1]['fundingTime']
        current_start = last_timestamp + 1
        
        if len(data) < limit:
            break
        
        time.sleep(0.1)
    
    if not all_data:
        logger.warning("No funding rate data collected")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df = df[['fundingTime', 'fundingRate']].copy()
    df.columns = ['timestamp', 'funding_rate']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
    
    logger.info(f"Collected {len(df)} funding rate records")
    
    return df


# ==================== Open Interest ====================

def collect_open_interest(
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    session,
    logger
) -> pd.DataFrame:
    """
    Collect open interest history.
    
    API limitation: Only recent 30 days available.
    
    Args:
        interval: '1h', '4h', '1d' (ingest 파이프라인에서는 1h만 사용할 예정)
        
    Returns:
        DataFrame with columns: timestamp, open_interest
    """
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    url = f"{FUTURES_BASE}/futures/data/openInterestHist"
    
    params = {
        'symbol': SYMBOL_FUTURES,
        'period': interval,
        'startTime': start_ms,
        'endTime': end_ms,
        'limit': 500
    }
    
    logger.info(f"Collecting open interest ({interval}): {start_dt.date()} to {end_dt.date()}")
    
    try:
        response = session.get(url, params=params, timeout=10)
        if not response.ok:
            logger.warning(f"Request failed: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching open interest: {e}")
        return pd.DataFrame()
    
    if not data:
        logger.warning("No open interest data collected")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df = df[['timestamp', 'sumOpenInterest']].copy()
    df.columns = ['timestamp', 'open_interest']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open_interest'] = pd.to_numeric(df['open_interest'], errors='coerce')
    
    logger.info(f"Collected {len(df)} open interest records")
    
    return df


# ==================== Long/Short Ratio ====================

def collect_long_short_ratio(
    interval: str,
    ratio_type: Literal['top', 'global'],
    start_dt: datetime,
    end_dt: datetime,
    session,
    logger
) -> pd.DataFrame:
    """
    Collect long/short ratio.
    
    API limitation: Only recent 30 days available.
    
    Args:
        interval: '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'
                  (ingest 파이프라인에서는 '1h'만 사용할 계획)
        ratio_type: 
            - 'top': Top trader long/short ratio (recommended)
            - 'global': Global long/short ratio
            
    Returns:
        DataFrame with columns: timestamp, long_short_ratio, long_account, short_account
    """
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    # Select endpoint
    if ratio_type == 'top':
        endpoint = "/futures/data/topLongShortAccountRatio"
    else:
        endpoint = "/futures/data/globalLongShortAccountRatio"
    
    url = f"{FUTURES_BASE}{endpoint}"
    
    params = {
        'symbol': SYMBOL_FUTURES,
        'period': interval,
        'startTime': start_ms,
        'endTime': end_ms,
        'limit': 500
    }
    
    logger.info(f"Collecting {ratio_type} long/short ratio ({interval}): {start_dt.date()} to {end_dt.date()}")
    
    try:
        response = session.get(url, params=params, timeout=10)
        if not response.ok:
            logger.warning(f"Request failed: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching long/short ratio: {e}")
        return pd.DataFrame()
    
    if not data:
        logger.warning(f"No {ratio_type} long/short ratio data collected")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df = df[['timestamp', 'longShortRatio', 'longAccount', 'shortAccount']].copy()
    df.columns = ['timestamp', 'long_short_ratio', 'long_account', 'short_account']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    for col in ['long_short_ratio', 'long_account', 'short_account']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    logger.info(f"Collected {len(df)} long/short ratio records")
    
    return df
