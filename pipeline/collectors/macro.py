# ingest/sources/macro.py
"""
Macro & Traditional Market data collection functions.

Pure functions for collecting:
- FRED: Interest rates, economic indicators
- Finnhub: Stock indices, DXY, Gold, VIX
- Alpha Vantage: FX rates
- Yahoo Finance: Backup source

All functions return pd.DataFrame without side effects.
"""

import os
from typing import Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import time
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ==================== Constants ====================

# API URLs
FRED_BASE = "https://api.stlouisfed.org/fred"
FINNHUB_BASE = "https://finnhub.io/api/v1"
ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"
YAHOO_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

# Data retention
MAX_DAYS = 540

# FRED series IDs
FRED_SERIES = {
    # 국채 금리
    "DGS10": "US 10-Year Treasury",
    "DGS2": "US 2-Year Treasury",
    "DGS5": "US 5-Year Treasury",
    "T10Y2Y": "10Y-2Y Spread",
    "T10Y3M": "10Y-3M Spread",
    
    # 기준금리
    "DFF": "Fed Funds Rate (Daily)",
    "FEDFUNDS": "Fed Funds Rate (Monthly)",
    
    # 경기 지표
    "UNRATE": "Unemployment Rate (Monthly)",
    "CPIAUCSL": "CPI All Urban (Monthly)",
    "PPIACO": "PPI All Commodities (Monthly)",
    
    # 환율 (USD 기준)
    "DEXUSEU": "USD to EUR",
    "DEXJPUS": "JPY to USD",
    "DEXUSUK": "USD to GBP",
    "DEXUSAL": "USD to AUD",
    "DEXCHUS": "USD to CNY",
}

# Finnhub symbols
FINNHUB_SYMBOLS = {
    # 미국 지수
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "^DJI": "Dow Jones Industrial",
    
    # 달러/금/변동성
    "DX-Y.NYB": "Dollar Index (DXY)",
    "GC=F": "Gold Futures",
    "^VIX": "CBOE Volatility Index",
    
    # 채권 ETF
    "TLT": "20+ Year Treasury Bond ETF",
    "IEF": "7-10 Year Treasury Bond ETF"
}


# ==================== FRED ====================

def collect_fred_series(
    series_id: str,
    start_dt: datetime,
    end_dt: datetime,
    api_key: str,
    session,
    logger
) -> pd.DataFrame:
    """
    Collect FRED economic data series.
    
    Args:
        series_id: FRED series ID (e.g., 'DGS10')
        start_dt: Start datetime
        end_dt: End datetime
        api_key: FRED API key
        session: requests.Session
        logger: Logger instance
        
    Returns:
        DataFrame with columns: date, value
    """
    if not api_key:
        logger.warning("FRED API key not provided, skipping")
        return pd.DataFrame()
    
    url = f"{FRED_BASE}/series/observations"
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_dt.strftime('%Y-%m-%d'),
        'observation_end': end_dt.strftime('%Y-%m-%d')
    }
    
    logger.info(f"Collecting FRED {series_id}: {start_dt.date()} to {end_dt.date()}")
    
    try:
        response = session.get(url, params=params, timeout=10)
        if not response.ok:
            logger.warning(f"FRED request failed: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching FRED {series_id}: {e}")
        return pd.DataFrame()
    
    # 데이터 없음 처리 (월별 데이터는 정상)
    if not data or 'observations' not in data or not data['observations']:
        logger.info(f"No new data for {series_id} (normal for monthly series)")
        return pd.DataFrame()
    
    observations = data['observations']
    
    # Convert to DataFrame
    df = pd.DataFrame(observations)
    
    # 컬럼 존재 확인
    if 'date' not in df.columns or 'value' not in df.columns:
        logger.warning(f"Unexpected columns in {series_id}: {df.columns.tolist()}")
        return pd.DataFrame()
    
    df = df[['date', 'value']].copy()
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Remove missing values (marked as '.')
    df = df.dropna()
    
    if len(df) > 0:
        logger.info(f"Collected {len(df)} observations for {series_id}")
    
    return df


def collect_all_fred(
    start_dt: datetime,
    end_dt: datetime,
    api_key: str,
    session,
    logger
) -> Dict[str, pd.DataFrame]:
    """Collect all FRED series."""
    results = {}
    
    for series_id, description in FRED_SERIES.items():
        df = collect_fred_series(series_id, start_dt, end_dt, api_key, session, logger)
        if not df.empty:
            results[series_id] = df
        time.sleep(0.5)  # Be nice to FRED API
    
    return results


# ==================== Finnhub ====================

def collect_finnhub_quote(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    api_key: str,
    session,
    logger
) -> pd.DataFrame:
    """
    Collect daily OHLC from Finnhub.
    
    Args:
        symbol: Stock symbol (e.g., '^GSPC', 'DX-Y.NYB')
        start_dt: Start datetime
        end_dt: End datetime
        api_key: Finnhub API key
        session: requests.Session
        logger: Logger instance
        
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    if not api_key:
        logger.warning("Finnhub API key not provided, skipping")
        return pd.DataFrame()
    
    # Convert to Unix timestamp
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    
    url = f"{FINNHUB_BASE}/stock/candle"
    params = {
        'symbol': symbol,
        'resolution': 'D',  # Daily
        'from': start_ts,
        'to': end_ts,
        'token': api_key
    }
    
    logger.info(f"Collecting Finnhub {symbol}: {start_dt.date()} to {end_dt.date()}")
    
    try:
        response = session.get(url, params=params, timeout=10)
        if not response.ok:
            logger.warning(f"Finnhub request failed: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching Finnhub {symbol}: {e}")
        return pd.DataFrame()
    
    if not data or data.get('s') != 'ok':
        logger.warning(f"No data for Finnhub {symbol}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(data['t'], unit='s'),
        'open': data['o'],
        'high': data['h'],
        'low': data['l'],
        'close': data['c'],
        'volume': data['v']
    })
    
    logger.info(f"Collected {len(df)} candles for {symbol}")
    
    return df


def collect_all_finnhub(
    start_dt: datetime,
    end_dt: datetime,
    api_key: str,
    session,
    logger
) -> Dict[str, pd.DataFrame]:
    """Collect all Finnhub symbols."""
    results = {}
    
    for symbol, description in FINNHUB_SYMBOLS.items():
        df = collect_finnhub_quote(symbol, start_dt, end_dt, api_key, session, logger)
        if not df.empty:
            results[symbol] = df
        time.sleep(1.1)  # Rate limit: 60/min
    
    return results


# ==================== Yahoo Finance (Backup) ====================

def collect_yahoo_finance(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    session,
    logger
) -> pd.DataFrame:
    """
    Collect data from Yahoo Finance (no API key needed).
    
    Backup source if other APIs fail.
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    # Yahoo uses Unix timestamps
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    
    url = f"{YAHOO_BASE}/{symbol}"
    params = {
        'period1': start_ts,
        'period2': end_ts,
        'interval': '1d'
    }
    
    logger.info(f"Collecting Yahoo Finance {symbol}: {start_dt.date()} to {end_dt.date()}")
    
    try:
        response = session.get(url, params=params, timeout=10)
        if not response.ok:
            logger.warning(f"Yahoo request failed: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching Yahoo {symbol}: {e}")
        return pd.DataFrame()
    
    if not data or 'chart' not in data:
        logger.warning(f"No Yahoo data for {symbol}")
        return pd.DataFrame()
    
    result = data['chart']['result'][0]
    timestamps = result['timestamp']
    quotes = result['indicators']['quote'][0]
    
    # Convert to DataFrame
    df = pd.DataFrame({
        'date': pd.to_datetime(timestamps, unit='s'),
        'open': quotes['open'],
        'high': quotes['high'],
        'low': quotes['low'],
        'close': quotes['close'],
        'volume': quotes['volume']
    })
    
    df = df.dropna()
    
    logger.info(f"Collected {len(df)} Yahoo candles for {symbol}")
    
    return df