# ingest/sources/sentiment.py
"""
Sentiment data collection functions.

Pure functions for collecting:
- Alternative.me Fear & Greed Index
- CoinGecko market sentiment

All functions return pd.DataFrame without side effects.
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import time
from dotenv import load_dotenv

# Load environment
load_dotenv()

# ==================== Constants ====================

FNG_BASE = os.getenv("FNG_BASE_URL", "https://api.alternative.me")
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINSTATS_BASE = "https://openapi.coinstats.app/public/v1"

MAX_DAYS = 540


# ==================== Fear & Greed Index ====================

def collect_fear_greed(
    start_dt: datetime,
    end_dt: datetime,
    session,
    logger
) -> pd.DataFrame:
    """
    Collect Fear & Greed Index from Alternative.me.
    
    No API key required.
    
    Returns:
        DataFrame with columns: timestamp, value, value_classification
    """
    # Calculate number of days
    days = (end_dt - start_dt).days + 1
    
    url = "https://api.alternative.me/fng/"
    params = {
        'limit': min(days, 365),
        'format': 'json',
        'date_format': 'us'
    }
    
    logger.info(f"Collecting Fear & Greed Index: {start_dt.date()} to {end_dt.date()}")
    
    try:
        response = session.get(url, params=params, timeout=10)
        if not response.ok:
            logger.warning(f"Fear & Greed request failed: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed: {e}")
        return pd.DataFrame()
    
    if not data or 'data' not in data:
        logger.warning("No Fear & Greed data returned")
        return pd.DataFrame()
    
    records = data['data']
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    if df.empty:
        return df
    
    # Clean up
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df[['timestamp', 'value', 'value_classification']].copy()
    
    # Filter date range
    df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Collected {len(df)} Fear & Greed records")
    
    return df


# ==================== CoinGecko Sentiment ====================

def collect_coingecko_sentiment(
    api_key: str,
    session,
    logger
) -> pd.DataFrame:
    """
    Collect sentiment and market data from CoinGecko for Bitcoin.
    This now replaces the CoinStats call by fetching more market data.
    
    Note: Current snapshot only (free tier limitation).
    """
    if not api_key:
        logger.warning("CoinGecko API key not provided, skipping")
        return pd.DataFrame()
    
    url = f"{COINGECKO_BASE}/coins/bitcoin"
    
    params = {
        'localization': 'false',
        'tickers': 'false',
        'market_data': 'true',  # Fetch market data to replace CoinStats
        'community_data': 'true',
        'developer_data': 'true',
        'sparkline': 'false'
    }
    
    headers = {
        'x-cg-demo-api-key': api_key
    }
    
    logger.info("Collecting CoinGecko sentiment & market data (current snapshot)")
    
    try:
        response = session.get(url, params=params, headers=headers, timeout=10)
        if not response.ok:
            logger.warning(f"CoinGecko request failed: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching CoinGecko sentiment: {e}")
        return pd.DataFrame()
    
    if not data:
        logger.warning("No CoinGecko data returned")
        return pd.DataFrame()
    
    # Extract sentiment and market data
    try:
        community = data.get('community_data', {})
        developer = data.get('developer_data', {})
        market = data.get('market_data', {})
        
        record = {
            'timestamp': pd.Timestamp.now().normalize(),
            'sentiment_up_percentage': data.get('sentiment_votes_up_percentage'),
            'sentiment_down_percentage': data.get('sentiment_votes_down_percentage'),
            'community_score': community.get('community_score'),
            'developer_score': developer.get('developer_score'),
            'public_interest_score': data.get('public_interest_score'),
            'coingecko_rank': data.get('coingecko_rank'),
            'market_cap_rank': data.get('market_cap_rank'),
            'price_change_percentage_24h': market.get('price_change_percentage_24h'),
            'price_change_percentage_7d': market.get('price_change_percentage_7d'),
            'total_volume_usd': market.get('total_volume', {}).get('usd'),
        }
        
        df = pd.DataFrame([record])
        
        # Convert to numeric
        for col in df.columns:
            if col != 'timestamp':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info("Collected CoinGecko sentiment snapshot")
        
        return df

    except Exception as e:
        logger.error(f"Failed to parse CoinGecko data: {e}")
        return pd.DataFrame()


# ==================== CoinGecko Historical Market Data ====================

def collect_coingecko_market_history(
    start_dt: datetime,
    end_dt: datetime,
    session,
    logger,
    coin_id: str = "bitcoin",
    vs_currency: str = "usd"
) -> pd.DataFrame:
    """
    Collect historical market data from CoinGecko.

    Free tier provides:
    - 7-90 days: Hourly data (perfect for 1h table)
    - 90+ days: Daily data

    Returns:
        DataFrame with columns: timestamp, market_cap, total_volume
    """
    days = (end_dt - start_dt).days + 1

    # Use hourly data if within 90 days
    if days <= 90:
        logger.info(f"Collecting CoinGecko hourly market data: {start_dt.date()} to {end_dt.date()}")
    else:
        logger.info(f"Collecting CoinGecko daily market data: {start_dt.date()} to {end_dt.date()}")

    url = f"{COINGECKO_BASE}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': vs_currency,
        'days': min(days, 365),
        'interval': 'hourly' if days <= 90 else 'daily'
    }

    try:
        response = session.get(url, params=params, timeout=15)
        if not response.ok:
            logger.warning(f"CoinGecko market history request failed: {response.status_code}")
            return pd.DataFrame()

        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching CoinGecko market history: {e}")
        return pd.DataFrame()

    if not data:
        logger.warning("No CoinGecko market history data returned")
        return pd.DataFrame()

    try:
        # Extract market caps and volumes
        market_caps = data.get('market_caps', [])
        volumes = data.get('total_volumes', [])

        if not market_caps or not volumes:
            logger.warning("CoinGecko returned empty market data")
            return pd.DataFrame()

        # Convert to DataFrame
        df_mc = pd.DataFrame(market_caps, columns=['timestamp_ms', 'market_cap'])
        df_vol = pd.DataFrame(volumes, columns=['timestamp_ms', 'total_volume'])

        # Merge on timestamp
        df = df_mc.merge(df_vol, on='timestamp_ms', how='inner')

        # Convert timestamp (milliseconds to datetime)
        df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms', utc=True)
        df = df.drop(columns=['timestamp_ms'])

        # Filter date range
        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Collected {len(df)} CoinGecko market records")

        return df

    except Exception as e:
        logger.error(f"Failed to parse CoinGecko market history: {e}")
        return pd.DataFrame()
