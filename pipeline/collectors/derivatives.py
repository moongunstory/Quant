# ingest/sources/derivatives.py
"""
Derivatives market data collection functions.

Pure functions for collecting:
- Deribit: Volatility index data for Bitcoin.

All functions return pd.DataFrame without side effects.
"""

import pandas as pd
from datetime import datetime

# ==================== Constants ====================

DERIBIT_BASE = "https://www.deribit.com/api/v2"

# Deribit metrics to collect
DERIBIT_METRICS = {
    "BTC": "BTC Volatility Index (DVOL)",
}

# ==================== Deribit ====================

def collect_deribit_dvol(
    currency: str,
    session,
    logger
) -> pd.DataFrame:
    """
    Collects historical volatility index (DVOL) data from Deribit.
    
    Args:
        currency: The currency to fetch (e.g., 'BTC').
        session: requests.Session
        logger: Logger instance
        
    Returns:
        DataFrame with columns: date, value
    """
    url = f"{DERIBIT_BASE}/public/get_volatility_index_data"
    
    # Deribit API requires start and end timestamps in milliseconds.
    # We fetch the maximum possible range and let the orchestrator filter it.
    end_timestamp = int(datetime.now().timestamp() * 1000)
    start_timestamp = int((datetime.now() - pd.Timedelta(days=365*5)).timestamp() * 1000) # 5 years back

    params = {
        'currency': currency,
        'start_timestamp': start_timestamp,
        'end_timestamp': end_timestamp,
        'resolution': '1D' # Daily resolution
    }
    
    logger.info(f"Collecting Deribit DVOL for '{currency}'")
    
    try:
        response = session.get(url, params=params, timeout=20)
        if not response.ok:
            logger.warning(f"Deribit DVOL request for {currency} failed: {response.status_code} - {response.text}")
            return pd.DataFrame()
        
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching Deribit DVOL for {currency}: {e}")
        return pd.DataFrame()
    
    if not data or 'result' not in data or 'data' not in data['result']:
        logger.warning(f"No DVOL data returned for Deribit {currency}")
        return pd.DataFrame()
    
    # Response format is a list of lists: [timestamp, open, high, low, close]
    records = data['result']['data']
    df = pd.DataFrame(records, columns=['timestamp', 'open', 'high', 'low', 'close'])
    
    if df.empty:
        return pd.DataFrame()

    # Convert timestamp and create a single 'value' column (close price of the index)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.normalize()
    df['value'] = df['close']
    df = df[['date', 'value']].copy()
    
    logger.info(f"Collected {len(df)} records for {currency} DVOL")
    
    return df