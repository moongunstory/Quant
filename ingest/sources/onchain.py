# ingest/sources/onchain.py
"""
On-chain data collection functions.

Pure functions for collecting:
- Blockchain.com: Network activity metrics for Bitcoin.

All functions return pd.DataFrame without side effects.
"""

from datetime import datetime, timedelta
import pandas as pd

# ==================== Constants ====================

BLOCKCHAIN_COM_BASE = "https://api.blockchain.info/charts"

# On-chain metrics from Blockchain.com
BLOCKCHAIN_COM_METRICS = {
    "n-transactions": "Daily Transaction Count",
    "n-unique-addresses": "Daily Active Addresses",
    "estimated-transaction-volume-usd": "Daily Estimated Transaction Volume (USD)",
    "hash-rate": "Network Hash Rate (TH/s)",
    "difficulty": "Mining Difficulty",
    "mempool-size": "Mempool Size (bytes)",
    "avg-block-size": "Average Block Size (bytes)",
}

# ==================== Blockchain.com ====================

def collect_blockchain_com_metric(
    metric: str,
    session,
    logger
) -> pd.DataFrame:
    """
    Collect a specific time-series metric from Blockchain.com Charts API.
    
    Args:
        metric: The metric to collect (e.g., 'n-transactions').
        session: requests.Session
        logger: Logger instance
        
    Returns:
        DataFrame with columns: date, value
    """
    url = f"{BLOCKCHAIN_COM_BASE}/{metric}"
    params = {
        'timespan': '2years', # Fetch a long timespan and let the orchestrator filter
        'format': 'json'
    }
    
    logger.info(f"Collecting Blockchain.com on-chain metric: {metric}")
    
    try:
        response = session.get(url, params=params, timeout=20) # Increased timeout for potentially large response
        if not response.ok:
            logger.warning(f"Blockchain.com request for {metric} failed: {response.status_code}")
            return pd.DataFrame()
        
        data = response.json()
    except Exception as e:
        logger.error(f"Error fetching Blockchain.com {metric}: {e}")
        return pd.DataFrame()
    
    if not data or 'values' not in data:
        logger.warning(f"No data for Blockchain.com {metric}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    records = data['values']
    df = pd.DataFrame(records)
    df = df.rename(columns={'x': 'timestamp', 'y': 'value'})
    
    # Convert Unix timestamp to datetime and keep only the date part
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.normalize()
    df = df[['date', 'value']].copy()
    
    logger.info(f"Collected {len(df)} records for {metric}")
    
    return df
