# train/prepare/fetch_binance.py

import os
import requests
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv

from config.paths import (
    get_ohlcv_path,
    get_funding_rate_path,
    get_index_price_path,
)

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
client = Client(API_KEY, API_SECRET)

BASE_URL = "https://fapi.binance.com/futures/data/openInterestHist"

def fetch_ohlcv(symbol: str, interval: str, start_date: str, end_date: str) -> pd.DataFrame:
    klines = client.get_historical_klines(symbol, interval, start_str=start_date, end_str=end_date)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def fetch_funding_rate(symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
    start = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)
    all_data = []

    while start < end:
        next_time = start + pd.Timedelta(days=30)
        if next_time > end:
            next_time = end

        data = client.futures_funding_rate(
            symbol=symbol,
            startTime=_to_ms(start),
            endTime=_to_ms(next_time)
        )
        if not data:
            break

        df = pd.DataFrame(data)
        df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df = df[['fundingTime', 'fundingRate']].rename(columns={'fundingTime': 'timestamp'})
        df['fundingRate'] = df['fundingRate'].astype(float)
        all_data.append(df)

        start = df['timestamp'].max() + pd.Timedelta(hours=8)

    if not all_data:
        raise ValueError(f"No funding data fetched for {symbol}")

    return pd.concat(all_data).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

def fetch_index_price(symbol: str, interval: str, start_time: str, end_time: str) -> pd.DataFrame:
    url = "https://fapi.binance.com/fapi/v1/indexPriceKlines"
    start = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)

    all_data = []
    current = start

    while current < end:
        next_time = current + pd.Timedelta(minutes=1500 * 5)
        if next_time > end:
            next_time = end

        params = {
            "pair": symbol,
            "interval": interval,
            "startTime": int(current.timestamp() * 1000),
            "endTime": int(next_time.timestamp() * 1000),
            "limit": 1500
        }

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if not data:
            break

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["timestamp", "open", "high", "low", "close"]].rename(columns={
            "open": "index_open",
            "high": "index_high",
            "low": "index_low",
            "close": "index_close"
        })
        df[["index_open", "index_high", "index_low", "index_close"]] = df[
            ["index_open", "index_high", "index_low", "index_close"]
        ].astype(float)

        all_data.append(df)
        current = df["timestamp"].max() + pd.Timedelta(minutes=5)

    if not all_data:
        raise ValueError(f"No index price data fetched for {symbol}")

    return pd.concat(all_data).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

def fetch_binance_data(start_date: str, end_date: str):
    for symbol in ['ETHUSDT', 'BTCUSDT']:
        print(f"[Binance] Fetching {symbol} data...")

        ohlcv = fetch_ohlcv(symbol, Client.KLINE_INTERVAL_5MINUTE, start_date, end_date)
        funding = fetch_funding_rate(symbol, start_date, end_date)
        index = fetch_index_price(symbol, interval="5m", start_time=start_date, end_time=end_date)

        ohlcv_path = get_ohlcv_path(symbol)
        funding_path = get_funding_rate_path(symbol)
        index_path = get_index_price_path(symbol)

        ohlcv_path.parent.mkdir(parents=True, exist_ok=True)
        funding_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        ohlcv.to_csv(ohlcv_path, index=False)
        funding.to_csv(funding_path, index=False)
        index.to_csv(index_path, index=False)

        print(f"[{symbol}] Saved ohlcv → {ohlcv_path}")
        print(f"[{symbol}] Saved funding → {funding_path}")
        print(f"[{symbol}] Saved index → {index_path}")

def _to_ms(dt) -> int:
    return int(pd.Timestamp(dt).timestamp() * 1000)
