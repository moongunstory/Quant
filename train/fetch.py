"""
Mini Module 1 — ETH Futures Ingest (UM) with Auto 3-Way Split

Ultra-lean version. No CLI. No metadata JSON. No gap checks. Minimal logs.
- Market: USDT‑M Futures (UM) only
- Symbol: ETHUSDT only
- Intervals: [5m, 15m, 1h, 4h]
- Range: START_DATE → END_DATE (edit constants below)
- Split: train/val/test = 70/15/15 (by time, per-interval)
- Output: ./ai_binance/data/raw/fut_{train|val|test}_data_{interval}.parquet

Run: python module_1_ingest.py
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd
import requests

# ===== User constants (edit as needed) =====
START_DATE = "2017-08-25"  # inclusive UTC
END_DATE = None            # exclusive UTC; None → today UTC
INTERVALS = ["5m", "15m", "1h", "4h"]
SPLIT = (0.70, 0.15, 0.15)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw"))
SLEEP = 0.25
TIMEOUT = 20

# ===== Binance UM Futures endpoints =====
BASE = "https://fapi.binance.com"
KLINES = "/fapi/v1/klines"


def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _fetch(symbol: str, interval: str, start_ms: int, end_ms: Optional[int]) -> pd.DataFrame:
    rows: List[List] = []
    next_start = start_ms
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": next_start, "limit": 1500}
        if end_ms is not None:
            params["endTime"] = end_ms
        r = requests.get(BASE + KLINES, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        rows.extend(page)
        last_close = page[-1][6]
        next_start = last_close + 1
        if end_ms is not None and next_start > end_ms:
            break
        time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame()
    cols = ["Open_time","Open","High","Low","Close","Volume","Close_time","Quote_asset_volume","Number_of_trades","Taker_buy_base","Taker_buy_quote","Ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True)
    df.set_index("Open_time", inplace=True)
    # cast to numeric
    for c in ["Open","High","Low","Close","Volume","Quote_asset_volume","Taker_buy_base","Taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Number_of_trades"] = pd.to_numeric(df["Number_of_trades"], errors="coerce")
    # drop exact duplicate index (minimal hygiene)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _split(df: pd.DataFrame, a: float, b: float, c: float):
    n = len(df)
    i1 = max(int(n * a) - 1, 0)
    i2 = max(int(n * (a + b)) - 1, 0)
    idx = df.index
    train = df.loc[: idx[i1]]
    val = df.loc[idx[i1 + 1] : idx[i2]] if n > 1 else df.iloc[0:0]
    test = df.loc[idx[i2 + 1] :] if n > 2 else df.iloc[0:0]
    return train, val, test


def main():
    symbol = "ETHUSDT"
    start_ms = _to_ms(START_DATE)
    if END_DATE is None:
        end_date = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
    else:
        end_date = END_DATE
    end_ms = _to_ms(end_date)

    print(f"Ingest {symbol} {INTERVALS} | {START_DATE} → {end_date} (UTC)")
    for itv in INTERVALS:
        print(f"  - {itv} fetching...")
        df = _fetch(symbol, itv, start_ms, end_ms)
        if df.empty:
            print(f"    [skip] no data")
            continue
        train, val, test = _split(df, *SPLIT)
        for name, part in ("train", train), ("val", val), ("test", test):
            path = os.path.join(OUT_DIR, f"fut_{name}_data_{itv}.parquet")
            _ensure_dir(os.path.dirname(path))
            part.to_parquet(path)
            print(f"    [ok] {name}: {len(part):,} → {path}")

    print("Done.")


if __name__ == "__main__":
    main()
