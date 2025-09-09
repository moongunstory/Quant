# ingest_futures.py (REV‑4)
from __future__ import annotations
import os, time, math
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
import requests

# ===== User constants =====
START_DATE = "2019-11-27"
END_DATE = None
SYMBOLS = ["ETHUSDT", "BTCUSDT"]
INTERVALS_PER_SYMBOL = {
    "ETHUSDT": ["5m", "15m", "1h", "4h"],
    "BTCUSDT": ["1h"],
}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "raw"))
SLEEP = 0.25
TIMEOUT = 20
MAX_RETRIES = 3

# ===== Binance UM Futures endpoints =====
BASE = "https://fapi.binance.com"
KLINES = "/fapi/v1/klines"
FUNDING_RATE = "/fapi/v1/fundingRate"
FREQ = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}

def _to_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _request_with_retry(url, params):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"    [warn] Request attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
            else:
                raise

def _fetch(symbol: str, interval: str, start_ms: int, end_ms: Optional[int]) -> pd.DataFrame:
    rows: List[List] = []
    next_start = start_ms
    limit = 1500
    interval_ms = int(pd.to_timedelta(FREQ[interval]).total_seconds() * 1000)
    while True:
        params = {"symbol": symbol, "interval": interval, "startTime": next_start, "limit": limit}
        if end_ms is not None:
            params["endTime"] = end_ms
        page = _request_with_retry(BASE + KLINES, params)
        if not page:
            break
        rows.extend(page)
        last_close = page[-1][6]
        next_start = last_close + interval_ms
        if end_ms is not None and next_start > end_ms:
            break
        time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame()
    cols = ["Open_time","Open","High","Low","Close","Volume","Close_time","Quote_asset_volume",
            "Number_of_trades","Taker_buy_base","Taker_buy_quote","Ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True)
    df.set_index("Open_time", inplace=True)
    df = df.astype({
        "Open":"float", "High":"float", "Low":"float", "Close":"float",
        "Volume":"float", "Quote_asset_volume":"float",
        "Number_of_trades":"int", "Taker_buy_base":"float", "Taker_buy_quote":"float"
    }, errors="ignore")
    dup = df.index.duplicated().sum()
    if dup:
        print(f"    [warn] {symbol} {interval}: duplicate index count = {dup}")
        df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    return df

def _fetch_funding_rates(symbol: str, start_ms: int, end_ms: Optional[int]) -> pd.DataFrame:
    rows = []
    next_start = start_ms
    limit = 1000
    while True:
        params = {"symbol": symbol, "startTime": next_start, "limit": limit}
        if end_ms:
            params["endTime"] = end_ms
        page = _request_with_retry(BASE + FUNDING_RATE, params)
        if not page:
            break
        rows.extend(page)
        last_ts = page[-1]["fundingTime"]
        next_start = last_ts + 1
        if end_ms and next_start > end_ms:
            break
        time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df.set_index("fundingTime").sort_index()
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df

def _warn_if_irregular(df: pd.DataFrame, itv: str, sym: str) -> pd.DatetimeIndex:
    if df.empty:
        return pd.DatetimeIndex([])
    full = pd.date_range(df.index[0], df.index[-1], freq=FREQ[itv], tz="UTC")
    missing = full.difference(df.index)
    if len(missing):
        print(f"    [warn] {sym} {itv}: missing bars = {len(missing)} (e.g., {missing[0]})")
    return full

def _attach_funding(df: pd.DataFrame, fr: Optional[pd.Series]) -> pd.DataFrame:
    if fr is not None and not df.empty:
        fr.name = "FundingRate"
        merged = pd.merge_asof(
            df.sort_index(),
            fr.sort_index().to_frame(),
            left_index=True,
            right_index=True,
            direction="backward",
            tolerance=pd.Timedelta(hours=12),
        )
        missing_ratio = merged["FundingRate"].isna().mean()
        if missing_ratio > 0.05:
            print(f"[WARNING] {missing_ratio:.1%} rows missing funding rate — check timestamps!")

        merged["FundingRate"] = merged["FundingRate"].fillna(0.0)
    else:
        merged = df.copy()
        merged["FundingRate"] = 0.0

    merged["Funding8h"] = merged["FundingRate"]
    merged["FundingSettle"] = (((merged.index.hour % 8 == 0) & (merged.index.minute == 0))).astype("int8")
    return merged

def main():
    start_ms = _to_ms(START_DATE)
    end_date = END_DATE or datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
    end_ms = _to_ms(end_date)
    print(f"Ingest {SYMBOLS} | {START_DATE} → {end_date} (UTC)")

    # 1. Fetch all data first
    print("=== Step 1: Fetching all data... ===")
    all_dfs = {}
    for sym in SYMBOLS:
        all_dfs[sym] = {}
        intervals = INTERVALS_PER_SYMBOL.get(sym, [])
        for itv in intervals:
            print(f"  - Fetching {sym} {itv}...")
            df = _fetch(sym, itv, start_ms, end_ms)
            if df.empty:
                print(f"    [warn] No data for {sym} {itv}, it will be skipped.")
            all_dfs[sym][itv] = df

    # 2. Find common date range
    print("=== Step 2: Finding common date range... ===")
    max_start = None
    min_end = None
    for sym in all_dfs:
        for itv in all_dfs[sym]:
            df = all_dfs[sym][itv]
            if not df.empty:
                if max_start is None or df.index.min() > max_start:
                    max_start = df.index.min()
                if min_end is None or df.index.max() < min_end:
                    min_end = df.index.max()
    
    if max_start is None or min_end is None or max_start >= min_end:
        raise ValueError("No overlapping data found across all symbols and intervals.")
    
    print(f"  - Common range found: {max_start} to {min_end}")

    # 3. Trim all dataframes to common range
    print("=== Step 3: Trimming data to common range... ===")
    trimmed_dfs = {}
    for sym in all_dfs:
        trimmed_dfs[sym] = {}
        for itv in all_dfs[sym]:
            if not all_dfs[sym][itv].empty:
                trimmed_df = all_dfs[sym][itv].loc[max_start:min_end]
                trimmed_dfs[sym][itv] = trimmed_df

    # 4. Attach funding and save
    print("=== Step 4: Processing and saving full dataframes... ===")
    for sym in trimmed_dfs:
        print(f"--- Processing {sym} ---")
        print("  - FundingRate fetching...")
        df_fund = _fetch_funding_rates(sym, int(max_start.timestamp()*1000), int(min_end.timestamp()*1000))
        fr = df_fund["fundingRate"] if not df_fund.empty else None

        for itv in trimmed_dfs[sym]:
            print(f"  - Processing {sym} {itv}...")
            df = trimmed_dfs[sym][itv]
            if df.empty:
                print("    [skip] no data in common range")
                continue

            full_index = _warn_if_irregular(df, itv, sym)
            if not full_index.empty:
                df = df.reindex(full_index).sort_index()

            df = _attach_funding(df, fr)
            
            out_dir = os.path.join(OUT_DIR, sym.lower())
            _ensure_dir(out_dir)
            
            path = os.path.join(out_dir, f"fut_data_{itv}.parquet")
            df.to_parquet(path)
            print(f"    [ok] Full data: {len(df):,} -> {path}")

    print("Done.")

if __name__ == "__main__":
    main()
