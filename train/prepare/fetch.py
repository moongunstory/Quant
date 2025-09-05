# ingest_futures.py (REV‑4)
from __future__ import annotations
import os, time, math
from datetime import datetime, timezone
from typing import List, Optional
import pandas as pd
import requests

# ===== User constants =====
START_DATE = "2019-09-10"
END_DATE = None
SYMBOLS = ["ETHUSDT", "BTCUSDT"]
INTERVALS_PER_SYMBOL = {
    "ETHUSDT": ["5m", "15m", "1h", "4h"],
    "BTCUSDT": ["1h"],
}
SPLIT = (0.70, 0.15, 0.15)
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

def _find_global_cuts_by_eth5m(start_ms: int, end_ms: int):
    print("  - ETHUSDT 5m fetching for global cuts...")
    df5 = _fetch("ETHUSDT", "5m", start_ms, end_ms)
    if df5.empty:
        print("    [warn] no ETHUSDT 5m data; fallback to per-interval split")
        return None, None
    n = len(df5)
    i1 = max(int(n * SPLIT[0]) - 1, 0)
    i2 = max(int(n * (SPLIT[0] + SPLIT[1])) - 1, 0)
    t1, t2 = df5.index[i1], df5.index[i2]
    print(f"    [ok] global cut times (UTC): t1={t1}, t2={t2}")
    return (t1, t2), df5

def _split_by_cuts(df, t1, t2):
    if df.empty:
        return df, df, df
    train = df.loc[:t1]
    val = df.loc[t1:].iloc[1:].loc[:t2]
    test = df.loc[t2:].iloc[1:]
    return train, val, test

def _split(df: pd.DataFrame, a: float, b: float, c: float):
    n = len(df)
    if n == 0:
        return df, df, df
    idx = df.index
    i1 = max(int(n * a) - 1, 0)
    i2 = max(int(n * (a + b)) - 1, 0)
    train = df.loc[:idx[i1]]
    val = df.loc[idx[i1]:idx[i2]].iloc[1:] if n > 1 else df.iloc[0:0]
    test = df.loc[idx[i2]:].iloc[1:] if n > 2 else df.iloc[0:0]
    return train, val, test

def _attach_funding(df: pd.DataFrame, fr: Optional[pd.Series]) -> pd.DataFrame:
    if fr is not None and not df.empty:
        fr.name = "FundingRate"  # <- 🔧 이름 직접 지정해서 rename() 피함
        merged = pd.merge_asof(
            df.sort_index(),
            fr.sort_index().to_frame(),  # <- 🔧 Series를 DataFrame으로 변환
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

    global_cuts, df5_eth = _find_global_cuts_by_eth5m(start_ms, end_ms)

    for sym in SYMBOLS:
        print(f"\n=== {sym} ===")
        print("  - FundingRate fetching...")
        df_fund = _fetch_funding_rates(sym, start_ms, end_ms)
        fr = df_fund["fundingRate"] if not df_fund.empty and "fundingRate" in df_fund.columns else None
        intervals = INTERVALS_PER_SYMBOL.get(sym, [])
        for itv in intervals:
            print(f"  - {sym} {itv} fetching...")
            if sym == "ETHUSDT" and itv == "5m" and df5_eth is not None and not df5_eth.empty:
                df = df5_eth.copy()
            else:
                df = _fetch(sym, itv, start_ms, end_ms)
            if df.empty:
                print("    [skip] no data")
                continue

            full_index = _warn_if_irregular(df, itv, sym)
            if not full_index.empty:
                df = df.reindex(full_index).sort_index()

            df = _attach_funding(df, fr)

            if global_cuts is not None:
                train, val, test = _split_by_cuts(df, *global_cuts)
            else:
                train, val, test = _split(df, *SPLIT)

            out_dir = os.path.join(OUT_DIR, sym.lower())
            _ensure_dir(out_dir)
            for name, part in (("train", train), ("val", val), ("test", test)):
                path = os.path.join(out_dir, f"fut_{name}_data_{itv}.parquet")
                part.to_parquet(path)
                print(f"    [ok] {name}: {len(part):,} → {path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
