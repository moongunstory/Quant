# ingest_futures.py  (REV-3)
from __future__ import annotations
import os, time
from datetime import datetime, timezone
from typing import List, Optional, Dict

import pandas as pd
import requests

# ===== User constants =====
START_DATE = "2019-09-10"   # UM 시작 이후 권장
END_DATE = None             # None → today (UTC)
SYMBOLS = ["ETHUSDT", "BTCUSDT"]
INTERVALS_PER_SYMBOL = {
    "ETHUSDT": ["5m", "15m", "1h", "4h"],
    "BTCUSDT": ["1h"],      # 요청 사항
}
SPLIT = (0.70, 0.15, 0.15)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw"))
SLEEP = 0.25
TIMEOUT = 20

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
        last_close = page[-1][6]  # Close_time(ms)
        next_start = last_close + 1
        if end_ms is not None and next_start > end_ms:
            break
        if len(page) < 1500:
            break
        time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame()
    cols = [
        "Open_time","Open","High","Low","Close","Volume",
        "Close_time","Quote_asset_volume","Number_of_trades",
        "Taker_buy_base","Taker_buy_quote","Ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True)
    df.set_index("Open_time", inplace=True)
    for c in ["Open","High","Low","Close","Volume","Quote_asset_volume","Taker_buy_base","Taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Number_of_trades"] = pd.to_numeric(df["Number_of_trades"], errors="coerce")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def _fetch_funding_rates(symbol: str, start_ms: int, end_ms: Optional[int]) -> pd.DataFrame:
    rows: List[dict] = []
    next_start = start_ms
    limit = 1000
    while True:
        params = {"symbol": symbol, "startTime": next_start, "limit": limit}
        if end_ms:
            params["endTime"] = end_ms
        r = requests.get(BASE + FUNDING_RATE, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        page = r.json()
        if not page:
            break
        rows.extend(page)
        last_ts = page[-1]["fundingTime"]
        next_start = last_ts + 1
        if end_ms and next_start > end_ms:
            break
        if len(page) < limit:
            break
        time.sleep(SLEEP)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df = df.set_index("fundingTime").sort_index()
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    return df

def _warn_if_irregular(df: pd.DataFrame, itv: str, sym: str):
    if df.empty:
        return
    freq = FREQ[itv]
    full = pd.date_range(df.index[0], df.index[-1], freq=freq, tz="UTC")
    missing = full.difference(df.index)
    if len(missing):
        print(f"    [warn] {sym} {itv}: missing bars = {len(missing)} (e.g., {missing[0]})")

def _find_global_cuts_by_eth5m(start_ms: int, end_ms: int):
    print("  - ETHUSDT 5m fetching for global cuts...")
    df5 = _fetch("ETHUSDT", "5m", start_ms, end_ms)
    if df5.empty:
        print("    [warn] no ETHUSDT 5m data; fallback to per-interval split")
        return None
    n = len(df5)
    i1 = max(int(n * SPLIT[0]) - 1, 0)
    i2 = max(int(n * (SPLIT[0] + SPLIT[1])) - 1, 0)
    t1, t2 = df5.index[i1], df5.index[i2]
    print(f"    [ok] global cut times (UTC): t1={t1}, t2={t2}")
    return (t1, t2), df5

def _split_by_cuts(df: pd.DataFrame, t1, t2):
    if df.empty:
        return df.iloc[0:0], df.iloc[0:0], df.iloc[0:0]
    train = df.loc[:t1]
    val = df.loc[t1:].iloc[1:].loc[:t2]
    test = df.loc[t2:].iloc[1:]
    return train, val, test

def _split(df: pd.DataFrame, a: float, b: float, c: float):
    n = len(df)
    if n == 0:
        return df, df, df
    i1 = max(int(n * a) - 1, 0)
    i2 = max(int(n * (a + b)) - 1, 0)
    idx = df.index
    train = df.loc[: idx[i1]]
    val = df.loc[idx[i1] : idx[i2]].iloc[1:] if n > 1 else df.iloc[0:0]
    test = df.loc[idx[i2] :].iloc[1:] if n > 2 else df.iloc[0:0]
    return train, val, test

def _attach_funding(df: pd.DataFrame, fr: Optional[pd.Series]):
    if fr is not None and not df.empty:
        df["FundingRate"] = fr.reindex(df.index, method="ffill").fillna(0.0)
    else:
        df["FundingRate"] = 0.0
    df["Funding8h"] = df["FundingRate"]
    df["FundingSettle"] = (((df.index.hour % 8 == 0) & (df.index.minute == 0))).astype("int8")
    return df

def main():
    start_ms = _to_ms(START_DATE)
    if END_DATE is None:
        end_date = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
    else:
        end_date = END_DATE
    end_ms = _to_ms(end_date)
    print(f"Ingest {SYMBOLS} | {START_DATE} → {end_date} (UTC)")

    # 글로벌 컷(ETH 5m)
    global_cuts, df5_eth = _find_global_cuts_by_eth5m(start_ms, end_ms)

    # 심볼 루프
    for sym in SYMBOLS:
        print(f"\n=== {sym} ===")
        # 심볼별 Funding
        print("  - FundingRate fetching...")
        df_fund = _fetch_funding_rates(sym, start_ms, end_ms)
        fr = df_fund["fundingRate"].sort_index() if not df_fund.empty else None
        if fr is None:
            print("    [warn] no funding rate data")
        else:
            print(f"    [ok] funding rates: {len(df_fund):,} records")

        # 인터벌 루프
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

            _warn_if_irregular(df, itv, sym)
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
