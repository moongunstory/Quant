"""
Mini Module 1 — ETH Futures Ingest (UM) with Auto 3-Way Split (REV-2)

- Market: USDT-M Futures (UM) only
- Symbol: ETHUSDT only
- Intervals: [5m, 15m, 1h, 4h]
- Range: START_DATE → END_DATE (UTC)
- Split: train/val/test = 70/15/15 (by time, **global cut based on 5m**)
- Output: ./ai_binance/data/raw/fut_{train|val|test}_data_{interval}.parquet

변경점(실무 안전화):
1) **글로벌 컷**: 5분봉으로 t1,t2 경계 계산 → 모든 인터벌 동일 시점으로 분할(누수 방지)
2) **갭 경고**: 정규 격자 여부만 경고 출력(자동 보정 없음)
3) **펀딩 이벤트 마커**: Funding8h=ffill된 펀딩율, FundingSettle(8시간 정시=1) 추가
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
FUNDING_RATE = "/fapi/v1/fundingRate"

FREQ = {"5m": "5min", "15m": "15min", "1h": "1H", "4h": "4H"}


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
    # cast to numeric
    for c in ["Open","High","Low","Close","Volume","Quote_asset_volume","Taker_buy_base","Taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Number_of_trades"] = pd.to_numeric(df["Number_of_trades"], errors="coerce")
    # drop exact duplicate index (minimal hygiene)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _fetch_funding_rates(symbol: str, start_ms: int, end_ms: Optional[int]) -> pd.DataFrame:
    """Fetches funding rate history for a symbol."""
    rows: List[dict] = []
    next_start = start_ms
    limit = 1000  # Max limit for fundingRate endpoint
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


def _find_global_cuts(df_5m: pd.DataFrame, split=(0.70, 0.15, 0.15)):
    """5분봉 기준으로 t1, t2 경계시각 산출."""
    n = len(df_5m)
    i1 = max(int(n * split[0]) - 1, 0)
    i2 = max(int(n * (split[0] + split[1])) - 1, 0)
    idx = df_5m.index
    t1 = idx[i1]
    t2 = idx[i2]
    return t1, t2


def _split_by_cuts(df: pd.DataFrame, t1, t2):
    """동일 시점 경계로 분할. 경계 중복 방지 위해 한 칸씩 밀기."""
    if df.empty:
        return df.iloc[0:0], df.iloc[0:0], df.iloc[0:0]
    train = df.loc[:t1]
    val = df.loc[t1:].iloc[1:].loc[:t2] if len(df) else df.iloc[0:0]
    test = df.loc[t2:].iloc[1:] if len(df) else df.iloc[0:0]
    return train, val, test


def _split(df: pd.DataFrame, a: float, b: float, c: float):
    n = len(df)
    i1 = max(int(n * a) - 1, 0)
    i2 = max(int(n * (a + b)) - 1, 0)
    idx = df.index
    train = df.loc[: idx[i1]]
    val = df.loc[idx[i1] + pd.Timedelta(0) : idx[i2]] if n > 1 else df.iloc[0:0]
    test = df.loc[idx[i2] + pd.Timedelta(0) :] if n > 2 else df.iloc[0:0]
    return train, val, test


def _warn_if_irregular(df: pd.DataFrame, itv: str):
    """정규 격자 여부 경고."""
    if df.empty:
        return
    freq = FREQ[itv]
    full = pd.date_range(df.index[0], df.index[-1], freq=freq, tz="UTC")
    missing = full.difference(df.index)
    if len(missing):
        print(f"    [warn] {itv}: missing bars = {len(missing)} (e.g., {missing[0]})")


def main():
    symbol = "ETHUSDT"
    start_ms = _to_ms(START_DATE)
    if END_DATE is None:
        end_date = datetime.now(timezone.utc).date().strftime("%Y-%m-%d")
    else:
        end_date = END_DATE
    end_ms = _to_ms(end_date)

    print(f"Ingest {symbol} {INTERVALS} | {START_DATE} → {end_date} (UTC)")

    # FundingRate: 8h 스냅샷 1회 수집
    print("  - FundingRate fetching...")
    df_fund = _fetch_funding_rates(symbol, start_ms, end_ms)
    if not df_fund.empty:
        fr = df_fund["fundingRate"].sort_index()
        print(f"    [ok] funding rates: {len(df_fund):,} records")
    else:
        fr = None
        print("    [warn] no funding rate data")

    # 5분봉으로 글로벌 컷 계산
    print("  - 5m fetching for global cuts...")
    df5 = _fetch(symbol, "5m", start_ms, end_ms)
    if df5.empty:
        print("    [warn] no 5m data; fallback to per-interval split")
        global_cuts = None
    else:
        _warn_if_irregular(df5, "5m")
        t1, t2 = _find_global_cuts(df5, SPLIT)
        print(f"    [ok] global cut times (UTC): t1={t1}, t2={t2}")
        global_cuts = (t1, t2)

    # 인터벌별 수집/분할/저장
    for itv in INTERVALS:
        print(f"  - {itv} fetching...")
        if itv == "5m" and df5 is not None and not df5.empty:
            df = df5.copy()
        else:
            df = _fetch(symbol, itv, start_ms, end_ms)

        if df.empty:
            print("    [skip] no data")
            continue

        _warn_if_irregular(df, itv)

        # 펀딩율 정렬 + 보조 컬럼
        if fr is not None:
            df["FundingRate"] = fr.reindex(df.index, method="ffill").fillna(0.0)
        else:
            df["FundingRate"] = 0.0
        df["Funding8h"] = df["FundingRate"]
        df["FundingSettle"] = (
            ((df.index.hour % 8 == 0) & (df.index.minute == 0))
        ).astype("int8")

        # 글로벌 컷 우선 적용
        if global_cuts is not None:
            train, val, test = _split_by_cuts(df, *global_cuts)
        else:
            train, val, test = _split(df, *SPLIT)

        for name, part in (("train", train), ("val", val), ("test", test)):
            path = os.path.join(OUT_DIR, f"fut_{name}_data_{itv}.parquet")
            _ensure_dir(os.path.dirname(path))
            part.to_parquet(path)
            print(f"    [ok] {name}: {len(part):,} → {path}")

    print("Done.")


if __name__ == "__main__":
    main()
