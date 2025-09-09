# ingest_futures.py (REV‑5) - with missing data retry
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

def _fetch_missing_data(symbol: str, interval: str, missing_times: pd.DatetimeIndex) -> pd.DataFrame:
    """누락된 특정 시점들의 데이터를 재요청"""
    if len(missing_times) == 0:
        return pd.DataFrame()
        
    print(f"    [retry] Attempting to fetch {len(missing_times)} missing bars")
    
    # 연속된 구간들로 그룹화하여 효율적으로 요청
    missing_groups = []
    current_group = [missing_times[0]]
    
    for i in range(1, len(missing_times)):
        time_diff = missing_times[i] - missing_times[i-1]
        expected_diff = pd.Timedelta(FREQ[interval])
        
        # 연속된 시점이면 같은 그룹에 추가
        if time_diff <= expected_diff * 1.5:  # 약간의 여유 허용
            current_group.append(missing_times[i])
        else:
            # 새로운 그룹 시작
            missing_groups.append(current_group)
            current_group = [missing_times[i]]
    
    missing_groups.append(current_group)
    
    all_recovered_data = []
    
    for group in missing_groups:
        if len(group) > 100:  # 너무 많은 누락은 건너뛰기
            print(f"    [skip] Skipping large gap of {len(group)} bars")
            continue
            
        try:
            # 그룹의 시작과 끝 시점 + 여유분으로 요청
            start_time = group[0] - pd.Timedelta(hours=2)
            end_time = group[-1] + pd.Timedelta(hours=2)
            
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            # 해당 구간 데이터 요청
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": min(500, len(group) * 3)  # 적절한 limit 설정
            }
            
            data = _request_with_retry(BASE + KLINES, params)
            if data:
                all_recovered_data.extend(data)
            
            time.sleep(SLEEP)
            
        except Exception as e:
            print(f"    [warn] Failed to fetch missing group: {e}")
            continue
    
    if all_recovered_data:
        # DataFrame으로 변환
        cols = ["Open_time","Open","High","Low","Close","Volume","Close_time",
                "Quote_asset_volume","Number_of_trades","Taker_buy_base","Taker_buy_quote","Ignore"]
        df_recovered = pd.DataFrame(all_recovered_data, columns=cols)
        df_recovered["Open_time"] = pd.to_datetime(df_recovered["Open_time"], unit="ms", utc=True)
        df_recovered.set_index("Open_time", inplace=True)
        
        # 타입 변환
        df_recovered = df_recovered.astype({
            "Open":"float", "High":"float", "Low":"float", "Close":"float",
            "Volume":"float", "Quote_asset_volume":"float",
            "Number_of_trades":"int", "Taker_buy_base":"float", "Taker_buy_quote":"float"
        }, errors="ignore")
        
        # 중복 제거
        df_recovered = df_recovered[~df_recovered.index.duplicated(keep="last")]
        df_recovered = df_recovered.sort_index()
        
        # 실제로 누락되었던 시점들만 필터링
        recovered_missing = df_recovered[df_recovered.index.isin(missing_times)]
        
        return recovered_missing
    
    return pd.DataFrame()

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

def _process_missing_data(df: pd.DataFrame, itv: str, sym: str) -> pd.DataFrame:
    """누락된 데이터 처리 - 재요청 → 보간 → 제거 순서"""
    if df.empty:
        return df
        
    full_index = pd.date_range(df.index[0], df.index[-1], freq=FREQ[itv], tz="UTC")
    missing = full_index.difference(df.index)
    
    if len(missing) == 0:
        return df
        
    print(f"    [warn] {sym} {itv}: missing bars = {len(missing)}")
    
    # 1단계: 재요청 시도
    recovered_data = _fetch_missing_data(sym, itv, missing)
    
    if not recovered_data.empty:
        # 기존 데이터와 병합
        df_combined = pd.concat([df, recovered_data]).sort_index()
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        
        print(f"    [ok] Recovered {len(recovered_data)} missing bars")
        
        # 다시 missing 체크
        remaining_missing = full_index.difference(df_combined.index)
        if len(remaining_missing) == 0:
            return df_combined
        else:
            df = df_combined
            missing = remaining_missing
            print(f"    [info] Still missing {len(missing)} bars after recovery")
    
    # 2단계: 남은 누락에 대해 보간 적용 (조건부)
    missing_ratio = len(missing) / len(full_index)
    
    if missing_ratio > 0.05:  # 5% 이상 누락이면 해당 구간 제외
        print(f"    [error] Too many missing bars ({missing_ratio:.1%}), cannot interpolate")
        # 연속된 데이터가 있는 구간만 사용
        return df
    else:
        print(f"    [fix] Using interpolation for remaining {len(missing)} bars ({missing_ratio:.1%})")
        df_full = df.reindex(full_index).sort_index()
        
        df_full = df_full.infer_objects(copy=False)
        df_full = df_full.interpolate(method='linear')
        df_full = df_full.ffill().bfill()

        # 최후 수단: 0으로 채우기
        if df_full.isnull().any().any():
            print(f"    [warn] Using zero-fill for remaining NaN in {sym} {itv}")
            df_full = df_full.fillna(0)
        
        return df_full

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

    # 4. Process missing data, attach funding and save
    print("=== Step 4: Processing missing data and saving... ===")
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

            # 누락 데이터 처리 (재요청 + 보간)
            df = _process_missing_data(df, itv, sym)
            
            # 펀딩 레이트 붙이기
            df = _attach_funding(df, fr)
            
            # 최종 품질 체크
            nan_count = df.isnull().sum().sum()
            if nan_count > 0:
                print(f"    [warn] Final data contains {nan_count} NaN values")
                df = df.fillna(0.0)  # 최후 안전장치
            
            out_dir = os.path.join(OUT_DIR, sym.lower())
            _ensure_dir(out_dir)
            
            path = os.path.join(out_dir, f"fut_data_{itv}.parquet")
            df.to_parquet(path)
            print(f"    [ok] Final data: {len(df):,} rows -> {path}")

    print("Done.")

if __name__ == "__main__":
    main()