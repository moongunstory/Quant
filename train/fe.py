# fe.py — Feature Engineering for ETHUSDT (MTF) + BTC1h (REV-6.4 / Final Fix)
"""
- NEW (REV-6.4):
  * CRITICAL FIX: f-string 내에 이중 중괄호({{}})가 사용된 치명적 버그 수정.

- NEW (REV-6.3):
  * WORKAROUND: f-string 포매팅을 .format() 방식으로 변경하여 이례적인 환경 문제에 대응.

- NEW (REV-6.2):
  * DEBUG: _load_raw 함수에 print 구문을 추가하여 파일 경로 생성 문제를 디버깅.

- NEW (REV-6.1):
  * BUGFIX: _load_raw 함수에서 val/test 데이터셋의 파일 경로를 잘못 생성하던 오류 수정.

- NEW (REV-6):
  * BTC 1h 리드-래그(Lead-Lag) 피처 추가.
"""

from __future__ import annotations

import os, json
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import joblib

# ===== Paths / Constants =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw"))
OUT_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
os.makedirs(OUT_DIR, exist_ok=True)

# MTF Setup
TIMEFRAMES     = ["5m", "15m", "1h", "4h", "btc1h"]
ETH_TIMEFRAMES = ["5m", "15m", "1h", "4h"]
BASE_INTERVAL  = "5m"

# === TF별 피처검색/스케일 설정 ===
FEATURE_SEARCH = True
RANDOM_STATE = 72
TOP_K_PER_TF = {"5m": 128, "15m": 128, "1h": 96, "4h": 64}
TF_FOR_SEARCH = ["5m", "15m", "1h", "4h"]

# Output path formats
FEATURE_LIST_PATH_FMT = os.path.join(OUT_DIR, "fe_feature_list_{tf}.json")
SCALER_PATH_FMT       = os.path.join(OUT_DIR, "scaler_{tf}.joblib")

# ===== Reference (unscaled) columns to keep =====
REF_COLS_CANON = ["Open", "High", "Low", "Close", "Volume", "FundingRate"]

# ===== Utilities =====

def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    std = df.std(numeric_only=True)
    zero_std_cols = std[std == 0].index.tolist()
    if len(zero_std_cols) > 0:
        df[zero_std_cols] = 0.0
    return df.fillna(0.0)

def zscore(s: pd.Series, win: int | None = None) -> pd.Series:
    if win is None:
        mu, sd = s.mean(), s.std()
        sd = sd if (sd and sd > 0) else np.nan
        out = (s - mu) / sd
    else:
        mu = s.rolling(win, min_periods=win).mean()
        sd = s.rolling(win, min_periods=win).std().replace(0, np.nan)
        out = (s - mu) / sd
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def _enforce_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        for tcol in ["Open_time", "open_time", "time"]:
            if tcol in df.columns:
                idx = pd.to_datetime(df[tcol], errors="coerce", utc=True)
                if idx.notna().any():
                    df = df.set_index(idx).drop(columns=[tcol])
                    break
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df.index.name = "time"
    return df

def _load_raw(split: str, interval: str) -> pd.DataFrame:
    if interval == "btc1h":
        p = os.path.join(RAW_DIR, "btcusdt", f"fut_{split}_data_1h.parquet")
    else:
        p = os.path.join(RAW_DIR, "ethusdt", f"fut_{split}_data_{interval}.parquet")

    if not os.path.exists(p):
        raise FileNotFoundError(f"Raw data file not found: {p}")

    df = pd.read_parquet(p)

    cols = {c.lower(): c for c in df.columns}
    for k in ["open","high","low","close","volume"]:
        if k in cols:
            real = cols[k]
            df[real] = pd.to_numeric(df[real], errors="coerce")

    if "FundingRate" not in df.columns and "funding_rate" in df.columns:
        df.rename(columns={"funding_rate": "FundingRate"}, inplace=True)
    if "FundingRate" not in df.columns:
        df["FundingRate"] = 0.0
    df["FundingRate"] = pd.to_numeric(df["FundingRate"], errors="coerce").fillna(0.0)

    for k in ["Volume", "Quote_asset_volume", "Taker_buy_base", "Taker_buy_quote"]:
        if k in df.columns:
            df[k] = np.log1p(np.clip(pd.to_numeric(df[k], errors="coerce"), 0, None))

    df = _enforce_dt_index(df)

    for c in REF_COLS_CANON:
        if c not in df.columns:
            df[c] = 0.0

    if interval == BASE_INTERVAL:
        if "FundingSettle" not in df.columns:
            df["FundingSettle"] = (((df.index.hour % 8 == 0) & (df.index.minute == 0))).astype("int8")
        else:
            df["FundingSettle"] = df["FundingSettle"].astype("int8")

    return df

# ===== Indicators =====

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _funding_phase_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    steps_per_8h = 96
    steps_since = (idx.hour % 8) * 12 + (idx.minute // 5)
    steps_to_next = (steps_per_8h - steps_since) % steps_per_8h
    phase = 2 * np.pi * (steps_since / steps_per_8h)
    out = pd.DataFrame(index=idx)
    out["time_to_funding_5m"] = steps_to_next.astype("int16")
    out["funding_phase_sin"] = np.sin(phase)
    out["funding_phase_cos"] = np.cos(phase)
    return out

def compute_heikin_ashi(df_ohlc: pd.DataFrame, o="Open", h="High", l="Low", c="Close") -> pd.DataFrame:
    if df_ohlc.empty: return pd.DataFrame(index=df_ohlc.index)
    O, H, L, C = df_ohlc[o].values, df_ohlc[h].values, df_ohlc[l].values, df_ohlc[c].values
    n = len(df_ohlc)
    HA_C = (O + H + L + C) / 4.0
    HA_O = np.empty(n); HA_O[0] = (O[0] + C[0]) / 2.0
    for i in range(1, n): HA_O[i] = (HA_O[i-1] + HA_C[i-1]) / 2.0
    HA_H, HA_L = np.maximum.reduce([H, HA_O, HA_C]), np.minimum.reduce([L, HA_O, HA_C])
    out = pd.DataFrame({"HA_O": HA_O, "HA_H": HA_H, "HA_L": HA_L, "HA_C": HA_C}, index=df_ohlc.index)
    out["HA_TR"], out["HA_BC"] = out["HA_H"] - out["HA_L"], out["HA_C"] - out["HA_O"]
    out["HA_R"]  = out["HA_C"].pct_change().fillna(0.0)
    return out

# ===== Feature Engines =====

def compute_features_for_tf(df: pd.DataFrame, interval: str, btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df.sort_index()
    out = pd.DataFrame(index=df.index)

    if interval == "btc1h":
        close = df["Close"].astype(float)
        out["ret_1h"] = close.pct_change()
        out["ret_4h"] = close.pct_change(4)
        out["atr14"]  = _atr(df, period=14)
        ha = compute_heikin_ashi(df[["Open","High","Low","Close"]])
        out = pd.concat([out, ha], axis=1)
        out.columns = [f"{c}_btc1h" for c in out.columns]
        ref = df[REF_COLS_CANON].copy()
        return _sanitize(pd.concat([out, ref], axis=1))

    close, high, low, volume = df["Close"].astype(float), df["High"].astype(float), df["Low"].astype(float), df["Volume"].astype(float)

    out["ret_1"] = close.pct_change()
    out["ret_3"] = close.pct_change(3)
    out["z_close_48"] = zscore(close, win=48)
    out["hl_spread"] = (high - low) / close
    out["vol_z_48"] = zscore(volume, win=48)

    ema_12, ema_26 = close.ewm(span=12, adjust=False).mean(), close.ewm(span=26, adjust=False).mean()
    out["macd"], out["macd_sig"] = ema_12 - ema_26, (ema_12 - ema_26).ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_sig"]

    delta = close.diff()
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = up.ewm(alpha=1/14, adjust=False).mean() / down.ewm(alpha=1/14, adjust=False).mean()
    out["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    out["atr14"] = _atr(df, period=14)

    ha = compute_heikin_ashi(df[["Open","High","Low","Close"]])
    out = pd.concat([out, ha], axis=1)

    if interval == BASE_INTERVAL:
        out['hour_sin'], out['hour_cos'] = np.sin(2*np.pi*df.index.hour/24), np.cos(2*np.pi*df.index.hour/24)
        out['day_sin'], out['day_cos'] = np.sin(2*np.pi*df.index.dayofweek/7), np.cos(2*np.pi*df.index.dayofweek/7)
        out = pd.concat([out, _funding_phase_features(out.index)], axis=1)
        out["is_funding_settle"] = df.get("FundingSettle", 0).astype("int8")
        out["funding_z_48"] = zscore(df["FundingRate"].astype(float), win=48)

    if btc_df is not None:
        btc_renamed = btc_df.rename(columns={c: f"{c}_btc1h" for c in btc_df.columns})
        merged_df = pd.merge_asof(df, btc_renamed, on="time", direction="backward")
        
        btc_close = merged_df["Close_btc1h"].astype(float)
        btc_vol = merged_df["Volume_btc1h"].astype(float)
        
        btc_lead_features = pd.DataFrame(index=df.index)
        btc_lead_features["btc_ret_1h"] = btc_close.pct_change()
        btc_lead_features["btc_vol_z_24"] = zscore(btc_vol, win=24)
        
        btc_ema_12 = btc_close.ewm(span=12, adjust=False).mean()
        btc_ema_26 = btc_close.ewm(span=26, adjust=False).mean()
        btc_lead_features["btc_macd"] = btc_ema_12 - btc_ema_26

        for lag in range(1, 7):
            lagged = btc_lead_features.shift(lag)
            lagged.columns = [f"{c}_lag{lag}" for c in lagged.columns]
            out = pd.concat([out, lagged], axis=1)

    out.columns = [f"{c}_{interval}" for c in out.columns]
    final_out = pd.concat([out, df[REF_COLS_CANON]], axis=1)
    return _sanitize(final_out)

# ===== Feature Search & Scaler =====

def _make_proxy_y(df: pd.DataFrame) -> pd.Series:
    return (df["Close"].astype(float).pct_change().shift(-1) > 0).astype(int).fillna(0)

def _feature_search_mi(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    X_ = _sanitize(X).astype(float)
    mi = mutual_info_classif(X_, y.values, random_state=RANDOM_STATE)
    scores = pd.Series(mi, index=X_.columns).sort_values(ascending=False)
    return scores.head(top_k).index.tolist()

def _feature_search_for_tf(train_df: pd.DataFrame, tf: str) -> List[str]:
    exclude = set(REF_COLS_CANON)
    feat_cols = [c for c in train_df.columns if c not in exclude]
    if not FEATURE_SEARCH or not feat_cols:
        keep = feat_cols
    else:
        y_tr = _make_proxy_y(train_df)
        k = min(TOP_K_PER_TF.get(tf, len(feat_cols)), len(feat_cols))
        keep = _feature_search_mi(train_df[feat_cols], y_tr, k)
    
    path = FEATURE_LIST_PATH_FMT.format(tf=tf)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)
    print(f"[ok] feature_list[{tf}] = {len(keep)} -> {path}")
    return keep

def _fit_scaler_for_tf(train_df: pd.DataFrame, feat_list: List[str], tf: str) -> StandardScaler:
    X = _sanitize(train_df[feat_list]).to_numpy(dtype=float)
    sc = StandardScaler().fit(X)
    path = SCALER_PATH_FMT.format(tf=tf)
    joblib.dump(sc, path)
    print(f"[ok] scaler[{tf}] saved -> {path}")
    return sc

# ===== Main =====

def main():
    print("[1/4] Loading all raw data...")
    raw_data = {s: {tf: _load_raw(s, tf) for tf in TIMEFRAMES} for s in ["train", "val", "test"]}

    print("\n[2/4] Pre-computing BTC features...")
    btc_features = {}
    for split in ["train", "val", "test"]:
        btc_df_raw = raw_data[split]["btc1h"]
        btc_features[split] = compute_features_for_tf(btc_df_raw, "btc1h")
        out_p = os.path.join(OUT_DIR, f"fe_{split}_btc1h.parquet")
        btc_features[split].to_parquet(out_p)
        print(f"  [ok] Saved standalone {split}/btc1h -> {out_p}")

    print("\n[3/4] Computing ETH features with BTC lead-lag...")
    feature_data = {s: {} for s in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for tf in ETH_TIMEFRAMES:
            print(f"  - Computing features for {split} / {tf}...")
            eth_df_raw = raw_data[split][tf]
            feature_data[split][tf] = compute_features_for_tf(eth_df_raw, tf, btc_df=btc_features[split])

    print(f"\n[4/4] Building TF-specific feature lists & scalers...")
    feature_list_per_tf = {}
    scalers = {}
    for tf in TF_FOR_SEARCH:
        tr_df = feature_data["train"][tf]
        feat_list = _feature_search_for_tf(tr_df, tf)
        feature_list_per_tf[tf] = feat_list
        scalers[tf] = _fit_scaler_for_tf(tr_df, feat_list, tf)

    print("\n[5/5] Processing and saving all ETH timeframe data...")
    for split in ["train", "val", "test"]:
        for tf in ETH_TIMEFRAMES:
            print(f"  - Processing {split} / {tf}...")
            df = feature_data[split][tf].copy()
            feat_list = feature_list_per_tf[tf]
            
            df_sel = df.reindex(columns=feat_list, fill_value=0.0)
            X = _sanitize(df_sel).to_numpy(dtype=float)
            Xs = scalers[tf].transform(X)
            df_scaled = pd.DataFrame(Xs, index=df.index, columns=feat_list)

            ref_cols = [c for c in REF_COLS_CANON if c in df.columns]
            final_df = pd.concat([df_scaled, df[ref_cols]], axis=1)

            out_p = os.path.join(OUT_DIR, f"fe_{split}_{tf}.parquet")
            final_df.to_parquet(out_p)
            print(f"    [ok] Saved {split}/{tf}: {len(final_df):,} x {final_df.shape[1]} -> {out_p}")

    print("\n[+] MTF Multi-Input Feature Engineering complete (with BTC Lead-Lag).")

if __name__ == "__main__":
    main()
