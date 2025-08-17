"""
Module 2 — Feature Engineering & Normalization for ETH Futures (No Leakage, No Warnings)

Goal: build RL-ready normalized features with zero data leakage and avoid noisy deps/warnings.
- No pandas_ta (pure pandas/numpy indicators)
- Fit StandardScaler **on train DataFrame** (keeps feature names)
- Persist scaler via joblib → reuse for val/test (no "fitted without feature names" warning)
- Auto-run (no CLI)

Added (per spec):
- Long returns: ret48, ret96
- Trend slope: ma20_slope
- Band position: bb_pos = (Close - bb_mid) / (bb_up - bb_dn)
- Volume normalization: vol_norm = Volume / atr14
- Multi RSI: rsi7, rsi14, rsi28
- log1p only for Volume/ATR family before StandardScaler
"""
from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

RAW_DIR = "./ai_binance/data/raw"
PROC_DIR = "./ai_binance/data/processed"
INTERVALS = ["5m", "15m", "1h", "4h"]
BASE_INTERVAL = "5m"
FEATURE_LIST_PATH = os.path.join(PROC_DIR, "feature_list.json")
SCALER_JOBLIB = os.path.join(PROC_DIR, "scaler.joblib")
SCALER_INFO_JSON = os.path.join(PROC_DIR, "normalize_stats.json")  # kept for compatibility note

# =========================
# IO helpers
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _load(split: str, interval: str) -> pd.DataFrame | None:
    path = os.path.join(RAW_DIR, f"fut_{split}_data_{interval}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

# =========================
# Indicators (minimal, fast, dependency-free)
# =========================
def _rsi(series: pd.Series, period: int) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(period, min_periods=period).mean()
    dn = (-d.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _indicators(df: pd.DataFrame, itv: str) -> pd.DataFrame:
    x = df.copy()

    # Returns (short)
    x[f"ret1_{itv}"] = x["Close"].pct_change()
    x[f"ret3_{itv}"] = x["Close"].pct_change(3)
    x[f"ret12_{itv}"] = x["Close"].pct_change(12)

    # Returns (long)
    x[f"ret48_{itv}"] = x["Close"].pct_change(48)
    x[f"ret96_{itv}"] = x["Close"].pct_change(96)

    # Volatility proxy
    x[f"hlv_{itv}"] = (x["High"] - x["Low"]) / x["Close"].replace(0, np.nan)

    # RSI: 7 / 14 / 28
    x[f"rsi7_{itv}"] = _rsi(x["Close"], 7)
    x[f"rsi14_{itv}"] = _rsi(x["Close"], 14)
    x[f"rsi28_{itv}"] = _rsi(x["Close"], 28)

    # MACD(12,26,9)
    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    x[f"macd_{itv}"] = macd
    x[f"macd_sig_{itv}"] = macd_sig
    x[f"macd_hist_{itv}"] = macd - macd_sig

    # ATR(14)
    prev_close = x["Close"].shift()
    tr = np.maximum(
        x["High"] - x["Low"],
        np.maximum((x["High"] - prev_close).abs(), (x["Low"] - prev_close).abs()),
    )
    x[f"atr14_{itv}"] = tr.rolling(14, min_periods=14).mean()

    # Bollinger Bands(20)
    ma20 = x["Close"].rolling(20, min_periods=20).mean()
    sd20 = x["Close"].rolling(20, min_periods=20).std()
    x[f"bb_mid_{itv}"] = ma20
    x[f"bb_up_{itv}"] = ma20 + 2 * sd20
    x[f"bb_dn_{itv}"] = ma20 - 2 * sd20

    # Trend slope (MA20 slope)
    x[f"ma20_slope_{itv}"] = ma20.diff()

    # Band position (≈[-1, 1] in-band)
    bb_span = (x[f"bb_up_{itv}"] - x[f"bb_dn_{itv}"]).replace(0, np.nan)
    x[f"bb_pos_{itv}"] = (x["Close"] - x[f"bb_mid_{itv}"]) / bb_span

    # Volume normalization (heavy-tail mitigated later via log1p)
    # Guard zero ATR to avoid inf
    x[f"vol_norm_{itv}"] = x["Volume"] / x[f"atr14_{itv}"].replace(0, np.nan)

    return x

# =========================
# Merge MTF → base index
# =========================
def _merge(split: str) -> pd.DataFrame:
    dfs: Dict[str, pd.DataFrame] = {}
    for itv in INTERVALS:
        df = _load(split, itv)
        if df is not None and not df.empty:
            dfs[itv] = _indicators(df, itv)
    if BASE_INTERVAL not in dfs:
        raise ValueError(f"[{split}] missing base interval {BASE_INTERVAL}")

    out = dfs[BASE_INTERVAL].copy()
    for itv, df in dfs.items():
        if itv == BASE_INTERVAL:
            continue
        # keep existing convention: add a second suffix to avoid collisions
        out = pd.merge_asof(
            out.sort_index(),
            df.sort_index().add_suffix(f"_{itv}"),
            left_index=True,
            right_index=True,
            direction="backward",
        )
    return out

# =========================
# Normalization (train fit → apply to val/test)
# =========================
def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    x = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).copy()

    # log1p for heavy-tail families: volume/atr (case-insensitive)
    cols = list(x.columns)
    for c in cols:
        lc = c.lower()
        if ("volume" in lc) or ("vol_" in lc) or (lc.startswith("vol")) or ("atr" in lc):
            # Ensure non-negative before log1p
            x[c] = np.log1p(x[c].clip(lower=0))
    return x

def _fit_train_scaler(train_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    numeric = _numeric(train_df).dropna()
    feature_cols = list(numeric.columns)
    scaler = StandardScaler()
    # fit on DataFrame to retain feature_names_in_
    norm_arr = scaler.fit_transform(numeric)
    norm = pd.DataFrame(norm_arr, index=numeric.index, columns=feature_cols)
    _ensure_dir(PROC_DIR)
    joblib.dump(scaler, SCALER_JOBLIB)
    return norm, feature_cols

def _apply_scaler(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    scaler: StandardScaler = joblib.load(SCALER_JOBLIB)
    x = _numeric(df)
    # align columns
    for c in feature_cols:
        if c not in x.columns:
            x[c] = np.nan
    x = x[feature_cols].dropna()
    arr = scaler.transform(x)
    return pd.DataFrame(arr, index=x.index, columns=feature_cols)

def _save_norm(df: pd.DataFrame, split: str) -> None:
    _ensure_dir(PROC_DIR)
    path = os.path.join(PROC_DIR, f"{split}_normalized.parquet")
    df.to_parquet(path)
    print(f"[ok] {split} normalized → {path}")

# =========================
# Pipeline
# =========================
def main() -> None:
    # Train
    train_merged = _merge("train")
    train_norm, feature_cols = _fit_train_scaler(train_merged)
    _save_norm(train_norm, "train")

    # Persist feature list and a tiny info file (backward compatibility)
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(SCALER_INFO_JSON, "w") as f:
        json.dump({"info": "Fitted StandardScaler persisted at scaler.joblib"}, f, indent=2)

    # Val/Test reuse
    for split in ["val", "test"]:
        try:
            merged = _merge(split)
        except ValueError:
            print(f"[skip] {split} missing base interval {BASE_INTERVAL}")
            continue
        norm = _apply_scaler(merged, feature_cols)
        _save_norm(norm, split)

    print("[OK] Processing completed without leakage and without pandas_ta warnings.]")

if __name__ == "__main__":
    main()
