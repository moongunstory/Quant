"""
fe.py — Feature Engineering for ETHUSDT (MTF) [REV-3 / Multi-Input]
- NEW: Saves each timeframe as a separate file for multi-input models.
- Raw: Loads 5m, 15m, 1h, 4h data from ./ai_binance/data/raw/
- Output: fe_{train|val|test}_{5m|15m|1h|4h}.parquet, fe_feature_list_5m.json, scaler_5m.joblib
- Logic:
  1. Load raw data for all specified timeframes (5m, 15m, 1h, 4h).
  2. Generate features for each timeframe independently.
  3. Perform feature selection based on the base interval (5m) training data.
  4. Fit a scaler on the base interval (5m) training data.
  5. Apply the selected features and scaler to all timeframes.
  6. Save each processed timeframe and split into its own file.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import joblib

# ===== Optional: LightGBM for model-based feature importance =====
try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

# ===== Paths / Constants =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw"))
OUT_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))

# MTF Setup
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
BASE_INTERVAL = "5m" # Feature selection and scaling will be based on this interval

# Output paths
FEATURE_LIST_PATH = os.path.join(OUT_DIR, f"fe_feature_list_{BASE_INTERVAL}.json")
SCALER_PATH       = os.path.join(OUT_DIR, f"scaler_{BASE_INTERVAL}.joblib")

os.makedirs(OUT_DIR, exist_ok=True)

# ===== Toggles =====
FEATURE_SEARCH = True
FEATURE_SEARCH_METHOD = "mi"
TOP_K_FEATURES = 128
RANDOM_STATE = 72

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
    p = os.path.join(RAW_DIR, f"fut_{split}_data_{interval}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Raw split not found: {p}")
    df = pd.read_parquet(p)

    cols = {c.lower(): c for c in df.columns}
    for k in ["open","high","low","close","volume"]:
        if k in cols:
            real = cols[k]
            df[real] = pd.to_numeric(df[real], errors="coerce")

    if "FundingRate" in df.columns:
        df["FundingRate"] = pd.to_numeric(df["FundingRate"], errors="coerce").fillna(0.0)
    elif "funding_rate" in df.columns:
        df["FundingRate"] = pd.to_numeric(df["funding_rate"], errors="coerce").fillna(0.0)
    else:
        df["FundingRate"] = 0.0

    for k in ["Volume", "Quote_asset_volume", "Taker_buy_base", "Taker_buy_quote"]:
        if k in df.columns:
            df[k] = np.log1p(np.clip(pd.to_numeric(df[k], errors="coerce"), 0, None))

    df = _enforce_dt_index(df)

    for c in REF_COLS_CANON:
        if c not in df.columns:
            df[c] = 0.0

    if interval == BASE_INTERVAL:
        if "FundingSettle" in df.columns:
            df["FundingSettle"] = df["FundingSettle"].astype("int8")
        else:
            df["FundingSettle"] = (((df.index.hour % 8 == 0) & (df.index.minute == 0))).astype("int8")

    return df

# ===== Technical Indicators / Engineered Features =====

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype("float64")
    low  = df["Low"].astype("float64")
    close= df["Close"].astype("float64")
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
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

def compute_features_for_tf(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Computes features for a single timeframe and suffixes column names."""
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype("float64")
    high = df["High"].astype("float64")
    low = df["Low"].astype("float64")
    volume = df["Volume"].astype("float64")

    # Basic returns and volatility
    out["ret_1"] = close.pct_change().replace([np.inf, -np.inf], 0.0)
    out["ret_3"] = close.pct_change(3).replace([np.inf, -np.inf], 0.0)
    out["z_close_48"] = zscore(close, win=48)
    out["hl_spread"] = (high - low) / close.replace(0, np.nan)
    out["vol_z_48"] = zscore(volume, win=48)

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema_12 - ema_26
    out["macd_sig"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_sig"]

    # RSI
    delta = close.diff()
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)
    roll_up, roll_down = up.ewm(alpha=1/14, adjust=False).mean(), down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    # ATR
    out["atr14"] = _atr(df, period=14)

    # Time features (only for base interval to avoid redundancy)
    if interval == BASE_INTERVAL:
        out['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        out['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        out['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        out['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Funding-related features
        fp = _funding_phase_features(out.index)
        out = pd.concat([out, fp], axis=1)
        out["is_funding_settle"] = df.get("FundingSettle", pd.Series(0, index=out.index)).astype("int8")
        out["funding_z_48"] = zscore(df["FundingRate"].astype("float64"), win=48)

    # Add suffix to all generated columns
    out.columns = [f"{c}_{interval}" for c in out.columns]
    
    # Add back original reference columns
    final_out = pd.concat([out, df[REF_COLS_CANON]], axis=1)
    
    return _sanitize(final_out)

# ===== Feature Search & Scaling =====

def _make_proxy_y(df: pd.DataFrame) -> pd.Series:
    y = df["Close"].astype("float64").pct_change().shift(-1)
    y = (y > 0).astype(int)
    return y.fillna(0)

def _feature_search_mi(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    X_ = _sanitize(X).astype("float64")
    y_ = y.astype(int).values
    mi = mutual_info_classif(X_, y_, random_state=RANDOM_STATE, discrete_features=False)
    scores = pd.Series(mi, index=X_.columns).sort_values(ascending=False)
    return scores.head(top_k).index.tolist()

def get_feature_list(train_df: pd.DataFrame) -> List[str]:
    """Performs feature selection based on the training data of the base interval."""
    exclude = set(REF_COLS_CANON)
    feat_cols = [c for c in train_df.columns if c not in exclude]

    if os.path.exists(FEATURE_LIST_PATH):
        print(f"[info] Found existing feature list. Reusing: {FEATURE_LIST_PATH}")
        with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
            keep = json.load(f)
        return keep

    if (not FEATURE_SEARCH) or (TOP_K_FEATURES is None) or (TOP_K_FEATURES >= len(feat_cols)):
        keep = feat_cols
    else:
        print(f"[info] Performing feature search with method: {FEATURE_SEARCH_METHOD} on base interval")
        y_tr = _make_proxy_y(train_df)
        # In this version, both lgbm (not implemented) and mi will use _feature_search_mi
        keep = _feature_search_mi(train_df[feat_cols], y_tr, TOP_K_FEATURES)

    print(f"[ok] Generated feature list with {len(keep)} features.")
    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)
    
    return keep

def get_and_fit_scaler(train_df: pd.DataFrame, feature_list: List[str]) -> StandardScaler:
    """Fits a scaler on the training data of the base interval."""
    print(f"[info] Fitting scaler on base interval training data.")
    scaler = StandardScaler(with_mean=True, with_std=True)
    
    # Ensure all features are present for fitting
    X_tr = train_df.copy()
    for col in feature_list:
        if col not in X_tr.columns:
            X_tr[col] = 0.0
    X_tr = X_tr[feature_list]
    
    scaler.fit(_sanitize(X_tr))
    joblib.dump(scaler, SCALER_PATH)
    print(f"[ok] Scaler fitted and saved to {SCALER_PATH}")
    return scaler

# ===== Main =====

def main():
    # 1. Load raw data for all timeframes
    print("[1/4] Loading all raw data...")
    raw_data = {split: {} for split in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for tf in TIMEFRAMES:
            print(f"  - Loading {split} / {tf}...")
            raw_data[split][tf] = _load_raw(split, tf)

    # 2. Compute features for each timeframe
    print("\n[2/4] Computing features for each timeframe...")
    feature_data = {split: {} for split in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for tf in TIMEFRAMES:
            print(f"  - Computing features for {split} / {tf}...")
            feature_data[split][tf] = compute_features_for_tf(raw_data[split][tf], tf)

    # 3. Get Feature List and Scaler from Base Interval (5m)
    print(f"\n[3/4] Getting feature list and scaler from base interval ({BASE_INTERVAL})...")
    base_train_df = feature_data["train"][BASE_INTERVAL]
    feature_list = get_feature_list(base_train_df)
    scaler = get_and_fit_scaler(base_train_df, feature_list)

    # 4. Process and save data for each timeframe and split
    print("\n[4/4] Processing and saving all timeframe data...")
    for split in ["train", "val", "test"]:
        for tf in TIMEFRAMES:
            print(f"  - Processing {split} / {tf}...")
            df = feature_data[split][tf].copy()

            # Select features
            current_features = [c for c in feature_list if c in df.columns]
            missing_features = [c for c in feature_list if c not in df.columns]
            
            ref_cols = [c for c in REF_COLS_CANON if c in df.columns]
            
            # 수정: .copy()를 사용하여 SettingWithCopyWarning 방지
            df_selected = df[current_features + ref_cols].copy()
            
            # 수정: missing features 추가 방법 개선
            for col in missing_features:
                df_selected[col] = 0.0
            
            # Scale features
            df_scaled = df_selected.copy()
            df_scaled[feature_list] = scaler.transform(df_selected[feature_list])
            
            # Reorder columns for consistency
            final_df = df_scaled[feature_list + ref_cols]
            
            # Save to file
            final_df = _sanitize(final_df)
            assert np.isfinite(final_df.select_dtypes(include=[np.number])).all().all(), f"Non-finite detected in {split}/{tf}"
            
            # 수정: f-string 포맷팅 적용
            out_p = os.path.join(OUT_DIR, f"fe_{split}_{tf}.parquet")
            final_df.to_parquet(out_p)
            print(f"    [ok] Saved {split}/{tf}: {len(final_df):,} x {final_df.shape[1]} -> {out_p}")

    print("\n[+] MTF Multi-Input Feature Engineering complete.")

if __name__ == "__main__":
    main()
