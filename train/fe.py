"""
fe.py — Feature Engineering for ETHUSDT (MTF) [REV-2]
- NEW: Multi-Time-Frame (MTF) feature generation.
- Raw: Loads 5m, 15m, 1h, 4h data from ./ai_binance/data/raw/
- Output: fe_{train|val|test}_5m.parquet, fe_feature_list_5m.json, scaler_5m.joblib
- Logic:
  1. Load raw data for all specified timeframes (5m, 15m, 1h, 4h).
  2. Generate features for each timeframe independently, suffixing columns with interval (e.g., "rsi_14_1h").
  3. Use 5m as the base index.
  4. Reindex higher timeframe features to the 5m index using forward-fill.
  5. Concatenate all features into a single dataframe.
  6. Perform feature selection and scaling on the combined MTF feature set.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Tuple

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
BASE_INTERVAL = "5m" # The final index will be based on this interval

# Output paths remain compatible with rl.py
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

# ===== Feature Search =====

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

def _select_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    exclude = set(REF_COLS_CANON + ["close_ref"])
    feat_cols = [c for c in train.columns if c not in exclude]

    if os.path.exists(FEATURE_LIST_PATH):
        print(f"[info] Found existing feature list. Reusing.")
        with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
            keep = json.load(f)
    else:
        if (not FEATURE_SEARCH) or (TOP_K_FEATURES is None) or (TOP_K_FEATURES >= len(feat_cols)):
            keep = feat_cols
        else:
            print(f"[info] Performing feature search with method: {FEATURE_SEARCH_METHOD}")
            y_tr = _make_proxy_y(train)
            if FEATURE_SEARCH_METHOD == "lgbm" and _HAS_LGB:
                # Not implemented in this version, falls back to MI
                keep = _feature_search_mi(train[feat_cols], y_tr, TOP_K_FEATURES)
            else:
                keep = _feature_search_mi(train[feat_cols], y_tr, TOP_K_FEATURES)

    for df in (train, val, test):
        add = [c for c in keep if c not in df.columns]
        for c in add:
            df[c] = 0.0

    train_sel = pd.concat([train[keep], train[REF_COLS_CANON + ["close_ref"]]], axis=1)
    val_sel   = pd.concat([val[keep],   val[REF_COLS_CANON + ["close_ref"]]], axis=1)
    test_sel  = pd.concat([test[keep],  test[REF_COLS_CANON + ["close_ref"]]], axis=1)
    return train_sel, val_sel, test_sel, keep

# ===== Scaling / Saving =====

def _scale_and_merge(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str]):
    scaler = StandardScaler(with_mean=True, with_std=True)

    X_tr, ref_tr = train[feature_cols], train.drop(columns=feature_cols)
    X_va, ref_va = val[feature_cols], val.drop(columns=feature_cols)
    X_te, ref_te = test[feature_cols], test.drop(columns=feature_cols)

    X_tr, X_va, X_te = _sanitize(X_tr), _sanitize(X_va), _sanitize(X_te)

    scaler.fit(X_tr)
    X_tr.loc[:, feature_cols] = scaler.transform(X_tr)
    X_va.loc[:, feature_cols] = scaler.transform(X_va)
    X_te.loc[:, feature_cols] = scaler.transform(X_te)

    tr_out = pd.concat([X_tr, ref_tr], axis=1)
    va_out = pd.concat([X_va, ref_va], axis=1)
    te_out = pd.concat([X_te, ref_te], axis=1)

    def _save(df: pd.DataFrame, split: str):
        df = _sanitize(df)
        assert np.isfinite(df.select_dtypes(include=[np.number])).all().all(), f"Non-finite detected in {split}"
        out_p = os.path.join(OUT_DIR, f"fe_{split}_{BASE_INTERVAL}.parquet")
        df.to_parquet(out_p)
        print(f"[ok] {split}: {len(df):,} x {df.shape[1]} -> {out_p}")

    _save(tr_out, "train")
    _save(va_out, "val")
    _save(te_out, "test")

    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[ok] feature list -> {FEATURE_LIST_PATH}")
    print(f"[ok] scaler -> {SCALER_PATH}")

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

    # 3. Merge MTF features
    print("\n[3/4] Merging MTF features...")
    merged_data = {}
    for split in ["train", "val", "test"]:
        print(f"  - Merging {split} data...")
        base_df = feature_data[split][BASE_INTERVAL].copy()
        
        for tf in TIMEFRAMES:
            if tf == BASE_INTERVAL:
                continue
            
            # Select only feature columns from higher TFs (not OHLCV)
            higher_tf_df = feature_data[split][tf]
            feature_cols_higher_tf = [c for c in higher_tf_df.columns if c not in REF_COLS_CANON]
            
            # Reindex and forward-fill
            resampled_features = higher_tf_df[feature_cols_higher_tf].reindex(base_df.index, method='ffill')
            
            base_df = pd.concat([base_df, resampled_features], axis=1)
        
        base_df["close_ref"] = base_df["Close"].astype("float64")
        merged_data[split] = _sanitize(base_df)

    # 4. Feature selection and scaling
    print("\n[4/4] Selecting features, scaling, and saving...")
    train_df, val_df, test_df = merged_data["train"], merged_data["val"], merged_data["test"]
    
    train_sel, val_sel, test_sel, feat_cols = _select_features(train_df, val_df, test_df)
    
    _scale_and_merge(train_sel, val_sel, test_sel, feat_cols)
    
    print("\n[+] MTF Feature Engineering complete.")

if __name__ == "__main__":
    main()