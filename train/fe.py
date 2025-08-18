# fe.py — Feature Engineering for ETHUSDT (5m)
# - Loads raw splits from ./ai_binance/data/raw
# - Keeps OHLCV + FundingRate (UNSCALED reference)
# - Builds technical features (SCALed inputs)
# - Optional feature search (MI or LightGBM) using a self-supervised proxy label (next 5m return sign)
# - Saves processed splits + feature list + scaler to ./ai_binance/data/processed
#
# Output files:
#   fe_train_5m.parquet, fe_val_5m.parquet, fe_test_5m.parquet
#   fe_feature_list_5m.json, scaler_5m.joblib

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
INTERVAL = "5m"

FEATURE_LIST_PATH = os.path.join(OUT_DIR, f"fe_feature_list_{INTERVAL}.json")
SCALER_PATH       = os.path.join(OUT_DIR, f"scaler_{INTERVAL}.joblib")

os.makedirs(OUT_DIR, exist_ok=True)

# ===== Toggles =====
FEATURE_SEARCH = True          # On/Off
FEATURE_SEARCH_METHOD = "mi"   # "mi" | "lgbm"
TOP_K_FEATURES = 128            # keep top-K features; set None to keep all
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


def _load_raw(split: str) -> pd.DataFrame:
    p = os.path.join(RAW_DIR, f"fut_{split}_data_{INTERVAL}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Raw split not found: {p}")
    df = pd.read_parquet(p)

    # Normalize column names present in Binance payloads
    # Ensure OHLCV exists (case tolerant)
    # Map lower to real case
    cols = {c.lower(): c for c in df.columns}
    # Ensure numeric types
    for k in ["open","high","low","close","volume"]:
        if k in cols:
            real = cols[k]
            df[real] = pd.to_numeric(df[real], errors="coerce")

    # FundingRate normalization (either FundingRate or funding_rate)
    if "FundingRate" in df.columns:
        df["FundingRate"] = pd.to_numeric(df["FundingRate"], errors="coerce").fillna(0.0)
    elif "funding_rate" in df.columns:
        df["FundingRate"] = pd.to_numeric(df["funding_rate"], errors="coerce").fillna(0.0)
    else:
        df["FundingRate"] = 0.0

    # Log-stabilize scale-heavy columns to reduce overflow warnings down the line
    for k in ["Volume", "Quote_asset_volume", "Taker_buy_base", "Taker_buy_quote"]:
        if k in df.columns:
            df[k] = np.log1p(np.clip(pd.to_numeric(df[k], errors="coerce"), 0, None))

    # Set datetime index if present
    for tcol in ["Open_time", "open_time", "time"]:
        if tcol in df.columns:
            idx = pd.to_datetime(df[tcol], errors="coerce", utc=True)
            if idx.notna().any():
                df = df.set_index(idx).drop(columns=[tcol])
                df.index.name = "time"
                break

    # Ensure canonical REF columns exist; if missing, backfill zeros
    for c in REF_COLS_CANON:
        if c not in df.columns:
            df[c] = 0.0

    return df.sort_index()


# ===== Technical Indicators / Engineered Features =====

def _ta_basic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Use canonical Close column for returns
    close = out["Close"].astype("float64")

    # Returns
    out["ret_1"]   = close.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["ret_3"]   = close.pct_change(3).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["ret_6"]   = close.pct_change(6).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["ret_12"]  = close.pct_change(12).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Rolling z-scores
    out["z_close_48"] = zscore(close, win=48)
    out["z_ret1_48"]  = zscore(out["ret_1"], win=48)

    # Moving averages
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["macd"]   = out["ema_12"] - out["ema_26"]
    out["macd_sig"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_sig"]

    # RSI (Wilder's)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    # Volatility / Range features
    high, low = out["High"].astype("float64"), out["Low"].astype("float64")
    out["hl_spread"]   = (high - low) / (close.replace(0, np.nan))
    out["atr14"]       = _atr(out, period=14)
    out["vol_z_48"]    = zscore(out["Volume"].astype("float64"), win=48)

    # Funding-related
    out["funding_z_48"] = zscore(out["FundingRate"].astype("float64"), win=48)

    # Clean
    out = _sanitize(out)
    return out


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


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = _ta_basic(df)
    # Ensure REF columns are present and unscaled; create explicit copies for clarity
    for c in REF_COLS_CANON:
        if c not in out.columns and c in df.columns:
            out[c] = df[c]
    # Reference copies (for downstream clarity); not scaled
    out["close_ref"] = out["Close"].astype("float64")
    return _sanitize(out)


# ===== Feature Search =====

def _make_proxy_y(df: pd.DataFrame) -> pd.Series:
    """Self-supervised proxy label: next-bar return sign in {0,1}."""
    y = df["Close"].astype("float64").pct_change().shift(-1)
    y = (y > 0).astype(int)
    return y.fillna(0)


def _feature_search_mi(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    # mutual_info_classif expects finite values
    X_ = _sanitize(X).astype("float64")
    y_ = y.astype(int).values
    mi = mutual_info_classif(X_, y_, random_state=RANDOM_STATE, discrete_features=False)
    scores = pd.Series(mi, index=X_.columns).sort_values(ascending=False)
    keep = scores.head(top_k).index.tolist()
    return keep


def _feature_search_lgbm(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    if not _HAS_LGB:
        # Fallback to MI if LightGBM is unavailable
        return _feature_search_mi(X, y, top_k)
    X_ = _sanitize(X).astype("float64")
    y_ = y.astype(int).values
    dtrain = lgb.Dataset(X_, label=y_)
    params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                  seed=RANDOM_STATE)
    gbm = lgb.train(params, dtrain, num_boost_round=300)
    imp = pd.Series(gbm.feature_importance(importance_type="gain"), index=X_.columns)
    keep = imp.sort_values(ascending=False).head(top_k).index.tolist()
    return keep


def _select_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    # Determine feature columns: exclude REF columns and explicit refs
    exclude = set(REF_COLS_CANON + ["close_ref"])
    feat_cols = [c for c in train.columns if c not in exclude]

    if (not FEATURE_SEARCH) or (TOP_K_FEATURES is None) or (TOP_K_FEATURES >= len(feat_cols)):
        return train, val, test, feat_cols

    # Build proxy label on TRAIN only
    y_tr = _make_proxy_y(train)

    if FEATURE_SEARCH_METHOD == "lgbm":
        keep = _feature_search_lgbm(train[feat_cols], y_tr, TOP_K_FEATURES)
    else:
        keep = _feature_search_mi(train[feat_cols], y_tr, TOP_K_FEATURES)

    for df in (train, val, test):
        for c in keep:
            if c not in df.columns:
                df[c] = 0.0

    train_sel = pd.concat([train[keep], train[REF_COLS_CANON + ["close_ref"]]], axis=1)
    val_sel   = pd.concat([val[keep],   val[REF_COLS_CANON + ["close_ref"]]], axis=1)
    test_sel  = pd.concat([test[keep],  test[REF_COLS_CANON + ["close_ref"]]], axis=1)
    return train_sel, val_sel, test_sel, keep


# ===== Scaling / Saving =====

def _split_X_ref(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ref = df[REF_COLS_CANON + ["close_ref"]].copy()
    X = df.drop(columns=[c for c in ref.columns if c in df.columns], errors="ignore")
    return X, ref


def _scale_and_merge(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: List[str]):
    scaler = StandardScaler(with_mean=True, with_std=True)

    X_tr, ref_tr = _split_X_ref(train)
    X_va, ref_va = _split_X_ref(val)
    X_te, ref_te = _split_X_ref(test)

    X_tr = _sanitize(X_tr.astype("float64"))
    X_va = _sanitize(X_va.astype("float64"))
    X_te = _sanitize(X_te.astype("float64"))

    scaler.fit(X_tr[feature_cols])
    X_tr[feature_cols] = scaler.transform(X_tr[feature_cols])
    X_va[feature_cols] = scaler.transform(X_va[feature_cols])
    X_te[feature_cols] = scaler.transform(X_te[feature_cols])

    tr_out = pd.concat([X_tr[feature_cols], ref_tr], axis=1)
    va_out = pd.concat([X_va[feature_cols], ref_va], axis=1)
    te_out = pd.concat([X_te[feature_cols], ref_te], axis=1)

    # Final safety & save
    def _save(df: pd.DataFrame, split: str):
        df = _sanitize(df)
        assert np.isfinite(df.select_dtypes(include=[np.number])).all().all(), f"Non-finite detected in {split}"
        out_p = os.path.join(OUT_DIR, f"fe_{split}_{INTERVAL}.parquet")
        df.to_parquet(out_p)
        print(f"[ok] {split}: {len(df):,} x {df.shape[1]} → {out_p}")

    _save(tr_out, "train")
    _save(va_out, "val")
    _save(te_out, "test")

    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[ok] feature list → {FEATURE_LIST_PATH}")
    print(f"[ok] scaler → {SCALER_PATH}")


# ===== Main =====

def main():
    # 1) Load raw splits
    df_tr = _load_raw("train")
    df_va = _load_raw("val")
    df_te = _load_raw("test")

    # 2) Compute features (OHLCV + Funding kept unscaled)
    fe_tr = compute_features(df_tr)
    fe_va = compute_features(df_va)
    fe_te = compute_features(df_te)

    # 3) Feature search (optional)
    fe_tr, fe_va, fe_te, feat_cols = _select_features(fe_tr, fe_va, fe_te)

    # 4) Scale features & save together with REF columns
    _scale_and_merge(fe_tr, fe_va, fe_te, feat_cols)


if __name__ == "__main__":
    main()
