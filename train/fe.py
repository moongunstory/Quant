"""
Module 2 — Feature Engineering & Normalization for ETH Futures (No Leakage, No Warnings)

Goal: compact, regime-aware features for RL without leakage.
- Inputs: raw parquet (5m, 15m, 1h, 4h) → UTC index, sorted asc
- Indicators (core only):
  • Returns: ret1, ret12, ret48
  • Volatility: atr14, hlv=(High-Low)/Close
  • Trend: sma20_slope, macd_hist(12,26,9), adx14
  • Position: bb_pos=(Close-mid)/(up-dn)
  • Momentum: rsi14
  • Volume: vol_norm=Volume/atr14  (log1p later)
- Regime tags (one-hot):
  • trend_strong(ADX14≥25)/trend_weak
  • vola_high(ATR%≥q70 on TRAIN)/vola_low
- Feature selection (noise cut):
  • Unsupervised PCA-loading ranking on TRAIN → top N features (keep regime tags)
- Normalize:
  • log1p on Volume/ATR family → StandardScaler fit on TRAIN, reuse for VAL/TEST
- Outputs:
  • processed/train_normalized.parquet, val_normalized.parquet, test_normalized.parquet
  • feature_list.json (selected features only)
  • scaler.joblib
  • normalize_stats.json (stores vola threshold & misc)
"""

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

RAW_DIR = "./ai_binance/data/raw"
PROC_DIR = "./ai_binance/data/processed"
INTERVALS = ["5m", "15m", "1h", "4h"]
BASE_INTERVAL = "5m"
FEATURE_LIST_PATH = os.path.join(PROC_DIR, "feature_list.json")
SCALER_JOBLIB = os.path.join(PROC_DIR, "scaler.joblib")
SCALER_INFO_JSON = os.path.join(PROC_DIR, "normalize_stats.json")  # stores thresholds etc.

# ===== Regime & Selection params =====
ADX_PERIOD = 14
ZSCORE_WIN = 48            # (kept for potential future use)
VOL_Q = 0.70               # 70th percentile on TRAIN atr_pct_5m
TOP_N = 30                 # number of selected features (excluding forced keeps)

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
# Indicators (dependency-free)
# =========================
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).rolling(period, min_periods=period).mean()
    dn = (-d.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _indicators(df: pd.DataFrame, itv: str) -> pd.DataFrame:
    """Return ONLY engineered features to keep merge clean."""
    x = pd.DataFrame(index=df.index)

    # Returns
    x[f"ret1_{itv}"] = df["Close"].pct_change()
    x[f"ret12_{itv}"] = df["Close"].pct_change(12)
    x[f"ret48_{itv}"] = df["Close"].pct_change(48)

    # Volatility
    prev_close = df["Close"].shift()
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum((df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()),
    )
    atr14 = pd.Series(tr, index=df.index).rolling(14, min_periods=14).mean()
    x[f"atr14_{itv}"] = atr14
    x[f"hlv_{itv}"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)

    # Trend: SMA20 slope
    sma20 = df["Close"].rolling(20, min_periods=20).mean()
    x[f"sma20_slope_{itv}"] = sma20.diff()

    # MACD hist (12,26,9)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    x[f"macd_hist_{itv}"] = macd - macd_sig

    # ADX(14)
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr_adx = pd.Series(tr, index=df.index).rolling(ADX_PERIOD, min_periods=ADX_PERIOD).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(ADX_PERIOD, min_periods=ADX_PERIOD).mean() / atr_adx.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(ADX_PERIOD, min_periods=ADX_PERIOD).mean() / atr_adx.replace(0, np.nan)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    x[f"adx14_{itv}"] = dx.rolling(ADX_PERIOD, min_periods=ADX_PERIOD).mean()

    # Bollinger position (20)
    bb_mid = sma20
    bb_std = df["Close"].rolling(20, min_periods=20).std()
    bb_up = bb_mid + 2 * bb_std
    bb_dn = bb_mid - 2 * bb_std
    span = (bb_up - bb_dn).replace(0, np.nan)
    x[f"bb_pos_{itv}"] = (df["Close"] - bb_mid) / span

    # Momentum
    x[f"rsi14_{itv}"] = _rsi(df["Close"], 14)

    # Volume (normalized by ATR14)
    x[f"vol_norm_{itv}"] = df["Volume"] / atr14.replace(0, np.nan)

    # For regime tagging on BASE interval: ATR% = atr14/Close
    if itv == BASE_INTERVAL:
        x[f"atr_pct_{itv}"] = (atr14 / df["Close"].replace(0, np.nan))

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

    out = dfs[BASE_INTERVAL].copy()  # base features are not suffixed (already include itv in names)
    for itv, df in dfs.items():
        if itv == BASE_INTERVAL:
            continue
        out = pd.merge_asof(
            out.sort_index(),
            df.sort_index(),  # every column already includes _{itv} suffix
            left_index=True,
            right_index=True,
            direction="backward",
        )
    return out

# =========================
# Regime Tags (TRAIN threshold → apply to all)
# =========================
def _add_regime_columns(df: pd.DataFrame, vola_thresh: float) -> pd.DataFrame:
    out = df.copy()
    adx = out.get(f"adx14_{BASE_INTERVAL}")
    atr_pct = out.get(f"atr_pct_{BASE_INTERVAL}")

    trend_strong = (adx >= 25).astype(float) if adx is not None else 0.0
    trend_weak = 1.0 - trend_strong

    vola_high = (atr_pct >= vola_thresh).astype(float) if atr_pct is not None else 0.0
    vola_low = 1.0 - vola_high

    out["trend_strong"] = trend_strong
    out["trend_weak"] = trend_weak
    out["vola_high"] = vola_high
    out["vola_low"] = vola_low
    return out

# =========================
# Normalization helpers
# =========================
def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    x = df.select_dtypes(include=["float64", "float32", "int64", "int32"]).copy()
    # log1p for heavy-tail families: volume/atr
    for c in list(x.columns):
        lc = c.lower()
        if ("volume" in lc) or ("vol_" in lc) or lc.startswith("vol") or ("atr" in lc):
            x[c] = np.log1p(x[c].clip(lower=0))
    return x

def _force_keep_features(df: pd.DataFrame) -> List[str]:
    """
    레짐 태그 + ADX 원시값(모든 TF)을 강제 포함.
    df: _numeric() 통과 후의 숫자형 DataFrame
    """
    base_keep = ["trend_strong", "trend_weak", "vola_high", "vola_low"]
    # 모든 타임프레임 ADX 보존 (예: adx14_5m, adx14_15m, ...)
    adx_cols = [c for c in df.columns if c.lower().startswith("adx14_")]

    seen, kept = set(), []
    for c in base_keep + adx_cols:
        if c in df.columns and c not in seen:
            kept.append(c); seen.add(c)
    return kept

def _pca_feature_rank(train_df: pd.DataFrame, force_keep: List[str], top_n: int = TOP_N) -> List[str]:
    """Unsupervised ranking by PCA loading energy weighted by explained variance."""
    X = train_df.dropna()
    if X.empty or X.shape[1] == 0:
        return list(dict.fromkeys(force_keep))

    # Z-score for PCA
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    n_samples, n_features = Z.shape
    # PCA 제약: n_components <= min(n_samples, n_features) and >= 1
    k = max(1, min(top_n, n_features, n_samples))

    # 'randomized' 인자 제거
    pca = PCA(n_components=k, svd_solver="auto")
    pca.fit(Z)

    loadings = pca.components_                 # (k, n_features)
    weights = pca.explained_variance_ratio_.reshape(-1, 1)  # (k, 1)
    imp = (loadings ** 2) * weights            # (k, n_features)
    scores = imp.sum(axis=0)                   # (n_features,)
    feats = list(X.columns)

    ranked = [f for _, f in sorted(zip(scores, feats), key=lambda t: float(t[0]), reverse=True)]

    # force_keep을 항상 포함(앞쪽 고정)
    forced = [f for f in force_keep if f in feats]
    ranked = [f for f in ranked if f not in set(forced)]
    selected = forced + ranked[:top_n]

    # 순서 유지 중복 제거
    seen, final = set(), []
    for f in selected:
        if f not in seen:
            final.append(f); seen.add(f)
    return final

def _fit_scaler_and_save(train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    numeric = train_df[feature_cols].dropna()
    scaler = StandardScaler()
    arr = scaler.fit_transform(numeric)
    norm = pd.DataFrame(arr, index=numeric.index, columns=feature_cols)
    _ensure_dir(PROC_DIR)
    joblib.dump(scaler, SCALER_JOBLIB)
    with open(FEATURE_LIST_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)
    return norm, scaler

def _apply_scaler(df: pd.DataFrame, feature_cols: List[str], scaler: StandardScaler) -> pd.DataFrame:
    x = df.copy()
    # align & dropna like TRAIN
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
def _build(split: str, vola_thresh: float) -> pd.DataFrame:
    merged = _merge(split)
    merged = _add_regime_columns(merged, vola_thresh)
    return merged

def main() -> None:
    _ensure_dir(PROC_DIR)

    # 1) TRAIN merge + regime thresholds
    train_merged = _merge("train")
    # derive vola threshold from TRAIN only
    atr_pct_col = f"atr_pct_{BASE_INTERVAL}"
    if atr_pct_col not in train_merged.columns:
        raise RuntimeError(f"Missing {atr_pct_col} for volatility regime threshold.")
    vola_thresh = float(np.nanquantile(train_merged[atr_pct_col].values, VOL_Q))
    # attach regimes to TRAIN
    train_merged = _add_regime_columns(train_merged, vola_thresh)

    # 2) Feature selection on TRAIN (unsupervised PCA ranking)
    # Candidates: numeric engineered + regime tags; exclude helper atr_pct column from selection
    train_numeric = _numeric(train_merged)
    force_keep = _force_keep_features(train_numeric)
    # ensure helper not considered
    if atr_pct_col in train_numeric.columns:
        train_numeric = train_numeric.drop(columns=[atr_pct_col])
    selected_feats = _pca_feature_rank(train_numeric, force_keep=force_keep, top_n=TOP_N)

    # 3) Fit StandardScaler on TRAIN(selected) and save artifacts
    # Keep only selected columns (dropna to align)
    train_selected = train_numeric[selected_feats]
    train_norm, scaler = _fit_scaler_and_save(train_selected, selected_feats)

    # Save stats JSON (for reproducibility & VAL/TEST)
    with open(SCALER_INFO_JSON, "w") as f:
        json.dump(
            {
                "info": "StandardScaler for FE; PCA-loading selection on TRAIN.",
                "vola_quantile": VOL_Q,
                "vola_threshold_on_train_atr_pct": vola_thresh,
                "base_interval": BASE_INTERVAL,
                "top_n": TOP_N,
                "feature_count": len(selected_feats),
            },
            f,
            indent=2,
        )

    _save_norm(train_norm, "train")

    # 4) VAL/TEST — build with same threshold, same features, same scaler
    for split in ["val", "test"]:
        merged = None
        try:
            merged = _build(split, vola_thresh)
        except ValueError:
            print(f"[skip] {split} missing base interval {BASE_INTERVAL}")
            continue
        if merged is None or merged.empty:
            print(f"[skip] {split} has no data")
            continue
        X = _numeric(merged)
        # drop helper
        if atr_pct_col in X.columns:
            X = X.drop(columns=[atr_pct_col])
        # align columns to selected
        for c in selected_feats:
            if c not in X.columns:
                X[c] = np.nan
        X = X[selected_feats]
        norm = _apply_scaler(X, selected_feats, scaler)
        _save_norm(norm, split)

    print("[OK] FE pipeline completed: compact features, regime tags, PCA-selected, normalized without leakage.")

if __name__ == "__main__":
    main()
