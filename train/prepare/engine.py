# engine.py — 데이터 로딩, 피처 생성, HPO 확장

import os
import numpy as np
import pandas as pd
import json
import joblib
from typing import Dict, List, Optional

from sklearn.feature_selection import mutual_info_classif
from .paths import RAW_DIR, OUT_DIR, REF_COLS_CANON, BASE_INTERVAL, HPO_FEATURE_LIST_FMT
from .utils import *

# === 데이터 로딩 ===
def load_raw(split: str, interval: str) -> pd.DataFrame:
    if interval == "btc1h":
        path = os.path.join(RAW_DIR, "btcusdt", f"fut_{split}_data_1h.parquet")
    else:
        path = os.path.join(RAW_DIR, "ethusdt", f"fut_{split}_data_{interval}.parquet")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found: {path}")

    df = pd.read_parquet(path)

    # 수치형 보정
    for col in ["open", "high", "low", "close", "volume"]:
        real = next((c for c in df.columns if c.lower() == col), None)
        if real:
            df[real] = pd.to_numeric(df[real], errors="coerce")

    # Funding
    if "FundingRate" not in df.columns:
        if "funding_rate" in df.columns:
            df.rename(columns={"funding_rate": "FundingRate"}, inplace=True)
        else:
            df["FundingRate"] = 0.0
    df["FundingRate"] = pd.to_numeric(df["FundingRate"], errors="coerce").fillna(0.0)

    # Volume + 로그
    def add_volume_logs(col: str):
        if col in df.columns:
            base = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}Raw"] = base
            df[f"{col}Log"] = np.log1p(np.clip(base, 0, None))

    for col in ["Volume", "Quote_asset_volume", "Taker_buy_base", "Taker_buy_quote"]:
        add_volume_logs(col)

    if "VolumeRaw" in df.columns:
        df["Volume"] = df["VolumeRaw"]

    df = enforce_dt_index(df)

    for c in REF_COLS_CANON:
        if c not in df.columns:
            df[c] = 0.0

    if interval == BASE_INTERVAL:
        df["FundingSettle"] = (((df.index.hour % 8 == 0) & (df.index.minute == 0))).astype("int8")

    return df

# === BTC merge + 검증 포함 ===
def merge_btc(df: pd.DataFrame, btc_df: pd.DataFrame, interval: str) -> pd.DataFrame:
    btc_slim = pd.DataFrame(index=btc_df.index)
    btc_slim["Close_btc1h"] = pd.to_numeric(btc_df["Close"], errors="coerce")
    if "Volume" in btc_df.columns:
        btc_slim["Volume_btc1h"] = pd.to_numeric(btc_df["Volume"], errors="coerce")

    tol = pd.Timedelta(hours=2)
    merged = pd.merge_asof(
        df.reset_index().sort_values("time"),
        btc_slim.reset_index().sort_values("time"),
        on="time", direction="backward", tolerance=tol
    ).set_index("time")

    btc_close = merged["Close_btc1h"].astype(float)
    missing_ratio = btc_close.isna().mean()
    if missing_ratio > 0.05:
        print(f"[WARNING] {missing_ratio:.1%} ETH rows missing BTC data in {interval}")
    merged["Close_btc1h"] = btc_close.ffill()

    return merged

# === HPO 피처 확장 ===
def add_hpo_candidates(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    df = df.copy()
    existing_cols = set(df.columns)
    base_cols = [c for c in df.columns if c not in REF_COLS_CANON]
    new_feats: Dict[str, pd.Series] = {}

    for c in base_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.dtype.kind not in ("i", "u", "f"):
            continue
        for w in HPO_EXPAND_WINDOWS:
            mean_name = f"{c}_mean{w}"
            z_name = f"{c}_z{w}"
            if mean_name in existing_cols:
                print(f"[WARNING] Duplicate feature name: {mean_name}")
                mean_name = f"hpo_{mean_name}"
            if z_name in existing_cols:
                z_name = f"hpo_{z_name}"
            m = s.rolling(w, min_periods=w).mean()
            sd = s.rolling(w, min_periods=w).std().replace(0, np.nan)
            new_feats[mean_name] = m
            new_feats[z_name] = (s - m) / (sd + 1e-9)

    if new_feats:
        df = pd.concat([df, pd.DataFrame(new_feats, index=df.index)], axis=1)

    return sanitize(df)

# === Feature Search (MI 기반) ===
def feature_search_mi(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    X_ = sanitize(X).astype(float)
    mi = mutual_info_classif(X_, y.values, random_state=72)
    scores = pd.Series(mi, index=X_.columns).sort_values(ascending=False)
    return scores.head(top_k).index.tolist()

# === 저장된 결과 로딩 ===
def load_processed(split: str, tf: str, mode: str = "auto") -> pd.DataFrame:
    base_p = os.path.join(OUT_DIR, f"fe_{split}_{tf}.parquet")
    hpo_p = os.path.join(OUT_DIR, f"feHPO_{split}_{tf}.parquet")
    if mode == "hpo":
        path = hpo_p
    elif mode == "base":
        path = base_p
    else:
        path = hpo_p if os.path.exists(hpo_p) else base_p
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found: {path}")
    df = pd.read_parquet(path)
    # Cast float columns to float32 to save memory
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)
    return df

def feature_universe(df: pd.DataFrame, prefix: str = "f_") -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]

def build_universe_from_processed(split: str = "train", tf: str = "5m", mode: str = "auto") -> List[str]:
    df = load_processed(split, tf, mode=mode)
    feats = feature_universe(df, prefix="f_")
    if len(feats) < 10:
        raise RuntimeError(f"Feature universe too small: {len(feats)}")
    return feats
