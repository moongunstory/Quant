# fe.py — Feature Engineering for ETHUSDT (MTF) + BTC1h (REV-5 / TF-specific search & scaler)
"""
- NEW (REV-5):
  * 타임프레임별(5m/15m/1h/4h) 피처검색 + 스케일러 각각 생성/적용.
  * BTCUSDT 1h 보조 시계열: 경량 지표 + HA, 스케일/피처선택 제외 그대로 저장.
  * Heikin-Ashi(HA) ETH 전 TF + BTC1h 포함.

- Raw in:
    ./ai_binance/data/raw/fut_{train|val|test}_data_{5m|15m|1h|4h}.parquet
    ./ai_binance/data/raw/fut_{train|val|test}_data_btc1h.parquet
- Out:
    ./ai_binance/data/processed/fe_{train|val|test}_{5m|15m|1h|4h|btc1h}.parquet
    ./ai_binance/data/processed/fe_feature_list_{5m|15m|1h|4h}.json
    ./ai_binance/data/processed/scaler_{5m|15m|1h|4h}.joblib
"""

from __future__ import annotations

import os, json
from typing import List, Dict
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
os.makedirs(OUT_DIR, exist_ok=True)

# MTF Setup
TIMEFRAMES     = ["5m", "15m", "1h", "4h", "btc1h"]  # 처리 대상 전체
BASE_INTERVAL  = "5m"  # 일부 시간/펀딩 위상 피처는 5m에서만 생성

# === TF별 피처검색/스케일 설정 ===
FEATURE_SEARCH = True
RANDOM_STATE = 72
# TF별 TOP-K (필요시 조정)
TOP_K_PER_TF = {"5m": 128, "15m": 128, "1h": 96, "4h": 64}
# 검색/스케일 대상 TF (BTC 보조 제외)
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
    # btc1h 파일명 매핑
    if interval == "btc1h":
        # BTC 데이터는 'btcusdt' 폴더의 '1h' 데이터를 사용
        p = os.path.join(RAW_DIR, "btcusdt", f"fut_{split}_data_1h.parquet")
    else:
        # ETH 데이터는 'ethusdt' 폴더의 각 timeframe 데이터를 사용
        p = os.path.join(RAW_DIR, "ethusdt", f"fut_{split}_data_{interval}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Raw split not found: {p}")
    df = pd.read_parquet(p)

    cols = {c.lower(): c for c in df.columns}
    for k in ["open","high","low","close","volume"]:
        if k in cols:
            real = cols[k]
            df[real] = pd.to_numeric(df[real], errors="coerce")

    # FundingRate 표준화
    if "FundingRate" in df.columns:
        df["FundingRate"] = pd.to_numeric(df["FundingRate"], errors="coerce").fillna(0.0)
    elif "funding_rate" in df.columns:
        df["FundingRate"] = pd.to_numeric(df["funding_rate"], errors="coerce").fillna(0.0)
    else:
        df["FundingRate"] = 0.0

    # 로그-스케일 가능한 양수 컬럼들(선택)
    for k in ["Volume", "Quote_asset_volume", "Taker_buy_base", "Taker_buy_quote"]:
        if k in df.columns:
            df[k] = np.log1p(np.clip(pd.to_numeric(df[k], errors="coerce"), 0, None))

    df = _enforce_dt_index(df)

    # 참조컬럼 보정
    for c in REF_COLS_CANON:
        if c not in df.columns:
            df[c] = 0.0

    # FundingSettle (5m 기준만 강제 필요, 나머지는 선택)
    if interval == BASE_INTERVAL:
        if "FundingSettle" in df.columns:
            df["FundingSettle"] = df["FundingSettle"].astype("int8")
        else:
            df["FundingSettle"] = (((df.index.hour % 8 == 0) & (df.index.minute == 0))).astype("int8")

    return df

# ===== Indicators =====

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
    steps_per_8h = 96  # 5m 기준
    steps_since = (idx.hour % 8) * 12 + (idx.minute // 5)
    steps_to_next = (steps_per_8h - steps_since) % steps_per_8h
    phase = 2 * np.pi * (steps_since / steps_per_8h)
    out = pd.DataFrame(index=idx)
    out["time_to_funding_5m"] = steps_to_next.astype("int16")
    out["funding_phase_sin"] = np.sin(phase)
    out["funding_phase_cos"] = np.cos(phase)
    return out

def compute_heikin_ashi(df_ohlc: pd.DataFrame, o="Open", h="High", l="Low", c="Close") -> pd.DataFrame:
    if df_ohlc.empty:
        return pd.DataFrame(index=df_ohlc.index)
    O = df_ohlc[o].to_numpy(dtype=float)
    H = df_ohlc[h].to_numpy(dtype=float)
    L = df_ohlc[l].to_numpy(dtype=float)
    C = df_ohlc[c].to_numpy(dtype=float)
    n = len(df_ohlc)
    HA_C = (O + H + L + C) / 4.0
    HA_O = np.empty(n, dtype=float)
    HA_O[0] = (O[0] + C[0]) / 2.0
    for i in range(1, n):
        HA_O[i] = (HA_O[i-1] + HA_C[i-1]) / 2.0
    HA_H = np.maximum.reduce([H, HA_O, HA_C])
    HA_L = np.minimum.reduce([L, HA_O, HA_C])
    out = pd.DataFrame({
        "HA_O": HA_O, "HA_H": HA_H, "HA_L": HA_L, "HA_C": HA_C
    }, index=df_ohlc.index)
    out["HA_TR"] = out["HA_H"] - out["HA_L"]
    out["HA_BC"] = out["HA_C"] - out["HA_O"]
    out["HA_R"]  = out["HA_C"].pct_change().fillna(0.0)
    return out

# ===== Feature Engines =====

def compute_features_for_tf(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    ETH TF: 리치 피처(리턴/변동성/MACD/RSI/ATR + 시간/펀딩 + HA)
    BTC1h: 경량 피처(ret_1h, ret_4h, ATR14, HA) + 참조컬럼 동봉. 스케일 비적용.
    """
    df = df.sort_index()
    out = pd.DataFrame(index=df.index)

    # --- BTC 1h (경량) ---
    if interval == "btc1h":
        close = df["Close"].astype("float64")

        out["ret_1h"] = close.pct_change().replace([np.inf, -np.inf], 0.0)
        out["ret_4h"] = close.pct_change(4).replace([np.inf, -np.inf], 0.0)
        out["atr14"]  = _atr(df, period=14)

        ha = compute_heikin_ashi(df[["Open","High","Low","Close"]])
        out = pd.concat([out, ha], axis=1)

        # 접미사
        out.columns = [f"{c}_btc1h" for c in out.columns]

        # 참조열 붙여 저장용으로 반환
        ref = df[REF_COLS_CANON].copy()
        return _sanitize(pd.concat([out, ref], axis=1))

    # --- ETH 타임프레임 ---
    close = df["Close"].astype("float64")
    high  = df["High"].astype("float64")
    low   = df["Low"].astype("float64")
    volume= df["Volume"].astype("float64")

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
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)

    # ATR
    out["atr14"] = _atr(df, period=14)

    # Heikin-Ashi
    ha = compute_heikin_ashi(df[["Open","High","Low","Close"]])
    out = pd.concat([out, ha], axis=1)

    # Time/Funding (base interval만)
    if interval == BASE_INTERVAL:
        out['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        out['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        out['day_sin']  = np.sin(2 * np.pi * df.index.dayofweek / 7)
        out['day_cos']  = np.cos(2 * np.pi * df.index.dayofweek / 7)
        fp = _funding_phase_features(out.index)
        out = pd.concat([out, fp], axis=1)
        out["is_funding_settle"] = df.get("FundingSettle", pd.Series(0, index=out.index)).astype("int8")
        out["funding_z_48"] = zscore(df["FundingRate"].astype("float64"), win=48)

    # 접미사
    out.columns = [f"{c}_{interval}" for c in out.columns]

    # 참조컬럼 동봉
    final_out = pd.concat([out, df[REF_COLS_CANON]], axis=1)
    return _sanitize(final_out)

# ===== Feature Search (per TF) & Scaler (per TF) =====

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

def _feature_search_for_tf(train_df: pd.DataFrame, tf: str) -> List[str]:
    exclude = set(REF_COLS_CANON)
    feat_cols = [c for c in train_df.columns if c not in exclude]
    if not FEATURE_SEARCH or len(feat_cols) == 0:
        keep = feat_cols
    else:
        y_tr = _make_proxy_y(train_df)  # 해당 TF의 다음 스텝 방향
        k = min(TOP_K_PER_TF.get(tf, len(feat_cols)), len(feat_cols))
        keep = _feature_search_mi(train_df[feat_cols], y_tr, k)
    with open(FEATURE_LIST_PATH_FMT.format(tf=tf), "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)
    print(f"[ok] feature_list[{tf}] = {len(keep)} → {FEATURE_LIST_PATH_FMT.format(tf=tf)}")
    return keep

def _fit_scaler_for_tf(train_df: pd.DataFrame, feat_list: List[str], tf: str) -> StandardScaler:
    X = _sanitize(train_df.reindex(columns=feat_list, fill_value=0.0))[feat_list].to_numpy(dtype=np.float64, copy=False)
    sc = StandardScaler(with_mean=True, with_std=True).fit(X)
    joblib.dump(sc, SCALER_PATH_FMT.format(tf=tf))
    print(f"[ok] scaler[{tf}] saved → {SCALER_PATH_FMT.format(tf=tf)}")
    return sc

# ===== Main =====

def main():
    # 1) Load
    print("[1/4] Loading all raw data...")
    raw_data: Dict[str, Dict[str, pd.DataFrame]] = {split: {} for split in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for tf in TIMEFRAMES:
            print(f"  - Loading {split} / {tf}...")
            raw_data[split][tf] = _load_raw(split, tf)

    # 2) Feature compute
    print("\n[2/4] Computing features for each timeframe...")
    feature_data: Dict[str, Dict[str, pd.DataFrame]] = {split: {} for split in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for tf in TIMEFRAMES:
            print(f"  - Computing features for {split} / {tf}...")
            feature_data[split][tf] = compute_features_for_tf(raw_data[split][tf], tf)

    # 3) TF-specific feature lists & scalers
    print(f"\n[3/4] Building TF-specific feature lists & scalers...")
    feature_list_per_tf: Dict[str, List[str]] = {}
    scalers: Dict[str, StandardScaler] = {}

    for tf in TF_FOR_SEARCH:
        tr_df = feature_data["train"][tf]
        feat_list = _feature_search_for_tf(tr_df, tf)
        feature_list_per_tf[tf] = feat_list
        scalers[tf] = _fit_scaler_for_tf(tr_df, feat_list, tf)

    # 4) Process & save
    print("\n[4/4] Processing and saving all timeframe data...")
    for split in ["train", "val", "test"]:
        for tf in TIMEFRAMES:
            print(f"  - Processing {split} / {tf}...")
            df = feature_data[split][tf].copy()

            if tf == "btc1h":
                # 보조 시계열: 스케일/선택 없이 저장 (경량 지표 + REF)
                out_btc = _sanitize(df)
                out_p = os.path.join(OUT_DIR, f"fe_{split}_btc1h.parquet")
                out_btc.to_parquet(out_p)
                print(f"    [ok] Saved {split}/btc1h (no scaling): {len(out_btc):,} x {out_btc.shape[1]} -> {out_p}")
                continue

            # ETH TF: TF별 feature_list + TF별 scaler 적용
            feat_list = feature_list_per_tf[tf]
            df_sel = df.reindex(columns=feat_list, fill_value=0.0)

            X = _sanitize(df_sel[feat_list]).to_numpy(dtype=np.float64, copy=False)
            Xs = scalers[tf].transform(X)
            df_scaled = pd.DataFrame(Xs, index=df.index, columns=feat_list)

            # 참조열 (비스케일)
            ref_cols = [c for c in REF_COLS_CANON if c in df.columns]
            final_df = _sanitize(pd.concat([df_scaled, df[ref_cols]], axis=1))

            out_p = os.path.join(OUT_DIR, f"fe_{split}_{tf}.parquet")
            final_df.to_parquet(out_p)
            print(f"    [ok] Saved {split}/{tf}: {len(final_df):,} x {final_df.shape[1]} -> {out_p}")

    print("\n[+] MTF Multi-Input Feature Engineering complete.")

if __name__ == "__main__":
    main()
