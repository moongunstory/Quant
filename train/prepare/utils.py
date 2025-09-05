# utils.py — 범용 유틸리티 + 지표 + 검증 함수들

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Optional
from .paths import REF_COLS_CANON, HPO_EXPAND_WINDOWS

# === 데이터 정리 ===
def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    std = df.std(numeric_only=True)
    zero_std_cols = std[std == 0].index.tolist()
    if zero_std_cols:
        print(f"[INFO] Zero std columns detected: {zero_std_cols[:5]}...")
        df[zero_std_cols] = 0.0
    return df.fillna(0.0)

# === 시계열 유틸 ===
def zscore(s: pd.Series, win: Optional[int] = None) -> pd.Series:
    if win is None:
        mu, sd = s.mean(), s.std()
        return ((s - mu) / (sd or 1e-9)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std().replace(0, np.nan)
    return ((s - mu) / sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)

def pct_slope(x: pd.Series, win: int) -> pd.Series:
    return x.pct_change(fill_method=None).rolling(win, min_periods=win).mean().fillna(0.0)

# === 날짜 처리 ===
def to_utc_dt(s: pd.Series) -> pd.DatetimeIndex:
    s = pd.Series(s)
    if np.issubdtype(s.dtype, np.number):
        vmax = float(s.dropna().max()) if s.dropna().size else 0.0
        unit = "ms" if 1e11 < vmax < 1e14 else "ns"
        return pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")

def enforce_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        for tcol in ["Open_time", "open_time", "time"]:
            if tcol in df.columns:
                idx = to_utc_dt(df[tcol])
                if idx.notna().any():
                    df = df.set_index(idx).drop(columns=[tcol])
                    break
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = to_utc_dt(df.index)
    df.index = pd.to_datetime(df.index, utc=True)

    # === 시간대 검증 ===
    if df.index.tz is None:
        print(f"[WARNING] Timezone info missing, assuming UTC")
    elif str(df.index.tz) != 'UTC':
        print(f"[WARNING] Non-UTC timezone detected: {df.index.tz}")

    df = df.sort_index()
    df.index.name = "time"
    return df

# === Feature Mask 적용 ===
def apply_feature_mask(df: pd.DataFrame, selected: List[str], ref_cols: Optional[List[str]] = None) -> pd.DataFrame:
    missing_cols = [c for c in selected if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing features in dataframe: {missing_cols[:10]}...")

    if len(selected) > 1000:
        print(f"[WARNING] Large feature set: {len(selected)} features selected")

    if ref_cols is None:
        ref_cols = [c for c in REF_COLS_CANON if c in df.columns]
    X = df.reindex(columns=selected, fill_value=0.0)
    return pd.concat([X, df[ref_cols]], axis=1)

# === Proxy Y 생성 ===
def make_proxy_y(df: pd.DataFrame, horizon: int) -> pd.Series:
    future_ret = df["Close"].astype(float).pct_change(horizon, fill_method=None).shift(-horizon)
    valid_mask = future_ret.notna()
    lost_samples = (~valid_mask).sum()
    print(f"[INFO] {lost_samples} samples lost due to future return calculation (horizon={horizon})")
    return (future_ret > 0).astype(int).fillna(0)

# === 컬럼 이름 보조 ===
def prefix_f(cols: List[str]) -> List[str]:
    return [c if c.startswith("f_") else f"f_{c}" for c in cols]

def rename_with_f_prefix(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    mapping = {c: c if c.startswith("f_") else f"f_{c}" for c in feat_cols}
    return df.rename(columns=mapping)

# === 스케일러 학습 ===
def fit_scaler(train_df: pd.DataFrame, feat_list: List[str]) -> StandardScaler:
    X = sanitize(train_df[feat_list]).astype(float)
    return StandardScaler().fit(X)  # 👈 DataFrame 유지

