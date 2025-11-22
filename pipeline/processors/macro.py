# pipeline/processors/macro.py

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from .utils import ensure_sorted_date


# 금리/스프레드 위주 시리즈 (delta_30d 중요)
FRED_RATE_SERIES = {
    "dgs10",
    "dgs2",
    "dgs5",
    "t10y2y",
    "t10y3m",
    "dff",
    "fedfunds",
}

# 레벨 자체가 중요한 애들 (z-score도 같이)
LEVEL_Z_SYMBOLS = {"vix", "dxy_nyb"}


def _fred_to_features(series_id: str, df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_sorted_date(df, "date")
    df = df[["date", "value"]].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    col_level = f"fred_{series_id}_level"
    df = df.rename(columns={"value": col_level})

    # 30일 변화량 (금리/스프레드/지표 공통으로 써도 크게 문제 없음)
    df[f"fred_{series_id}_delta_30d"] = df[col_level] - df[col_level].shift(30)

    return df


def _yahoo_to_features(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_sorted_date(df, "date")

    # close / adjclose 중 하나 사용
    close_col = None
    for cand in ("close", "adjclose", "adj_close"):
        if cand in df.columns:
            close_col = cand
            break
    if close_col is None:
        # 못 찾으면 스킵
        return pd.DataFrame()

    close = pd.to_numeric(df[close_col], errors="coerce")
    prefix = f"yahoo_{symbol}"

    df_out = pd.DataFrame({"date": df["date"].copy()})

    df_out[f"{prefix}_ret_1d"] = close.pct_change(1)
    df_out[f"{prefix}_ret_5d"] = close.pct_change(5)
    df_out[f"{prefix}_ret_20d"] = close.pct_change(20)

    log_ret = np.log(close / close.shift(1))
    df_out[f"{prefix}_vol_20d"] = log_ret.rolling(20).std()

    if symbol in LEVEL_Z_SYMBOLS:
        df_out[f"{prefix}_level"] = close
        roll = close.rolling(252)
        df_out[f"{prefix}_z_1y"] = (close - roll.mean()) / roll.std()

    return df_out


def _fx_to_features(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    fx_aud_usd 같은 것들: close를 환율로 보고 수익률/변동성만 만든다.
    """
    df = ensure_sorted_date(df, "date")
    # 'value' 대신 'close' 컬럼을 사용
    if 'close' not in df.columns:
        return pd.DataFrame() # close 컬럼이 없으면 빈 DataFrame 반환

    price = pd.to_numeric(df['close'], errors="coerce")

    prefix = f"fx_{symbol}"

    out = pd.DataFrame({"date": df["date"].copy()})
    out[f"{prefix}_ret_1d"] = price.pct_change(1)
    out[f"{prefix}_ret_5d"] = price.pct_change(5)
    out[f"{prefix}_ret_20d"] = price.pct_change(20)

    log_ret = np.log(price / price.shift(1))
    out[f"{prefix}_vol_20d"] = log_ret.rolling(20).std()

    return out


def build_macro_features(
    fred_series: Dict[str, pd.DataFrame],
    yahoo_series: Dict[str, pd.DataFrame],
    fx_series: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    매크로 raw DataFrame 묶음을 받아서 일단위 피처 생성.

    fred_series : {"dgs10": df, "t10y2y": df, ...}
    yahoo_series: {"gspc": df, "vix": df, ...}
    fx_series    : {"aud_usd": df, "eur_usd": df, ...}
    """
    features = None

    def _merge(features: pd.DataFrame | None, df_add: pd.DataFrame) -> pd.DataFrame:
        if df_add is None or df_add.empty:
            return features
        if features is None:
            return df_add
        return features.merge(df_add, on="date", how="outer")

    # 1) FRED
    for series_id, df in fred_series.items():
        if df is None or df.empty:
            continue
        df_feat = _fred_to_features(series_id.lower(), df)
        features = _merge(features, df_feat)

    # 2) Yahoo
    for sym, df in yahoo_series.items():
        if df is None or df.empty:
            continue
        df_feat = _yahoo_to_features(sym.lower(), df)
        if df_feat.empty:
            continue
        features = _merge(features, df_feat)

    # 3) FX
    for sym, df in fx_series.items():
        if df is None or df.empty:
            continue
        df_feat = _fx_to_features(sym.lower(), df)
        features = _merge(features, df_feat)

    if features is None:
        raise RuntimeError("No macro data provided to build_macro_features")

    features = ensure_sorted_date(features, "date")
    return features
