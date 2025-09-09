
import os
import numpy as np
import pandas as pd
import talib
import heapq

from typing import List, Optional, Tuple
from sklearn.feature_selection import mutual_info_classif

from .paths import RAW_DIR, OUT_DIR, REF_COLS_CANON, BASE_INTERVAL
from .feature_engineering import get_feature_specs_for_tf, generate_feature

# === रॉ डेटा लोड हो रहा है ===

def load_raw(tf: str) -> pd.DataFrame:
    """
    지정된 시간 프레임에 대한 전체 원시 데이터 로드
    - btc1h -> btcusdt, 나머지는 ethusdt로 매핑
    """
    symbol = "btcusdt" if tf == "btc1h" else "ethusdt"
    suffix = "1h" if tf == "btc1h" else tf
    
    path = os.path.join(RAW_DIR, symbol, f"fut_data_{suffix}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw data not found at {path}")
    return pd.read_parquet(path)

def add_y_class(df: pd.DataFrame, lookahead: int = 24, threshold: float = 0.005) -> pd.DataFrame:
    """
    미래 가격 변동에 기반한 목표 변수 (y_class) 생성
    - 0: Sell (가격 하락)
    - 1: Hold (큰 변화 없음)
    - 2: Buy (가격 상승)
    """
    df = df.copy()
    future_returns = df['Close'].pct_change(periods=lookahead, fill_method=None).shift(-lookahead)

    df['y_class'] = 1  # Default to Hold
    df.loc[future_returns > threshold, 'y_class'] = 2  # Buy
    df.loc[future_returns < -threshold, 'y_class'] = 0  # Sell
    
    # Drop rows where y_class could not be calculated
    df.dropna(subset=['y_class'], inplace=True)
    df['y_class'] = df['y_class'].astype(int)

    return df

# === 유틸 함수들 ===

def sanitize(df: pd.DataFrame, verbose: bool = False,
             drop_zero_std: bool = True, std_thresh: float = 1e-8) -> pd.DataFrame:
    """
    학습에 안전한 상태로 정리:
    - Inf/-Inf는 NaN으로 전환
    - 완전히 NaN이거나 상수 컬럼은 제거
    - 분산이 거의 없는 컬럼도 (원하면) 제거
    - fillna(0)은 제거 → 왜곡 방지
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    all_nan_cols = df.columns[df.isna().all()].tolist()
    constant_cols = df.columns[df.nunique(dropna=False) <= 1].tolist()
    std = df.std(numeric_only=True)
    zero_std_cols = std[std <= std_thresh].index.tolist()

    cols_to_drop = set()
    if drop_zero_std:
        cols_to_drop.update(all_nan_cols + constant_cols + zero_std_cols)

    if verbose and cols_to_drop:
        print(f"[sanitize] Dropping columns: {cols_to_drop}")

    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df  # 더 이상 fillna 처리 없슴

def analyze_features(df: pd.DataFrame, prefix: str = "f_") -> pd.DataFrame:
    """
    피처별 통계 진단:
    - 평균, 표준, 최소/최댓값, NaN/Inf 개수, 1%, 99% quantile
    -> 문제 컬럼 식별 용도
    """
    result = []
    for col in df.columns:
        if not col.startswith(prefix):
            continue
        s = df[col]
        result.append({
            "feature": col,
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "is_inf": np.isinf(s).sum(),
            "is_nan": s.isna().sum(),
            "q01": s.quantile(0.01),
            "q99": s.quantile(0.99),
        })
    return pd.DataFrame(result).sort_values("std", ascending=False)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    기술지표 생성:
    - NaN이 생기더라도 유지
    - bfill(), ffill() 제거 → 초기 결측 왜곡 방지
    """
    df = df.copy()
    close, high, low, open_, volume = df["Close"], df["High"], df["Low"], df["Open"], df.get("Volume", None)

    df["bb_mid"] = close.rolling(20).mean()
    df["bb_std"] = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    for period in [5, 10, 20, 60, 120]:
        df[f"ema_{period}"] = talib.EMA(close, timeperiod=period)

    df["rsi_14"] = talib.RSI(close, timeperiod=14)
    macd, macd_signal, macd_hist = talib.MACD(close, 12, 26, 9)
    df["macd"], df["macd_signal"], df["macd_hist"] = macd, macd_signal, macd_hist

    k, d = talib.STOCH(high, low, close, 14, 3, 0, 3, 0)
    df["stoch_k"], df["stoch_d"] = k, d
    df["cci_20"] = talib.CCI(high, low, close, 20)

    ha_close = (open_ + high + low + close) / 4
    ha_open = (open_.shift(1) + ha_close.shift(1)) / 2
    df["ha_close"], df["ha_open"] = ha_close, ha_open
    df["ha_high"] = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
    df["ha_low"] = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)

    period1 = (high.rolling(9).max() + low.rolling(9).min()) / 2
    period2 = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df["tenkan_sen"], df["kijun_sen"] = period1, period2
    df["senkou_a"] = ((period1 + period2) / 2).shift(26)
    df["senkou_b"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df["chikou_span"] = close.shift(-26)

    df["adx_14"] = talib.ADX(high, low, close, timeperiod=14)
    df["aroondown"], df["aroonup"] = talib.AROON(high, low, timeperiod=14)
    df["aroon_osc"] = talib.AROONOSC(high, low, timeperiod=14)

    df["apo"] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
    df["ppo"] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
    df["ultosc"] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    df["willr_14"] = talib.WILLR(high, low, close, timeperiod=14)
    df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
    df["natr_14"] = talib.NATR(high, low, close, timeperiod=14)

    if volume is not None:
        df["obv"] = talib.OBV(close, volume)
        df["ad"] = talib.AD(high, low, close, volume)
        df["adosc"] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)

    df["psar"] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)

    return df  # no ffill/bfill

# === HPO용 피처 생성 및 피처 진단 통합 ===

def add_hpo_candidates(df: pd.DataFrame, interval: str,
                       top_k: int = 300, verbose: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    y = df["y_class"]
    ref_cols = df[REF_COLS_CANON].copy()

    df = add_technical_indicators(df)

    specs = get_feature_specs_for_tf(interval)
    top_feats = []

    batch_size = 1000
    for i in range(0, len(specs), batch_size):
        batch = {}
        for spec in specs[i:i + batch_size]:
            name, series = generate_feature(df, spec)
            batch[name] = series

        X = pd.DataFrame(batch)  # 더 이상 fillna(0) 하지 않음
        try:
            mi_scores = mutual_info_classif(X.fillna(0), y, discrete_features=False)
        except ValueError:
            continue

        for score, name in zip(mi_scores, X.columns):
            if len(top_feats) < top_k:
                heapq.heappush(top_feats, (score, name, X[name]))
            elif score > top_feats[0][0]:
                heapq.heappushpop(top_feats, (score, name, X[name]))

    selected_feats = sorted(top_feats, reverse=True)
    feat_names = [f[1] for f in selected_feats]
    feat_data = {f[1]: f[2] for f in selected_feats}

    df_selected = pd.concat([df, pd.DataFrame(feat_data, index=df.index)], axis=1)
    df_selected = sanitize(df_selected, verbose=verbose)

    for col in REF_COLS_CANON:
        if col not in df_selected.columns:
            df_selected[col] = ref_cols.get(col, 0.0)

    return df_selected, feat_names

# === 통합 전처리 파이프라인 함수 ===

def generate_clean_features(df: pd.DataFrame, interval: str,
                            top_k: int = 300, verbose: bool = True) -> pd.DataFrame:
    original_len = len(df)
    df = df.copy()
    ref_cols = df[REF_COLS_CANON].copy()

    # 피처 생성 전 입력 데이터 NaN 체크
    input_nan_count = df.isnull().sum().sum()
    if input_nan_count > 0 and verbose:
        print(f"Input data contains {input_nan_count} NaN values before feature generation")

    df, feat_names = add_hpo_candidates(df, interval, top_k=top_k, verbose=verbose)

    if verbose:
        print("Feature diagnostics (top issue features):")
        print(analyze_features(df).head(30))

    # 피처 생성 후 NaN 체크
    feature_cols = [c for c in df.columns if c.startswith('f_')]
    nan_summary = df[feature_cols].isnull().sum()
    nan_cols = nan_summary[nan_summary > 0]
    
    if len(nan_cols) > 0:
        if verbose:
            print(f"Features with NaN after generation: {dict(nan_cols.head(10))}")
        
        # NaN이 있는 행들을 제거하되, 손실률 체크
        df_before_drop = df.copy()
        df = df.dropna(subset=feature_cols)
        dropped_rows = len(df_before_drop) - len(df)
        loss_ratio = dropped_rows / original_len
        
        if verbose:
            print(f"Dropped {dropped_rows} rows due to NaN features (loss ratio: {loss_ratio:.1%})")
        
        # 너무 많은 데이터 손실시 경고
        if loss_ratio > 0.1:  # 10% 이상 손실
            print(f"[WARNING] High data loss in {interval}: {loss_ratio:.1%} of original data dropped")
    
    # 추가 데이터 정제
    df = sanitize(df, verbose=verbose)

    # 참조 컬럼 복원 (NaN 제거된 인덱스에 맞춰서)
    for col in REF_COLS_CANON:
        if col not in df.columns:
            if col in ref_cols.columns:
                # 동일한 인덱스의 참조 데이터만 사용
                matching_ref = ref_cols.loc[df.index, col] if col in ref_cols.columns else 0.0
                df[col] = matching_ref.fillna(0.0)  # 참조 컬럼에 NaN이 있다면 0으로 채움
            else:
                df[col] = 0.0

    # 최종 NaN 체크
    final_nan_count = df.isnull().sum().sum()
    if final_nan_count > 0:
        if verbose:
            print(f"[WARNING] Final data still contains {final_nan_count} NaN values")
            nan_cols_final = df.columns[df.isnull().any()].tolist()
            print(f"Columns with NaN: {nan_cols_final}")
        
        # 최후의 안전장치: 모든 NaN을 0으로 대체
        df = df.fillna(0.0)
        if verbose:
            print("Applied zero-fill to all remaining NaN values")

    if verbose:
        final_len = len(df)
        total_loss_ratio = (original_len - final_len) / original_len
        print(f"Final data: {final_len:,} rows (total loss: {total_loss_ratio:.1%})")

    return df

# === 결과 로딩 & 유니버스 관련 함수 ===

def load_processed(split: str, tf: str, mode: str = "auto") -> pd.DataFrame:
    base_p = os.path.join(OUT_DIR, f"fe_{split}_{tf}.parquet")
    hpo_p = os.path.join(OUT_DIR, f"feHPO_{split}_{tf}.parquet")
    path = hpo_p if mode == "hpo" or (mode == "auto" and os.path.exists(hpo_p)) else base_p
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found: {path}")
    df = pd.read_parquet(path)
    float_cols = df.select_dtypes(include=["float64"]).columns
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
