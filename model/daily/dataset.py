from __future__ import annotations

from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import DailyConfig


def ensure_datetime_index(df: pd.DataFrame, cfg: DailyConfig) -> pd.DataFrame:
    """timestamp 컬럼 또는 index를 DatetimeIndex로 정리."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if cfg.timestamp_col not in df.columns:
            raise ValueError(
                f"'{cfg.timestamp_col}' 컬럼도 없고 index도 DatetimeIndex가 아닙니다."
            )
        df[cfg.timestamp_col] = pd.to_datetime(
            df[cfg.timestamp_col], utc=True, errors="coerce"
        )
        df = df.set_index(cfg.timestamp_col)

    return df.sort_index()


def coerce_numeric_features(
    df: pd.DataFrame,
    drop_cols: List[str],
) -> tuple[pd.DataFrame, List[str]]:
    """
    - 숫자/불리언 컬럼은 그대로 사용
    - object 컬럼은 숫자로 캐스팅 시도 (문자 → float)
    - 그래도 숫자가 안 되면 피처에서 제외
    """
    work = df.copy()

    # object → 숫자로 바꿀 수 있으면 최대한 바꿔줌
    for col in work.columns:
        if col in drop_cols:
            continue
        s = work[col]
        if s.dtype == "object":
            # 숫자처럼 생긴 문자열이면 float로 변환, 아니면 NaN
            work[col] = pd.to_numeric(s, errors="coerce")

    # 숫자/불리언 타입만 남기고, 라벨/리턴 컬럼은 제거
    feat_df = (
        work
        .select_dtypes(include=[np.number, "bool"])
        .drop(columns=drop_cols, errors="ignore")
    )

    feature_names = feat_df.columns.tolist()
    return feat_df, feature_names


def build_supervised_for_horizon(
    df: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    horizon_hours: int,
    cfg: DailyConfig,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    특정 horizon(시간 단위)에 대해:
    - 뒤로 window_days 윈도우에서
    - 미래 horizon만큼의 수익률로 라벨 만들고
    - X, y, feature_names 리턴
    """
    window_start_ts = as_of_ts - timedelta(days=cfg.window_days)
    last_label_ts = as_of_ts - timedelta(hours=horizon_hours)

    df_win = df.loc[window_start_ts:last_label_ts].copy()
    if df_win.empty:
        raise ValueError("해당 horizon에 사용할 윈도우 데이터가 없습니다.")

    if cfg.close_col not in df_win.columns:
        raise KeyError(f"종가 컬럼 '{cfg.close_col}' 을(를) 찾을 수 없습니다.")

    close = df_win[cfg.close_col].astype(float)
    future_close = close.shift(-horizon_hours)
    ret = (future_close - close) / close

    ret_col = f"ret_{horizon_hours}h"
    label_col = f"label_{horizon_hours}h"

    df_win[ret_col] = ret
    df_win[label_col] = 0
    df_win.loc[ret >= cfg.threshold, label_col] = 1
    df_win.loc[ret <= -cfg.threshold, label_col] = -1

    # 미래 수익률 없는 구간 제거
    df_train = df_win.dropna(subset=[ret_col])
    if len(df_train) < cfg.min_samples:
        raise ValueError(f"학습 샘플이 너무 적습니다: {len(df_train)}개")

    y = df_train[label_col].to_numpy()

    # 클래스가 2개 미만이면 학습 의미 없음 → 스킵
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"라벨 클래스가 1개뿐입니다 (classes={classes}). "
            f"horizon={horizon_hours}h 구간은 스킵."
        )

    # 숫자 피처만 깔끔하게 정리
    feat_df, feature_names = coerce_numeric_features(
        df_train,
        drop_cols=[ret_col, label_col],
    )
    if not feature_names:
        raise ValueError(
            f"사용 가능한 숫자 피처가 없습니다. horizon={horizon_hours}h"
        )

    # LightGBM에 들어갈 X (순수 float 배열)
    X = feat_df.to_numpy(dtype=float)
    return X, y, feature_names
