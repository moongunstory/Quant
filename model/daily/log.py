from __future__ import annotations

import os
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from .config import DailyConfig
from .dataset import ensure_datetime_index


def append_predictions_log(pred_df: pd.DataFrame, cfg: DailyConfig) -> pd.DataFrame:
    """금일 예측을 pred_log_path에 append."""
    os.makedirs(os.path.dirname(cfg.pred_log_path), exist_ok=True)

    if os.path.exists(cfg.pred_log_path):
        old = pd.read_parquet(cfg.pred_log_path)
        combined = pd.concat([old, pred_df], ignore_index=True)
    else:
        combined = pred_df.copy()

    combined.to_parquet(cfg.pred_log_path)
    return combined


def update_realized_outcomes(
    df_log: pd.DataFrame,
    df_master: pd.DataFrame,
    cfg: DailyConfig,
) -> pd.DataFrame:
    """
    예측 로그 중에서 아직 realized_*가 없는 것들 중,
    지금 시점 기준으로 horizon 만큼 지난 것 → 실제 수익률 계산해서 채움.
    """
    df_master = ensure_datetime_index(df_master, cfg)

    if cfg.close_col not in df_master.columns:
        raise KeyError(f"종가 컬럼 '{cfg.close_col}' 을(를) 찾을 수 없습니다.")

    close = df_master[cfg.close_col].astype(float)

    # 컬럼 보장
    for col in ["realized_return", "realized_label", "realized_at"]:
        if col not in df_log.columns:
            df_log[col] = np.nan

    for idx, row in df_log.iterrows():
        if not pd.isna(row["realized_return"]):
            continue  # 이미 채워진 건 패스

        start_ts = pd.to_datetime(row["as_of_ts"], utc=True)
        H = int(row["horizon_hours"])
        end_ts = start_ts + timedelta(hours=H)

        # 아직 미래 데이터가 충분치 않으면 스킵
        if end_ts > df_master.index.max():
            continue

        if start_ts not in close.index or end_ts not in close.index:
            continue

        r = (close.loc[end_ts] - close.loc[start_ts]) / close.loc[start_ts]
        df_log.at[idx, "realized_return"] = float(r)

        if r >= cfg.threshold:
            lab = 1
        elif r <= -cfg.threshold:
            lab = -1
        else:
            lab = 0

        df_log.at[idx, "realized_label"] = lab
        df_log.at[idx, "realized_at"] = pd.Timestamp.utcnow()

    return df_log
