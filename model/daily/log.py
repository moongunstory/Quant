from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd

from .config import DailyConfig
from .dataset import ensure_datetime_index


def append_predictions_log(pred_df: pd.DataFrame, cfg: DailyConfig) -> pd.DataFrame:
    """
    오늘 예측(pred_df)을 기존 pred_log_path에 이어붙이고,
    합쳐진 전체 로그를 반환한다.
    """
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
    예측 로그(df_log) 중 아직 realized_*가 비어 있는 것들에 대해,
    horizon_hours만큼 시간이 지난 뒤의 실제 수익률을 계산해 채운다.

    - 수익률: (end_close - start_close) / start_close
    - 라벨: horizon별 threshold(있으면) 또는 기본 threshold 기준으로 -1/0/1
    """
    df_master = ensure_datetime_index(df_master, cfg)

    if cfg.close_col not in df_master.columns:
        raise KeyError(f"종가 컬럼 '{cfg.close_col}' 을(를) 찾을 수 없습니다.")

    close = df_master[cfg.close_col].astype(float)

    # 필요한 컬럼이 없으면 만들어 둔다.
    for col in ["realized_return", "realized_label", "realized_at"]:
        if col not in df_log.columns:
            df_log[col] = np.nan

    master_last_ts = df_master.index.max()

    for idx, row in df_log.iterrows():
        # 이미 채워진 건 패스
        if not pd.isna(row["realized_return"]):
            continue

        start_ts = pd.to_datetime(row["as_of_ts"], utc=True)
        H = int(row["horizon_hours"])
        end_ts = start_ts + timedelta(hours=H)

        # 아직 미래 데이터가 충분치 않으면 스킵
        if end_ts > master_last_ts:
            continue

        # 시가/종가 시점이 둘 다 있어야 계산 가능
        if start_ts not in close.index or end_ts not in close.index:
            continue

        # 실제 수익률 계산
        r = (close.loc[end_ts] - close.loc[start_ts]) / close.loc[start_ts]
        df_log.at[idx, "realized_return"] = float(r)

        # horizon별 threshold 적용 (없으면 기본값)
        horizon_days = int(row.get("horizon_days", 0) or 0)
        if hasattr(cfg, "get_threshold_for") and horizon_days > 0:
            threshold = cfg.get_threshold_for(horizon_days)
        else:
            threshold = cfg.threshold

        if r >= threshold:
            lab = 1
        elif r <= -threshold:
            lab = -1
        else:
            lab = 0

        df_log.at[idx, "realized_label"] = lab
        df_log.at[idx, "realized_at"] = pd.Timestamp.utcnow()

    return df_log
