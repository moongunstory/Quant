# model/daily/policy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .config import DailyConfig


def _compute_key_prob(df: pd.DataFrame) -> pd.Series:
    """
    각 예측 레코드에서
    - 상승 예측이면 상승 확률
    - 하락 예측이면 하락 확률
    - 중립 예측이면 중립 확률
    만 따로 뽑아서 'key_prob'로 씀.
    """
    def _row_prob(row):
        lbl = row.get("pred_label")
        if lbl == 1:
            return row.get("proba_up", np.nan)
        elif lbl == -1:
            return row.get("proba_down", np.nan)
        else:
            return row.get("proba_flat", np.nan)

    return df.apply(_row_prob, axis=1)


def _add_prob_bins(
    df: pd.DataFrame,
    bin_width: float = 0.1,
) -> pd.DataFrame:
    """
    key_prob(0~1)를 0.1 단위 구간으로 나눠서
    - prob_bin_low
    - prob_bin_high
    라는 칸을 만들어줌.
    """
    df = df.copy()
    df["key_prob"] = _compute_key_prob(df)

    # 0~1 사이로 잘라주기
    df["key_prob"] = df["key_prob"].clip(lower=0.0, upper=1.0)

    # 몇 번째 구간인지 (0~9)
    bin_idx = np.floor(df["key_prob"] / bin_width).astype("Int64")
    max_idx = int(1.0 / bin_width) - 1
    bin_idx = bin_idx.clip(lower=0, upper=max_idx)

    df["prob_bin_low"] = (bin_idx * bin_width).astype(float)
    df["prob_bin_high"] = df["prob_bin_low"] + bin_width

    return df


def build_expectation_table(
    df_log: pd.DataFrame,
    cfg: DailyConfig,
    bin_width: float = 0.1,
    min_samples: int = 20,
) -> pd.DataFrame:
    """
    과거 예측 로그 + 실적을 가지고
    (horizon, pred_label, 확률구간)별 평균 수익률을 계산.

    → 이게 '성적표 테이블'
    """
    if df_log.empty:
        return pd.DataFrame()

    # 실적이 채워진 행만 사용
    df_hist = df_log.dropna(subset=["realized_return"]).copy()
    if df_hist.empty:
        return pd.DataFrame()

    df_hist = _add_prob_bins(df_hist, bin_width=bin_width)

    grouped = (
        df_hist
        .groupby(
            ["horizon_days", "pred_label", "prob_bin_low", "prob_bin_high"],
            dropna=False,
        )["realized_return"]
        .agg(["mean", "count"])
        .reset_index()
    )

    # 샘플 수가 너무 적으면 버림
    grouped = grouped[grouped["count"] >= min_samples].copy()
    grouped = grouped.rename(
        columns={
            "mean": "exp_return",
            "count": "exp_return_samples",
        }
    )
    return grouped


def attach_expected_return(
    today_pred: pd.DataFrame,
    df_log_full: pd.DataFrame,
    cfg: DailyConfig,
    bin_width: float = 0.1,
    min_samples: int = 20,
) -> pd.DataFrame:
    """
    오늘 예측(today_pred)에
    - exp_return (예상 수익률, 비율)
    - exp_return_samples (그 근거가 된 과거 샘플 수)
    를 붙여서 리턴.
    """
    if today_pred.empty:
        return today_pred

    # 과거 전체 로그로 성적표 만들기
    table = build_expectation_table(
        df_log=df_log_full,
        cfg=cfg,
        bin_width=bin_width,
        min_samples=min_samples,
    )
    if table.empty:
        # 아직 실적 데이터가 거의 없으면 그대로 리턴 (NaN 유지)
        today_pred = today_pred.copy()
        today_pred["exp_return"] = np.nan
        today_pred["exp_return_samples"] = 0
        return today_pred

    # 오늘 예측에도 확률 구간 붙이기
    today = _add_prob_bins(today_pred.copy(), bin_width=bin_width)

    # (horizon_days, pred_label, prob_bin_low, prob_bin_high) 기준으로 성적표 머지
    merged = pd.merge(
        today,
        table,
        how="left",
        on=["horizon_days", "pred_label", "prob_bin_low", "prob_bin_high"],
    )

    # 성적표가 없으면 NaN으로 남음
    if "exp_return" not in merged.columns:
        merged["exp_return"] = np.nan
    if "exp_return_samples" not in merged.columns:
        merged["exp_return_samples"] = 0

    return merged
