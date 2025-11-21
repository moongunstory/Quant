from __future__ import annotations

import os
from datetime import datetime
from typing import List

import pandas as pd

from .config import DailyConfig
from .dataset import ensure_datetime_index, build_supervised_for_horizon
from .trainer import train_horizon_model
from .predictor import predict_for_horizon
from .log import append_predictions_log, update_realized_outcomes
from .report import print_and_save_report
from .policy import attach_expected_return

def run_daily_cycle(
    cfg: DailyConfig,
    df_master: pd.DataFrame,
    as_of_ts: datetime | None = None,
) -> pd.DataFrame:
    """
    1) window_days 윈도우로 horizon별 LGBM 학습
    2) as_of_ts 시점에서 horizon별 예측 1건씩
    3) daily_predictions.parquet에 append
    4) 과거 예측들 중 실적 채울 수 있는 것들 업데이트
    5) 한국어 텍스트 리포트 출력 + (선택) 저장
    """
    df_master = ensure_datetime_index(df_master, cfg)

    if as_of_ts is None:
        as_of_ts = df_master.index.max()

    as_of_ts = pd.to_datetime(as_of_ts, utc=True)
    as_of_date = as_of_ts.normalize()

    # 오늘 이미 실행된 적 있으면 학습 스킵하고 실적만 업데이트
    existing_log = None
    if os.path.exists(cfg.pred_log_path):
        existing_log = pd.read_parquet(cfg.pred_log_path)
        if cfg.skip_if_exists and (existing_log["as_of_date"] == as_of_date).any():
            print(
                f"오늘({as_of_date.date()}) 데일리 사이클은 이미 실행됨. "
                "학습은 스킵하고, 실적(realized)만 업데이트합니다."
            )
            updated_log = update_realized_outcomes(existing_log, df_master, cfg)
            updated_log.to_parquet(cfg.pred_log_path)
            print_and_save_report(updated_log, cfg, as_of_date)

            # ✅ 오늘(as_of_date) 예측만 잘라서
            today_mask = updated_log["as_of_date"] == as_of_date
            today_pred = updated_log.loc[today_mask].copy()

            # ✅ 여기서 예상 수익률 붙여서 리턴
            today_pred = attach_expected_return(
                today_pred=today_pred,
                df_log_full=updated_log,   # 전체 로그로 성적표 만듦
                cfg=cfg,
                bin_width=0.1,
                min_samples=20,
            )

            return today_pred

    all_rows: List[dict] = []

    for d in cfg.horizons_days:
        H = d * 24
        try:
            X, y, feature_names = build_supervised_for_horizon(
                df_master, as_of_ts, H, cfg
            )
        except ValueError as e:
            print(f"[{as_of_date.date()}] horizon {d}일({H}h) 스킵:", e)
            continue

        # 학습 + 저장
        model, model_id = train_horizon_model(
            X=X,
            y=y,
            feature_names=feature_names,
            cfg=cfg,
            horizon_days=d,
            horizon_hours=H,
            as_of_date=as_of_date,
        )

        # 마지막 시점 1개로 예측
        pred_label, proba = predict_for_horizon(
            model=model,
            df_master=df_master,
            as_of_ts=as_of_ts,
            feature_names=feature_names,
        )

        new_row = {
            "as_of_ts": as_of_ts,
            "as_of_date": as_of_date,
            "model_id": model_id,
            "horizon_days": d,
            "horizon_hours": H,
            "pred_label": pred_label,
            "proba_down": float(proba[0]),
            "proba_flat": float(proba[1]),
            "proba_up": float(proba[2]),
            "created_at": pd.Timestamp.utcnow(),
        }
        all_rows.append(new_row)

    if not all_rows:
        print(f"[{as_of_date.date()}] 학습된 horizon이 없습니다.")
        if existing_log is not None:
            updated_log = update_realized_outcomes(existing_log, df_master, cfg)
            updated_log.to_parquet(cfg.pred_log_path)
            print_and_save_report(updated_log, cfg, as_of_date)
        return pd.DataFrame()

    pred_df = pd.DataFrame(all_rows)

    # 오늘 예측 append
    combined = append_predictions_log(pred_df, cfg)
    # 과거 예측들 중 실적 채울 수 있는 것들 업데이트
    combined = update_realized_outcomes(combined, df_master, cfg)
    combined.to_parquet(cfg.pred_log_path)

    # ---- 여기서 오늘 예측에 예상 수익률 붙이기 ----
    # as_of_date 기준 오늘 레코드만 따로 뽑고
    today_mask = combined["as_of_date"] == as_of_date
    today_pred = combined.loc[today_mask].copy()

    # 성적표 기반 예상 수익률 컬럼(exp_return) 붙이기
    today_pred = attach_expected_return(
        today_pred=today_pred,
        df_log_full=combined,
        cfg=cfg,
        bin_width=0.1,      # 0.1 (=10%) 확률 구간
        min_samples=20,     # 성적표 최소 샘플 수
    )

    # 한국어 리포트 출력 + 저장 (이건 전체 로그 기준)
    print_and_save_report(combined, cfg, as_of_date)

    # run.py에서 오늘 예측만 요약할 때 쓰라고 오늘 것만 리턴
    return today_pred

