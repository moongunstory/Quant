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
    전체 일일 사이클:
    1) horizon별 학습
    2) as_of_ts 시점에서 horizon별 예측 1건 생성
    3) pred_log_path에 append
    4) 실현 수익률(realized_return, realized_label) 업데이트
    5) 전체 로그 기준 리포트 출력 + 오늘 예측에 예상 수익률 붙여서 리턴
    """
    # 인덱스 정리
    df_master = ensure_datetime_index(df_master, cfg)

    # 기준 시점 결정
    if as_of_ts is None:
        as_of_ts = df_master.index.max()

    as_of_ts = pd.to_datetime(as_of_ts, utc=True)
    as_of_date = as_of_ts.normalize()

    # 이미 오늘 기록이 있으면 학습 스킵 + 실적만 업데이트
    existing_log: pd.DataFrame | None = None
    if os.path.exists(cfg.pred_log_path):
        existing_log = pd.read_parquet(cfg.pred_log_path)

        if cfg.skip_if_exists and "as_of_date" in existing_log.columns:
            if (existing_log["as_of_date"] == as_of_date).any():
                print(
                    f"[INFO] {as_of_date.date()} 데일리 사이클 이미 실행됨 → "
                    "학습 스킵, 실현 수익률만 업데이트"
                )
                updated_log = update_realized_outcomes(existing_log, df_master, cfg)
                updated_log.to_parquet(cfg.pred_log_path)

                # 전체 로그 기준 리포트
                print_and_save_report(updated_log, cfg, as_of_date)

                # 오늘 예측만 분리
                today = updated_log.loc[updated_log["as_of_date"] == as_of_date].copy()
                if today.empty:
                    return today

                # 오늘 예측에 예상 수익률 붙여서 리턴
                today = attach_expected_return(
                    today_pred=today,
                    df_log_full=updated_log,
                    cfg=cfg,
                    bin_width=0.1,
                    min_samples=20,
                )
                return today

    # ─────────────────────────────────────────
    # horizon별 학습 + 예측
    # ─────────────────────────────────────────
    all_rows: List[dict] = []

    for d in cfg.horizons_days:
        H = d * 24  # 일 → 시간

        try:
            X, y, feature_names = build_supervised_for_horizon(
                df=df_master,
                as_of_ts=as_of_ts,
                horizon_hours=H,
                cfg=cfg,
            )
        except ValueError as e:
            print(f"[WARN] {as_of_date.date()} horizon {d}d({H}h) 스킵: {e}")
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

        all_rows.append(
            {
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
        )

    # 학습된 horizon이 하나도 없으면
    if not all_rows:
        print(f"[WARN] {as_of_date.date()} 학습된 horizon 없음.")
        if existing_log is not None:
            updated_log = update_realized_outcomes(existing_log, df_master, cfg)
            updated_log.to_parquet(cfg.pred_log_path)
            print_and_save_report(updated_log, cfg, as_of_date)
        return pd.DataFrame()

    # 오늘 새 예측들
    pred_df = pd.DataFrame(all_rows)

    # 로그에 append → 전체 로그
    combined = append_predictions_log(pred_df, cfg)
    combined = update_realized_outcomes(combined, df_master, cfg)
    combined.to_parquet(cfg.pred_log_path)

    # 오늘 예측만 분리
    today_mask = combined["as_of_date"] == as_of_date
    today_pred = combined.loc[today_mask].copy()

    # 예상 수익률(exp_return 등) 붙이기
    today_pred = attach_expected_return(
        today_pred=today_pred,
        df_log_full=combined,
        cfg=cfg,
        bin_width=0.1,
        min_samples=20,
    )

    # 전체 로그 기준 리포트 출력 + 저장
    print_and_save_report(combined, cfg, as_of_date)

    # run.py 쪽에서 오늘 예측만 쓰기 좋게, 오늘 것만 리턴
    return today_pred
