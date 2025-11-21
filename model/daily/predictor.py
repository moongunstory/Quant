from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def make_last_feature_row(
    df_master: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    as_of_ts 이전 구간 중 마지막 1개 시점의 피처 1행 만들기.
    """
    last_row = df_master.loc[:as_of_ts].iloc[-1]
    # 문자열일 수 있는 값들도 숫자로 바꾸고, 1행짜리 DF로
    last_feat = pd.to_numeric(last_row[feature_names], errors="coerce")
    return last_feat.to_frame().T


def predict_for_horizon(
    model,
    df_master: pd.DataFrame,
    as_of_ts: pd.Timestamp,
    feature_names: List[str],
) -> tuple[int, np.ndarray]:
    """
    한 horizon 모델에 대해 마지막 시점 1개 예측.
    """
    X_pred = make_last_feature_row(df_master, as_of_ts, feature_names)
    pred_label = int(model.predict(X_pred)[0])
    proba = model.predict_proba(X_pred)[0]
    return pred_label, proba
