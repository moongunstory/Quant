# train/prepare/process/feature_cleaning.py

import pandas as pd


def clean_and_align_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    단일 주기 데이터에 대해:
    - 기술지표 초반 결측치: 모든 컬럼의 가장 늦은 첫 유효값 기준으로 시작
    - 기술지표 끝 결측치: 모든 컬럼의 가장 이른 마지막 유효값 기준으로 종료
    - 중간 결측치: 시간 기준 보간
    """
    df = df.set_index("timestamp").sort_index()

    # 모든 컬럼에서 유효값 시작/종료 시점 수집
    start_times = []
    end_times = []

    for col in df.columns:
        if df[col].notna().any():
            start_times.append(df[col].first_valid_index())
            end_times.append(df[col].last_valid_index())

    # 유효한 공통 구간 추출
    start = max(start_times)
    end = min(end_times)

    # 해당 구간 자르기
    df = df.loc[start:end]

    # 중간 결측치 시간 기준 보간
    df = df.interpolate(method="time", limit_area="inside")

    return df.reset_index()


def align_feature_ends(feature_dfs: dict) -> dict:
    """
    여러 데이터프레임을 가장 짧은 종료 시점 기준으로 정렬
    """
    last_timestamps = {
        name: df["timestamp"].max() for name, df in feature_dfs.items()
    }
    min_last_timestamp = min(last_timestamps.values())

    trimmed_dfs = {
        name: df[df["timestamp"] <= min_last_timestamp].copy()
        for name, df in feature_dfs.items()
    }

    return trimmed_dfs
