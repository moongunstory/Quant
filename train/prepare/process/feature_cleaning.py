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


def merge_and_clean_features(
    df_main: pd.DataFrame,
    df_background_list: list[pd.DataFrame],
    on: str = "timestamp"
) -> pd.DataFrame:
    """
    다양한 주기 데이터 병합 + 결측 정리

    - 기준 df_main 시간축 기준으로 나머지 df들을 asof 병합
    - 병합된 전체 데이터에서 유효 시간 구간 추출
    - 중간 결측치는 시간 기준 보간
    """
    df_main = df_main.sort_values(on).reset_index(drop=True)
    df_main = df_main.set_index(on)

    for df in df_background_list:
        df = df.sort_values(on).reset_index(drop=True).set_index(on)
        df_main = pd.merge_asof(
            df_main.reset_index(),
            df.reset_index(),
            on=on,
            direction="backward"
        ).set_index(on)

    # 유효 구간 추출
    start = max(df_main[col].first_valid_index() for col in df_main.columns if df_main[col].notna().any())
    end = min(df_main[col].last_valid_index() for col in df_main.columns if df_main[col].notna().any())
    df_main = df_main.loc[start:end]

    # 중간 결측치 보간
    df_main = df_main.interpolate(method="time", limit_area="inside")

    return df_main.reset_index()
