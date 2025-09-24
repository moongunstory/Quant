# train/prepare/process/data_splitter.py

import pandas as pd

def split_data(df: pd.DataFrame, train_pct: float = 0.7, val_pct: float = 0.15):
    """
    시간순으로 정렬된 데이터프레임을 Train, Validation, Test 세트로 분할합니다.

    Args:
        df (pd.DataFrame): 'timestamp' 컬럼을 포함한 전체 데이터프레임.
        train_pct (float): 훈련 세트의 비율.
        val_pct (float): 검증 세트의 비율.

    Returns:
        tuple: (df_train, df_val, df_test) 세 개의 데이터프레임.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")

    # 타임스탬프 기준으로 정렬
    df = df.sort_values('timestamp').reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    print(f"--- Data Split ---")
    print(f"Total: {n} rows")
    print(f" - Train: {len(df_train)} rows ({train_pct:.0%})")
    print(f" - Validation: {len(df_val)} rows ({val_pct:.0%})")
    print(f" - Test: {len(df_test)} rows (~{1 - train_pct - val_pct:.0%})")

    return df_train, df_val, df_test
