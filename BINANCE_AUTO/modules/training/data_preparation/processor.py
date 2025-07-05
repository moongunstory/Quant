import pandas as pd
import numpy as np
import os
import sys
from typing import Dict
import pickle
import logging

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    RAW_DATA_PATH, TRAIN_PICKLE_PATHS,
    TP_THRESHOLD, SL_THRESHOLD, LABEL_HORIZON,
    TIMEFRAMES
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def log_ohlcv_nans(df: pd.DataFrame, tf: str, stage: str = "") -> None:
    """Log NaN counts for OHLCV columns."""
    prefix = f"[{tf}]"
    if stage:
        prefix += f" {stage}"
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            logger.info(f"{prefix} {col} NaNs: {df[col].isna().sum()}")

def create_dune_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """DUNE 파생 피처 생성 - DUNE DataFrame만 처리"""
    df_processed = df.copy()
    
    # 수치형 파생 피처
    if 'eth_to_cex' in df.columns and 'eth_from_cex' in df.columns:
        df_processed['cex_netflow'] = df['eth_to_cex'] - df['eth_from_cex']
    
    if 'whale_to_cex' in df.columns and 'cex_to_whale' in df.columns:
        df_processed['whale_netflow'] = df['whale_to_cex'] - df['cex_to_whale']
    
    if 'deposit_amount' in df.columns and 'withdraw_amount' in df.columns:
        df_processed['staking_netflow'] = df['deposit_amount'] - df['withdraw_amount']
    
    # 이벤트 flag 피처 (00:00에만 값 설정, 나머지는 NaN)
    midnight_mask = (df_processed.index.hour == 0) & (df_processed.index.minute == 0)
    
    if 'cex_netflow' in df_processed.columns:
        df_processed['cex_increase_flag'] = np.nan
        df_processed.loc[midnight_mask, 'cex_increase_flag'] = (
            df_processed.loc[midnight_mask, 'cex_netflow'] > 0
        ).astype(int)
    
    if 'whale_netflow' in df_processed.columns:
        df_processed['whale_increase_flag'] = np.nan
        df_processed.loc[midnight_mask, 'whale_increase_flag'] = (
            df_processed.loc[midnight_mask, 'whale_netflow'] > 0
        ).astype(int)
    
    if 'staking_netflow' in df_processed.columns:
        df_processed['staking_increase_flag'] = np.nan
        df_processed.loc[midnight_mask, 'staking_increase_flag'] = (
            df_processed.loc[midnight_mask, 'staking_netflow'] > 0
        ).astype(int)
    
    return df_processed

def apply_feature_processing(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """피처별 처리 정책 적용 - BTC/DUNE DataFrame만 처리"""
    df_processed = df.copy()
    log_ohlcv_nans(df_processed, data_type, stage="before_process")
    
    if data_type == "btc":
        # BTC 피처 .ffill() 적용
        btc_features = [col for col in df.columns if col.startswith('btc_')]
        for col in btc_features:
            df_processed[col] = df_processed[col].ffill()
    
    elif data_type == "dune":
        # DUNE 수치형 피처 .ffill() 적용
        dune_numeric_features = ['cex_netflow', 'whale_netflow', 'staking_netflow']
        for col in dune_numeric_features:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].ffill()
        # DUNE flag 피처는 .ffill() 금지 (그대로 유지)
    
    log_ohlcv_nans(df_processed, data_type, stage="after_process")
    return df_processed

def create_labels(df_dict: Dict[str, pd.DataFrame], label_horizon: int = 4, direction: str = "long") -> pd.DataFrame:
    """
    15분봉 기준 라벨 생성 (5분봉 기준 TP/SL 도달 판별) -> 라벨만 반환
    direction: 'long' 또는 'short'에 따라 라벨링 로직 변경
    """
    if "15min" not in df_dict or "5min" not in df_dict:
        raise ValueError("Label creation requires both '15min' and '5min' data.")

    df_15m = df_dict["15min"].copy()
    df_5m = df_dict["5min"].copy()

    # Find close, high, low columns robustly
    close_col = next((col for col in df_15m.columns if "close" in col.lower()), None)
    high_col = next((col for col in df_5m.columns if "high" in col.lower()), None)
    low_col = next((col for col in df_5m.columns if "low" in col.lower()), None)

    if not all([close_col, high_col, low_col]):
        raise ValueError("Could not find required 'close', 'high', 'low' columns.")

    labels = pd.Series('hold', index=df_15m.index, name='label')

    for i in range(len(df_15m) - label_horizon):
        entry_time = df_15m.index[i]
        entry_price = df_15m.iloc[i][close_col]

        # 방향에 따른 TP/SL 가격 계산
        if direction == "long":
            tp_price = entry_price * (1 + TP_THRESHOLD)
            sl_price = entry_price * (1 + SL_THRESHOLD)
        elif direction == "short":
            tp_price = entry_price * (1 + SL_THRESHOLD)
            sl_price = entry_price * (1 + TP_THRESHOLD)
        else:
            raise ValueError("Direction must be 'long' or 'short'.")

        end_time = df_15m.index[i + label_horizon]
        future_5m = df_5m[(df_5m.index > entry_time) & (df_5m.index <= end_time)]

        if future_5m.empty:
            continue

        # 방향에 따른 TP/SL 도달 조건
        if direction == "long":
            tp_reached_times = future_5m.index[future_5m[high_col] >= tp_price]
            sl_reached_times = future_5m.index[future_5m[low_col] <= sl_price]
        else:
            tp_reached_times = future_5m.index[future_5m[low_col] <= tp_price]
            sl_reached_times = future_5m.index[future_5m[high_col] >= sl_price]

        tp_first_time = tp_reached_times.min() if not tp_reached_times.empty else pd.NaT
        sl_first_time = sl_reached_times.min() if not sl_reached_times.empty else pd.NaT

        # 라벨 할당
        if pd.notna(tp_first_time) and pd.notna(sl_first_time):
            if tp_first_time <= sl_first_time:
                labels.iloc[i] = direction
            else:
                labels.iloc[i] = 'hold'
        elif pd.notna(tp_first_time):
            labels.iloc[i] = direction
        elif pd.notna(sl_first_time):
            labels.iloc[i] = 'hold'

    return labels.to_frame()

def create_balanced_ppo_dataset(df_labeled: pd.DataFrame, success_label: str) -> pd.DataFrame:
    """Create a balanced PPO dataset with success, fail and hold scenarios."""
    fail_label = "short" if success_label == "long" else "long"

    success_df = df_labeled[df_labeled["label"] == success_label]
    fail_df = df_labeled[df_labeled["label"] == fail_label]
    hold_df = df_labeled[df_labeled["label"] == "hold"]

    total_n = len(df_labeled)
    target_success = int(total_n * 0.3)
    target_fail = int(total_n * 0.3)
    target_hold = int(total_n * 0.4)

    success_sample = success_df.sample(min(len(success_df), target_success), random_state=42)
    fail_sample = fail_df.sample(min(len(fail_df), target_fail), random_state=42)
    hold_sample = hold_df.sample(min(len(hold_df), target_hold), random_state=42)

    balanced_df = pd.concat([success_sample, fail_sample, hold_sample]).sort_index()
    return balanced_df

def load_mtf_data() -> Dict[str, pd.DataFrame]:
    """MTF 개별 pickle 파일들을 로드"""
    save_dir = os.path.dirname(os.path.join(PROJECT_ROOT, RAW_DATA_PATH))
    mtf_data = {}
    
    # 각 타임프레임 + BTC/DUNE 데이터 로드
    data_keys = TIMEFRAMES + ["btc", "dune"]
    
    for key in data_keys:
        file_path = os.path.join(save_dir, f"market_data_{key}.pkl")
        if os.path.exists(file_path):
            df = pd.read_pickle(file_path)
            mtf_data[key] = df
            logger.info(f"[로드] {key}: {len(df)} rows, {len(df.columns)} columns")
            log_ohlcv_nans(df, key, stage="loaded")
        else:
            logger.warning(f"[경고] {key} 파일 없음: {file_path}")
    
    return mtf_data

def main():
    """메인 처리 함수"""
    print("[MTF 데이터 전처리 및 라벨링 시작]")

    # 1. MTF 데이터 로딩 (유출 없는 원본 피처)
    mtf_data = load_mtf_data()
    if not mtf_data:
        raise ValueError("MTF 데이터를 로드할 수 없습니다.")

    # (DUNE 및 BTC 관련 처리는 그대로 유지)
    if "dune" in mtf_data and not mtf_data["dune"].empty:
        mtf_data["dune"] = create_dune_derived_features(mtf_data["dune"])
        print("[DUNE 파생 피처 생성 완료]")

    if "btc" in mtf_data and not mtf_data["btc"].empty:
        mtf_data["btc"] = apply_feature_processing(mtf_data["btc"], "btc")
        print("[BTC 피처 처리 완료]")

    if "dune" in mtf_data and not mtf_data["dune"].empty:
        mtf_data["dune"] = apply_feature_processing(mtf_data["dune"], "dune")
        print("[DUNE 피처 처리 완료]")

    # 2. 라벨 생성 (방향별 라벨링)
    print("[방향별 라벨링 시작]")
    df_labels_long = create_labels(mtf_data, LABEL_HORIZON, direction="long")
    df_labels_short = create_labels(mtf_data, LABEL_HORIZON, direction="short")
    print("[방향별 라벨링 완료]")

    # 3. 깨끗한 15분봉 데이터에 라벨 병합 (데이터 유출 방지)
    df_15m_clean = mtf_data["15min"]

    df_labeled_long = df_15m_clean.join(df_labels_long, how='inner')
    df_labeled_short = df_15m_clean.join(df_labels_short, how='inner')
    print("[라벨-피처 안전한 병합 완료]")

    # 라벨 분포 확인
    print(f"[Long 라벨 분포] {dict(df_labeled_long['label'].value_counts())}")
    print(f"[Short 라벨 분포] {dict(df_labeled_short['label'].value_counts())}")

    # 4. PPO 학습을 위한 균형 데이터셋 구성
    df_long_binary = create_balanced_ppo_dataset(df_labeled_long, "long")
    df_long_binary['label'] = (df_long_binary['label'] == 'long').astype(int)

    df_short_binary = create_balanced_ppo_dataset(df_labeled_short, "short")
    df_short_binary['label'] = (df_short_binary['label'] == 'short').astype(int)

    print(f"[이진분류 데이터 준비]")
    print(f"  - Long 모델용: {len(df_long_binary)}행 (long={sum(df_long_binary['label'])}, hold={len(df_long_binary)-sum(df_long_binary['label'])})")
    print(f"  - Short 모델용: {len(df_short_binary)}행 (short={sum(df_short_binary['label'])}, hold={len(df_short_binary)-sum(df_short_binary['label'])})")

    # 5. 최종 데이터 저장
    long_path = TRAIN_PICKLE_PATHS["long"]
    short_path = TRAIN_PICKLE_PATHS["short"]

    os.makedirs(os.path.dirname(long_path), exist_ok=True)
    os.makedirs(os.path.dirname(short_path), exist_ok=True)

    long_data_to_save = mtf_data.copy()
    long_data_to_save["15min"] = df_long_binary
    with open(long_path, "wb") as f:
        pickle.dump(long_data_to_save, f)

    short_data_to_save = mtf_data.copy()
    short_data_to_save["15min"] = df_short_binary
    with open(short_path, "wb") as f:
        pickle.dump(short_data_to_save, f)

    print(f"[저장 완료]")
    print(f"  - Long 이진분류 데이터: {len(df_long_binary)}행 → {long_path}")
    print(f"  - Short 이진분류 데이터: {len(df_short_binary)}행 → {short_path}")

    return df_labeled_long, df_labeled_short

if __name__ == "__main__":
    labeled_df, processed_mtf_data = main()