import pandas as pd
import numpy as np
import os
import sys
from typing import Dict
import pickle
import logging
from sklearn.preprocessing import StandardScaler

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    RAW_DATA_PATH, TRAIN_PICKLE_PATHS, SCALER_PATH,
    TP_THRESHOLD, SL_THRESHOLD, LABEL_HORIZON,
    TIMEFRAMES, AUX_TIMEFRAMES
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

# ✅ 수정: DUNE 피처 엔지니어링 확장
def create_dune_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """DUNE 데이터로부터 추세, 변동성, 비율 등 파생 피처를 생성합니다."""
    df_processed = df.copy()
    
    # 기본 순유입량 피처
    if 'eth_to_cex' in df.columns and 'eth_from_cex' in df.columns:
        df_processed['cex_netflow'] = df['eth_to_cex'] - df['eth_from_cex']
    
    if 'whale_to_cex' in df.columns and 'cex_to_whale' in df.columns:
        df_processed['whale_netflow'] = df['whale_to_cex'] - df['cex_to_whale']
    
    if 'deposit_amount' in df.columns and 'withdraw_amount' in df.columns:
        df_processed['staking_netflow'] = df['deposit_amount'] - df['withdraw_amount']

    netflow_features = ['cex_netflow', 'whale_netflow', 'staking_netflow']
    
    for col in netflow_features:
        if col in df_processed.columns:
            epsilon = 1e-6
            df_processed[f'{col}_ma_3d'] = df_processed[col].rolling(window=3).mean()
            df_processed[f'{col}_ma_7d'] = df_processed[col].rolling(window=7).mean()
            df_processed[f'{col}_diff_1d'] = df_processed[col].diff(periods=1)
            df_processed[f'{col}_std_7d'] = df_processed[col].rolling(window=7).std()
            df_processed[f'{col}_vs_ma_7d_ratio'] = df_processed[col] / (df_processed[f'{col}_ma_7d'] + epsilon)

    if 'cex_netflow' in df_processed and 'whale_netflow' in df_processed:
        epsilon = 1e-6
        total_cex_flow = (df['eth_to_cex'] + df['eth_from_cex']).abs()
        df_processed['whale_flow_ratio_in_cex'] = df_processed['whale_netflow'].abs() / (total_cex_flow + epsilon)

    new_cols = [c for c in df_processed.columns if c not in df.columns]
    df_processed[new_cols] = df_processed[new_cols].ffill()
    df_processed = df_processed.fillna(0)

    return df_processed

def apply_feature_processing(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """피처별 처리 정책 적용 - BTC/DUNE DataFrame만 처리"""
    df_processed = df.copy()
    log_ohlcv_nans(df_processed, data_type, stage="before_process")
    
    if data_type == "btc":
        btc_features = [col for col in df.columns if col.startswith('btc_')]
        for col in btc_features:
            df_processed[col] = df_processed[col].ffill()
    
    elif data_type == "dune":
        # 이제 모든 DUNE 피처를 ffill
        df_processed = df_processed.ffill()
    
    log_ohlcv_nans(df_processed, data_type, stage="after_process")
    return df_processed

def create_labels(df_dict: Dict[str, pd.DataFrame], direction: str = "long") -> pd.DataFrame:
    """
    15분봉 기준 라벨 생성
    """
    if "15min" not in df_dict:
        raise ValueError("Label creation requires '15min' data.")

    df_15m = df_dict["15min"].copy()
    close_col = next((col for col in df_15m.columns if "close" in col.lower()), None)

    if not close_col:
        raise ValueError("Could not find required 'close' column in 15min data.")

    labels = pd.Series('hold', index=df_15m.index, name='label')
    future_close_prices = df_15m[close_col].shift(-LABEL_HORIZON)

    for i in range(len(df_15m) - LABEL_HORIZON):
        entry_price = df_15m.iloc[i][close_col]
        future_close_price = future_close_prices.iloc[i]

        if pd.isna(future_close_price):
            continue

        if direction == "long":
            if future_close_price > entry_price:
                labels.iloc[i] = direction
        elif direction == "short":
            if future_close_price < entry_price:
                labels.iloc[i] = direction

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
    
    data_keys = TIMEFRAMES + AUX_TIMEFRAMES
    
    for key in data_keys:
        file_path = os.path.join(save_dir, f"market_data_{key}.pkl")
        if os.path.exists(file_path):
            df = pd.read_pickle(file_path)
            mtf_data[key] = df
            logger.info(f"[로드] {key}: {len(df)} rows, {len(df.columns)} columns")
        else:
            logger.warning(f"[경고] {key} 파일 없음: {file_path}")
    
    return mtf_data

# ✅ 추가: 모든 피처를 결합하고 스케일러를 학습시키는 함수
def fit_and_save_scaler(data_dict: Dict[str, pd.DataFrame]):
    """모든 타임프레임의 피처를 결합하여 StandardScaler를 학습하고 저장합니다."""
    # 모든 타임프레임의 데이터프레임을 외부 조인으로 결합
    combined_df = pd.DataFrame()
    for key, df in data_dict.items():
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.join(df, how='outer', rsuffix=f'_{key}')
    
    # 모든 수치형 데이터에 대해 ffill을 적용하여 NaN 최소화
    combined_df.ffill(inplace=True)
    combined_df.bfill(inplace=True) # 시작 부분의 NaN도 채움
    
    # 라벨과 같은 비-피처 컬럼 제외
    if 'label' in combined_df.columns:
        combined_df = combined_df.drop(columns=['label'])

    # 스케일러 학습
    scaler = StandardScaler()
    scaler.fit(combined_df)
    
    # 스케일러 저장
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✅ 스케일러 학습 완료 및 저장: {SCALER_PATH}")
    return scaler

def main():
    """메인 처리 함수"""
    print("[MTF 데이터 전처리 및 라벨링 시작]")

    mtf_data = load_mtf_data()
    if not mtf_data:
        raise ValueError("MTF 데이터를 로드할 수 없습니다.")

    if "dune" in mtf_data and not mtf_data["dune"].empty:
        mtf_data["dune"] = create_dune_derived_features(mtf_data["dune"])
        print("[DUNE 파생 피처 생성 완료]")

    if "btc" in mtf_data and not mtf_data["btc"].empty:
        mtf_data["btc"] = apply_feature_processing(mtf_data["btc"], "btc")
        print("[BTC 피처 처리 완료]")

    if "dune" in mtf_data and not mtf_data["dune"].empty:
        mtf_data["dune"] = apply_feature_processing(mtf_data["dune"], "dune")
        print("[DUNE 피처 처리 완료]")

    # ✅ 추가: 스케일러 학습 및 저장 단계
    # 라벨링 전, 순수 피처 데이터만으로 스케일러를 학습
    fit_and_save_scaler(mtf_data)

    print("[방향별 라벨링 시작]")
    df_labels_long = create_labels(mtf_data, direction="long")
    df_labels_short = create_labels(mtf_data, direction="short")
    print("[방향별 라벨링 완료]")

    df_15m_clean = mtf_data["15min"]
    df_labeled_long = df_15m_clean.join(df_labels_long, how='inner')
    df_labeled_short = df_15m_clean.join(df_labels_short, how='inner')
    print("[라벨-피처 안전한 병합 완료]")

    print(f"[Long 라벨 분포] {dict(df_labeled_long['label'].value_counts())}")
    print(f"[Short 라벨 분포] {dict(df_labeled_short['label'].value_counts())}")

    df_long_binary = create_balanced_ppo_dataset(df_labeled_long, "long")
    df_long_binary['label'] = (df_long_binary['label'] == 'long').astype(int)

    df_short_binary = create_balanced_ppo_dataset(df_labeled_short, "short")
    df_short_binary['label'] = (df_short_binary['label'] == 'short').astype(int)

    print(f"[이진분류 데이터 준비]")
    print(f"  - Long 모델용: {len(df_long_binary)}행 (long={sum(df_long_binary['label'])}, hold={len(df_long_binary)-sum(df_long_binary['label'])})")
    print(f"  - Short 모델용: {len(df_short_binary)}행 (short={sum(df_short_binary['label'])}, hold={len(df_short_binary)-sum(df_short_binary['label'])})")

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
