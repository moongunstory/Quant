import pandas as pd
import numpy as np
import os
import sys
from typing import Dict
import pickle

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    RAW_DATA_PATH, TRAIN_PICKLE_PATHS,
    TP_THRESHOLD, SL_THRESHOLD, LABEL_HORIZON,
    TIMEFRAMES
)

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
    
    return df_processed

def create_labels(df_dict: Dict[str, pd.DataFrame], label_horizon: int = 4) -> pd.DataFrame:
    """15분봉 기준 라벨 생성 (5분봉 기준 TP/SL 도달 판별)"""
    if "15min" not in df_dict:
        raise ValueError("15min timeframe not found in df_dict")
    
    if "5min" not in df_dict:
        raise ValueError("5min timeframe not found in df_dict")
    
    # 15분봉을 베이스로 라벨링
    df_15m = df_dict["15min"].copy()
    df_5m = df_dict["5min"]
    
    # 15분봉에서 close 컬럼 찾기
    close_col = None
    for col in df_15m.columns:
        if "close" in col.lower():
            close_col = col
            break
    if close_col is None:
        raise ValueError("No 'close' column found in 15min timeframe")
    
    # 5분봉에서 high/low 컬럼 찾기
    high_col = low_col = None
    for col in df_5m.columns:
        if "high" in col.lower():
            high_col = col
        elif "low" in col.lower():
            low_col = col
    if high_col is None or low_col is None:
        raise ValueError("No 'high' or 'low' columns found in 5min timeframe")
    
    # 라벨 컬럼 초기화
    df_15m['label'] = 'hold'
    
    # 15분봉 인덱스를 기준으로 반복
    for i in range(len(df_15m) - label_horizon):
        # 진입 시점 (15분봉 기준)
        entry_time = df_15m.index[i]
        entry_price = df_15m.iloc[i][close_col]
        
        # TP/SL 가격 계산
        tp_price = entry_price * (1 + TP_THRESHOLD)
        sl_price = entry_price * (1 + SL_THRESHOLD)
        
        # horizon 기간 설정 (다음 label_horizon개 15분봉)
        if i + label_horizon < len(df_15m):
            end_time = df_15m.index[i + label_horizon]
        else:
            end_time = df_15m.index[-1]
        
        # 해당 기간의 5분봉 데이터 추출
        future_5m = df_5m[(df_5m.index > entry_time) & (df_5m.index <= end_time)]
        
        if len(future_5m) == 0:
            continue
        
        # TP/SL 도달 시점 찾기 (5분봉 high/low 사용)
        tp_reached = future_5m[high_col] >= tp_price
        sl_reached = future_5m[low_col] <= sl_price
        
        tp_first_idx = future_5m[tp_reached].index.min() if tp_reached.any() else pd.NaT
        sl_first_idx = future_5m[sl_reached].index.min() if sl_reached.any() else pd.NaT
        
        # 라벨 결정
        if pd.notna(tp_first_idx) and pd.notna(sl_first_idx):
            # 둘 다 도달 - 먼저 도달한 것으로 결정
            if sl_first_idx < tp_first_idx:
                df_15m.iloc[i, df_15m.columns.get_loc('label')] = 'short'
            else:
                df_15m.iloc[i, df_15m.columns.get_loc('label')] = 'long'
        elif pd.notna(tp_first_idx):
            # TP만 도달
            df_15m.iloc[i, df_15m.columns.get_loc('label')] = 'long'
        elif pd.notna(sl_first_idx):
            # SL만 도달
            df_15m.iloc[i, df_15m.columns.get_loc('label')] = 'short'
        # 둘 다 도달하지 않음 - 'hold' 유지
    
    return df_15m

def mask_future_5m_values(df_dict: Dict[str, pd.DataFrame], df_labeled: pd.DataFrame, 
                         label_horizon: int = 4) -> Dict[str, pd.DataFrame]:
    """미래 5분봉 high/low 값 마스킹 - 데이터 누출 방지"""
    if "5min" not in df_dict:
        return df_dict
    
    df_dict_masked = df_dict.copy()
    df_5m = df_dict_masked["5min"].copy()
    
    # 5분봉에서 high/low 컬럼 찾기
    high_col = low_col = None
    for col in df_5m.columns:
        if "high" in col.lower():
            high_col = col
        elif "low" in col.lower():
            low_col = col
    
    if high_col is None or low_col is None:
        print("[WARNING] No high/low columns found in 5min data for masking")
        return df_dict_masked
    
    # 각 15분봉 라벨링 시점에 대해 미래 5분봉 값 마스킹
    for i in range(len(df_labeled) - label_horizon):
        entry_time = df_labeled.index[i]
        
        # horizon 기간 설정
        if i + label_horizon < len(df_labeled):
            end_time = df_labeled.index[i + label_horizon]
        else:
            end_time = df_labeled.index[-1]
        
        # 미래 5분봉 구간 마스킹
        future_mask = (df_5m.index > entry_time) & (df_5m.index <= end_time)
        df_5m.loc[future_mask, [high_col, low_col]] = np.nan
    
    df_dict_masked["5min"] = df_5m
    print(f"[MASK] 미래 5분봉 high/low 값 마스킹 완료")
    
    return df_dict_masked

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
            print(f"[로드] {key}: {len(df)} rows, {len(df.columns)} columns")
        else:
            print(f"[경고] {key} 파일 없음: {file_path}")
    
    return mtf_data

def main():
    """메인 처리 함수"""
    print("[🚀 MTF 데이터 전처리 및 라벨링 시작]")
    
    # 1. MTF 데이터 로딩
    mtf_data = load_mtf_data()
    if not mtf_data:
        raise ValueError("MTF 데이터를 로드할 수 없습니다.")
    
    # 2. DUNE 파생 피처 생성 (DUNE DataFrame만)
    if "dune" in mtf_data and not mtf_data["dune"].empty:
        mtf_data["dune"] = create_dune_derived_features(mtf_data["dune"])
        print("[⛓️ DUNE 파생 피처 생성 완료]")
    
    # 3. 개별 피처 처리
    if "btc" in mtf_data and not mtf_data["btc"].empty:
        mtf_data["btc"] = apply_feature_processing(mtf_data["btc"], "btc")
        print("[🔧 BTC 피처 처리 완료]")
    
    if "dune" in mtf_data and not mtf_data["dune"].empty:
        mtf_data["dune"] = apply_feature_processing(mtf_data["dune"], "dune")
        print("[🔧 DUNE 피처 처리 완료]")
    
    # 4. 라벨 생성 (15분봉 기준, 5분봉 TP/SL 사용)
    df_labeled_15m = create_labels(mtf_data, LABEL_HORIZON)
    print("[🎯 라벨링 완료]")
    
    # 라벨 분포 확인
    label_counts = df_labeled_15m['label'].value_counts()
    print(f"[📊 라벨 분포] {dict(label_counts)}")
    
    # 5. 미래 정보 마스킹 (데이터 누출 방지)
    mtf_data_masked = mask_future_5m_values(mtf_data, df_labeled_15m, LABEL_HORIZON)
    
    # 6. 이진분류용 데이터 준비 (라벨링된 15분봉 DataFrame 직접 사용)
    # Long 모델용: long + hold (short 제외)
    df_long_binary = df_labeled_15m[df_labeled_15m['label'].isin(['long', 'hold'])].copy()
    # 라벨 변환: long=1, hold=0
    df_long_binary['label'] = (df_long_binary['label'] == 'long').astype(int)
    
    # Short 모델용: short + hold (long 제외)  
    df_short_binary = df_labeled_15m[df_labeled_15m['label'].isin(['short', 'hold'])].copy()
    # 라벨 변환: short=1, hold=0
    df_short_binary['label'] = (df_short_binary['label'] == 'short').astype(int)
    
    print(f"[🎯 이진분류 데이터 준비]")
    print(f"  - Long 모델용: {len(df_long_binary)}행 (long={sum(df_long_binary['label'])}, hold={len(df_long_binary)-sum(df_long_binary['label'])})")
    print(f"  - Short 모델용: {len(df_short_binary)}행 (short={sum(df_short_binary['label'])}, hold={len(df_short_binary)-sum(df_short_binary['label'])})")
    
    # 7. 저장
    long_path = TRAIN_PICKLE_PATHS["long"]
    short_path = TRAIN_PICKLE_PATHS["short"]
    
    os.makedirs(os.path.dirname(long_path), exist_ok=True)
    os.makedirs(os.path.dirname(short_path), exist_ok=True)
    
    # ✅ long
    with open(long_path, "wb") as f:
        pickle.dump({**mtf_data_masked, "15min": df_long_binary}, f)

    # ✅ short
    with open(short_path, "wb") as f:
        pickle.dump({**mtf_data_masked, "15min": df_long_binary}, f)

    print(f"[💾 저장 완료]")
    print(f"  - Long 이진분류 데이터: {len(df_long_binary)}행 → {long_path}")
    print(f"  - Short 이진분류 데이터: {len(df_short_binary)}행 → {short_path}")
    
    return df_labeled_15m, mtf_data_masked

if __name__ == "__main__":
    labeled_df, masked_mtf_data = main()