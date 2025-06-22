import pandas as pd
import numpy as np
import os
import sys

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    RAW_DATA_PATH, TRAIN_LABEL_PATHS,
    TP_THRESHOLD, SL_THRESHOLD, LABEL_HORIZON
)

def create_dune_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """DUNE 파생 피처 생성"""
    df_processed = df.copy()
    
    # 수치형 파생 피처
    df_processed['cex_netflow'] = df['eth_to_cex'] - df['eth_from_cex']
    df_processed['whale_netflow'] = df['whale_to_cex'] - df['cex_to_whale']
    df_processed['staking_netflow'] = df['deposit_amount'] - df['withdraw_amount']
    
    # 이벤트 flag 피처 (00:00에만 값 설정, 나머지는 NaN)
    midnight_mask = (df_processed.index.hour == 0) & (df_processed.index.minute == 0)
    
    df_processed['cex_increase_flag'] = np.nan
    df_processed.loc[midnight_mask, 'cex_increase_flag'] = (
        df_processed.loc[midnight_mask, 'cex_netflow'] > 0
    ).astype(int)
    
    df_processed['whale_increase_flag'] = np.nan
    df_processed.loc[midnight_mask, 'whale_increase_flag'] = (
        df_processed.loc[midnight_mask, 'whale_netflow'] > 0
    ).astype(int)
    
    df_processed['staking_increase_flag'] = np.nan
    df_processed.loc[midnight_mask, 'staking_increase_flag'] = (
        df_processed.loc[midnight_mask, 'staking_netflow'] > 0
    ).astype(int)
    
    return df_processed

def apply_feature_processing(df: pd.DataFrame) -> pd.DataFrame:
    """피처별 처리 정책 적용"""
    df_processed = df.copy()
    
    # BTC 피처 .ffill() 적용
    btc_features = [col for col in df.columns if col.startswith('btc_')]
    for col in btc_features:
        df_processed[col] = df_processed[col].ffill()
    
    # DUNE 수치형 피처 .ffill() 적용
    dune_numeric_features = ['cex_netflow', 'whale_netflow', 'staking_netflow']
    for col in dune_numeric_features:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].ffill()
    
    # DUNE flag 피처는 .ffill() 금지 (그대로 유지)
    
    return df_processed

def create_labels(df: pd.DataFrame, label_horizon: int = 4) -> pd.DataFrame:
    """15분봉 기준 라벨 생성 (5분봉 기준 TP/SL 도달 판별)"""
    df_labeled = df.copy()
    df_labeled['label'] = 'hold'
    
    for i in range(len(df) - label_horizon):
        # 진입 시점
        entry_time = df.index[i]
        entry_price = df.iloc[i]['15m_close']
        
        # TP/SL 가격 계산
        tp_price = entry_price * (1 + TP_THRESHOLD)
        sl_price = entry_price * (1 + SL_THRESHOLD)
        
        # horizon 기간 설정 (다음 4개 15분봉)
        end_time = df.index[i + label_horizon]
        
        # 해당 기간의 5분봉 데이터 추출 (같은 데이터프레임에서)
        future_5m = df[(df.index > entry_time) & (df.index <= end_time)]
        
        if len(future_5m) == 0:
            continue
        
        # TP/SL 도달 시점 찾기 (5분봉 high/low 사용)
        tp_reached = future_5m['5m_high'] >= tp_price
        sl_reached = future_5m['5m_low'] <= sl_price
        
        tp_first_idx = future_5m[tp_reached].index.min() if tp_reached.any() else pd.NaT
        sl_first_idx = future_5m[sl_reached].index.min() if sl_reached.any() else pd.NaT
        
        # 라벨 결정
        if pd.notna(tp_first_idx) and pd.notna(sl_first_idx):
            # 둘 다 도달 - 먼저 도달한 것으로 결정
            if sl_first_idx < tp_first_idx:
                df_labeled.iloc[i, df_labeled.columns.get_loc('label')] = 'short'
            else:
                df_labeled.iloc[i, df_labeled.columns.get_loc('label')] = 'long'
        elif pd.notna(tp_first_idx):
            # TP만 도달
            df_labeled.iloc[i, df_labeled.columns.get_loc('label')] = 'long'
        elif pd.notna(sl_first_idx):
            # SL만 도달
            df_labeled.iloc[i, df_labeled.columns.get_loc('label')] = 'short'
        # 둘 다 도달하지 않음 - 'hold' 유지
    
    return df_labeled

def remove_ohlc_features(df: pd.DataFrame) -> pd.DataFrame:
    """5분봉 OHLC 컬럼만 제거 (지표는 유지)"""
    # 5분봉 OHLC만 제거 (지표는 유지)
    ohlc_cols = ['5m_open', '5m_high', '5m_low', '5m_close', '5m_volume']
    existing_ohlc_cols = [col for col in ohlc_cols if col in df.columns]
    df_clean = df.drop(columns=existing_ohlc_cols)
    
    print(f"[🧹 5분봉 OHLC만 제거] {len(existing_ohlc_cols)}개 컬럼 제거: {existing_ohlc_cols}")
    
    # 남은 5분봉 지표 확인
    remaining_5m = [col for col in df_clean.columns if col.startswith('5m_')]
    print(f"[✅ 5분봉 지표 유지] {len(remaining_5m)}개 컬럼: {remaining_5m}")
    
    return df_clean

def main():
    """메인 처리 함수"""
    print("[🚀 데이터 전처리 및 라벨링 시작]")
    
    # 1. 원시 데이터 로딩
    raw_data_path = os.path.join(PROJECT_ROOT, RAW_DATA_PATH)
    df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)
    print(f"[📊 원시 데이터 로딩 완료] 행: {len(df)}, 컬럼: {len(df.columns)}")
    
    # 2. DUNE 파생 피처 생성
    df = create_dune_derived_features(df)
    print("[⛓️ DUNE 파생 피처 생성 완료]")
    
    # 3. 피처별 처리 정책 적용
    df = apply_feature_processing(df)
    print("[🔧 피처 처리 정책 적용 완료]")
    
    # 4. 라벨 생성 (5분봉 OHLC 사용)
    df_labeled = create_labels(df, LABEL_HORIZON)
    print("[🎯 라벨링 완료]")
    
    # 라벨 분포 확인
    label_counts = df_labeled['label'].value_counts()
    print(f"[📊 라벨 분포] {dict(label_counts)}")
    
    # 5. 5분봉 OHLC 컬럼 제거 (라벨링 완료 후)
    df_clean = remove_ohlc_features(df_labeled)
    print(f"[✅ 최종 데이터] 행: {len(df_clean)}, 컬럼: {len(df_clean.columns)}")
    
    # 6. 이진분류용 데이터 준비
    # Long 모델용: long + hold (short 제외)
    df_long_binary = df_clean[df_clean['label'].isin(['long', 'hold'])].copy()
    # 라벨 변환: long=1, hold=0
    df_long_binary['label'] = (df_long_binary['label'] == 'long').astype(int)
    
    # Short 모델용: short + hold (long 제외)  
    df_short_binary = df_clean[df_clean['label'].isin(['short', 'hold'])].copy()
    # 라벨 변환: short=1, hold=0
    df_short_binary['label'] = (df_short_binary['label'] == 'short').astype(int)
    
    print(f"[🎯 이진분류 데이터 준비]")
    print(f"  - Long 모델용: {len(df_long_binary)}행 (long={sum(df_long_binary['label'])}, hold={len(df_long_binary)-sum(df_long_binary['label'])})")
    print(f"  - Short 모델용: {len(df_short_binary)}행 (short={sum(df_short_binary['label'])}, hold={len(df_short_binary)-sum(df_short_binary['label'])})")
    
    # 7. 저장
    long_path = TRAIN_LABEL_PATHS["long"]
    short_path = TRAIN_LABEL_PATHS["short"]
        
    os.makedirs(os.path.dirname(long_path), exist_ok=True)
    os.makedirs(os.path.dirname(short_path), exist_ok=True)
    
    df_long_binary.to_csv(long_path)
    df_short_binary.to_csv(short_path)
    
    print(f"[💾 저장 완료]")
    print(f"  - Long 이진분류 데이터: {len(df_long_binary)}행 → {long_path}")
    print(f"  - Short 이진분류 데이터: {len(df_short_binary)}행 → {short_path}")
    
    return df_clean

if __name__ == "__main__":
    result = main()