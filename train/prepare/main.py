
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ai_binance.train.prepare.paths import *
from ai_binance.train.prepare.engine import generate_clean_features, load_raw, add_y_class
from ai_binance.train.prepare.feature_engineering import filter_features
from sklearn.preprocessing import StandardScaler

def check_zero_std_consistency(feature_data: Dict[str, Dict[str, pd.DataFrame]]):
    """Checks for columns with zero standard deviation across splits."""
    for tf in ETH_TIMEFRAMES:
        if tf not in feature_data["train"] or feature_data["train"][tf].empty:
            continue
        train = feature_data["train"][tf]
        val = feature_data["val"][tf]
        test = feature_data["test"][tf]

        train_std = train.std(numeric_only=True)
        val_std = val.std(numeric_only=True)
        test_std = test.std(numeric_only=True)

        train_zero = set(train_std[train_std == 0].index)
        val_zero = set(val_std[val_std == 0].index)
        test_zero = set(test_std[test_std == 0].index)

        if train_zero != val_zero or val_zero != test_zero:
            print(f"[WARNING] Zero std columns inconsistent across splits in {tf}")

def main():
    # 1. Load full raw data
    print("--- [1/4] Loading full raw data... ---")
    full_raw_data = {tf: load_raw(tf) for tf in TIMEFRAMES}
    for tf, df in full_raw_data.items():
        print(f"  - Loaded {tf}: {len(df):,} rows")

    # 2. Add y_class and generate features on full data
    print("--- [2/4] Generating features on full dataframes... ---")
    full_features = {}
    for tf, df in full_raw_data.items():
        df_with_y = add_y_class(df)
        if tf == 'btc1h':
            full_features[tf] = df_with_y
            print(f"  - Processed btc1h: {len(full_features[tf]):,} rows")
            continue
        
        print(f"  - Generating features for {tf}...")
        full_features[tf] = generate_clean_features(df_with_y, tf, top_k=300, verbose=False)
        print(f"  - Features for {tf}: {len(full_features[tf]):,} rows")

    # 3. Split data into train, val, test
    print("--- [3/4] Splitting data into train/val/test sets... ---")

    # Find the full date range across all timeframes
    all_start = min(df.index.min() for df in full_features.values() if not df.empty)
    all_end = max(df.index.max() for df in full_features.values() if not df.empty)
    print(f"  - Full data range: {all_start} to {all_end}")

    # Split by date range (70:15:15)
    total_days = (all_end - all_start).days
    train_days = int(total_days * SPLIT[0])
    val_days = int(total_days * SPLIT[1])

    t1 = all_start + pd.Timedelta(days=train_days)
    t2 = t1 + pd.Timedelta(days=val_days)

    print(f"  - Split points: t1={t1}, t2={t2}")
    print(f"  - Train: {all_start} ~ {t1} ({train_days} days)")
    print(f"  - Val:   {t1} ~ {t2} ({val_days} days)")
    print(f"  - Test:  {t2} ~ {all_end} ({total_days - train_days - val_days} days)")

    features = {"train": {}, "val": {}, "test": {}}
    for tf, df in full_features.items():
        features["train"][tf] = df.loc[df.index <= t1]
        features["val"][tf] = df.loc[(df.index > t1) & (df.index <= t2)]
        features["test"][tf] = df.loc[df.index > t2]
        # Log the split sizes
        print(f"  - Split {tf}: train={len(features['train'][tf]):,}, val={len(features['val'][tf]):,}, test={len(features['test'][tf]):,}")

    # 4. Save all features and scalers
    print("--- [4/4] Saving all features & scalers ---")
    # ETH
    for tf in ETH_TIMEFRAMES:
        print(f"-- Saving ETH {tf} --")
        tr_df = features["train"][tf]
        if tr_df.empty:
            print(f"[ERROR] Train set for {tf} is empty. Halting for this timeframe.")
            continue

        common_feats = [c for c in tr_df.columns if c.startswith("f_")]
        filtered_df = filter_features(tr_df[common_feats], target=tr_df["y_class"], top_k=300)
        selected_feats = filtered_df.columns.tolist()

        scaler = StandardScaler().fit(filtered_df[selected_feats])
        joblib.dump(scaler, HPO_SCALER_PATH_FMT.format(tf=tf))
        with open(HPO_FEATURE_LIST_FMT.format(tf=tf), "w") as f: 
            json.dump(selected_feats, f, indent=2)

        for split in ["train", "val", "test"]:
            df_split = features[split][tf]
            if df_split.empty: 
                print(f"    [skip] {split}/{tf} is empty")
                continue
                
            # 모든 필요한 컬럼에서 NaN 제거
            all_required_cols = selected_feats + REF_COLS_CANON
            available_ref_cols = [col for col in REF_COLS_CANON if col in df_split.columns]
            all_cols_to_check = selected_feats + available_ref_cols
            
            # NaN이 있는 행을 완전히 제거
            original_len = len(df_split)
            df_clean = df_split[all_cols_to_check].dropna()
            
            if df_clean.empty:
                print(f"    [error] {split}/{tf} became empty after NaN removal")
                continue
                
            # 데이터 손실 체크
            loss_ratio = (original_len - len(df_clean)) / original_len
            if loss_ratio > 0.05:  # 5% 이상 손실
                print(f"    [warn] High data loss in {split}/{tf}: {loss_ratio:.1%} ({original_len} -> {len(df_clean)} rows)")
            
            # 피처와 참조 컬럼 분리
            df_selected = df_clean[selected_feats]
            ref_cols_clean = df_clean[available_ref_cols]
            
            # 스케일링
            df_scaled = pd.DataFrame(
                scaler.transform(df_selected), 
                index=df_clean.index, 
                columns=selected_feats
            )
            
            # 최종 데이터 결합
            final_df = pd.concat([df_scaled, ref_cols_clean], axis=1)
            
            # 누락된 참조 컬럼을 0으로 채우기
            for col in REF_COLS_CANON:
                if col not in final_df.columns:
                    final_df[col] = 0.0
            
            # 최종 NaN 검증
            nan_count = final_df.isnull().sum().sum()
            if nan_count > 0:
                print(f"    [error] Final data for {split}/{tf} still contains {nan_count} NaN values")
                nan_cols = final_df.columns[final_df.isnull().any()].tolist()
                print(f"    [error] NaN columns: {nan_cols}")
                # 최후의 안전장치
                final_df = final_df.fillna(0.0)
                print(f"    [fix] Applied zero-fill to {split}/{tf}")
            
            # 저장
            out_path = os.path.join(OUT_DIR, f"{HPO_OUT_PREFIX}_{split}_{tf}.parquet")
            final_df.to_parquet(out_path)
            print(f"    [ok] Saved {split}/{tf}: {len(final_df):,} rows")

    # BTC
    print("-- Saving BTC 1h --")
    for split in ["train", "val", "test"]:
        df = features[split]["btc1h"]
        if df.empty: 
            print(f"    [skip] BTC {split}/1h is empty")
            continue
            
        # BTC는 피처 생성을 안했으므로 모든 숫자형 컬럼 사용
        original_len = len(df)
        df_numeric = df.select_dtypes(include=[np.number])
        
        # NaN 제거
        df_clean = df_numeric.dropna()
        
        if df_clean.empty:
            print(f"    [error] BTC {split}/1h became empty after NaN removal")
            continue
        
        # 데이터 손실 체크
        loss_ratio = (original_len - len(df_clean)) / original_len
        if loss_ratio > 0.05:
            print(f"    [warn] High data loss in BTC {split}/1h: {loss_ratio:.1%} ({original_len} -> {len(df_clean)} rows)")

        # 스케일링
        scaler = StandardScaler().fit(df_clean)
        df_scaled = pd.DataFrame(
            scaler.transform(df_clean), 
            index=df_clean.index, 
            columns=df_clean.columns
        )
        
        # 최종 NaN 검증
        nan_count = df_scaled.isnull().sum().sum()
        if nan_count > 0:
            print(f"    [error] BTC final data for {split}/1h contains {nan_count} NaN values")
            df_scaled = df_scaled.fillna(0.0)
            print(f"    [fix] Applied zero-fill to BTC {split}/1h")
        
        # 저장
        out_path = os.path.join(OUT_DIR, f"{HPO_OUT_PREFIX}_{split}_btc1h.parquet")
        df_scaled.to_parquet(out_path)
        print(f"    [ok] Saved BTC {split}/1h: {len(df_scaled):,} rows")

    print("--- Checking std consistency ---")
    check_zero_std_consistency(features)
    print("[DONE] All complete.")

if __name__ == "__main__":
    main()
