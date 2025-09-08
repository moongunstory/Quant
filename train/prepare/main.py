import os
import json
import pandas as pd
import joblib
import numpy as np
from typing import Dict
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ai_binance.train.prepare.paths import *
from ai_binance.train.prepare.engine import *
from ai_binance.train.prepare.feature_engineering import filter_features
from sklearn.preprocessing import StandardScaler

def check_zero_std_consistency(feature_data: Dict[str, Dict[str, pd.DataFrame]]):
    for tf in ETH_TIMEFRAMES:
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
    print("[1/4] Loading raw data...")
    raw_data = {split: {tf: load_raw(split, tf) for tf in TIMEFRAMES} for split in ["train", "val", "test"]}
    btc_data = {split: raw_data[split]["btc1h"] for split in ["train", "val", "test"]}

    print("\n[2/4] Adding technical indicators & features (ETH only)...")
    features: Dict[str, Dict[str, pd.DataFrame]] = {s: {} for s in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for tf in ETH_TIMEFRAMES:
            df = raw_data[split][tf]
            df, feat_names = add_hpo_candidates(df, tf)
            features[split][tf] = df

    print("\n[3/4] Saving HPO features & scalers (ETH)...")
    for tf in ETH_TIMEFRAMES:
        tr_df = features["train"][tf]
        val_df = features["val"][tf]
        test_df = features["test"][tf]

        raw_cols = ["Open", "High", "Low", "Close", "Volume"]  # 또는 실제 사용 중인 컬럼명으로
        base_feats = raw_cols + [c for c in tr_df.columns if c.startswith("f_")]
        common_feats = [f for f in base_feats if f in val_df.columns and f in test_df.columns]

        filtered_df = filter_features(tr_df[common_feats], target=tr_df["y_class"], top_k=300, vif_thresh=10.0)
        selected_feats = filtered_df.columns.tolist()

        scaler = StandardScaler().fit(filtered_df.astype(float))
        joblib.dump(scaler, HPO_SCALER_PATH_FMT.format(tf=tf))

        with open(HPO_FEATURE_LIST_FMT.format(tf=tf), "w", encoding="utf-8") as f:
            json.dump(selected_feats, f, indent=2)

        for split in ["train", "val", "test"]:
            df = features[split][tf]
            df_selected = df[selected_feats].copy()
            df_scaled = pd.DataFrame(scaler.transform(df_selected), index=df.index, columns=selected_feats)
            final_df = pd.concat([df_scaled, df[REF_COLS_CANON]], axis=1)

            out_path = os.path.join(OUT_DIR, f"{HPO_OUT_PREFIX}_{split}_{tf}.parquet")
            final_df.to_parquet(out_path)
            print(f"    [ok] Saved HPO {split}/{tf}: {len(final_df):,} rows")

    # Get the aligned indices from the processed 1h ETH data
    eth_1h_indices = {
        split: features[split]['1h'].index for split in ["train", "val", "test"]
    }

    print("\n[3.5/4] Saving BTC 1h features (aligned with ETH 1h)...")
    for split in ["train", "val", "test"]:
        df = btc_data[split]

        # Align BTC index with the corresponding processed ETH 1h index
        aligned_index = eth_1h_indices[split]
        df = df[df.index.isin(aligned_index)]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df_numeric = df[numeric_cols].copy()
        scaler = StandardScaler().fit(df_numeric.astype(float))
        df_scaled = pd.DataFrame(scaler.transform(df_numeric), index=df.index, columns=numeric_cols)

        out_path = os.path.join(OUT_DIR, f"{HPO_OUT_PREFIX}_{split}_btc1h.parquet")
        df_scaled.to_parquet(out_path)
        print(f"    [ok] Saved BTC {split}/1h: {len(df_scaled):,} rows")

    print("\n[4/4] Checking std consistency...")
    check_zero_std_consistency(features)
    print("[ok] All complete.")

if __name__ == "__main__":
    main()
