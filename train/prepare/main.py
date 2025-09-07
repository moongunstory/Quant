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

    print("\n[2/4] Adding technical indicators...")
    features: Dict[str, Dict[str, pd.DataFrame]] = {s: {} for s in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for tf in ETH_TIMEFRAMES:
            df = raw_data[split][tf]
            btc_df = btc_data[split]
            extended = add_hpo_candidates(df, tf)
            features[split][tf] = sanitize(extended)

    print("\n[3/4] Saving HPO features & scalers...")
    for tf in ETH_TIMEFRAMES:
        tr_df = features["train"][tf]
        all_feats = [c for c in tr_df.columns if c not in REF_COLS_CANON]

        # 접두어 f_ 붙이기
        feat_list = [c if c.startswith("f_") else f"f_{c}" for c in all_feats]

        with open(HPO_FEATURE_LIST_FMT.format(tf=tf), "w", encoding="utf-8") as f:
            json.dump(feat_list, f, indent=2)

        scaler = StandardScaler().fit(sanitize(tr_df[all_feats]).astype(float))
        joblib.dump(scaler, HPO_SCALER_PATH_FMT.format(tf=tf))

        for split in ["train", "val", "test"]:
            df = features[split][tf]
            X = sanitize(df[all_feats])
            X_scaled = scaler.transform(X)

            # f_ 접두어 컬럼으로 변환
            renamed_cols = [c if c.startswith("f_") else f"f_{c}" for c in all_feats]
            df_scaled = pd.DataFrame(X_scaled, index=df.index, columns=renamed_cols)

            final_df = pd.concat([df_scaled, df[REF_COLS_CANON]], axis=1)
            out_path = os.path.join(OUT_DIR, f"{HPO_OUT_PREFIX}_{split}_{tf}.parquet")
            final_df.to_parquet(out_path)
            print(f"    [ok] Saved HPO {split}/{tf}: {len(final_df):,} rows")

    print("\n[4/4] Checking std consistency...")
    check_zero_std_consistency(features)
    print("[✓] All complete.")

if __name__ == "__main__":
    main()
