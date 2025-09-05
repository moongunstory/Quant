# main.py — 전체 피처 엔지니어링 파이프라인 실행 (BASE + HPO)

import os
import json
import pandas as pd
import joblib
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from ai_binance.train.prepare.paths import *
from ai_binance.train.prepare.utils import *
from ai_binance.train.prepare.engine import *

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
    print("[1/6] Loading raw data...")
    raw_data = {split: {tf: load_raw(split, tf) for tf in TIMEFRAMES} for split in ["train", "val", "test"]}
    btc_data = {split: raw_data[split]["btc1h"] for split in ["train", "val", "test"]}

    print("\n[2/6] Generating ETH features with BTC context...")
    features: Dict[str, Dict[str, pd.DataFrame]] = {s: {} for s in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for tf in ETH_TIMEFRAMES:
            df = raw_data[split][tf]
            btc_df = btc_data[split]
            merged = merge_btc(df, btc_df, tf)
            features[split][tf] = sanitize(merged)

    print("\n[3/6] Performing feature search and scaler fitting...")
    feature_list: Dict[str, list] = {}
    scalers: Dict[str, StandardScaler] = {}

    for tf in TF_FOR_SEARCH:
        tr_df = features["train"][tf]
        horizon = {"5m": 12, "15m": 4, "1h": 1, "4h": 1}.get(tf, 1)
        y = make_proxy_y(tr_df, horizon)
        feat_cols = [c for c in tr_df.columns if c not in REF_COLS_CANON]
        top_k = min(TOP_K_PER_TF.get(tf, len(feat_cols)), len(feat_cols))
        top_feats = feature_search_mi(tr_df[feat_cols], y, top_k)
        feature_list[tf] = prefix_f(top_feats)

        with open(FEATURE_LIST_PATH_FMT.format(tf=tf), "w", encoding="utf-8") as f:
            json.dump(feature_list[tf], f, indent=2)

        sc = fit_scaler(tr_df, top_feats)
        joblib.dump(sc, SCALER_PATH_FMT.format(tf=tf))
        scalers[tf] = sc

    print("\n[4/6] Saving BASE Top-K processed data...")
    for split in ["train", "val", "test"]:
        for tf in ETH_TIMEFRAMES:
            df = features[split][tf]
            feats = [f[2:] if f.startswith("f_") else f for f in feature_list[tf]]
            X = sanitize(df[feats])
            Xs = scalers[tf].transform(X)
            df_scaled = pd.DataFrame(Xs, index=df.index, columns=feats)
            df_scaled = rename_with_f_prefix(df_scaled, feats)

            final_df = pd.concat([df_scaled, df[REF_COLS_CANON]], axis=1)
            out_path = os.path.join(OUT_DIR, f"fe_{split}_{tf}.parquet")
            final_df.to_parquet(out_path)
            print(f"    [ok] Saved BASE {split}/{tf}: {len(final_df):,} rows")

    print("\n[5/6] Generating HPO-extended features...")
    hpo_feats: Dict[str, list] = {}
    hpo_scalers: Dict[str, StandardScaler] = {}

    for tf in ETH_TIMEFRAMES:
        tr = add_hpo_candidates(features["train"][tf], tf)
        all_feats = [c for c in tr.columns if c not in REF_COLS_CANON]
        hpo_feats[tf] = prefix_f(all_feats)

        with open(HPO_FEATURE_LIST_FMT.format(tf=tf), "w", encoding="utf-8") as f:
            json.dump(hpo_feats[tf], f, indent=2)

        sc = fit_scaler(tr, all_feats)
        joblib.dump(sc, HPO_SCALER_PATH_FMT.format(tf=tf))
        hpo_scalers[tf] = sc

    print("\n[6/6] Saving all HPO-extended processed data...")
    for split in ["train", "val", "test"]:
        for tf in ETH_TIMEFRAMES:
            base = features[split][tf]
            hpo = add_hpo_candidates(base, tf)
            feats = [f[2:] if f.startswith("f_") else f for f in hpo_feats[tf]]
            X = sanitize(hpo[feats])
            Xs = hpo_scalers[tf].transform(X)
            df_scaled = pd.DataFrame(Xs, index=hpo.index, columns=feats)
            df_scaled = rename_with_f_prefix(df_scaled, feats)

            final_df = pd.concat([df_scaled, hpo[REF_COLS_CANON]], axis=1)
            out_path = os.path.join(HPO_DIR, f"{HPO_OUT_PREFIX}_{split}_{tf}.parquet")
            final_df.to_parquet(out_path)
            print(f"    [ok] Saved HPO {split}/{tf}: {len(final_df):,} rows")

    print("\n[+] Done. Checking zero std consistency...")
    check_zero_std_consistency(features)
    print("[✓] All complete.")

if __name__ == "__main__":
    main()
