# train/hpo/core/feature_builder.py

import pandas as pd
from pathlib import Path
from typing import Dict, List
from ai_binance.config.paths import PROCESSED_DATA_DIR

def build_feature_dfs(symbol: str, selected_features: List[str]) -> Dict[str, pd.DataFrame]:
    """
    processed/train_set.parquet에서 선택된 피처만 골라 그룹별 DataFrame 반환
    """
    symbol_lower = symbol.lower()
    processed_path = PROCESSED_DATA_DIR / symbol_lower / "train_set.parquet"
    df = pd.read_parquet(processed_path)

    # timestamp 컬럼 처리
    if "timestamp" not in df.columns:
        df = df.reset_index().rename(columns={"index": "timestamp"})

    # 선택된 피처 + timestamp만 유지
    cols_to_keep = ["timestamp"] + [f for f in selected_features if f in df.columns]
    df_sel = df[cols_to_keep].copy()

    # 그룹별로 나누기
    feature_dfs: Dict[str, pd.DataFrame] = {}
    for f in selected_features:
        if f not in df_sel.columns:
            continue
        grp = f.split("_")[0]  # 피처 이름 접두사로 그룹 나눔
        if grp not in feature_dfs:
            feature_dfs[grp] = df_sel[["timestamp", f]].copy()
        else:
            # concat으로 한 번에 병합 → fragmentation 방지
            feature_dfs[grp] = pd.concat(
                [feature_dfs[grp], df_sel[[f]]], axis=1
            )

    return feature_dfs
