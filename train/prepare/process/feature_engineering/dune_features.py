# train/hpo/feature_engineering/dune_features.py

import pandas as pd
from ai_binance.config.paths import get_raw_dir

def load_dune_raw_data(symbol: str) -> pd.DataFrame:
    """
    Dune raw CSV 데이터를 로드하고 일 단위로 사전 처리합니다.
    - 여러 CSV 파일 로드 및 병합
    - 타임스탬프 처리
    - 일 단위 리샘플링
    - 피처 엔지니어링에 필요한 기본 컬럼 생성
    """
    dune_dir = get_raw_dir(symbol, "dune")
    files = sorted(dune_dir.glob("query_*.csv"))
    if not files:
        return pd.DataFrame()

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if 'day' in df.columns:
                df = df.rename(columns={'day': 'timestamp'})
                # 🔥 타임존과 밀리초 제거 후 datetime 변환
                df['timestamp'] = df['timestamp'].astype(str).str.replace(r'\.000.*', '', regex=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                dfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to process file {f}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    required_cols = [
        "eth_to_cex", "whale_to_cex", "eth_from_cex",
        "cex_to_whale", "deposit_amount", "withdraw_amount"
    ]
    if not all(col in df.columns for col in required_cols):
        print("Warning: Dune data is missing required columns. Returning empty DataFrame.")
        return pd.DataFrame()

    df = df.resample("1D").mean()

    df["dune_cex_inflow"] = df["eth_to_cex"] + df["whale_to_cex"]
    df["dune_cex_outflow"] = df["eth_from_cex"] + df["cex_to_whale"]
    df["dune_netflow"] = df["dune_cex_inflow"] - df["dune_cex_outflow"]
    df["dune_inflow_outflow_ratio"] = df["dune_cex_inflow"] / (df["dune_cex_outflow"] + 1e-8)
    df["dune_deposit_withdraw_ratio"] = df["deposit_amount"] / (df["withdraw_amount"] + 1e-8)

    df = df.drop(columns=required_cols, errors='ignore')

    return df.reset_index()


def compute_dune_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    사전 처리된 Dune 데이터에 HPO 설정에 따라 동적으로 피처를 추가합니다.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.set_index("timestamp").copy()
    
    features_to_keep = []
    
    base_features = {
        "dune_netflow": "use_dune_netflow",
        "dune_inflow_outflow_ratio": "use_dune_inflow_outflow_ratio",
        "dune_deposit_withdraw_ratio": "use_dune_deposit_withdraw_ratio"
    }

    for feature, use_flag in base_features.items():
        if config.get(use_flag):
            features_to_keep.append(feature)

    window_features = {
        "ma": "use_dune_ma",
        "momentum": "use_dune_momentum",
        "zscore": "use_dune_zscore"
    }
    
    window_target_cols = []
    if config.get("use_dune_netflow_for_window"):
        window_target_cols.append("dune_netflow")
    
    windows = config.get("windows", [])

    for feature_type, use_flag in window_features.items():
        if config.get(use_flag):
            for col in window_target_cols:
                for w in windows:
                    new_feat_name = f"{col}_{feature_type}_{w}d"
                    if feature_type == "ma":
                        df[new_feat_name] = df[col].rolling(window=w).mean()
                    elif feature_type == "momentum":
                        df[new_feat_name] = df[col].diff(periods=w)
                    elif feature_type == "zscore":
                        ma = df[col].rolling(window=w).mean()
                        std = df[col].rolling(window=w).std()
                        df[new_feat_name] = (df[col] - ma) / (std + 1e-8)
                    features_to_keep.append(new_feat_name)

    if not features_to_keep:
        return pd.DataFrame()

    all_features = ["timestamp"] + features_to_keep
    df = df.reset_index()
    
    existing_cols_to_keep = [col for col in all_features if col in df.columns]
    df = df[existing_cols_to_keep]

    return df

def get_available_dune_features() -> dict:
    """HPO 탐색을 위한 Dune 피처 목록 반환"""
    return {
        "simple": [
            "use_dune_netflow",
            "use_dune_inflow_outflow_ratio",
            "use_dune_deposit_withdraw_ratio"
        ],
        "window_applies_to": [
            "use_dune_netflow_for_window"
        ],
        "window": [
            "use_dune_ma",
            "use_dune_momentum",
            "use_dune_zscore"
        ]
    }