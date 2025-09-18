# train/hpo/core/feature_builder.py

from ai_binance.train.hpo.feature_engineering import (
    ohlcv_features,
    funding_index_features,
    dune_features
)
from ai_binance.train.hpo.core import feature_cleaning
from ai_binance.config.paths import get_processed_feature_path


def save_feature_to_parquet(df, symbol: str, category: str, name: str = None):
    path = get_processed_feature_path(symbol, category, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"✅ Saved {category} features to {path}")


def build_feature_dfs(symbol: str, config: dict) -> dict:
    """
    선택된 피처 config를 바탕으로 실제 피처 데이터프레임 생성
    반환값: {
        "ohlcv": pd.DataFrame,
        "funding": pd.DataFrame,
        "index": pd.DataFrame,
        "dune": pd.DataFrame
    }
    """
    feature_dfs = {}

    # 1. OHLCV
    if "ohlcv" in config and config["ohlcv"]:
        raw_df = ohlcv_features.load_ohlcv_data(symbol)
        df = ohlcv_features.compute_ohlcv_features(raw_df, config["ohlcv"])
        df = feature_cleaning.clean_and_align_features(df)
        # save_feature_to_parquet(df, symbol, "ohlcv")
        if not df.empty:
            feature_dfs["ohlcv"] = df

    # 2. Funding + Index
    if ("funding" in config and config["funding"]) or ("index" in config and config["index"]):
        fund_df = funding_index_features.load_funding_data(symbol)
        index_df = funding_index_features.load_index_data(symbol)

        if "funding" in config and config["funding"]:
            fund_feat_cfg = config["funding"]
            window = fund_feat_cfg.get("window", 96) if isinstance(fund_feat_cfg, dict) else 96
            fund_feat_df = funding_index_features.compute_funding_features(fund_df, window=window)
            fund_feat_df = feature_cleaning.clean_and_align_features(fund_feat_df)
            save_feature_to_parquet(fund_feat_df, symbol, "funding_index", f"{symbol.lower()}_funding")
            feature_dfs["funding"] = fund_feat_df

        if "index" in config and config["index"]:
            index_feat_cfg = config["index"]
            windows = index_feat_cfg.get("windows", [96]) if isinstance(index_feat_cfg, dict) else [96]
            index_feat_df = funding_index_features.compute_index_features(index_df, windows=windows)
            index_feat_df = feature_cleaning.clean_and_align_features(index_feat_df)
            save_feature_to_parquet(index_feat_df, symbol, "funding_index", f"{symbol.lower()}_index")
            feature_dfs["index"] = index_feat_df

    # 3. Dune
    if "dune" in config and config["dune"]:
        raw_df = dune_features.load_dune_raw_data(symbol)
        if not raw_df.empty:
            df = dune_features.compute_dune_features(raw_df, config["dune"])
            df = feature_cleaning.clean_and_align_features(df)
            # save_feature_to_parquet(df, symbol, "dune") # 저장 로직 제거
            if not df.empty:
                feature_dfs["dune"] = df

    return feature_dfs
