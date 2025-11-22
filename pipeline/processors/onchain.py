# process/modules/onchain.py

from __future__ import annotations
from typing import Optional
import pandas as pd

from .utils import ensure_sorted_date


def _add_basic_transforms(df: pd.DataFrame, value_col: str, prefix: str) -> pd.DataFrame:
    s = df[value_col].astype(float)

    # 365일 기준 z-score
    roll = s.rolling(365)
    df[f"{prefix}_z_365d"] = (s - roll.mean()) / roll.std()

    # 변화율
    df[f"{prefix}_ret_1d"] = s.pct_change(1)
    df[f"{prefix}_ret_7d"] = s.pct_change(7)

    # 롤링 평균
    df[f"{prefix}_ma_7d"] = s.rolling(7).mean()
    df[f"{prefix}_ma_30d"] = s.rolling(30).mean()

    return df


def build_onchain_features(
    df_n_txn: Optional[pd.DataFrame] = None,
    df_n_unique_addr: Optional[pd.DataFrame] = None,
    df_est_tx_volume_usd: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    온체인 raw DataFrame 들을 모아서 일단위 피처 생성.

    각 df는 최소 ['date', 'value'] 컬럼을 가진다고 가정.
    """
    features = None

    def _merge(df_base: Optional[pd.DataFrame], df_add: pd.DataFrame) -> pd.DataFrame:
        if df_base is None:
            return df_add
        return df_base.merge(df_add, on="date", how="outer")

    # 1) n-transactions
    if df_n_txn is not None and not df_n_txn.empty:
        df = ensure_sorted_date(df_n_txn, "date").copy()
        df = df.rename(columns={"value": "onchain_txn_level"})
        df = _add_basic_transforms(df, "onchain_txn_level", "onchain_txn")
        features = _merge(features, df)

    # 2) n-unique-addresses
    if df_n_unique_addr is not None and not df_n_unique_addr.empty:
        df = ensure_sorted_date(df_n_unique_addr, "date").copy()
        df = df.rename(columns={"value": "onchain_active_addr_level"})
        df = _add_basic_transforms(df, "onchain_active_addr_level", "onchain_active_addr")
        features = _merge(features, df)

    # 3) estimated-transaction-volume-usd
    if df_est_tx_volume_usd is not None and not df_est_tx_volume_usd.empty:
        df = ensure_sorted_date(df_est_tx_volume_usd, "date").copy()
        df = df.rename(columns={"value": "onchain_est_tx_volume_usd_level"})
        df = _add_basic_transforms(df, "onchain_est_tx_volume_usd_level", "onchain_est_tx_volume_usd")
        features = _merge(features, df)

    if features is None:
        raise RuntimeError("No on-chain data provided to build_onchain_features")

    features = ensure_sorted_date(features, "date")
    return features
