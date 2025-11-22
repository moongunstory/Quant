#!/usr/bin/env python
from __future__ import annotations

# pipeline/builder.py

import os
import sys

# === 프로젝트 루트 경로 자동 설정 ===
# 현재 파일: Quant/pipeline/builder.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../Quant/pipeline
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)                # .../Quant

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
from typing import Dict
import pandas as pd

from pipeline.processors import (
    binance as binance_mod,
    onchain as onchain_mod,
    macro as macro_mod,
    derivatives as derivatives_mod,
    news as news_mod
)


RAW_DEFAULT = Path("data/raw")
PROCESSED_DEFAULT = Path("data/processed")


def _read_parquet_safe(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty:
        return None
    return df


# ==================== Binance ====================

def build_binance(raw_root: Path = RAW_DEFAULT,
                  processed_root: Path = PROCESSED_DEFAULT,
                  save: bool = True) -> pd.DataFrame:
    bdir = raw_root / "binance"

    df_fut = _read_parquet_safe(bdir / "ohlcv_futures_1h.parquet")
    df_spot = _read_parquet_safe(bdir / "ohlcv_spot_1h.parquet")
    if df_fut is None or df_spot is None:
        raise FileNotFoundError("Binance futures/spot parquet not found or empty")

    df_oi = _read_parquet_safe(bdir / "oi_1h.parquet")
    df_ls = _read_parquet_safe(bdir / "ls_ratio_top_1h.parquet")
    df_fr = _read_parquet_safe(bdir / "funding_rate.parquet")

    feat = binance_mod.build_binance_features(
        df_fut_1h=df_fut,
        df_spot_1h=df_spot,
        df_oi_1h=df_oi,
        df_ls_1h=df_ls,
        df_funding=df_fr,
    )

    if save:
        processed_root.mkdir(parents=True, exist_ok=True)
        out = processed_root / "binance_features_1h.parquet"
        feat.to_parquet(out, index=False, compression="snappy")

    return feat


# ==================== On-chain ====================

def build_onchain(raw_root: Path = RAW_DEFAULT,
                  processed_root: Path = PROCESSED_DEFAULT,
                  save: bool = True) -> pd.DataFrame:
    odir = raw_root / "onchain"

    df_n_txn = _read_parquet_safe(odir / "blockchain_com_n-transactions.parquet")
    df_n_unique = _read_parquet_safe(odir / "blockchain_com_n-unique-addresses.parquet")
    df_est_vol = _read_parquet_safe(
        odir / "blockchain_com_estimated-transaction-volume-usd.parquet"
    )

    feat = onchain_mod.build_onchain_features(
        df_n_txn=df_n_txn,
        df_n_unique_addr=df_n_unique,
        df_est_tx_volume_usd=df_est_vol,
    )

    if save:
        processed_root.mkdir(parents=True, exist_ok=True)
        out = processed_root / "onchain_features_daily.parquet"
        feat.to_parquet(out, index=False, compression="snappy")

    return feat


# ==================== Macro ====================

def build_macro(raw_root: Path = RAW_DEFAULT,
                processed_root: Path = PROCESSED_DEFAULT,
                save: bool = True) -> pd.DataFrame:
    mdir = raw_root / "macro"

    fred_series: Dict[str, pd.DataFrame] = {}
    yahoo_series: Dict[str, pd.DataFrame] = {}
    fx_series: Dict[str, pd.DataFrame] = {}

    # FRED
    for path in mdir.glob("fred_*.parquet"):
        sid = path.stem.replace("fred_", "").lower()
        df = _read_parquet_safe(path)
        if df is not None:
            fred_series[sid] = df

    # Yahoo
    for path in mdir.glob("yahoo_*.parquet"):
        sym = path.stem.replace("yahoo_", "").lower()
        df = _read_parquet_safe(path)
        if df is not None:
            yahoo_series[sym] = df

    # FX
    for path in mdir.glob("fx_*.parquet"):
        sym = path.stem.replace("fx_", "").lower()
        df = _read_parquet_safe(path)
        if df is not None:
            fx_series[sym] = df

    feat = macro_mod.build_macro_features(
        fred_series=fred_series,
        yahoo_series=yahoo_series,
        fx_series=fx_series,
    )

    if save:
        processed_root.mkdir(parents=True, exist_ok=True)
        out = processed_root / "macro_features_daily.parquet"
        feat.to_parquet(out, index=False, compression="snappy")

    return feat


# ==================== Derivatives (DVOL) ====================

def build_derivatives(raw_root: Path = RAW_DEFAULT,
                      processed_root: Path = PROCESSED_DEFAULT,
                      save: bool = True) -> pd.DataFrame:
    ddir = raw_root / "derivatives"
    df_dvol = _read_parquet_safe(ddir / "deribit_btc_dvol.parquet")
    if df_dvol is None:
        raise FileNotFoundError("deribit_btc_dvol.parquet not found or empty")

    feat = derivatives_mod.build_derivatives_features(df_dvol)

    if save:
        processed_root.mkdir(parents=True, exist_ok=True)
        out = processed_root / "derivatives_features_daily.parquet"
        feat.to_parquet(out, index=False, compression="snappy")

    return feat


# ==================== News ====================

def build_news(raw_root: Path = RAW_DEFAULT,
               processed_root: Path = PROCESSED_DEFAULT,
               save: bool = True) -> pd.DataFrame:
    ndir = raw_root / "news"
    df_news = _read_parquet_safe(ndir / "news_raw.parquet")
    if df_news is None:
        raise FileNotFoundError("news_raw.parquet not found or empty")

    feat = news_mod.build_news_features(df_news)

    if save:
        processed_root.mkdir(parents=True, exist_ok=True)
        out = processed_root / "news_features_daily.parquet"
        feat.to_parquet(out, index=False, compression="snappy")

    return feat


# ==================== MASTER (1h 레벨 병합) ====================

def build_master_1h(
    processed_root: Path = PROCESSED_DEFAULT,
    save: bool = True,
) -> pd.DataFrame:
    """
    1) binance 1h 피처 읽고
    2) onchain/macro/derivatives/news/sentiment daily 피처들을
       'date' 기준으로 붙여서
    3) 최종 1h 마스터 테이블 생성
    """

    # ---- 1) 필수: binance 1h ----
    binance_path = processed_root / "binance_features_1h.parquet"
    binance = pd.read_parquet(binance_path)
    binance = binance.copy()
    binance["timestamp"] = pd.to_datetime(binance["timestamp"])
    binance = binance.sort_values("timestamp").reset_index(drop=True)

    # 1h → 일자 키
    binance["date"] = binance["timestamp"].dt.normalize()

    master = binance

    # ---- 2) daily 피처들 읽기 ----
    daily_files = {
        "onchain": processed_root / "onchain_features_daily.parquet",
        "macro": processed_root / "macro_features_daily.parquet",
        "derivatives": processed_root / "derivatives_features_daily.parquet",
        "news": processed_root / "news_features_daily.parquet",
        "sentiment": processed_root / "sentiment_features_daily.parquet",
    }

    for name, path in daily_files.items():
        if not path.exists():
            # 아직 안 만든 건 그냥 건너뜀
            continue

        df = pd.read_parquet(path)
        if df.empty:
            continue

        df = df.copy()
        if "date" not in df.columns:
            raise RuntimeError(f"{name} features parquet has no 'date' column: {path}")

        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)

        # 이미 'date' 를 공유하니까 그냥 머지해도 됨 (동일한 키)
        master = master.merge(df, on="date", how="left")

    # ---- 3) 정리 ----
    master = master.sort_values("timestamp").reset_index(drop=True)

    if save:
        out_path = processed_root / "master_features_1h.parquet"
        processed_root.mkdir(parents=True, exist_ok=True)
        master.to_parquet(out_path, index=False, compression="snappy")

    return master


# ==================== ALL ====================

def build_all_features(
    raw_root: Path = RAW_DEFAULT,
    processed_root: Path = PROCESSED_DEFAULT,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    전체 피처 한 번에 빌드.

    반환:
        {
          "binance": df_binance_1h,
          "onchain": df_onchain_daily,
          "macro": df_macro_daily,
          "derivatives": df_derivatives_daily,
          "news": df_news_daily,
          "sentiment": df_sentiment_daily,
        }
    """
    out: dict[str, pd.DataFrame] = {}

    out["binance"] = build_binance(raw_root, processed_root, save)
    out["onchain"] = build_onchain(raw_root, processed_root, save)
    out["macro"] = build_macro(raw_root, processed_root, save)
    out["derivatives"] = build_derivatives(raw_root, processed_root, save)
    out["news"] = build_news(raw_root, processed_root, save)
    out["master_1h"] = build_master_1h(processed_root, save)

    return out
