# pipeline/processors/binance.py

from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

from .utils import ensure_sorted_datetime, rsi


def build_binance_features(
    df_fut_1h: pd.DataFrame,
    df_spot_1h: pd.DataFrame,
    df_oi_1h: Optional[pd.DataFrame] = None,
    df_ls_1h: Optional[pd.DataFrame] = None,
    df_funding: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Binance raw 데이터들을 합쳐서 1h 피처 테이블 생성.

    입력:
        df_fut_1h  : futures ohlcv 1h
        df_spot_1h : spot ohlcv 1h
        df_oi_1h   : open interest 1h
        df_ls_1h   : long/short ratio 1h
        df_funding : funding rate (원래 8h)

    반환:
        timestamp 기준 정렬된 feature DataFrame
    """
    # ----- 1) 기본 정리 -----
    fut = ensure_sorted_datetime(df_fut_1h, "timestamp").copy()
    spot = ensure_sorted_datetime(df_spot_1h, "timestamp").copy()

    fut = fut.rename(
        columns={
            "open": "fut_open",
            "high": "fut_high",
            "low": "fut_low",
            "close": "fut_close",
            "volume": "fut_volume",
            "taker_buy_base": "fut_taker_buy_base",
            "quote_volume": "fut_quote_volume",
            "taker_buy_ratio": "fut_taker_buy_ratio",
        }
    )

    spot = spot.rename(
        columns={
            "open": "spot_open",
            "high": "spot_high",
            "low": "spot_low",
            "close": "spot_close",
            "volume": "spot_volume",
        }
    )

    spot = spot[
        ["timestamp", "spot_open", "spot_high", "spot_low", "spot_close", "spot_volume"]
    ].copy()

    df = fut.merge(spot, on="timestamp", how="left")

    # ----- 2) 수익률 / 변동성 -----
    df["ret_1h"] = df["fut_close"].pct_change()
    df["log_ret_1h"] = np.log(df["fut_close"] / df["fut_close"].shift(1))
    df["ret_4h"] = df["fut_close"].pct_change(4)
    df["ret_24h"] = df["fut_close"].pct_change(24)

    df["vol_24h"] = df["log_ret_1h"].rolling(24).std()
    df["vol_7d"] = df["log_ret_1h"].rolling(24 * 7).std()

    # ----- 3) MA / Bollinger / RSI -----
    for win in (24, 168):  # 1일, 7일
        df[f"ma_{win}h"] = df["fut_close"].rolling(win).mean()

    df["ma_24h_gap"] = df["fut_close"] / df["ma_24h"] - 1

    win = 20
    ma20 = df["fut_close"].rolling(win).mean()
    std20 = df["fut_close"].rolling(win).std()

    df["bb_upper_20"] = ma20 + 2 * std20
    df["bb_lower_20"] = ma20 - 2 * std20
    df["bb_width_20"] = (df["bb_upper_20"] - df["bb_lower_20"]) / ma20

    df["rsi_14"] = rsi(df["fut_close"], 14)

    # ----- 4) Futures - Spot 베이시스 -----
    df["basis_pct"] = (df["fut_close"] - df["spot_close"]) / df["spot_close"]

    # ----- 5) Open Interest -----
    if df_oi_1h is not None and not df_oi_1h.empty:
        oi = ensure_sorted_datetime(df_oi_1h, "timestamp")
        oi = oi[["timestamp", "open_interest"]].copy()

        df = df.merge(oi, on="timestamp", how="left")
        df["oi_pct_change_1h"] = df["open_interest"].pct_change()

        roll = df["open_interest"].rolling(24 * 30)  # 30일
        df["oi_z_30d"] = (df["open_interest"] - roll.mean()) / roll.std()

    # ----- 6) Long / Short Ratio -----
    if df_ls_1h is not None and not df_ls_1h.empty:
        ls = ensure_sorted_datetime(df_ls_1h, "timestamp")
        ls = ls.rename(
            columns={
                "long_short_ratio": "ls_ratio",
                "long_account": "ls_long_account",
                "short_account": "ls_short_account",
            }
        )
        keep = ["timestamp", "ls_ratio", "ls_long_account", "ls_short_account"]
        ls = ls[[c for c in keep if c in ls.columns]]

        df = df.merge(ls, on="timestamp", how="left")

    # ----- 7) Funding (8h → 1h ffill) -----
    if df_funding is not None and not df_funding.empty:
        fr = ensure_sorted_datetime(df_funding, "timestamp")
        fr_1h = fr.set_index("timestamp").resample("1h").ffill().reset_index()
        df = df.merge(fr_1h, on="timestamp", how="left")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
