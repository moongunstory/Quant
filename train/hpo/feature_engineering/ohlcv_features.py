# train/prepare/process/ohlcv_features.py

import pandas as pd
import numpy as np
import ta
from ai_binance.config.paths import get_ohlcv_path


def load_ohlcv_data(symbol: str, filename: str = "ohlcv.csv") -> pd.DataFrame:
    path = get_ohlcv_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"❌ OHLCV 데이터 없음: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.resample("5min").mean()
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    return df


def compute_ohlcv_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()

    df["ohlcv_log_return"] = np.log(df["close"] / df["close"].shift(1))

    if config.get("sma"):
        for w in config["sma"]:
            if isinstance(w, int):
                df[f"ohlcv_sma_{w}"] = df["close"].rolling(window=w).mean()

    if config.get("ema"):
        for w in config["ema"]:
            if isinstance(w, int):
                df[f"ohlcv_ema_{w}"] = df["close"].ewm(span=w, adjust=False).mean()

    if config.get("macd", True):
        macd = ta.trend.MACD(close=df["close"])
        df["ohlcv_macd"] = macd.macd()
        df["ohlcv_macd_signal"] = macd.macd_signal()
        df["ohlcv_macd_diff"] = macd.macd_diff()

    if config.get("rsi"):
        for w in config["rsi"]:
            if isinstance(w, int):
                df[f"ohlcv_rsi_{w}"] = ta.momentum.RSIIndicator(close=df["close"], window=w).rsi()

    if config.get("bbands"):
        for w in config["bbands"]:
            if isinstance(w, int):
                bb = ta.volatility.BollingerBands(close=df["close"], window=w, window_dev=2)
                df[f"ohlcv_bb_upper_{w}"] = bb.bollinger_hband()
                df[f"ohlcv_bb_lower_{w}"] = bb.bollinger_lband()
                width = df[f"ohlcv_bb_upper_{w}"] - df[f"ohlcv_bb_lower_{w}"]
                df[f"ohlcv_bb_width_{w}"] = width
                df[f"ohlcv_bb_band_pos_{w}"] = (df["close"] - df[f"ohlcv_bb_lower_{w}"]) / (width + 1e-8)

    if config.get("stoch"):
        for w in config["stoch"]:
            if isinstance(w, int):
                sto = ta.momentum.StochasticOscillator(
                    high=df["high"], low=df["low"], close=df["close"], window=w, smooth_window=3)
                df[f"ohlcv_sto_k_{w}"] = sto.stoch()
                df[f"ohlcv_sto_d_{w}"] = sto.stoch_signal()

    if config.get("atr"):
        for w in config["atr"]:
            if isinstance(w, int):
                atr = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=w)
                df[f"ohlcv_atr_{w}"] = atr.average_true_range()

    if config.get("adr"):
        for w in config["adr"]:
            if isinstance(w, int):
                df["ohlcv_daily_range"] = df["high"] - df["low"]
                df[f"ohlcv_adr_{w}"] = df["ohlcv_daily_range"].rolling(window=288 * w).mean()

    if config.get("vwap", True):
        df["ohlcv_vwap"] = (df["close"] * df["volume"]).cumsum() / (df["volume"].cumsum() + 1e-8)

    if config.get("heikin_ashi", True):
        ha = pd.DataFrame(index=df.index)
        ha["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha["ha_open"] = (df["open"].shift(1) + df["close"].shift(1)) / 2
        ha["ha_high"] = pd.concat([df["high"], ha["ha_open"], ha["ha_close"]], axis=1).max(axis=1)
        ha["ha_low"] = pd.concat([df["low"], ha["ha_open"], ha["ha_close"]], axis=1).min(axis=1)
        df["ohlcv_ha_close"] = ha["ha_close"]
        df["ohlcv_ha_open"] = ha["ha_open"]
        df["ohlcv_ha_high"] = ha["ha_high"]
        df["ohlcv_ha_low"] = ha["ha_low"]

    if config.get("ichimoku", True):
        high = df["high"]
        low = df["low"]
        close = df["close"]

        period9_high = high.rolling(window=9).max()
        period9_low = low.rolling(window=9).min()
        df["ohlcv_ichimoku_tenkan"] = (period9_high + period9_low) / 2

        period26_high = high.rolling(window=26).max()
        period26_low = low.rolling(window=26).min()
        df["ohlcv_ichimoku_kijun"] = (period26_high + period26_low) / 2

        df["ohlcv_ichimoku_senkou_a"] = ((df["ohlcv_ichimoku_tenkan"] + df["ohlcv_ichimoku_kijun"]) / 2).shift(26)

        period52_high = high.rolling(window=52).max()
        period52_low = low.rolling(window=52).min()
        df["ohlcv_ichimoku_senkou_b"] = ((period52_high + period52_low) / 2).shift(26)

        df["ohlcv_ichimoku_chikou"] = close.shift(-26)

    if config.get("candlestick", True):
        df["ohlcv_candle_body"] = (df["close"] - df["open"]).abs()
        df["ohlcv_candle_upper_shadow"] = df["high"] - df[["close", "open"]].max(axis=1)
        df["ohlcv_candle_lower_shadow"] = df[["close", "open"]].min(axis=1) - df["low"]
        df["ohlcv_candle_body_ratio"] = df["ohlcv_candle_body"] / (df["high"] - df["low"] + 1e-8)

    if config.get("fibonacci", True):
        lookback = config.get("fib_lookback", 100)
        if isinstance(lookback, int):
            recent_high = df["high"].rolling(window=lookback).max()
            recent_low = df["low"].rolling(window=lookback).min()
            diff = recent_high - recent_low + 1e-8
            for lvl in [0.236, 0.382, 0.5, 0.618, 0.786]:
                name = f"{int(lvl * 1000)}/1000"
                fib = recent_high - diff * lvl
                df[f"ohlcv_fib_{name}"] = fib
                df[f"ohlcv_fib_dist_{name}"] = (df["close"] - fib) / (recent_high + 1e-8)

    df = df.rename(columns={
        "open": "ohlcv_open",
        "high": "ohlcv_high",
        "low": "ohlcv_low",
        "close": "ohlcv_close",
        "volume": "ohlcv_volume"
    })

    return df.reset_index()
