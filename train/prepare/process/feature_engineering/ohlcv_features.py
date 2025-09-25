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
    return df.reset_index()


def compute_ohlcv_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    # 원본 데이터프레임의 인덱스를 보존하기 위해 복사
    original_df = df.copy()
    # 피처 계산을 위한 기본 데이터 준비
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    
    # 계산된 피처(Series)를 저장할 리스트
    features = []

    if config.get("log_return"): 
        log_return = np.log(close / close.shift(1))
        log_return.name = "ohlcv_log_return"
        features.append(log_return)

    if config.get("sma"):
        for w in config["sma"]:
            sma = close.rolling(window=w).mean()
            sma.name = f"ohlcv_sma_{w}"
            features.append(sma)

    if config.get("ema"):
        for w in config["ema"]:
            ema = close.ewm(span=w, adjust=False).mean()
            ema.name = f"ohlcv_ema_{w}"
            features.append(ema)

    if config.get("macd"):
        macd_indicator = ta.trend.MACD(close=close)
        features.extend([
            macd_indicator.macd().rename("ohlcv_macd"),
            macd_indicator.macd_signal().rename("ohlcv_macd_signal"),
            macd_indicator.macd_diff().rename("ohlcv_macd_diff")
        ])

    if config.get("rsi"):
        for w in config["rsi"]:
            rsi = ta.momentum.RSIIndicator(close=close, window=w).rsi()
            rsi.name = f"ohlcv_rsi_{w}"
            features.append(rsi)

    if config.get("bbands"):
        for w in config["bbands"]:
            bb = ta.volatility.BollingerBands(close=close, window=w, window_dev=2)
            bb_h = bb.bollinger_hband()
            bb_l = bb.bollinger_lband()
            bb_w = bb_h - bb_l
            bb_p = (close - bb_l) / (bb_w + 1e-8)
            features.extend([
                bb_h.rename(f"ohlcv_bb_upper_{w}"),
                bb_l.rename(f"ohlcv_bb_lower_{w}"),
                bb_w.rename(f"ohlcv_bb_width_{w}"),
                bb_p.rename(f"ohlcv_bb_band_pos_{w}")
            ])

    if config.get("stoch"):
        for w in config["stoch"]:
            sto = ta.momentum.StochasticOscillator(high=high, low=low, close=close, window=w, smooth_window=3)
            features.extend([
                sto.stoch().rename(f"ohlcv_sto_k_{w}"),
                sto.stoch_signal().rename(f"ohlcv_sto_d_{w}")
            ])

    if config.get("atr"):
        for w in config["atr"]:
            atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=w).average_true_range()
            atr.name = f"ohlcv_atr_{w}"
            features.append(atr)

    if config.get("vwap"):
        vwap = (close * volume).cumsum() / (volume.cumsum() + 1e-8)
        vwap.name = "ohlcv_vwap"
        features.append(vwap)

    if config.get("heikin_ashi"):
        ha = pd.DataFrame(index=df.index)
        ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        ha["open"] = (df["open"].shift(1) + df["close"].shift(1)) / 2
        ha["high"] = df[["high", "open", "close"]].max(axis=1)
        ha["low"] = df[["low", "open", "close"]].min(axis=1)
        ha = ha.rename(columns=lambda x: f"ohlcv_heikin_ashi_{x}")
        features.extend([ha[col] for col in ha.columns])

    if config.get("ichimoku"):
        ichimoku = ta.trend.IchimokuIndicator(high=high, low=low, window1=9, window2=26, window3=52)
        features.extend([
            ichimoku.ichimoku_conversion_line().rename("ohlcv_ichimoku_tenkan"),
            ichimoku.ichimoku_base_line().rename("ohlcv_ichimoku_kijun"),
            ichimoku.ichimoku_a().shift(26).rename("ohlcv_ichimoku_span_a"),   # 선행 스팬 A
            ichimoku.ichimoku_b().shift(26).rename("ohlcv_ichimoku_span_b"),   # 선행 스팬 B
            close.shift(26).rename("ohlcv_ichimoku_chikou")                    # Chikou Span — 과거 26스텝으로 정렬
        ])

    # 원본 데이터프레임의 이름을 규칙에 맞게 변경
    original_df = original_df.rename(columns={
        "open": "ohlcv_open",
        "high": "ohlcv_high",
        "low": "ohlcv_low",
        "close": "ohlcv_close",
        "volume": "ohlcv_volume"
    })

    # 원본 + 모든 피처를 한 번에 병합 후 timestamp 인덱스 리셋
    final_df = pd.concat([original_df] + features, axis=1)
    return final_df.reset_index()