import os
import pandas as pd
import re
import requests
from datetime import timedelta
from binance.um_futures import UMFutures
from dotenv import load_dotenv

from modules.config import (
    BINANCE_API_KEY,
    BINANCE_SECRET_KEY,
    TIMEFRAMES,
    REQUIRED_CANDLE_COUNTS,
    BINANCE_INTERVAL_MAP,
    DUNE_API_KEY,
    CACHE_DIR,
    ONCHAIN_CACHE_DIR,
    TZ,
)

from modules.training.data_preparation.collector import add_indicators_with_validation, fetch_btc_historical_features
from modules.training.data_preparation.processor import create_dune_derived_features


client = UMFutures(key=BINANCE_API_KEY, secret=BINANCE_SECRET_KEY)


def fetch_ohlcv_from_binance(symbol, tf, now, count):
    end_time_utc = now.tz_convert("UTC")
    end_naive = end_time_utc.tz_localize(None)

    unit = "minutes" if "min" in tf.lower() else "hours"
    value = int(re.findall(r"\d+", tf)[0])
    start_time = end_time_utc - pd.Timedelta(**{unit: value * count})
    start_naive = start_time.tz_localize(None)

    api_tf = BINANCE_INTERVAL_MAP[tf]
    klines = client.klines(
        symbol=symbol,
        interval=api_tf,
        startTime=int(start_naive.timestamp() * 1000),
        endTime=int(end_naive.timestamp() * 1000),
        limit=count,
    )
    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(
        klines,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "num_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ],
    )

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.index = df.index.tz_convert("UTC")
    return df.astype(float).sort_index()


def update_cache(symbol, tf, new_df, cache_dir, max_len):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_{tf}.pkl")

    if new_df.index.tz is None:
        new_df.index = new_df.index.tz_localize("UTC")

    if os.path.exists(path):
        old_df = pd.read_pickle(path)
        if old_df.index.tz is None:
            old_df.index = old_df.index.tz_localize("UTC")
        combined = pd.concat([old_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_df

    combined = combined.sort_index().iloc[-max_len:]
    combined.to_pickle(path)
    return combined


def fetch_latest_dune_row():
    url = "https://api.dune.com/api/v1/query/5182378/results"
    headers = {"x-dune-api-key": DUNE_API_KEY}
    response = requests.get(url, headers=headers)
    data = response.json()

    if "result" not in data or "rows" not in data["result"]:
        return pd.DataFrame()

    df = pd.DataFrame(data["result"]["rows"])
    if df.empty:
        return pd.DataFrame()

    df["day"] = pd.to_datetime(df["day"], utc=True).dt.floor("D")
    df.set_index("day", inplace=True)
    df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    latest_index = df.index.max()
    return df.loc[[latest_index]]


class RealTimeDataCollector:
    def __init__(self):
        # 항상 직전 캔들까지만 요청하도록 함 (30분 단위 실행 기준)
        self.now = pd.Timestamp.utcnow().floor("30min") - pd.Timedelta(minutes=1)
        self.symbol = "ETHUSDT"
        self.btc_symbol = "BTCUSDT"
        self.cache_dir = CACHE_DIR

    def collect_eth_features(self):
        result_df = None
        unified_ts = self.now.floor("5min")  # 모든 타임프레임의 마지막 row 인덱스를 통일

        for tf in TIMEFRAMES:
            count = REQUIRED_CANDLE_COUNTS[tf]
            new_df = fetch_ohlcv_from_binance(self.symbol, tf, self.now, count)
            if new_df.empty:
                raise ValueError(f"{tf} Binance 응답 없음")

            value = int(re.findall(r"\d+", tf)[0])
            expected_last_ts = (
                (self.now.tz_convert("UTC") - pd.Timedelta(minutes=value)).floor(tf.lower())
            )

            if new_df.index.max() < expected_last_ts:
                raise ValueError(f"{tf} 캔들 누락됨: {new_df.index.max()} < {expected_last_ts}")

            updated_df = update_cache(self.symbol, tf, new_df, self.cache_dir, count)
            df_with_ind = add_indicators_with_validation(updated_df, tf)
            if tf == "5min":
                df_with_ind["5m_close"] = updated_df["close"]  # ✅ 진입가 용도
            latest_row = df_with_ind.iloc[[-1]]
            latest_row.index = pd.DatetimeIndex([unified_ts])  # ✅ 인덱스 통일

            result_df = latest_row if result_df is None else result_df.join(latest_row, how="outer")

        return result_df

    def collect_btc_features(self):
        start = (self.now.floor("30min") - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        end = self.now.floor("30min").strftime("%Y-%m-%d %H:%M:%S")
        df = fetch_btc_historical_features(start, end, interval="30m")

        print(f"[BTC] 수집된 데이터 수: {len(df)}")
        if not df.empty:
            print(f"[BTC] 최근 캔들 timestamp: {df.index.max()}")
            print(f"[BTC] 최근 행 결측 수:\n{df.tail(1).isna().sum()}")
            print(f"[BTC] 컬럼명:\n{df.columns.tolist()}")

        if df.empty:
            raise ValueError("BTC 데이터 없음")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        latest = df.loc[[df.index.max()]]
        latest.index = pd.DatetimeIndex([self.now.floor("5min")])  # ✅ 인덱스 강제 통일

        return latest

    def collect_dune_features(self):
        dune_raw = fetch_latest_dune_row()
        if dune_raw.empty:
            raise ValueError("Dune 결과 없음")

        dune_df = create_dune_derived_features(dune_raw)
        if dune_df.index.tz is None:
            dune_df.index = dune_df.index.tz_localize("UTC")

        dune_latest = dune_df.loc[[dune_df.index.max()]]
        dune_latest.index = pd.DatetimeIndex([self.now.floor("5min")])  # ✅ 인덱스 통일

        date_str = dune_latest.index.max().strftime("%m.%d")
        os.makedirs(ONCHAIN_CACHE_DIR, exist_ok=True)
        dune_latest.to_json(os.path.join(ONCHAIN_CACHE_DIR, f"{date_str}.json"), orient="records", date_format="iso")

        files = sorted(os.listdir(ONCHAIN_CACHE_DIR))
        if len(files) > 3:
            for f in files[:-3]:
                os.remove(os.path.join(ONCHAIN_CACHE_DIR, f))

        return dune_latest

    def get_recent_market_df(self, tf="5m", seq_len=32, horizon=4):
        """
        PPO reward 평가용 5분봉 캔들 36줄 반환
        - seq_len: 상태 관측용
        - horizon: 보상 평가용
        - tf: 사용 타임프레임 ('5m')
        """
        count = seq_len + horizon  # PPO 학습 기준 총 필요 수
        df = fetch_ohlcv_from_binance(self.symbol, tf, self.now, count)
        if df is None or df.empty or len(df) < count:
            raise ValueError(f"📉 {tf} 캔들 수 부족: {len(df)} < {count}")

        return df.iloc[-count:]

    def run(self):
        self.now = pd.Timestamp.utcnow().floor("30min") - pd.Timedelta(minutes=1)

        eth_df = self.collect_eth_features()
        btc_df = self.collect_btc_features()
        dune_df = self.collect_dune_features()

        final_df = eth_df.join(btc_df, how="left").join(dune_df, how="left")

        if final_df.isna().any().any():
            print("\n🚨 결측값 존재 - 추론 skip")
            print("🚨 결측 컬럼 목록:")
            print(final_df.isna().sum()[final_df.isna().sum() > 0])

            print("\n📋 마지막 추론 대상 행 (최신 데이터):")
            print(final_df.tail(1).T)

            return None

        return final_df.iloc[-1]
