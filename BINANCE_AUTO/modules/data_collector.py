import os
import pandas as pd
import re
import requests
import json
from datetime import timedelta
from binance.client import Client
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

from modules.training.data_preparation.collector import fetch_btc_historical_features
from modules.training.data_preparation.processor import create_dune_derived_features


client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)


def fetch_ohlcv_from_binance(symbol, tf, now, count):

    api_tf = BINANCE_INTERVAL_MAP[tf]
    klines = client.futures_klines(
        symbol=symbol,
        interval=api_tf,
        limit=count  # 가장 최근 count개만 요청
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
    df = df.astype(float).sort_index()
    print(
        f"[DEBUG] {tf} OHLCV count = {len(df)} / index range: {df.index.min()} ~ {df.index.max()}"
    )
    return df


def update_cache(symbol, tf, new_df, cache_dir, max_len):
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_{tf}.pkl")

    if new_df.index.tz is None:
        new_df.index = new_df.index.tz_localize("UTC")

    use_old_cache = True
    if os.path.exists(path):
        old_df = pd.read_pickle(path)
        if old_df.index.tz is None:
            old_df.index = old_df.index.tz_localize("UTC")

        # 최신성 비교
        old_max = old_df.index.max()
        new_max = new_df.index.max()

        # 기준: old가 new보다 1 interval 이상 과거면 버린다
        tf_minutes = int(re.findall(r"\d+", tf)[0]) if "min" in tf else int(re.findall(r"\d+", tf)[0]) * 60
        max_allowed_delay = pd.Timedelta(minutes=tf_minutes)
        
        if old_max < new_max - max_allowed_delay:
            print(f"[INFO] {tf} 캐시가 너무 구식입니다 → 무시하고 새로 덮어씁니다.")
            use_old_cache = False

    if use_old_cache and os.path.exists(path):
        combined = pd.concat([old_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        combined = new_df

    combined = combined.sort_index().iloc[-max_len:]
    print(f"[DEBUG] {tf} cache combined rows: {len(combined)} / max index: {combined.index.max()}")
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


def add_indicators_for_live(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
    
    df = df.copy()
    
    if 'close' not in df.columns or df['close'].isna().all():
        return pd.DataFrame()

    prefix = f"{tf}_" if tf != "5min" else ""

    df[f"{prefix}rsi"] = RSIIndicator(df['close']).rsi()
    df[f"{prefix}stoch_k"] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    
    macd = MACD(df['close'])
    df[f"{prefix}macd"] = macd.macd()
    df[f"{prefix}macd_signal"] = macd.macd_signal()
    
    df[f"{prefix}ema_20"] = EMAIndicator(df['close'], window=20).ema_indicator()
    sma_col = f"{prefix}sma_50"
    df[sma_col] = SMAIndicator(df['close'], window=50).sma_indicator()
    df[f"{prefix}adx"] = ADXIndicator(df['high'], df['low'], df['close']).adx()
    print(
        f"[DEBUG] {sma_col} NaNs: {df[sma_col].isna().sum()} / Valid: {df[sma_col].notna().sum()}"
    )
    
    df = df.ffill().bfill()
    
    return df


class RealTimeDataCollector:
    def __init__(self):
        self.now = pd.Timestamp.utcnow().floor("30min") - pd.Timedelta(minutes=1)
        self.symbol = "ETHUSDT"
        self.btc_symbol = "BTCUSDT"
        self.cache_dir = CACHE_DIR

    def collect_btc_features(self):
        start = (self.now.floor("30min") - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
        end = self.now.floor("30min").strftime("%Y-%m-%d %H:%M:%S")
        df = fetch_btc_historical_features(start, end, interval="30m")

        if df.empty:
            raise ValueError("BTC 데이터 없음")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        latest = df.loc[[df.index.max()]]
        latest.index = pd.DatetimeIndex([self.now.floor("5min")])
        return latest

    def collect_dune_features(self):
        dune_raw = fetch_latest_dune_row()
        if dune_raw.empty:
            raise ValueError("Dune 결과 없음")

        dune_df = create_dune_derived_features(dune_raw)
        if dune_df.index.tz is None:
            dune_df.index = dune_df.index.tz_localize("UTC")

        dune_latest = dune_df.loc[[dune_df.index.max()]]
        dune_latest.index = pd.DatetimeIndex([self.now.floor("5min")])

        date_str = dune_latest.index.max().strftime("%m.%d")
        os.makedirs(ONCHAIN_CACHE_DIR, exist_ok=True)
        dune_latest.to_json(os.path.join(ONCHAIN_CACHE_DIR, f"{date_str}.json"), orient="records", date_format="iso")

        files = sorted(os.listdir(ONCHAIN_CACHE_DIR))
        if len(files) > 3:
            for f in files[:-3]:
                os.remove(os.path.join(ONCHAIN_CACHE_DIR, f))

        return dune_latest

    def get_recent_market_df(self, tf="5m", seq_len=32, horizon=4):
        count = seq_len + horizon
        df = fetch_ohlcv_from_binance(self.symbol, tf, self.now, count)
        if df is None or df.empty or len(df) < count:
            raise ValueError(f"📉 {tf} 캔들 수 부족: {len(df)} < {count}")

        return df.iloc[-count:]

    def refresh_caches(self):
        print("[디버그] 🔄 캐시 새로고침 중...")
        for tf in TIMEFRAMES:
            new_df = fetch_ohlcv_from_binance(self.symbol, tf, self.now, REQUIRED_CANDLE_COUNTS[tf])
            if not new_df.empty:
                update_cache(self.symbol, tf, new_df, self.cache_dir, REQUIRED_CANDLE_COUNTS[tf])

    def run(self, seq_len=32, auto_refresh_cache=False):
        print(f"[디버그] 🕒 추론 기준 시간: {self.now}")
        
        self.now = pd.Timestamp.utcnow().floor("30min") - pd.Timedelta(minutes=1)
        unified_ts = self.now.floor("5min")

        if auto_refresh_cache:
            self.refresh_caches()

        # 1. 5분봉 기준 로드
        path_5m = os.path.join(self.cache_dir, "ETHUSDT_5min.pkl")
        if not os.path.exists(path_5m):
            print(f"🚨 5분봉 캐시 누락: {path_5m}")
            return None

        df_5m = pd.read_pickle(path_5m).sort_index()
        df_5m = add_indicators_for_live(df_5m, "5min")
        df_5m = df_5m.loc[df_5m.index <= unified_ts].iloc[-seq_len:]

        if len(df_5m) < seq_len:
            print(f"🚨 5분봉 시퀀스 부족: {len(df_5m)} < {seq_len}")
            return None

        base_df = df_5m.copy()

        # 2. 멀티 타임프레임 지표 추가
        for tf in ["15min", "30min", "1H"]:
            path_tf = os.path.join(self.cache_dir, f"ETHUSDT_{tf}.pkl")
            if not os.path.exists(path_tf):
                print(f"🚨 {tf} 캐시 누락: {path_tf}")
                return None

            df_tf = pd.read_pickle(path_tf).sort_index()
            df_tf = add_indicators_for_live(df_tf, tf)

            # 미래 인덱스 체크
            max_index = df_tf.index.max()
            if max_index > base_df.index.max():
                print(f"🚨 {tf} 미래 인덱스 포함: {max_index} > {base_df.index.max()}")

            # Forward fill
            df_tf = df_tf[df_tf.index <= base_df.index.max()]
            df_tf = df_tf.reindex(base_df.index, method='ffill')
            
            # 컬럼 충돌 방지
            for col in df_tf.columns:
                if col not in base_df.columns:
                    base_df[col] = df_tf[col]

        # 3. BTC 및 Dune 피처 추가
        btc_row = self.collect_btc_features().iloc[0]
        dune_row = self.collect_dune_features().iloc[0]
        
        for col in btc_row.index:
            if col not in base_df.columns:
                base_df[col] = btc_row[col]
        for col in dune_row.index:
            if col not in base_df.columns:
                base_df[col] = dune_row[col]

        # 4. 피처 검증
        total_cols = len(base_df.columns)
        unique_cols = len(set(base_df.columns))
        print(f"[디버그] 전체 피처: {total_cols}, 고유 피처: {unique_cols}")

        duplicates = base_df.columns[base_df.columns.duplicated()].tolist()
        if duplicates:
            print(f"🚨 중복 피처명: {duplicates}")
            base_df = base_df.loc[:, ~base_df.columns.duplicated(keep='last')]

        # 5. 피처 순서 검증
        try:
            with open("trained_feature_order.json") as f:
                trained_order = json.load(f)
                current_order = base_df.columns.tolist()
                if current_order != trained_order:
                    print("🚨 피처 순서 불일치 → 재정렬 시도")
                    base_df = base_df[trained_order]
        except:
            print("⚠️ 피처 순서 검증 실패")

        # 6. 결측값 체크
        if base_df.isna().any().any():
            print("🚨 결측값 발견 → 건너뛰기")
            print(base_df.isna().sum()[base_df.isna().sum() > 0])
            return None

        print(f"[디버그] ✅ 최종 상태 형태: {base_df.shape}")
        return base_df