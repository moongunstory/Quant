# ai_binance/live/realtime_ingest.py
"""
Realtime Ingest for ETHUSDT Futures (5m)
- REST 폴링 기반(간단/안정). 종가 '확정된' 5m 바만 내보냄.
- fe.py와 동일 피처 생성 → 저장된 scaler/feature_list로 정규화까지 수행.
- 최신 패킷을 Queue로 푸시:
    {"X": DataFrame(normalized), "close": Series, "funding": Series, "funding_rate": float, "ts": Timestamp}

조절은 아래 상수로. CLI 없음.
"""
from __future__ import annotations

import os
import time
import math
import json
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import requests
import joblib
from queue import Queue

# =====================
# 설정 (필요시만 수정)
# =====================
SYMBOL = "ETHUSDT"
INTERVAL = "5m"
BASE_URL = "https://fapi.binance.com"    # UM Futures (공식) 
KLINES_EP = "/fapi/v1/klines"            # 캔들 데이터
FUNDING_EP = "/fapi/v1/fundingRate"      # 펀딩 이력 (8h 이벤트)
# 참고: 프리미엄/마크가격 & 마지막 펀딩레이트/다음 시간 → /fapi/v1/premiumIndex (미사용) 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/ai_binance
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

WINDOW = 48                 # 관찰 윈도우(학습과 동일)
POLL_SEC = 2.0              # 폴링 주기(초). WS로 바꾸면 미사용.
BACKFILL_LIMIT = 1500       # 시작시 가져올 과거 바 개수 (~5.2일)
REQ_TIMEOUT = 15
MAX_RETRY = 5
RETRY_SLEEP = 1.0

# ---- 펀딩 분배 단위 ----
FUNDING_SPLIT = 96          # 8h / 5m = 96

# =====================
# 내부 유틸
# =====================

def _get_now_ms() -> int:
    return int(pd.Timestamp.utcnow().timestamp() * 1000)

def _ceil_to_interval_ms(ts_ms: int, interval: str = "5m") -> int:
    unit = interval[-1]
    n = int(interval[:-1])
    if unit == "m":
        step_ms = n * 60_000
    elif unit == "h":
        step_ms = n * 3_600_000
    else:
        raise ValueError("Unsupported interval")
    return ((ts_ms // step_ms) + 1) * step_ms

def _is_closed(close_time_ms: int) -> bool:
    return close_time_ms <= _get_now_ms() - 1000  # 1초 버퍼

def _request_with_retry(url: str, params: Dict) -> dict | list:
    retry = 0
    while True:
        try:
            r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception:
            retry += 1
            if retry >= MAX_RETRY:
                raise
            time.sleep(RETRY_SLEEP)

def _request_klines(symbol: str, interval: str, limit: int) -> list:
    url = BASE_URL + KLINES_EP
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    return _request_with_retry(url, params)

def _request_funding(symbol: str, limit: int = 200) -> list:
    """
    최근 펀딩 이력(8h 이벤트)을 가져온다.
    """
    url = BASE_URL + FUNDING_EP
    params = {"symbol": symbol, "limit": limit}
    return _request_with_retry(url, params)

def _klines_to_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    cols = [
        "Open_time","Open","High","Low","Close","Volume","Close_time",
        "Quote_asset_volume","Number_of_trades","Taker_buy_base",
        "Taker_buy_quote","Ignore"
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True)
    df["Close_time"] = pd.to_datetime(df["Close_time"], unit="ms", utc=True)
    df.set_index("Open_time", inplace=True)
    for c in ["Open","High","Low","Close","Volume","Quote_asset_volume","Taker_buy_base","Taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Number_of_trades"] = pd.to_numeric(df["Number_of_trades"], errors="coerce")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def _funding_to_df(rows: list) -> pd.DataFrame:
    """
    fundingRate API → DataFrame(rate: float, time index)
    - fundingRate는 문자열이므로 float 변환
    - fundingTime(ms) → UTC Timestamp
    """
    if not rows:
        return pd.DataFrame(columns=["rate"])
    df = pd.DataFrame(rows)
    # 레거시/문서간 키 케이스 차이를 가드
    time_key = "fundingTime" if "fundingTime" in df.columns else "funding_time"
    rate_key = "fundingRate" if "fundingRate" in df.columns else "funding_rate"
    df["ts"] = pd.to_datetime(df[time_key].astype(np.int64), unit="ms", utc=True)
    df["rate"] = pd.to_numeric(df[rate_key], errors="coerce")
    df = df[["ts","rate"]].dropna()
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    return df

# =====================
# 피처 & 정규화 (fe.py와 일치)
# =====================

def _featurize_5m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """훈련 시와 동일한 방식으로 5m, 15m, 1h, 4h 피처를 생성한다."""
    def _indicators_like_training(df: pd.DataFrame, itv: str) -> pd.DataFrame:
        x = df.copy()
        x[f"ret1_{itv}"] = x["Close"].pct_change()
        x[f"ret3_{itv}"] = x["Close"].pct_change(3)
        x[f"ret12_{itv}"] = x["Close"].pct_change(12)
        x[f"hlv_{itv}"] = (x["High"] - x["Low"]) / x["Close"].replace(0, np.nan)
        d = x["Close"].diff()
        up = d.clip(lower=0).rolling(14, min_periods=14).mean()
        dn = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = up / dn.replace(0, np.nan)
        x[f"rsi14_{itv}"] = 100 - 100 / (1 + rs)
        ema12 = x["Close"].ewm(span=12, adjust=False).mean()
        ema26 = x["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        x[f"macd_{itv}"] = macd
        x[f"macd_sig_{itv}"] = macd_sig
        x[f"macd_hist_{itv}"] = macd - macd_sig
        tr = np.maximum(
            x["High"] - x["Low"],
            np.maximum((x["High"] - x["Close"].shift()).abs(),
                       (x["Low"] - x["Close"].shift()).abs())
        )
        x[f"atr14_{itv}"] = tr.rolling(14, min_periods=14).mean()
        ma20 = x["Close"].rolling(20, min_periods=20).mean()
        sd20 = x["Close"].rolling(20, min_periods=20).std()
        x[f"bb_mid_{itv}"] = ma20
        x[f"bb_up_{itv}"] = ma20 + 2 * sd20
        x[f"bb_dn_{itv}"] = ma20 - 2 * sd20
        return x
    
    base_df = _indicators_like_training(df_5m, "5m")
    timeframes = {"15m": "15min", "1h": "1h", "4h": "4h"}
    agg_rules = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 
        'Volume': 'sum', 'Quote_asset_volume': 'sum', 
        'Number_of_trades': 'sum', 'Taker_buy_base': 'sum', 'Taker_buy_quote': 'sum'
    }
    for tf_name, tf_rule in timeframes.items():
        df_resampled = df_5m.resample(tf_rule).agg(agg_rules).dropna()
        if df_resampled.empty:
            continue
        tf_features = _indicators_like_training(df_resampled, tf_name)
        base_df = pd.merge_asof(
            base_df.sort_index(),
            tf_features.sort_index().add_suffix(f"_{tf_name}"),
            left_index=True,
            right_index=True,
            direction="backward"
        )
    return base_df

def _normalize(df_feat: pd.DataFrame, scaler, feature_cols: List[str]) -> pd.DataFrame:
    X = df_feat.select_dtypes(include=["float64","float32","int64","int32"]).copy()
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols]
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    if X.isnull().any().any():
        X = X.fillna(0)
    arr = scaler.transform(X)
    return pd.DataFrame(arr, index=df_feat.index, columns=feature_cols)

# =====================
# 펀딩 분배 시리즈 생성
# =====================

def _distribute_funding_to_5m(index_5m: pd.DatetimeIndex, funding_df: pd.DataFrame) -> pd.Series:
    """
    funding_df: index= funding event time (UTC), column 'rate' (decimal, e.g., 0.0001)
    반환: index_5m에 정렬된 per-bar funding_cost = rate/FUNDING_SPLIT
    룰: 이벤트 시각 t_i 의 rate_i 는 구간 (t_{i-1}, t_i] 에 균등 분배.
        마지막 이벤트 이후 현재까지도 rate_last/FUNDING_SPLIT로 추정 분배.
    """
    s = pd.Series(0.0, index=index_5m, dtype=float)
    if funding_df is None or funding_df.empty or len(index_5m) == 0:
        return s

    events = funding_df.sort_index()
    # 시작 구간 커버를 위해, 인덱스 시작 이전의 가장 최근 이벤트도 포함되도록 트림
    events = events.loc[events.index <= index_5m[-1]]
    if events.empty:
        return s

    # 첫 구간 시작점: 인덱스 시작보다 과거의 이벤트가 없다면, 첫 이벤트 rate를 인덱스 시작~첫 이벤트까지 분배
    prev_time = index_5m[0] - pd.Timedelta(minutes=5)  # 경계 보정
    prev_rate = float(events.iloc[0]["rate"])
    for t_event, row in events.iterrows():
        rate = float(row["rate"])
        mask = (index_5m > prev_time) & (index_5m <= t_event)
        if mask.any():
            s.loc[mask] = rate / FUNDING_SPLIT
        prev_time = t_event
        prev_rate = rate

    # 마지막 이벤트 이후 구간: 마지막 rate로 추정 분배
    mask_tail = (index_5m > prev_time)
    if mask_tail.any():
        s.loc[mask_tail] = prev_rate / FUNDING_SPLIT

    return s

# =====================
# 인젝트 클래스
# =====================

class RealtimeIngest:
    """
    run()을 호출하면, 확정된 최신 5m 캔들이 생길 때마다 Queue로 패킷을 푸시한다.
    packet = {
        "X": normalized_features_df,
        "close": close_series,
        "funding": per_bar_funding_series,      # 5m 분배 시리즈 (rate/96)
        "funding_rate": float,                  # 최신 바의 분배 단위
        "ts": latest_open_timestamp
    }
    """
    def __init__(self, out_queue: Queue, symbol: str = SYMBOL, interval: str = INTERVAL):
        self.q = out_queue
        self.symbol = symbol
        self.interval = interval
        self.scaler = joblib.load(os.path.join(PROC_DIR, "scaler.joblib"))
        self.features = json.load(open(os.path.join(PROC_DIR, "feature_list.json"), "r"))
        # 초기 백필
        self.df = self._backfill()
        # 펀딩 캐시(최근 이력)
        self.funding_df = self._backfill_funding()

    def _backfill(self) -> pd.DataFrame:
        rows = _request_klines(self.symbol, self.interval, BACKFILL_LIMIT)
        df = _klines_to_df(rows)
        if df.empty:
            raise RuntimeError("과거 데이터 로드 실패: klines가 비어있습니다")
        return df

    def _backfill_funding(self) -> pd.DataFrame:
        try:
            rows = _request_funding(self.symbol, limit=200)
            return _funding_to_df(rows)
        except Exception as e:
            print(f"[수집기] 펀딩 이력 로드 실패(초기): {e}")
            return pd.DataFrame(columns=["rate"])

    def _refresh_funding(self) -> None:
        """최근 펀딩 이벤트 몇 개만 갱신(가벼움)"""
        try:
            rows = _request_funding(self.symbol, limit=50)
            df_new = _funding_to_df(rows)
            if df_new.empty:
                return
            if self.funding_df is None or self.funding_df.empty:
                self.funding_df = df_new
            else:
                # 인덱스 병합 후 최신만 남김
                self.funding_df = (
                    pd.concat([self.funding_df, df_new])
                    .sort_index()
                    .drop_duplicates(keep="last")
                )
                # 메모리 가드: 최근 400개만 유지(약 133일치)
                self.funding_df = self.funding_df.iloc[-400:]
        except Exception as e:
            print(f"[수집기] 펀딩 이력 갱신 실패: {e}")

    def _emit_if_closed_bar(self) -> Optional[pd.Timestamp]:
        if self.df.empty:
            return None
        latest_open = self.df.index[-1]
        latest_close_time = int(self.df.iloc[-1]["Close_time"].value // 10**6)  # ns→ms
        if not _is_closed(latest_close_time):
            return None

        # 피처 & 정규화
        feat = _featurize_5m(self.df)
        X = _normalize(feat, self.scaler, self.features)
        close = self.df["Close"].reindex(X.index).ffill().bfill()

        # 펀딩 시리즈(5m 분배)
        funding_series = _distribute_funding_to_5m(X.index, self.funding_df)
        funding_series = funding_series.reindex(X.index).fillna(0.0)
        current_funding = float(funding_series.iloc[-1]) if len(funding_series) else 0.0

        packet = {"X": X, "close": close, "funding": funding_series, "funding_rate": current_funding, "ts": latest_open}
        try:
            if self.q.full():
                self.q.get_nowait()
            self.q.put_nowait(packet)
        except Exception:
            pass
        return latest_open

    def _fetch_tail(self) -> None:
        # 캔들 테일 갱신
        rows = _request_klines(self.symbol, self.interval, 200)
        tail = _klines_to_df(rows)
        if not tail.empty:
            df = pd.concat([self.df.iloc[:-200], tail]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
            self.df = df
        # 펀딩 이력도 짧게 갱신
        self._refresh_funding()

    def run(self) -> None:
        print(f"[수집기] 심볼={self.symbol} 인터벌={self.interval} (REST 폴링) | 백필={len(self.df)}")
        last_emitted_ts = None
        
        emitted_ts = self._emit_if_closed_bar()
        if emitted_ts:
            last_emitted_ts = emitted_ts
            print(f"[수집기] 초기 확정봉 전송: {emitted_ts}")

        while True:
            try:
                self._fetch_tail()
                latest_open_ts = self.df.index[-1]
                if last_emitted_ts is None or latest_open_ts > last_emitted_ts:
                    emitted_ts = self._emit_if_closed_bar()
                    if emitted_ts:
                        last_emitted_ts = emitted_ts
                        print(f"[수집기] 새 확정봉 전송: {emitted_ts}")
                time.sleep(POLL_SEC)
            except KeyboardInterrupt:
                print("[수집기] 사용자에 의해 중지됨")
                break
            except Exception as e:
                print(f"[수집기] 경고: {e}")
                time.sleep(5.0)
