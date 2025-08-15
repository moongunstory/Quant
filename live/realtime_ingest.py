# ai_binance/live/realtime_ingest.py
"""
Realtime Ingest for ETHUSDT Futures (5m)
- REST 폴링 기반(간단/안정). 종가 '확정된' 5m 바만 내보냄.
- fe.py와 동일 피처 생성 → 저장된 scaler/feature_list로 정규화까지 수행.
- 최신 패킷을 Queue로 푸시: {"X": DataFrame(normalized), "close": Series, "ts": Timestamp}

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
BASE_URL = "https://fapi.binance.com"    # UM Futures
KLINES_EP = "/fapi/v1/klines"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/ai_binance
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

WINDOW = 48                 # 관찰 윈도우(학습과 동일)
POLL_SEC = 2.0              # 폴링 주기(초). WS로 바꾸면 미사용.
BACKFILL_LIMIT = 1500       # 시작시 가져올 과거 바 개수
REQ_TIMEOUT = 15
MAX_RETRY = 5
RETRY_SLEEP = 1.0

# =====================
# 내부 유틸
# =====================

def _get_now_ms() -> int:
    return int(pd.Timestamp.utcnow().timestamp() * 1000)

def _ceil_to_interval_ms(ts_ms: int, interval: str = "5m") -> int:
    # 캔들 close_time(미래) 계산
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
    # close_time이 현재시각보다 과거면 확정
    return close_time_ms <= _get_now_ms() - 1000  # 1초 버퍼

def _request_klines(symbol: str, interval: str, limit: int) -> list:
    url = BASE_URL + KLINES_EP
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    retry = 0
    while True:
        try:
            r = requests.get(url, params=params, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            retry += 1
            if retry >= MAX_RETRY:
                raise
            time.sleep(RETRY_SLEEP)

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
    # 중복 제거(드물지만 가드)
    df = df[~df.index.duplicated(keep="last")]
    return df

# =====================
# 피처 & 정규화 (fe.py와 일치)
# =====================

def _featurize_5m(df_5m: pd.DataFrame) -> pd.DataFrame:
    """훈련 시와 동일한 방식으로 5m, 15m, 1h, 4h 피처를 생성한다."""
    
    def _indicators_like_training(df: pd.DataFrame, itv: str) -> pd.DataFrame:
        """훈련 시와 완전히 동일한 피처명 생성"""
        x = df.copy()
        # Returns
        x[f"ret1_{itv}"] = x["Close"].pct_change()
        x[f"ret3_{itv}"] = x["Close"].pct_change(3)
        x[f"ret12_{itv}"] = x["Close"].pct_change(12)
        # Volatility proxy
        x[f"hlv_{itv}"] = (x["High"] - x["Low"]) / x["Close"].replace(0, np.nan)
        # RSI(14)
        d = x["Close"].diff()
        up = d.clip(lower=0).rolling(14, min_periods=14).mean()
        dn = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
        rs = up / dn.replace(0, np.nan)
        x[f"rsi14_{itv}"] = 100 - 100 / (1 + rs)
        # MACD(12,26,9)
        ema12 = x["Close"].ewm(span=12, adjust=False).mean()
        ema26 = x["Close"].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        x[f"macd_{itv}"] = macd
        x[f"macd_sig_{itv}"] = macd_sig
        x[f"macd_hist_{itv}"] = macd - macd_sig
        # ATR(14)
        tr = np.maximum(
            x["High"] - x["Low"],
            np.maximum((x["High"] - x["Close"].shift()).abs(),
                       (x["Low"] - x["Close"].shift()).abs())
        )
        x[f"atr14_{itv}"] = tr.rolling(14, min_periods=14).mean()
        # Bollinger Bands(20)
        ma20 = x["Close"].rolling(20, min_periods=20).mean()
        sd20 = x["Close"].rolling(20, min_periods=20).std()
        x[f"bb_mid_{itv}"] = ma20
        x[f"bb_up_{itv}"] = ma20 + 2 * sd20
        x[f"bb_dn_{itv}"] = ma20 - 2 * sd20
        return x
    
    # 1. 5분봉 피처 생성 (훈련과 동일)
    base_df = _indicators_like_training(df_5m, "5m")
    
    # 2. 상위 타임프레임 리샘플링 + 피처 생성
    timeframes = {"15m": "15min", "1h": "1h", "4h": "4h"}  # pandas 경고 수정
    agg_rules = {
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 
        'Volume': 'sum', 'Quote_asset_volume': 'sum', 
        'Number_of_trades': 'sum', 'Taker_buy_base': 'sum', 'Taker_buy_quote': 'sum'
    }
    
    for tf_name, tf_rule in timeframes.items():
        # 리샘플링
        df_resampled = df_5m.resample(tf_rule).agg(agg_rules).dropna()
        if df_resampled.empty:
            continue
            
        # 훈련과 동일한 방식으로 피처 생성
        tf_features = _indicators_like_training(df_resampled, tf_name)
        
        # 훈련 로직과 완전히 동일하게, 전체 컬럼에 접미사를 붙여 병합
        # 이렇게 하면 Open_15m, ret1_15m_15m 같은 훈련 시 피처가 생성됨
        base_df = pd.merge_asof(
            base_df.sort_index(),
            tf_features.sort_index().add_suffix(f"_{tf_name}"),
            left_index=True,
            right_index=True,
            direction="backward"
        )
    
    return base_df

def _normalize(df_feat: pd.DataFrame, scaler, feature_cols: List[str]) -> pd.DataFrame:
    """NaN 처리 강화"""
    X = df_feat.select_dtypes(include=["float64","float32","int64","int32"]).copy()
    
    # 누락된 피처는 0으로 초기화 (NaN 대신)
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    
    X = X[feature_cols]
    # 강화된 NaN/Inf 처리
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    
    # 최종 검증
    if X.isnull().any().any():
        print(f"경고: NaN이 여전히 존재합니다: {X.isnull().sum().sum()}")
        X = X.fillna(0)
    
    arr = scaler.transform(X)
    return pd.DataFrame(arr, index=df_feat.index, columns=feature_cols)

# =====================
# 인젝트 클래스
# =====================

class RealtimeIngest:
    """
    run()을 호출하면, 확정된 최신 5m 캔들이 생길 때마다 Queue로 패킷을 푸시한다.
    packet = {"X": normalized_features_df, "close": close_series, "ts": latest_timestamp}
    """
    def __init__(self, out_queue: Queue, symbol: str = SYMBOL, interval: str = INTERVAL):
        self.q = out_queue
        self.symbol = symbol
        self.interval = interval
        # scaler & features
        self.scaler = joblib.load(os.path.join(PROC_DIR, "scaler.joblib"))
        self.features = json.load(open(os.path.join(PROC_DIR, "feature_list.json"), "r"))
        # 초기 백필
        self.df = self._backfill()

    def _backfill(self) -> pd.DataFrame:
        rows = _request_klines(self.symbol, self.interval, BACKFILL_LIMIT)
        df = _klines_to_df(rows)
        if df.empty:
            raise RuntimeError("과거 데이터 로드 실패: klines가 비어있습니다")
        return df

    def _emit_if_closed_bar(self) -> Optional[pd.Timestamp]:
        # 최신 바의 close_time이 진짜로 지난 상태인지 확인
        if self.df.empty:
            return None
        latest_open = self.df.index[-1]
        latest_close_time = int(self.df.iloc[-1]["Close_time"].value // 10**6)  # ns→ms
        if not _is_closed(latest_close_time):
            return None

        # 피처 & 정규화 구성
        feat = _featurize_5m(self.df)
        X = _normalize(feat, self.scaler, self.features)
        close = self.df["Close"].reindex(X.index).ffill().bfill()

        # 최신 1바 기준으로도 트레이더가 WINDOW 슬라이스를 구성할 수 있도록 전체 전달
        packet = {"X": X, "close": close, "ts": latest_open}
        try:
            # 큐가 가득이면 오래된 것 버리고 최신만 남긴다(지연 방지)
            if self.q.full():
                self.q.get_nowait()
            self.q.put_nowait(packet)
        except Exception:
            pass
        return latest_open

    def _fetch_tail(self) -> None:
        # 최근 100~200개만 재요청해서 테일 업데이트(중복 제거)
        rows = _request_klines(self.symbol, self.interval, 200)
        tail = _klines_to_df(rows)
        if tail.empty:
            return
        # 기존 df와 병합(인덱스 기준 dedup)
        df = pd.concat([self.df.iloc[:-200], tail]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        self.df = df

    def run(self) -> None:
        print(f"[수집기] 심볼={self.symbol} 인터벌={self.interval} (REST 폴링) | 백필={len(self.df)}")
        last_emitted_ts = None
        
        # 초기 실행 시, 이미 닫힌 가장 최신 바를 찾아 한 번만 전송
        emitted_ts = self._emit_if_closed_bar()
        if emitted_ts:
            last_emitted_ts = emitted_ts
            print(f"[수집기] 초기 확정봉 전송: {emitted_ts}")

        while True:
            try:
                self._fetch_tail()
                
                # 새로운 확정봉이 있고, 이전에 보낸 봉과 다를 경우에만 전송
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
                time.sleep(5.0)  # 에러 발생 시 대기 시간 증가