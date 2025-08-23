# ai_binance/live/realtime_ingest.py
"""
Realtime Ingest (MTF: 5m / 15m / 1h / 4h)
- REST 폴링, '확정된' 5m 캔들만 내보냄(WS 미사용).
- 학습 시와 동일한 피처 생성(5m 기반 → 15m/1h/4h resample).
- 저장된 scaler/feature_list로 'TF별' 정규화.
- 각 TF의 정규화 피처를 5m 인덱스에 as-of(=ffill) 정렬하여 내보냄.
- 8h 펀딩 이벤트를 5m 단위로 균등 분배(rate/96)하여 per-bar 시리즈 제공.

Queue로 푸시되는 packet 스키마:
{
    "X": {
        "5m":  DataFrame(normalized features; index=5m UTC),
        "15m": DataFrame(normalized features; index=5m UTC, asof-aligned),
        "1h":  DataFrame(normalized features; index=5m UTC, asof-aligned),
        "4h":  DataFrame(normalized features; index=5m UTC, asof-aligned),
    },
    "close": Series(float) aligned to 5m index (ETHUSDT 5m 종가),
    "funding": Series(float) aligned to 5m index (rate/96 per 5m),
    "funding_rate": float,       # 최신 5m 바의 분배 단위
    "ts": Timestamp(UTC)         # 최신 확정봉의 Open_time
}
"""
from __future__ import annotations

import os
import time
import json
from typing import List, Optional, Dict
from queue import Queue

import numpy as np
import pandas as pd
import requests
import joblib

# =====================
# 설정
# =====================
SYMBOL = "ETHUSDT"
BASE_URL = "https://fapi.binance.com"
KLINES_EP = "/fapi/v1/klines"
FUNDING_EP = "/fapi/v1/fundingRate"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ~/ai_binance
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

POLL_SEC = 2.0
BACKFILL_LIMIT = 1500                # ~5.2일 (5m)
REQ_TIMEOUT = 15
MAX_RETRY = 5
RETRY_SLEEP = 1.0

FUNDING_SPLIT = 96                   # 8h / 5m

TF_RULES = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
}

# =====================
# HTTP
# =====================
_session = requests.Session()

def _now_ms() -> int:
    return int(pd.Timestamp.utcnow().timestamp() * 1000)

def _is_closed_bar(close_time_ms: int) -> bool:
    # 클로즈 타임 + 1초 버퍼 <= 현재
    return close_time_ms <= _now_ms() - 1000

def _get(url: str, params: Dict) -> dict | list:
    tries = 0
    while True:
        try:
            r = _session.get(url, params=params, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception:
            tries += 1
            if tries >= MAX_RETRY:
                raise
            time.sleep(RETRY_SLEEP)

def _fetch_klines(symbol: str, interval: str, limit: int) -> list:
    return _get(BASE_URL + KLINES_EP, {"symbol": symbol, "interval": interval, "limit": limit})

def _fetch_funding(symbol: str, limit: int = 200) -> list:
    return _get(BASE_URL + FUNDING_EP, {"symbol": symbol, "limit": limit})

# =====================
# 변환
# =====================
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
    df = df.set_index("Open_time").sort_index()
    for c in ["Open","High","Low","Close","Volume","Quote_asset_volume","Taker_buy_base","Taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Number_of_trades"] = pd.to_numeric(df["Number_of_trades"], errors="coerce")
    df = df[~df.index.duplicated(keep="last")]
    return df

def _funding_to_df(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["rate"])
    df = pd.DataFrame(rows)
    time_key = "fundingTime" if "fundingTime" in df.columns else "funding_time"
    rate_key = "fundingRate" if "fundingRate" in df.columns else "funding_rate"
    df["ts"] = pd.to_datetime(pd.to_numeric(df[time_key], errors="coerce").astype("Int64"), unit="ms", utc=True)
    df["rate"] = pd.to_numeric(df[rate_key], errors="coerce")
    df = df[["ts","rate"]].dropna()
    df = df.sort_values("ts").drop_duplicates("ts", keep="last").set_index("ts")
    return df

# =====================
# 피처 생성
# =====================
def _indicators(df: pd.DataFrame, itv: str) -> pd.DataFrame:
    x = df.copy()
    x[f"ret1_{itv}"]  = x["Close"].pct_change()
    x[f"ret3_{itv}"]  = x["Close"].pct_change(3)
    x[f"ret12_{itv}"] = x["Close"].pct_change(12)
    x[f"hlv_{itv}"]   = (x["High"] - x["Low"]) / x["Close"].replace(0, np.nan)

    d = x["Close"].diff()
    up = d.clip(lower=0).rolling(14, min_periods=14).mean()
    dn = (-d.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = up / dn.replace(0, np.nan)
    x[f"rsi14_{itv}"] = 100 - 100 / (1 + rs)

    ema12 = x["Close"].ewm(span=12, adjust=False).mean()
    ema26 = x["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    x[f"macd_{itv}"]      = macd
    x[f"macd_sig_{itv}"]  = macd_sig
    x[f"macd_hist_{itv}"] = macd - macd_sig

    tr = np.maximum(
        x["High"] - x["Low"],
        np.maximum((x["High"] - x["Close"].shift()).abs(), (x["Low"] - x["Close"].shift()).abs())
    )
    x[f"atr14_{itv}"] = tr.rolling(14, min_periods=14).mean()

    ma20 = x["Close"].rolling(20, min_periods=20).mean()
    sd20 = x["Close"].rolling(20, min_periods=20).std()
    x[f"bb_mid_{itv}"] = ma20
    x[f"bb_up_{itv}"]  = ma20 + 2 * sd20
    x[f"bb_dn_{itv}"]  = ma20 - 2 * sd20
    return x

def _normalize(df_feat: pd.DataFrame, scaler, feature_cols: List[str]) -> pd.DataFrame:
    X = df_feat.select_dtypes(include=["float64","float32","int64","int32"]).copy()
    # 누락 컬럼 0으로 채우고 순서 정렬
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0
    X = X[feature_cols]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = scaler.transform(X.values)
    return pd.DataFrame(arr, index=df_feat.index, columns=feature_cols)

# =====================
# 펀딩 8h → 5m 균등 분배
# =====================
def _distribute_funding_to_5m(index_5m: pd.DatetimeIndex, funding_df: pd.DataFrame) -> pd.Series:
    """
    funding_df: index=funding event time(UTC), column 'rate'
    반환: per-bar funding rate share (rate / 96), index=index_5m
    룰: 이벤트 시각 t_i 의 rate_i 는 구간 (t_{i-1}, t_i] 에 균등 분배.
        마지막 이벤트 이후 구간은 마지막 rate로 유지.
    """
    s = pd.Series(0.0, index=index_5m, dtype=float)
    if index_5m.empty:
        return s
    if funding_df is None or funding_df.empty:
        return s

    events = funding_df.sort_index()
    events = events.loc[events.index <= index_5m[-1]]
    if events.empty:
        return s

    prev_time = index_5m[0] - pd.Timedelta(minutes=5)  # 경계 보정
    prev_rate = float(events.iloc[0]["rate"])
    for t_event, row in events.iterrows():
        rate = float(row["rate"])
        mask = (index_5m > prev_time) & (index_5m <= t_event)
        if mask.any():
            s.loc[mask] = rate / FUNDING_SPLIT
        prev_time = t_event
        prev_rate = rate

    # 마지막 이벤트 이후
    tail_mask = (index_5m > prev_time)
    if tail_mask.any():
        s.loc[tail_mask] = prev_rate / FUNDING_SPLIT

    return s

# =====================
# Realtime Ingestor
# =====================
class RealtimeIngest:
    """
    run() 호출 시, 확정된 최신 5m 캔들이 생길 때마다 Queue로 패킷 푸시.

    packet = {
        "X": {
            "5m":  DataFrame(normalized features; index=5m),
            "15m": DataFrame(normalized features; index=5m, asof-aligned),
            "1h":  DataFrame(normalized features; index=5m, asof-aligned),
            "4h":  DataFrame(normalized features; index=5m, asof-aligned),
        },
        "close": close_series(5m 기준),
        "funding": funding_series_5m,   # rate/96
        "funding_rate": float,          # 최신 5m 바의 분배 단위
        "ts": Timestamp(UTC)            # 최신 확정봉의 Open_time
    }
    """
    TF_RULES = TF_RULES

    def __init__(self, out_queue: Queue, symbol: str = SYMBOL):
        self.q = out_queue
        self.symbol = symbol

        # TF별 스케일러/피처 로드 (존재하는 TF만 사용)
        self.scalers: Dict[str, object] = {}
        self.features: Dict[str, List[str]] = {}
        self.tf_used: List[str] = []
        for tf in self.TF_RULES.keys():
            scaler_path = os.path.join(PROC_DIR, f"scaler_{tf}.joblib")
            feats_path  = os.path.join(PROC_DIR, f"fe_feature_list_{tf}.json")
            if os.path.exists(scaler_path) and os.path.exists(feats_path):
                self.scalers[tf] = joblib.load(scaler_path)
                with open(feats_path, "r", encoding="utf-8") as f:
                    self.features[tf] = json.load(f)
                self.tf_used.append(tf)

        if "5m" not in self.tf_used:
            raise FileNotFoundError(
                f"[ingest] 최소 5m 아티팩트가 필요합니다: {os.path.join(PROC_DIR, 'scaler_5m.joblib')} & "
                f"{os.path.join(PROC_DIR, 'fe_feature_list_5m.json')}"
            )
        if len(self.tf_used) < len(self.TF_RULES):
            missing = [tf for tf in self.TF_RULES if tf not in self.tf_used]
            print(f"[ingest] 경고: 다음 TF 아티팩트가 없어 제외됩니다 → {missing}")

        # 초기 백필 (5m only → 상위 TF는 resample)
        self.df5m = self._backfill_5m()
        self.funding_df = self._backfill_funding()

    # ----- Backfill -----
    def _backfill_5m(self) -> pd.DataFrame:
        rows = _fetch_klines(self.symbol, "5m", BACKFILL_LIMIT)
        df = _klines_to_df(rows)
        if df.empty:
            raise RuntimeError("초기 5m 캔들 로드 실패")
        return df

    def _backfill_funding(self) -> pd.DataFrame:
        try:
            rows = _fetch_funding(self.symbol, limit=200)
            return _funding_to_df(rows)
        except Exception as e:
            print(f"[ingest] funding backfill fail: {e}")
            return pd.DataFrame(columns=["rate"])

    def _refresh_funding(self) -> None:
        try:
            rows = _fetch_funding(self.symbol, limit=50)
            df_new = _funding_to_df(rows)
            if not df_new.empty:
                self.funding_df = (
                    pd.concat([self.funding_df, df_new])
                    .sort_index()
                    .drop_duplicates(keep="last")
                ).iloc[-400:]
        except Exception as e:
            print(f"[ingest] funding refresh warn: {e}")

    # ----- Feature Build -----
    def _build_features(self, df5: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        TF별 indicator 생성 + 정규화 + 5m 인덱스에 as-of 정렬(ffill).
        반환 DataFrame들은 모두 index=df5.index(5m) 를 가짐.
        """
        out: Dict[str, pd.DataFrame] = {}

        # 5m
        feat_5m = _indicators(df5, "5m")
        X_5m = _normalize(feat_5m, self.scalers["5m"], self.features["5m"])
        out["5m"] = X_5m

        base_index = df5.index

        # 상위 TF
        agg = {
            'Open':'first','High':'max','Low':'min','Close':'last',
            'Volume':'sum','Quote_asset_volume':'sum',
            'Number_of_trades':'sum','Taker_buy_base':'sum','Taker_buy_quote':'sum'
        }
        for tf, rule in self.TF_RULES.items():
            if tf == "5m" or tf not in self.tf_used:
                continue
            higher = df5.resample(rule).agg(agg).dropna()
            if higher.empty:
                # 상위 TF가 아직 시작되지 않은 경우(백필 짧음)
                out[tf] = pd.DataFrame(0.0, index=base_index, columns=self.features[tf])
                continue

            feat_h = _indicators(higher, tf)
            X_h = _normalize(feat_h, self.scalers[tf], self.features[tf])

            # 5m 인덱스로 as-of 정렬: ffill
            X_h_aligned = X_h.reindex(base_index, method="ffill").fillna(0.0)
            out[tf] = X_h_aligned

        return out

    # ----- Emit -----
    def _emit_if_closed(self) -> Optional[pd.Timestamp]:
        if self.df5m.empty:
            return None
        latest_open = self.df5m.index[-1]
        close_ms = int(self.df5m.iloc[-1]["Close_time"].value // 10**6)  # ns→ms
        if not _is_closed_bar(close_ms):
            return None

        # TF별 피처(모두 5m 인덱스)
        X_dict = self._build_features(self.df5m)
        close = self.df5m["Close"].reindex(X_dict["5m"].index).ffill().bfill()

        # 펀딩 per 5m
        funding_series = _distribute_funding_to_5m(X_dict["5m"].index, self.funding_df)
        funding_series = funding_series.reindex(X_dict["5m"].index).fillna(0.0)
        current_funding = float(funding_series.iloc[-1]) if len(funding_series) else 0.0

        packet = {
            "X": X_dict,
            "close": close,
            "funding": funding_series,
            "funding_rate": current_funding,
            "ts": latest_open
        }
        # 비차단 push (가득 차 있으면 오래된 것 버림)
        try:
            if self.q.full():
                self.q.get_nowait()
            self.q.put_nowait(packet)
        except Exception:
            pass
        return latest_open

    # ----- Tail fetch -----
    def _fetch_tail(self) -> None:
        rows = _fetch_klines(self.symbol, "5m", 200)
        tail = _klines_to_df(rows)
        if not tail.empty:
            merged = pd.concat([self.df5m.iloc[:-200], tail]).sort_index()
            merged = merged[~merged.index.duplicated(keep="last")]
            self.df5m = merged
        # 펀딩 갱신
        self._refresh_funding()

    # ----- Run -----
    def run(self) -> None:
        print(f"[ingest] symbol={self.symbol} base_tf=5m | backfill={len(self.df5m)}")
        last_emitted = self._emit_if_closed()
        if last_emitted is not None:
            print(f"[ingest] initial closed bar emitted: {last_emitted}")

        while True:
            try:
                self._fetch_tail()
                latest_open = self.df5m.index[-1]
                if last_emitted is None or latest_open > last_emitted:
                    em = self._emit_if_closed()
                    if em is not None:
                        last_emitted = em
                        print(f"[ingest] new closed bar emitted: {em}")
                time.sleep(POLL_SEC)
            except KeyboardInterrupt:
                print("[ingest] stopped by user")
                break
            except Exception as e:
                print(f"[ingest] warn: {e}")
                time.sleep(5.0)
