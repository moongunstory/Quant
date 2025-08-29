# ai_binance/live/realtime_ingest.py
"""
Realtime Ingest (MTF: 5m / 15m / 1h / 4h + BTC 1h)
- NEW: fe.py와 동일한 피처 생성 로직 사용.
- REST 폴링, '확정된' 5m 캔들만 내보냄(WS 미사용).
- 학습 시와 동일한 '원시 피처' 생성 (정규화 X).
- 5m 기반 → 15m/1h/4h resample, BTC 1h 별도 fetch.
- 각 TF의 원시 피처를 5m 인덱스에 as-of(=ffill) 정렬하여 내보냄.

Queue로 푸시되는 packet 스키마:
{
    "X": {
        "5m":    DataFrame(raw features; index=5m UTC),
        "15m":   DataFrame(raw features; index=5m UTC, asof-aligned),
        "1h":    DataFrame(raw features; index=5m UTC, asof-aligned),
        "4h":    DataFrame(raw features; index=5m UTC, asof-aligned),
        "btc1h": DataFrame(raw features; index=5m UTC, asof-aligned),
    },
    "close": Series(float) aligned to 5m index (ETHUSDT 5m 종가),
    "funding": Series(float) aligned to 5m index (rate/96 per 5m),
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

# =====================
# 설정
# =====================
SYMBOL = "ETHUSDT"
SYMBOL_BTC = "BTCUSDT"
BASE_URL = "https://fapi.binance.com"
KLINES_EP = "/fapi/v1/klines"
FUNDING_EP = "/fapi/v1/fundingRate"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

POLL_SEC = 2.0
BACKFILL_LIMIT = 1500
REQ_TIMEOUT = 15
MAX_RETRY = 5
RETRY_SLEEP = 1.0
FUNDING_SPLIT = 96

TF_RULES = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}

# =====================
# HTTP
# =====================
_session = requests.Session()

def _now_ms() -> int: return int(pd.Timestamp.utcnow().timestamp() * 1000)
def _is_closed_bar(close_time_ms: int) -> bool: return close_time_ms <= _now_ms() - 1000

def _get(url: str, params: Dict) -> dict | list:
    for _ in range(MAX_RETRY):
        try:
            r = _session.get(url, params=params, timeout=REQ_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(RETRY_SLEEP)
    raise ConnectionError(f"Failed to fetch {url} after {MAX_RETRY} retries")

def _fetch_klines(symbol: str, interval: str, limit: int) -> list:
    return _get(BASE_URL + KLINES_EP, {"symbol": symbol, "interval": interval, "limit": limit})

def _fetch_funding(symbol: str, limit: int = 200) -> list:
    return _get(BASE_URL + FUNDING_EP, {"symbol": symbol, "limit": limit})

# =====================
# 변환
# =====================
def _klines_to_df(rows: list) -> pd.DataFrame:
    if not rows: return pd.DataFrame()
    cols = ["Open_time","Open","High","Low","Close","Volume","Close_time","Quote_asset_volume","Number_of_trades","Taker_buy_base","Taker_buy_quote","Ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df["Open_time"] = pd.to_datetime(df["Open_time"], unit="ms", utc=True)
    df = df.set_index("Open_time").sort_index()
    for c in ["Open","High","Low","Close","Volume","Quote_asset_volume","Taker_buy_base","Taker_buy_quote", "Number_of_trades"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Close_time"] = pd.to_numeric(df["Close_time"], errors="coerce").astype("Int64")
    df["FundingRate"] = 0.0 # Placeholder for compatibility with fe.py
    return df[~df.index.duplicated(keep="last")]

# ===================================================================
# 피처 생성 (fe.py에서 로직 복사 및 수정)
# ===================================================================
def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype("float64")
    low  = df["Low"].astype("float64")
    close= df["Close"].astype("float64")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def compute_heikin_ashi(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    if df_ohlc.empty: return pd.DataFrame(index=df_ohlc.index)
    O, H, L, C = df_ohlc["Open"].values, df_ohlc["High"].values, df_ohlc["Low"].values, df_ohlc["Close"].values
    n = len(df_ohlc)
    HA_C = (O + H + L + C) / 4.0
    HA_O = np.empty(n); HA_O[0] = (O[0] + C[0]) / 2.0
    for i in range(1, n): HA_O[i] = (HA_O[i-1] + HA_C[i-1]) / 2.0
    HA_H = np.maximum.reduce([H, HA_O, HA_C])
    HA_L = np.minimum.reduce([L, HA_O, HA_C])
    out = pd.DataFrame({"HA_O": HA_O, "HA_H": HA_H, "HA_L": HA_L, "HA_C": HA_C}, index=df_ohlc.index)
    out["HA_TR"] = out["HA_H"] - out["HA_L"]
    out["HA_BC"] = out["HA_C"] - out["HA_O"]
    out["HA_R"]  = out["HA_C"].pct_change().fillna(0.0)
    return out

def zscore(s: pd.Series, win: int | None = None) -> pd.Series:
    if win is None: mu, sd = s.mean(), s.std()
    else: mu, sd = s.rolling(win).mean(), s.rolling(win).std()
    return ((s - mu) / sd.replace(0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

def _funding_phase_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    steps_since = (idx.hour % 8) * 12 + (idx.minute // 5)
    phase = 2 * np.pi * (steps_since / 96)
    out = pd.DataFrame(index=idx)
    out["time_to_funding_5m"] = (96 - steps_since) % 96
    out["funding_phase_sin"] = np.sin(phase)
    out["funding_phase_cos"] = np.cos(phase)
    return out

def compute_features_for_tf(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    df = df.sort_index()
    out = pd.DataFrame(index=df.index)
    
    if interval == "btc1h":
        # Pass-through original columns for merging
        out["Close"] = df["Close"].astype("float64")
        out["Volume"] = df["Volume"].astype("float64")

        close = df["Close"].astype("float64")
        out["ret_1h"] = close.pct_change()
        out["ret_4h"] = close.pct_change(4)
        out["atr14"]  = _atr(df, period=14)
        ha = compute_heikin_ashi(df[["Open","High","Low","Close"]])
        out = pd.concat([out, ha], axis=1)
    else: # ETH Timeframes
        close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]
        out["ret_1"] = close.pct_change()
        out["ret_3"] = close.pct_change(3)
        out["z_close_48"] = zscore(close, win=48)
        out["hl_spread"] = (high - low) / close
        out["vol_z_48"] = zscore(volume, win=48)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        out["macd"] = ema_12 - ema_26
        out["macd_sig"] = out["macd"].ewm(span=9, adjust=False).mean()
        out["macd_hist"] = out["macd"] - out["macd_sig"]
        delta = close.diff()
        up, down = delta.clip(lower=0), (-delta).clip(lower=0)
        roll_up = up.ewm(alpha=1/14, adjust=False).mean()
        roll_down = down.ewm(alpha=1/14, adjust=False).mean()
        out["rsi_14"] = 100 - (100 / (1 + roll_up / roll_down.replace(0, np.nan))).fillna(50)
        out["atr14"] = _atr(df, period=14)
        ha = compute_heikin_ashi(df[["Open","High","Low","Close"]])
        out = pd.concat([out, ha], axis=1)
        if interval == "5m":
            out['hour_sin'], out['hour_cos'] = np.sin(2*np.pi*df.index.hour/24), np.cos(2*np.pi*df.index.hour/24)
            out['day_sin'], out['day_cos'] = np.sin(2*np.pi*df.index.dayofweek/7), np.cos(2*np.pi*df.index.dayofweek/7)
            out = pd.concat([out, _funding_phase_features(out.index)], axis=1)
            out["is_funding_settle"] = (((df.index.hour % 8 == 0) & (df.index.minute == 0))).astype("int8")
            out["funding_z_48"] = zscore(df["FundingRate"], win=48)

    out.columns = [f"{c}_{interval}" for c in out.columns]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

# =====================
# Realtime Ingestor
# =====================
class RealtimeIngest:
    def __init__(self, out_queue: Queue, symbol: str = SYMBOL):
        self.q = out_queue
        self.symbol = symbol
        self.features: Dict[str, List[str]] = {}
        for tf in TF_RULES.keys():
            feats_path  = os.path.join(PROC_DIR, f"fe_feature_list_{tf}.json")
            if os.path.exists(feats_path):
                with open(feats_path, "r", encoding="utf-8") as f: self.features[tf] = json.load(f)
            else: raise FileNotFoundError(f"[ingest] 피처 리스트 파일 누락: {feats_path}")
        
        self.df5m = self._backfill_5m()
        self.df_btc1h = self._backfill_btc()

    def _backfill_5m(self) -> pd.DataFrame:
        return _klines_to_df(_fetch_klines(self.symbol, "5m", BACKFILL_LIMIT))
    def _backfill_btc(self) -> pd.DataFrame:
        return _klines_to_df(_fetch_klines(SYMBOL_BTC, "1h", BACKFILL_LIMIT))

    def _build_features(self, df5: pd.DataFrame, df_btc1h: pd.DataFrame) -> dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        base_index = df5.index
        
        # ETH TFs
        agg = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
        for tf, rule in TF_RULES.items():
            df_tf = df5 if tf == "5m" else df5.resample(rule).agg(agg).dropna()
            if df_tf.empty: continue
            feat_tf = compute_features_for_tf(df_tf, tf)
            out[tf] = feat_tf.reindex(base_index, method="ffill").fillna(0.0)

        # BTC 1h
        if df_btc1h is not None and not df_btc1h.empty:
            feat_btc = compute_features_for_tf(df_btc1h, "btc1h")
            out["btc1h"] = feat_btc.reindex(base_index, method="ffill").fillna(0.0)

            # --- BTC Lead-Lag Feature Integration (for all TFs) ---
            if ('5m' in out and not out['5m'].empty):
                merged_df = pd.merge_asof(
                    out['5m'].sort_index(),
                    feat_btc.sort_index(),
                    left_index=True, right_index=True,
                    direction="backward",
                    allow_exact_matches=True,
                                                                                tolerance=pd.Timedelta("8h"),
                )
                
                btc_close = merged_df["Close_btc1h"].astype(float)
                btc_vol = merged_df["Volume_btc1h"].astype(float)
                
                btc_lead_features = pd.DataFrame(index=out['5m'].index)
                btc_lead_features["btc_ret_1h"] = btc_close.pct_change()
                btc_lead_features["btc_vol_z_24"] = zscore(btc_vol, win=24)
                
                btc_ema_12 = btc_close.ewm(span=12, adjust=False).mean()
                btc_ema_26 = btc_close.ewm(span=26, adjust=False).mean()
                btc_lead_features["btc_macd"] = btc_ema_12 - btc_ema_26

                # Generate lagged features and add them to ALL available ETH TF feature sets
                for lag in range(1, 7):
                    lagged = btc_lead_features.shift(lag)
                    for tf in ['5m', '15m', '1h', '4h']:
                        if tf in out:
                            lagged_tf = lagged.copy()
                            lagged_tf.columns = [f"{c}_lag{lag}_{tf}" for c in lagged_tf.columns]
                            out[tf] = pd.concat([out[tf], lagged_tf], axis=1)

                # Fill NA for all affected dataframes
                for tf in ['5m', '15m', '1h', '4h']:
                    if tf in out: out[tf].fillna(0.0, inplace=True)

        return out

    def _emit(self) -> Optional[pd.Timestamp]:
        latest_open = self.df5m.index[-1]
        close_time_ms = int(self.df5m["Close_time"].iloc[-1])
        if not _is_closed_bar(close_time_ms): return None
        
        X_dict = self._build_features(self.df5m, self.df_btc1h)
        
        # Select final features based on JSON lists
        for tf, feat_list in self.features.items():
            if tf in X_dict: X_dict[tf] = X_dict[tf][feat_list]
        
        # For btc1h, select the 10 features from fe.py
        if "btc1h" in X_dict:
            btc_cols = ['ret_1h_btc1h', 'ret_4h_btc1h', 'atr14_btc1h', 'HA_O_btc1h', 'HA_H_btc1h', 'HA_L_btc1h', 'HA_C_btc1h', 'HA_TR_btc1h', 'HA_BC_btc1h', 'HA_R_btc1h']
            X_dict["btc1h"] = X_dict["btc1h"][btc_cols]

        packet = {"X": X_dict, "close": self.df5m["Close"], "ts": latest_open}
        try:
            if self.q.full(): self.q.get_nowait()
            self.q.put_nowait(packet)
        except Exception: pass
        return latest_open

    def _fetch_tails(self):
        # ETH tail
        tail_eth = _klines_to_df(_fetch_klines(self.symbol, "5m", 200))
        if not tail_eth.empty: self.df5m = pd.concat([self.df5m, tail_eth]).groupby(level=0).last()
        # BTC tail
        tail_btc = _klines_to_df(_fetch_klines(SYMBOL_BTC, "1h", 50))
        if not tail_btc.empty: self.df_btc1h = pd.concat([self.df_btc1h, tail_btc]).groupby(level=0).last()

    def run(self) -> None:
        print(f"[ingest] symbol={self.symbol}, btc_symbol={SYMBOL_BTC} | backfill_eth={len(self.df5m)}, backfill_btc={len(self.df_btc1h)}")
        last_emitted = self._emit()
        if last_emitted: print(f"[ingest] initial closed bar emitted: {last_emitted}")

        while True:
            try:
                self._fetch_tails()
                if last_emitted is None or self.df5m.index[-1] > last_emitted:
                    emitted_ts = self._emit()
                    if emitted_ts:
                        last_emitted = emitted_ts
                        print(f"[ingest] new closed bar emitted: {emitted_ts}")
                time.sleep(POLL_SEC)
            except KeyboardInterrupt: print("[ingest] stopped by user"); break
            except Exception as e: print(f"[ingest] warn: {e}"); time.sleep(5.0)
