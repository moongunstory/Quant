# fe.py — Feature Engineering for ETHUSDT (MTF) + BTC1h
# (REV-10.0, leak-free / volume-consistent / HPO-extended)
from __future__ import annotations

import os, json
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import joblib

# ===== Paths / Constants =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw"))
OUT_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
os.makedirs(OUT_DIR, exist_ok=True)

# MTF Setup
TIMEFRAMES     = ["5m", "15m", "1h", "4h", "btc1h"]
ETH_TIMEFRAMES = ["5m", "15m", "1h", "4h"]
BASE_INTERVAL  = "5m"

# === TF별 피처검색/스케일 설정 (기존 파이프라인 유지) ===
FEATURE_SEARCH = True
RANDOM_STATE = 72
TOP_K_PER_TF = {"5m": 128, "15m": 128, "1h": 96, "4h": 64}
TF_FOR_SEARCH = ["5m", "15m", "1h", "4h"]  # 옵션 A: btc1h는 선정/스케일 대상 아님

# === HPO 확장 아티팩트 ===
HPO_OUT_PREFIX          = "feHPO"      # feHPO_{split}_{tf}.parquet
HPO_FEATURE_LIST_FMT    = os.path.join(OUT_DIR, "feHPO_feature_list_{tf}.json")
HPO_SCALER_PATH_FMT     = os.path.join(OUT_DIR, "scaler_hpo_{tf}.joblib")
HPO_EXPAND_WINDOWS      = [12, 24, 48, 96]  # rolling window 확장
HPO_MAX_FEATURES_HINT   = 2000  # 안전 가이드 (하드 컷은 하지 않음)

# Output path formats (기존)
FEATURE_LIST_PATH_FMT = os.path.join(OUT_DIR, "fe_feature_list_{tf}.json")
SCALER_PATH_FMT       = os.path.join(OUT_DIR, "scaler_{tf}.joblib")

# ===== Reference (unscaled) columns to keep =====
REF_COLS_CANON = ["Open", "High", "Low", "Close", "Volume", "FundingRate"]

# ===== Utilities =====
def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    std = df.std(numeric_only=True)
    zero_std_cols = std[std == 0].index.tolist()
    if len(zero_std_cols) > 0:
        df[zero_std_cols] = 0.0
    return df.fillna(0.0)

def _as_series(x):
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        return x.mean(axis=1)
    return x

def zscore(s: pd.Series, win: int | None = None) -> pd.Series:
    if win is None:
        mu, sd = s.mean(), s.std()
        sd = sd if (sd and sd > 0) else np.nan
        out = (s - mu) / sd
    else:
        mu = s.rolling(win, min_periods=win).mean()
        sd = s.rolling(win, min_periods=win).std().replace(0, np.nan)
        out = (s - mu) / sd
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def pct_slope(x: pd.Series, win: int) -> pd.Series:
    return x.pct_change(fill_method=None).rolling(win, min_periods=win).mean().fillna(0.0)

# --- robust datetime handling (ms/ns) ---
def _to_utc_dt(s: pd.Series) -> pd.DatetimeIndex:
    s = pd.Series(s)
    if np.issubdtype(s.dtype, np.number):
        vmax = float(s.dropna().max()) if s.dropna().size else 0.0
        unit = "ms" if 1e11 < vmax < 1e14 else "ns"
        return pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")

def _enforce_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        for tcol in ["Open_time", "open_time", "time"]:
            if tcol in df.columns:
                idx = _to_utc_dt(df[tcol])
                if idx.notna().any():
                    df = df.set_index(idx).drop(columns=[tcol])
                    break
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = _to_utc_dt(df.index)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df.index.name = "time"
    return df

def _load_raw(split: str, interval: str) -> pd.DataFrame:
    """
    - Volume: RAW 유지 (Volume), 로그 필요 시 별도 컬럼(VolumeLog) 생성
    - Quote/Taker 계열도 Raw/Log 병행 보존 (향후 선택적 사용)
    """
    if interval == "btc1h":
        p = os.path.join(RAW_DIR, "btcusdt", f"fut_{split}_data_1h.parquet")
    else:
        p = os.path.join(RAW_DIR, "ethusdt", f"fut_{split}_data_{interval}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Raw data file not found: {p}")

    df = pd.read_parquet(p)

    # 숫자형 강제
    cols = {c.lower(): c for c in df.columns}
    for k in ["open","high","low","close","volume"]:
        if k in cols:
            real = cols[k]
            df[real] = pd.to_numeric(df[real], errors="coerce")

    # Funding
    if "FundingRate" not in df.columns and "funding_rate" in df.columns:
        df.rename(columns={"funding_rate": "FundingRate"}, inplace=True)
    if "FundingRate" not in df.columns:
        df["FundingRate"] = 0.0
    df["FundingRate"] = pd.to_numeric(df["FundingRate"], errors="coerce").fillna(0.0)

    # --- Volume & flows: RAW 보존 + LOG 별도 ---
    def _mk_raw_log(col: str):
        if col in df.columns:
            base = pd.to_numeric(df[col], errors="coerce")
            df[f"{col}Raw"] = base
            df[f"{col}Log"] = np.log1p(np.clip(base, 0, None))

    _mk_raw_log("Volume")
    _mk_raw_log("Quote_asset_volume")
    _mk_raw_log("Taker_buy_base")
    _mk_raw_log("Taker_buy_quote")

    # 관습적 이름: 가급적 Volume = Raw 로 유지
    if "VolumeRaw" in df.columns:
        df["Volume"] = df["VolumeRaw"]

    df = _enforce_dt_index(df)

    for c in REF_COLS_CANON:
        if c not in df.columns:
            df[c] = 0.0

    if interval == BASE_INTERVAL:
        if "FundingSettle" not in df.columns:
            df["FundingSettle"] = (((df.index.hour % 8 == 0) & (df.index.minute == 0))).astype("int8")
        else:
            df["FundingSettle"] = df["FundingSettle"].astype("int8")
    return df

# ===== Core Indicators & Blocks =====
def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def _funding_phase_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    steps_per_8h = 96  # 8h = 96 x 5m
    steps_since = (idx.hour % 8) * 12 + (idx.minute // 5)
    steps_to_next = (steps_per_8h - steps_since) % steps_per_8h
    phase = 2 * np.pi * (steps_since / steps_per_8h)
    out = pd.DataFrame(index=idx)
    out["time_to_funding_5m"] = steps_to_next.astype("int16")
    out["funding_phase_sin"] = np.sin(phase)
    out["funding_phase_cos"] = np.cos(phase)
    return out

def compute_heikin_ashi(df_ohlc: pd.DataFrame, o="Open", h="High", l="Low", c="Close") -> pd.DataFrame:
    if df_ohlc.empty: return pd.DataFrame(index=df_ohlc.index)
    O, H, L, C = df_ohlc[o].values, df_ohlc[h].values, df_ohlc[l].values, df_ohlc[c].values
    n = len(df_ohlc)
    HA_C = (O + H + L + C) / 4.0
    HA_O = np.empty(n); HA_O[0] = (O[0] + C[0]) / 2.0
    for i in range(1, n): HA_O[i] = (HA_O[i-1] + HA_C[i-1]) / 2.0
    HA_H, HA_L = np.maximum.reduce([H, HA_O, HA_C]), np.minimum.reduce([L, HA_O, HA_C])
    out = pd.DataFrame({"HA_O": HA_O, "HA_H": HA_H, "HA_L": HA_L, "HA_C": HA_C}, index=df_ohlc.index)
    out["HA_TR"], out["HA_BC"] = out["HA_H"] - out["HA_L"], out["HA_C"] - out["HA_O"]
    out["HA_R"]  = out["HA_C"].pct_change(fill_method=None).fillna(0.0)
    return out

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up, down = delta.clip(lower=0), (-delta).clip(lower=0)
    rs = up.ewm(alpha=1/period, adjust=False).mean() / down.ewm(alpha=1/period, adjust=False).mean()
    return (100 - (100 / (1 + rs))).fillna(50)

def bollinger(series: pd.Series, win: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    m = series.rolling(win, min_periods=win).mean()
    s = series.rolling(win, min_periods=win).std()
    upper, lower = m + k*s, m - k*s
    return m, upper, lower

def keltner(df: pd.DataFrame, ema_win: int = 20, atr_win: int = 20, mult: float = 1.5) -> Tuple[pd.Series, pd.Series]:
    mid = ema(df["Close"].astype(float), ema_win)
    rng = _atr(df, period=atr_win) * mult
    upper, lower = mid + rng, mid - rng
    return upper, lower

def stoch_rsi(series: pd.Series, rsi_win: int = 14, stoch_win: int = 14) -> Tuple[pd.Series, pd.Series]:
    r = rsi(series, rsi_win)
    low = r.rolling(stoch_win, min_periods=stoch_win).min()
    high = r.rolling(stoch_win, min_periods=stoch_win).max()
    k = (r - low) / (high - low + 1e-9)
    d = k.rolling(3, min_periods=3).mean()
    return k.fillna(0.5), d.fillna(0.5)

def supertrend(df: pd.DataFrame, atr_period: int = 10, multiplier: float = 3.0) -> pd.Series:
    atr = _atr(df, period=atr_period)
    hl2 = (df["High"] + df["Low"]) / 2.0
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    dir_ = pd.Series(1, index=df.index, dtype="int8")
    st = pd.Series(index=df.index, dtype="float64")
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = lower.iloc[i]
            dir_.iloc[i] = 1
        else:
            if df["Close"].iloc[i] > st.iloc[i-1]:
                dir_.iloc[i] = 1
            elif df["Close"].iloc[i] < st.iloc[i-1]:
                dir_.iloc[i] = -1
            else:
                dir_.iloc[i] = dir_.iloc[i-1]
            st.iloc[i] = upper.iloc[i] if dir_.iloc[i] == 1 else lower.iloc[i]
    return dir_

def donchian(df: pd.DataFrame, win: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    hi = df["High"].rolling(win, min_periods=win).max()
    lo = df["Low"].rolling(win, min_periods=win).min()
    mid = (hi + lo) / 2.0
    return hi, mid, lo

def kama(series: pd.Series, n: int = 30, fast: int = 2, slow: int = 30) -> pd.Series:
    change = series.diff(n).abs()
    volatility = series.diff().abs().rolling(n, min_periods=n).sum()
    er = (change / (volatility + 1e-9)).fillna(0.0)
    fast_sc = 2/(fast+1); slow_sc = 2/(slow+1)
    sc = (er*(fast_sc - slow_sc) + slow_sc)**2
    kama = series.copy()
    for i in range(1, len(series)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i]*(series.iloc[i] - kama.iloc[i-1])
    return kama

def anchored_vwap(df: pd.DataFrame, anchor: str = "W") -> pd.Series:
    price = df["Close"].astype(float)
    vol = np.clip(pd.to_numeric(df.get("VolumeRaw", df["Volume"]), errors="coerce"), 0, None)

    # 빈도 매핑
    a = anchor.upper()
    if a == "M":       freq = "ME"
    elif a == "W":     freq = "W"
    elif a == "D":     freq = "D"
    else:              freq = anchor

    key = pd.Grouper(freq=freq, label="left", closed="left")
    g = df.groupby(key, group_keys=False)

    pv_cum = g.apply(lambda x: (x["Close"] * np.clip(pd.to_numeric(x.get("VolumeRaw", x["Volume"]), errors="coerce"), 0, None)).cumsum())
    v_cum  = g.apply(lambda x: np.clip(pd.to_numeric(x.get("VolumeRaw", x["Volume"]), errors="coerce"), 0, None).cumsum())
    vwap = (pv_cum / (v_cum + 1e-9)).reindex(df.index).ffill().fillna(price)
    return vwap

def tails_ratio(df: pd.DataFrame, win: int = 20) -> pd.Series:
    up_tail = (df["High"] - df[["Open","Close"]].max(axis=1)).clip(lower=0)
    dn_tail = (df[["Open","Close"]].min(axis=1) - df["Low"]).clip(lower=0)
    up_m = up_tail.rolling(win, min_periods=win).mean()
    dn_m = dn_tail.rolling(win, min_periods=win).mean()
    return (up_m / (dn_m + 1e-9)).fillna(1.0)

def realized_variance(ret: pd.Series, win: int = 12) -> pd.Series:
    return (ret.fillna(0.0)**2).rolling(win, min_periods=win).sum()

def cmf_proxy(df: pd.DataFrame, win: int = 20) -> pd.Series:
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-9)
    vol = np.clip(pd.to_numeric(df.get("VolumeRaw", df["Volume"]), errors="coerce"), 0, None)
    mfv = mfm * vol
    return mfv.rolling(win, min_periods=win).sum()

# ===== Feature Engines (Role-Tuned) =====
def compute_features_for_tf(df: pd.DataFrame, interval: str, btc_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = df.sort_index()
    out = pd.DataFrame(index=df.index)

    # ===== BTC 1h (standalone) — 옵션 A에서는 최종 저장하지 않음 =====
    if interval == "btc1h":
        close = df["Close"].astype(float)
        out["ret_1h"] = close.pct_change(fill_method=None)
        out["ret_4h"] = close.pct_change(4, fill_method=None)
        out["atr14"]  = _atr(df, period=14)
        ha = compute_heikin_ashi(df[["Open","High","Low","Close"]])
        out = pd.concat([out, ha], axis=1)
        out.columns = [f"{c}_btc1h" for c in out.columns]
        ref = df[REF_COLS_CANON].copy()
        return _sanitize(pd.concat([out, ref], axis=1))

    # ===== 공통 기본 =====
    close, high, low, volume = (
        df["Close"].astype(float),
        df["High"].astype(float),
        df["Low"].astype(float),
        df["Volume"].astype(float),  # RAW
    )
    ret1 = close.pct_change(fill_method=None).fillna(0.0)
    out["ret_1"] = ret1
    out["ret_3"] = close.pct_change(3, fill_method=None)
    out["hl_spread"] = (high - low) / (close + 1e-9)
    out["vol_z_48"] = zscore(volume, win=48)
    out["atr14"] = _atr(df, period=14)
    ha = compute_heikin_ashi(df[["Open","High","Low","Close"]])
    out = pd.concat([out, ha], axis=1)

    # ===== 역할 맞춤 추가 =====
    if interval == "4h":
        ema50, ema100, ema200 = ema(close,50), ema(close,100), ema(close,200)
        out["ema50_slope"]  = pct_slope(ema50, 5)
        out["ema100_slope"] = pct_slope(ema100, 5)
        out["ema200_slope"] = pct_slope(ema200, 5)
        out["ema50_200_spread"] = (ema50 - ema200) / (ema200 + 1e-9)

        cross_up = ((ema50.shift(1) < ema200.shift(1)) & (ema50 >= ema200)).astype(int)
        cross_dn = ((ema50.shift(1) > ema200.shift(1)) & (ema50 <= ema200)).astype(int)
        cross = (cross_up - cross_dn)
        idx = np.arange(len(df), dtype=int)
        mark = np.where(cross != 0, idx, -1)
        last = np.maximum.accumulate(mark)
        bars = idx - last
        bars[last < 0] = 0
        out["bars_since_cross"] = bars.astype(np.int32)

        di_win = 14
        dm_plus  = (high - high.shift(1)).clip(lower=0)
        dm_minus = (low.shift(1) - low).clip(lower=0)
        tr = (pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1)).max(axis=1)
        di_plus = 100 * (dm_plus.ewm(alpha=1/di_win, adjust=False).mean() / (tr.ewm(alpha=1/di_win, adjust=False).mean() + 1e-9))
        di_minus= 100 * (dm_minus.ewm(alpha=1/di_win, adjust=False).mean() / (tr.ewm(alpha=1/di_win, adjust=False).mean() + 1e-9))
        out["adx_14"] = 100 * ( (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-9) )

        out["supertrend_dir"] = supertrend(df, atr_period=10, multiplier=3.0)

        hi, mid, lo = donchian(df, win=20)
        width = (hi - lo).replace(0, np.nan)
        out["donch_mid_dist"] = (close - mid) / (width + 1e-9)
        out["donch_pos"] = (close - lo) / (width + 1e-9)

        out["kama_slope"] = pct_slope(kama(close, 30), 5)

        macd = ema(close,12) - ema(close,26)
        sig  = ema(macd,9)
        out["macd_hist"] = macd - sig

    elif interval == "1h":
        ema20, ema50 = ema(close,20), ema(close,50)
        out["ema20_slope"] = pct_slope(ema20, 5)
        out["ema50_slope"] = pct_slope(ema50, 5)
        out["price_ema20_z"] = zscore((close - ema20) / (ema20 + 1e-9), win=48)

        m, up, lo = bollinger(close, win=20, k=2.0)
        out["bb_width_pct"] = (up - lo) / (m + 1e-9)

        r = rsi(close, 14)
        out["rsi_14"] = r
        out["rsi_neutral_stay"] = ((r.between(40,60)).astype(int).rolling(24, min_periods=1).mean())

        vwap_w = _as_series(anchored_vwap(df, "W"))
        vwap_m = _as_series(anchored_vwap(df, "M"))
        out["dist_vwap_w"] = (close - vwap_w) / (vwap_w + 1e-9)
        out["dist_vwap_m"] = (close - vwap_m) / (vwap_m + 1e-9)

        atr = _atr(df, 14)
        q = pd.qcut(atr.replace(0, np.nan).ffill(), 3, labels=False, duplicates="drop").fillna(1).astype(int)
        out["atr_regime_low"]  = (q==0).astype(int)
        out["atr_regime_mid"]  = (q==1).astype(int)
        out["atr_regime_high"] = (q>=2).astype(int)

    elif interval == "15m":
        m, up, lo = bollinger(close, win=20, k=2.0)
        ku, kl = keltner(df, ema_win=20, atr_win=20, mult=1.5)
        bb_w = (up - lo); kc_w = (ku - kl)
        out["squeeze_ratio"] = (bb_w / (kc_w + 1e-9)) - 1.0

        k, d_ = stoch_rsi(close, 14, 14)
        out["stochrsi_k"], out["stochrsi_d"] = k, d_

        out["tails_ratio_20"] = tails_ratio(df, win=20)
        comp = (bb_w.pct_change(fill_method=None).rolling(10, min_periods=1).mean() < 0).astype(int)
        out["compress_run"] = comp.groupby((comp != comp.shift()).cumsum()).cumsum()

        out["ret_5"]  = close.pct_change(5, fill_method=None)
        out["ret_10"] = close.pct_change(10, fill_method=None)

    elif interval == "5m":
        out["atr3"] = _atr(df, period=3)
        out["rv_12"] = realized_variance(ret1, win=12)

        vwap_d = _as_series(anchored_vwap(df, "D")) if "D" else _as_series(anchored_vwap(df, "W"))
        vwap_w = _as_series(anchored_vwap(df, "W"))
        out["dist_vwap_d"] = (close - vwap_d) / (vwap_d + 1e-9)
        out["dist_vwap_w"] = (close - vwap_w) / (vwap_w + 1e-9)
        out["vwap_d_slope"] = pct_slope(vwap_d, 12)

        out["momo_1"] = ret1
        out["momo_3"] = close.pct_change(3, fill_method=None)
        out["momo_5"] = close.pct_change(5, fill_method=None)
        n_break = 36
        out["break_up"]   = (close > close.rolling(n_break, min_periods=n_break).max().shift(1)).astype(int)
        out["break_down"] = (close < close.rolling(n_break, min_periods=n_break).min().shift(1)).astype(int)

        out["obv_like"] = (np.sign(close.diff().fillna(0.0)) * volume).cumsum()
        out["cmf_proxy"] = cmf_proxy(df, win=20)

        out['hour_sin'], out['hour_cos'] = np.sin(2*np.pi*df.index.hour/24), np.cos(2*np.pi*df.index.hour/24)
        out['day_sin'],  out['day_cos']  = np.sin(2*np.pi*df.index.dayofweek/7), np.cos(2*np.pi*df.index.dayofweek/7)
        out = pd.concat([out, _funding_phase_features(out.index)], axis=1)

        settle_series = df["FundingSettle"] if "FundingSettle" in df.columns else pd.Series(0, index=df.index)
        out["is_funding_settle"] = settle_series.astype("int8")
        out["funding_z_48"] = zscore(df["FundingRate"].astype(float), win=48)

    # ===== BTC 리드/래그 & 상관/베타 =====
    if btc_df is not None:
        btc_slim = pd.DataFrame(index=btc_df.index)
        btc_slim["Close_btc1h"]  = pd.to_numeric(btc_df["Close"], errors="coerce")
        if "Volume" in btc_df.columns:
            btc_slim["Volume_btc1h"] = pd.to_numeric(btc_df["Volume"], errors="coerce")

        tol_map = {"5m": pd.Timedelta(hours=1), "15m": pd.Timedelta(hours=1),
                   "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}
        tol = tol_map.get(interval, pd.Timedelta(hours=1))

        merged = pd.merge_asof(
            df.reset_index().sort_values("time"),
            btc_slim.reset_index().sort_values("time"),
            on="time", direction="backward", tolerance=tol
        ).set_index("time")

        btc_close = merged["Close_btc1h"].astype(float)
        btc_vol   = merged.get("Volume_btc1h", pd.Series(0, index=merged.index)).astype(float)

        out["btc_ret_1h"]    = btc_close.pct_change(fill_method=None)
        out["btc_vol_z_24"]  = zscore(btc_vol, win=24)
        btc_macd = ema(btc_close,12) - ema(btc_close,26)
        out["btc_macd"] = btc_macd

        for lag in range(1, 3):
            out[f"btc_ret_1h_lag{lag}"] = out["btc_ret_1h"].shift(lag)

        if interval in ("1h","4h"):
            w = 24 if interval=="1h" else 6
        elif interval=="15m":
            w = 96
        else:
            w = 288
        e_ret = close.pct_change(fill_method=None)
        b_ret = btc_close.pct_change(fill_method=None)
        cov = e_ret.rolling(w).cov(b_ret)
        var_b = b_ret.rolling(w).var()
        beta = (cov / (var_b + 1e-9)).fillna(0.0)
        out["btc_beta"] = beta
        spread = e_ret - beta*b_ret
        out["btc_spread_z"] = zscore(spread, win=w)
        out["btc_corr_win"] = e_ret.rolling(w).corr(b_ret).fillna(0.0)

    out.columns = [f"{c}_{interval}" for c in out.columns]
    final_out = pd.concat([out, df[REF_COLS_CANON]], axis=1)
    return _sanitize(final_out)

# ===== HPO 전용: 후보 피처 확장 & 변환 =====
def _add_hpo_candidates_local(df_tf: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    - 기존 피처에서 누적/변환으로 유니버스를 확장 (leak-free: 과거 기반 rolling)
    - 범용 변환: rolling mean/std, zscore, pct_change, 바이너리 이벤트
    """
    df = df_tf.copy()
    # (1) 변화율/변동성 정규화
    if "Close" in df.columns:
        px = df["Close"].astype(float)
        ret = px.pct_change(fill_method=None).fillna(0.0)
        for n in (2,3,5,8,12,24,36,48):
            df[f"ret_{n}_{interval}"] = px.pct_change(n, fill_method=None)
        # ATR 대비 수익률
        atr_col = f"atr14_{interval}" if f"atr14_{interval}" in df.columns else None
        if atr_col:
            df[f"ret_over_atr_{interval}"] = (ret / (df[atr_col] + 1e-9)).clip(-10,10)

    # (2) 거래량/레인지 스파이크
    if "Volume" in df.columns:
        df[f"vol_z_96_{interval}"] = zscore(pd.to_numeric(df["Volume"], errors="coerce"), win=96)
    hi, lo = df.get(f"High", None), df.get(f"Low", None)
    if hi is not None and lo is not None:
        rng = (pd.to_numeric(hi, errors="coerce") - pd.to_numeric(lo, errors="coerce")).abs()
        df[f"range_z_96_{interval}"] = zscore(rng, win=96)

    # (3) EMA 크로스 바이너리 (자체 계산)
    if "Close" in df.columns:
        c = pd.to_numeric(df["Close"], errors="coerce")
        ema_fast = ema(c, 12); ema_slow = ema(c, 26)
        cross_up   = ((ema_fast.shift(1) <= ema_slow.shift(1)) & (ema_fast > ema_slow)).astype("int8")
        cross_down = ((ema_fast.shift(1) >= ema_slow.shift(1)) & (ema_fast < ema_slow)).astype("int8")
        df[f"bin_cross_up_{interval}"] = cross_up
        df[f"bin_cross_dn_{interval}"] = cross_down

    # (4) RSI 다이버전스 근사
    rsi_col = f"rsi_14_{interval}"
    if rsi_col in df.columns and "Close" in df.columns:
        r = pd.to_numeric(df[rsi_col], errors="coerce")
        c = pd.to_numeric(df["Close"], errors="coerce")
        mom_up = (r.diff() > 0).rolling(5, min_periods=5).sum()
        px_dn  = (c.diff() < 0).rolling(5, min_periods=5).sum()
        mom_dn = (r.diff() < 0).rolling(5, min_periods=5).sum()
        px_up  = (c.diff() > 0).rolling(5, min_periods=5).sum()
        df[f"div_bull_{interval}"] = ((mom_up>=3) & (px_dn>=3)).astype("int8")
        df[f"div_bear_{interval}"] = ((mom_dn>=3) & (px_up>=3)).astype("int8")

    # (5) 윈도우 통계 확장: 기존 수치 피처에 rolling mean/std zscore
    #  └ 반복적인 df[col] 추가로 인한 fragmentation 방지: 딕셔너리에 모아 한 번에 concat
    base_cols = [c for c in df.columns if c not in REF_COLS_CANON]
    new_feats: Dict[str, pd.Series] = {}

    for c in base_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.dtype.kind not in ("i","u","f"):  # 숫자만
            continue
        for w in HPO_EXPAND_WINDOWS:
            m  = s.rolling(w, min_periods=w).mean()
            sd = s.rolling(w, min_periods=w).std().replace(0, np.nan)
            new_feats[f"{c}_mean{w}"] = m
            new_feats[f"{c}_z{w}"]    = (s - m) / (sd + 1e-9)

    if new_feats:
        df = pd.concat([df, pd.DataFrame(new_feats, index=df.index)], axis=1).copy()  # defrag

    return _sanitize(df)

# ===== Feature Search & Scaler =====
def _make_proxy_y(df: pd.DataFrame, horizon: int) -> pd.Series:
    """RL 의사결정 지평선과 정합: 5m→+12, 15m→+4, 1h→+1, 4h→+1"""
    return (df["Close"].astype(float).pct_change(horizon, fill_method=None).shift(-horizon) > 0).astype(int).fillna(0)

def _feature_search_mi(X: pd.DataFrame, y: pd.Series, top_k: int) -> List[str]:
    X_ = _sanitize(X).astype(float)
    mi = mutual_info_classif(X_, y.values, random_state=RANDOM_STATE)
    scores = pd.Series(mi, index=X_.columns).sort_values(ascending=False)
    return scores.head(top_k).index.tolist()

def _feature_search_for_tf(train_df: pd.DataFrame, tf: str) -> List[str]:
    exclude = set(REF_COLS_CANON)
    feat_cols = [c for c in train_df.columns if c not in exclude]
    if not FEATURE_SEARCH or not feat_cols:
        keep = feat_cols
    else:
        H = {"5m": 12, "15m": 4, "1h": 1, "4h": 1}
        y_tr = _make_proxy_y(train_df, H.get(tf, 1))
        k = min(TOP_K_PER_TF.get(tf, len(feat_cols)), len(feat_cols))
        keep = _feature_search_mi(train_df[feat_cols], y_tr, k)
    path = FEATURE_LIST_PATH_FMT.format(tf=tf)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=2)
    print(f"[ok] feature_list[{tf}] = {len(keep)} -> {path}")
    return keep

def _fit_scaler_for_tf(train_df: pd.DataFrame, feat_list: List[str], tf: str, path_fmt: str = SCALER_PATH_FMT) -> StandardScaler:
    X = _sanitize(train_df[feat_list]).to_numpy(dtype=float)
    sc = StandardScaler().fit(X)
    path = path_fmt.format(tf=tf)
    joblib.dump(sc, path)
    print(f"[ok] scaler[{tf}] saved -> {path}")
    return sc

# ===== Helper: 접두 f_ 보장 =====
def _prefix_f(cols: List[str]) -> List[str]:
    return [c if c.startswith("f_") else f"f_{c}" for c in cols]

def _rename_with_f_prefix(df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    mapping = {}
    for c in feat_cols:
        fc = c if c.startswith("f_") else f"f_{c}"
        if fc != c:
            mapping[c] = fc
    return df.rename(columns=mapping)

# ===== Public Utils (HPO/Train에서 사용) =====
def load_processed(split: str, tf: str, mode: str = "auto") -> pd.DataFrame:
    """
    mode:
      - "auto": HPO 파일(feHPO_*)이 있으면 우선, 없으면 기본(fe_*)
      - "hpo" : HPO 전용 프레임 강제 로딩
      - "base": 기존 Top-K 프레임 로딩
    """
    base_p = os.path.join(OUT_DIR, f"fe_{split}_{tf}.parquet")
    hpo_p  = os.path.join(OUT_DIR, f"{HPO_OUT_PREFIX}_{split}_{tf}.parquet")
    path = None
    if mode == "hpo":
        path = hpo_p
    elif mode == "base":
        path = base_p
    else:
        path = hpo_p if os.path.exists(hpo_p) else base_p
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed frame not found: {path}")
    return pd.read_parquet(path)

def feature_universe(df: pd.DataFrame, prefix: str = "f_") -> List[str]:
    return [c for c in df.columns if c.startswith(prefix)]

def build_universe_from_processed(split: str = "train", tf: str = "5m", mode: str = "auto") -> List[str]:
    df = load_processed(split, tf, mode=mode)
    feats = feature_universe(df, prefix="f_")
    if len(feats) < 10:
        raise RuntimeError(f"Feature universe too small: {len(feats)} (check FE expansion).")
    return feats

def apply_feature_mask(df: pd.DataFrame, selected: List[str], ref_cols: List[str] | None = None) -> pd.DataFrame:
    if ref_cols is None:
        ref_cols = [c for c in REF_COLS_CANON if c in df.columns]
    X = df.reindex(columns=selected, fill_value=0.0)
    return pd.concat([X, df[ref_cols]], axis=1)

# ===== Main =====
def main():
    print("[1/6] Loading all raw data...")
    raw_data = {s: {tf: _load_raw(s, tf) for tf in TIMEFRAMES} for s in ["train", "val", "test"]}

    print("\n[2/6] Prepare BTC raw frames in-memory (no save).")
    btc_raw = {s: raw_data[s]["btc1h"] for s in ["train", "val", "test"]}

    print("\n[3/6] Computing ETH features with BTC lag/corr (no-leak)...")
    feature_data = {s: {} for s in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        for tf in ETH_TIMEFRAMES:
            print(f"  - Computing features for {split} / {tf}...")
            eth_df_raw = raw_data[split][tf]
            base = compute_features_for_tf(eth_df_raw, tf, btc_df=btc_raw[split])
            feature_data[split][tf] = base

    print(f"\n[4/6] Building TF-specific feature lists & scalers (BASE pipeline)...")
    feature_list_per_tf: Dict[str, List[str]] = {}
    scalers: Dict[str, StandardScaler] = {}
    for tf in TF_FOR_SEARCH:
        tr_df = feature_data["train"][tf]
        # BASE: Top-K 선택
        feat_list = _feature_search_for_tf(tr_df, tf)
        # 접두 f_ 보장
        feat_list_f = _prefix_f(feat_list)
        feature_list_per_tf[tf] = feat_list_f

        # 스케일러는 원래 컬럼명 기준으로 학습 후, 저장만 함 (적용 시엔 f_로 rename)
        sc = _fit_scaler_for_tf(tr_df, feat_list, tf=tf, path_fmt=SCALER_PATH_FMT)
        scalers[tf] = sc

    print("\n[5/6] Processing and saving all ETH timeframe data (BASE, Top-K)...")
    for split in ["train", "val", "test"]:
        for tf in ETH_TIMEFRAMES:
            print(f"  - [BASE] Processing {split} / {tf}...")
            df = feature_data[split][tf].copy()
            feat_list_no_f = [c[2:] if c.startswith("f_") else c for c in feature_list_per_tf[tf]]
            df_sel = df.reindex(columns=feat_list_no_f, fill_value=0.0)
            X = _sanitize(df_sel).to_numpy(dtype=float)
            Xs = scalers[tf].transform(X)
            df_scaled = pd.DataFrame(Xs, index=df.index, columns=feat_list_no_f)
            # f_ 접두로 rename
            df_scaled = _rename_with_f_prefix(df_scaled, feat_list_no_f)

            ref_cols = [c for c in REF_COLS_CANON if c in df.columns]
            final_df = pd.concat([df_scaled, df[ref_cols]], axis=1)

            out_p = os.path.join(OUT_DIR, f"fe_{split}_{tf}.parquet")
            final_df.to_parquet(out_p)
            print(f"    [ok] Saved BASE {split}/{tf}: {len(final_df):,} x {final_df.shape[1]} -> {out_p}")

    # ===== HPO 확장: 후보 추가 + NO Top-K + 별도 스케일 =====
    print("\n[6/6] Building & saving HPO-extended frames (NO Top-K, expanded features)...")
    # 6-1) train split에서 확장 후보 포함 전체 리스트 만들고 스케일러 학습
    hpo_feat_lists: Dict[str, List[str]] = {}
    hpo_scalers: Dict[str, StandardScaler] = {}
    expanded_train_frames: Dict[str, pd.DataFrame] = {}

    for tf in ETH_TIMEFRAMES:
        tr_base = feature_data["train"][tf]
        tr_hpo  = _add_hpo_candidates_local(tr_base, tf)
        # REF 제거 후 전체 컬럼
        feat_all = [c for c in tr_hpo.columns if c not in REF_COLS_CANON]
        # 접두 f_ 보장
        feat_all_f = _prefix_f(feat_all)
        hpo_feat_lists[tf] = feat_all_f
        # 스케일러 학습 (원래명 기준)
        sc_hpo = _fit_scaler_for_tf(tr_hpo, feat_all, tf=tf, path_fmt=HPO_SCALER_PATH_FMT)
        hpo_scalers[tf] = sc_hpo
        expanded_train_frames[tf] = tr_hpo
        # 목록 저장
        with open(HPO_FEATURE_LIST_FMT.format(tf=tf), "w", encoding="utf-8") as f:
            json.dump(feat_all_f, f, ensure_ascii=False, indent=2)
        print(f"    [ok] HPO feature universe [{tf}] = {len(feat_all_f)} (hint≤{HPO_MAX_FEATURES_HINT})")

    # 6-2) 모든 split/TF에 대해 HPO 프레임 스케일 & 저장
    for split in ["train", "val", "test"]:
        for tf in ETH_TIMEFRAMES:
            print(f"  - [HPO] Processing {split} / {tf}...")
            base_df = feature_data[split][tf]
            df_hpo  = _add_hpo_candidates_local(base_df, tf)

            feat_all_no_f = [c[2:] if c.startswith("f_") else c for c in hpo_feat_lists[tf]]
            df_sel = df_hpo.reindex(columns=feat_all_no_f, fill_value=0.0)
            X = _sanitize(df_sel).to_numpy(dtype=float)
            Xs = hpo_scalers[tf].transform(X)
            df_scaled = pd.DataFrame(Xs, index=df_hpo.index, columns=feat_all_no_f)
            df_scaled = _rename_with_f_prefix(df_scaled, feat_all_no_f)

            ref_cols = [c for c in REF_COLS_CANON if c in df_hpo.columns]
            final_df = pd.concat([df_scaled, df_hpo[ref_cols]], axis=1)

            out_p = os.path.join(OUT_DIR, f"{HPO_OUT_PREFIX}_{split}_{tf}.parquet")
            final_df.to_parquet(out_p)
            print(f"    [ok] Saved HPO {split}/{tf}: {len(final_df):,} x {final_df.shape[1]} -> {out_p}")

    print("\n[+] MTF Feature Engineering complete (BASE + HPO extended, leak-free, BTC integrated).")

if __name__ == "__main__":
    main()
