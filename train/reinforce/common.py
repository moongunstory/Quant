from __future__ import annotations
import os, json
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

# ===== Paths =====
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
# processed는 ai_binance/data/processed (원본 그대로)
PROC_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "processed"))
# ★ FIX: model도 ai_binance/data/model로 통일 (기존 train/data/model → 불일치)
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "data", "model"))
os.makedirs(MODEL_DIR, exist_ok=True)

# ===== Core Config (전액 진입/전액 청산) =====
FEE_RATE   = 0.0005
SLIP_BP    = 1.0
TIMING_K   = 4                # 5m 기준 20분
TIMING_K_COEF = 1.0           # 타점 shaping 강화
TURN_PENALTY  = 1.5 * FEE_RATE
FLIP_PENALTY  = 3.0 * FEE_RATE

# Manager 보상 가중치
M_W1 = 1.0
M_W3 = 0.5
M_FLIP = 0.2
M_MIS  = 0.2

# Worker 탐색/게이트
GATE_WARMUP_STEPS     = 20_000
ALIGN_EPS             = 0.08
OPPORTUNITY_COST      = 0.50 * FEE_RATE
GATE_SOFT_PENALTY_MULT= 0.5      # 15m 게이트 불일치 시 패널티(수수료 배수)

# ===== Utils: IO =====
def _path_fe(split: str, tf: str) -> str:
    fn = f"fe_{split}_{tf}.parquet" if tf != "btc1h" else f"fe_{split}_btc1h.parquet"
    return os.path.join(PROC_DIR, fn)

def _load_fe(split: str, tf: str) -> pd.DataFrame:
    p = _path_fe(split, tf)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()

def _feat_list(tf: str) -> List[str]:
    if tf == "btc1h":
        return []
    p = os.path.join(PROC_DIR, f"fe_feature_list_{tf}.json")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ===== Indicators =====
def _zscore(s: pd.Series, win: int = 100) -> pd.Series:
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std().replace(0, np.nan)
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def _heikin_ashi_ohlc(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    O = df["Open"].astype(float).to_numpy()
    H = df["High"].astype(float).to_numpy()
    L = df["Low"].astype(float).to_numpy()
    C = df["Close"].astype(float).to_numpy()
    n = len(df)
    HA_C = (O+H+L+C)/4.0
    HA_O = np.empty(n, dtype=float)
    HA_O[0] = (O[0] + C[0]) / 2.0
    for i in range(1, n):
        HA_O[i] = (HA_O[i-1] + HA_C[i-1]) / 2.0
    return pd.Series(HA_C, index=df.index), pd.Series(HA_O, index=df.index)

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    pc = c.shift(1)
    tr = pd.concat([(h-l), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# ===== Data assembly =====
def build_worker_inputs(split: str) -> Dict[str, pd.Series | pd.DataFrame]:
    df5  = _load_fe(split, "5m")
    df15 = _load_fe(split, "15m")
    df1h = _load_fe(split, "1h")
    df4h = _load_fe(split, "4h")
    btc1 = _load_fe(split, "btc1h")

    f5  = df5[_feat_list("5m")]
    f15 = df15[_feat_list("15m")].reindex(df5.index, method="ffill")
    f1h = df1h[_feat_list("1h")].reindex(df5.index, method="ffill")
    f4h = df4h[_feat_list("4h")].reindex(df5.index, method="ffill")
    cols_b = [c for c in btc1.columns if c.endswith("_btc1h")]
    fb  = btc1[cols_b].reindex(df5.index, method="ffill")

    X = pd.concat([f5, f15, f1h, f4h, fb], axis=1).replace([np.inf,-np.inf],0.0).fillna(0.0)

    # OHLC for rewards & gates
    ohlc5  = df5[["Open","High","Low","Close"]]
    ohlc15 = df15[["Open","High","Low","Close"]].reindex(df5.index, method="ffill")
    ohlc4h = df4h[["Open","High","Low","Close"]].reindex(df5.index, method="ffill")

    # 15m gate (HA_BC sign)
    ha_c_15, ha_o_15 = _heikin_ashi_ohlc(ohlc15)
    gate15_sign = np.sign(ha_c_15 - ha_o_15).astype(int)

    # 4h regime (완화된 약레짐 기준)
    ha_c_4h, ha_o_4h = _heikin_ashi_ohlc(ohlc4h)
    bc4h  = ha_c_4h - ha_o_4h
    atr4h = _atr(ohlc4h)
    weak4h = (bc4h.abs() < (bc4h.abs().rolling(200, min_periods=50).median()*0.15).fillna(np.inf)) \
             & (_zscore(atr4h, 200) < -0.8)
    reg4h_sign = np.sign(bc4h).astype(int)

    return dict(
        X=X,
        price=ohlc5["Close"].astype(float),
        gate15_sign=gate15_sign.reindex(df5.index).fillna(0),
        reg4h_weak=weak4h.reindex(df5.index).fillna(True),
        reg4h_sign=reg4h_sign.reindex(df5.index).fillna(0)
    )

def build_manager_inputs(split: str) -> Dict[str, pd.Series | pd.DataFrame]:
    df1h = _load_fe(split, "1h")
    df4h = _load_fe(split, "4h")

    XH = pd.concat(
        [df1h[_feat_list("1h")], df4h[_feat_list("4h")].reindex(df1h.index, method="ffill")],
        axis=1
    ).replace([np.inf,-np.inf],0.0).fillna(0.0)

    ohlc4h = df4h[["Open","High","Low","Close"]]
    ha_c_4h, ha_o_4h = _heikin_ashi_ohlc(ohlc4h)
    bc4h  = ha_c_4h - ha_o_4h
    atr4h = _atr(ohlc4h)
    weak4h = (bc4h.abs() < (bc4h.abs().rolling(200, min_periods=50).median()*0.15).fillna(np.inf)) \
             & (_zscore(atr4h, 200) < -0.8)
    reg4h_sign = np.sign(bc4h).astype(int)

    return dict(
        XH=XH,
        price=df1h["Close"].astype(float),
        reg4h_weak=weak4h.reindex(XH.index, method="ffill").fillna(True),
        reg4h_sign=reg4h_sign.reindex(XH.index, method="ffill").fillna(0)
    )

# ===== Goal Bridge =====
class Goal:
    __slots__ = ("dir","conf")
    def __init__(self, dir_: int = 0, conf: float = 0.0):
        self.dir = int(np.sign(dir_))
        self.conf = float(np.clip(conf, 0.0, 1.0))

class GoalBridge:
    def __init__(self):
        self.cur = Goal(0, 0.0)
    def set(self, g: Goal): self.cur = g
    def vec(self) -> np.ndarray:
        return np.array([self.cur.dir, self.cur.conf], dtype=np.float32)

# ===== Entropy Decay (탐색 → 수렴) =====
class EntropyDecay(BaseCallback):
    def __init__(self, start=0.10, end=0.03, decay_steps=600_000, verbose=0):
        super().__init__(verbose)
        self.start, self.end, self.decay = start, end, decay_steps
    def _on_training_start(self):
        self.model.ent_coef = self.start
    def _on_step(self):
        step = self.num_timesteps
        frac = min(1.0, step / self.decay)
        self.model.ent_coef = float(self.start + (self.end - self.start) * frac)
        return True
