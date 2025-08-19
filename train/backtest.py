# backtest.py — Auto-run backtester (MTF/STF autodetect) with action diagnostics
# - Run directly: finds latest PPO .zip, builds data, runs backtest on split="test"
# - New:
#     * EVAL_DET=0 -> stochastic actions (like training exploration)
#     * ACTION_GAIN=1.0 -> multiply policy output to test amplitude
#     * EVAL_SMOOTH_A=0.25 -> action smoothing (lower = 더 과감)
#     * LOG_ACTION_CSV=1 -> save per-step action/pos/equity CSV
# - Outputs:
#     reports/backtest_trades.csv, backtest_chart.png, (optional) backtest_actions.csv

from __future__ import annotations
import os, json, math, warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ===== Paths =====
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
PROC_DIR   = os.path.join(DATA_DIR, "processed")
MODEL_DIR  = os.path.join(DATA_DIR, "model")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ===== Defaults / Evaluation toggles =====
SPLIT = os.getenv("BACKTEST_SPLIT", "test")
SAVE_CSV = True
SAVE_CHART = True

# Trading / Costs
WINDOW_STF = 48
FEE_BPS    = 0.0006
SLIP_BPS   = 0.0003
EVAL_MIN_DPOS = float(os.getenv("EVAL_MIN_DPOS", "0.00"))  # 0.00~0.02 권장
EVAL_COOLDOWN = int(os.getenv("EVAL_COOLDOWN", "0"))
EVAL_SMOOTH_A = float(os.getenv("EVAL_SMOOTH_A", "0.25"))
EVAL_DET      = int(os.getenv("EVAL_DET", "1"))            # 1: deterministic / 0: stochastic
ACTION_GAIN   = float(os.getenv("ACTION_GAIN", "1.0"))     # 행동 출력 스케일
LOG_ACTION_CSV= int(os.getenv("LOG_ACTION_CSV", "0"))

LEVERAGE   = 1.0
START_CAP  = 100_000.0

# MTF setup
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
WINDOWS_MTF = {"5m": 48, "15m": 32, "1h": 24, "4h": 12}

STEPS_PER_YEAR = 365 * 24 * 12  # 5m bars/year = 105,120

# ===== Utils =====
def _find_latest_model(model_dir: str) -> Optional[str]:
    if not os.path.isdir(model_dir):
        return None
    zips = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".zip")]
    if not zips:
        return None
    def score(p):
        name = os.path.basename(p).lower()
        bonus = (2 if "best" in name else 0) + (1 if "final" in name else 0)
        return (bonus, os.path.getmtime(p))
    zips.sort(key=score, reverse=True)
    return zips[0]

def _load_feature_list() -> List[str]:
    with open(os.path.join(PROC_DIR, "fe_feature_list_5m.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def _load_split_parquet(split: str, tf: str) -> pd.DataFrame:
    p = os.path.join(PROC_DIR, f"fe_{split}_{tf}.parquet")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_parquet(p).sort_index()
    if "Close" not in df.columns:
        df["Close"] = df.get("close_ref", 0.0)
    if "FundingRate" not in df.columns:
        df["FundingRate"] = 0.0
    if "Funding8h" not in df.columns:
        df["Funding8h"] = df["FundingRate"]
    return df

def _save_chart(ts: pd.DatetimeIndex, eq: np.ndarray, out_path: str):
    if len(eq) == 0: return
    eq_norm = eq / eq[0]
    plt.figure(figsize=(10, 4))
    plt.plot(ts, eq_norm, linewidth=1.0)
    plt.title("Equity Curve (normalized)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity (×)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()

# ===== Build data (STF) =====
def _build_data_stf(split: str, feat_cols: List[str], window: int) -> Dict[str, np.ndarray]:
    df = _load_split_parquet(split, "5m")
    X = df[[c for c in feat_cols if c in df.columns]].astype("float32").to_numpy(copy=False)
    close = df["Close"].astype("float64").to_numpy(copy=False)
    fund8h = df["Funding8h"].astype("float64").to_numpy(copy=False)
    N = len(df)
    if N <= window + 1:
        raise ValueError(f"Not enough rows for STF window={window}: {N}")
    T = N - window
    obs = np.empty((T, window * X.shape[1]), dtype=np.float32)
    for t in range(T):
        obs[t] = X[t:t+window].reshape(-1)
    ret  = (close[window:] - close[window-1:-1]) / np.maximum(close[window-1:-1], 1e-12)
    fund = fund8h[window:] / 96.0
    ts   = pd.to_datetime(df.index[window:], utc=True)
    return dict(obs=obs, ret=ret.astype(np.float64), fund=fund.astype(np.float64), ts=ts, price=close[window:].astype(np.float64))

# ===== Build data (MTF) =====
def _first_ready_ts(df: pd.DataFrame, feat_cols: List[str], window: int) -> pd.Timestamp:
    cols = [c for c in feat_cols if c in df.columns]
    if not cols:
        raise ValueError("No matching features")
    arr = df[cols].to_numpy(dtype=float, copy=False)
    finite = np.isfinite(arr).all(axis=1)
    mask = pd.Series(finite, index=df.index)
    ready = mask.rolling(window, min_periods=window).apply(lambda x: 1.0 if bool(np.all(x)) else 0.0)
    ts = ready[ready == 1.0].index.min()
    if ts is None: raise ValueError("No valid warm-up segment")
    return ts

def _align_mtf(dfs: Dict[str, pd.DataFrame], feat_cols: List[str]) -> Dict[str, pd.DataFrame]:
    start_ts = max(_first_ready_ts(dfs[tf], feat_cols, WINDOWS_MTF[tf]) for tf in TIMEFRAMES)
    base = dfs["5m"].loc[dfs["5m"].index >= start_ts]
    base_times = base.index
    out = {}
    for tf in TIMEFRAMES:
        d = dfs[tf].loc[dfs[tf].index >= start_ts]
        d = d.reindex(base_times).ffill()
        d = d.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        out[tf] = d
    common_idx = out["5m"].index
    for tf in TIMEFRAMES:
        common_idx = common_idx.intersection(out[tf].index)
    for tf in TIMEFRAMES:
        out[tf] = out[tf].reindex(common_idx)
    return out

def _build_data_mtf(split: str, feat_cols: List[str]) -> Dict[str, np.ndarray]:
    dfs = {tf: _load_split_parquet(split, tf) for tf in TIMEFRAMES}
    aligned = _align_mtf(dfs, feat_cols)
    base = aligned["5m"]
    max_w = max(WINDOWS_MTF.values())
    N = len(base)
    if N <= max_w + 1:
        raise ValueError(f"Not enough rows after MTF alignment: {N}")
    mtf_obs = {}
    total_dim = 0
    for tf in TIMEFRAMES:
        d = aligned[tf]
        w = WINDOWS_MTF[tf]
        cols = [c for c in feat_cols if c in d.columns]
        X = d[cols].astype("float32").to_numpy(copy=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        T = N - max_w
        obs = np.empty((T, w * len(cols)), dtype=np.float32)
        for t in range(T):
            s = max_w - w + t
            e = max_w + t
            obs[t] = X[s:e].reshape(-1)
        mtf_obs[tf] = obs
        total_dim += obs.shape[1]
        print(f"[MTF] {tf}: {len(cols)} features × {w} window = {obs.shape[1]} dims")
    T = min(arr.shape[0] for arr in mtf_obs.values())
    combined = np.empty((T, total_dim), dtype=np.float32)
    i = 0
    for tf in TIMEFRAMES:
        d = mtf_obs[tf][:T]
        combined[:, i:i+d.shape[1]] = d
        i += d.shape[1]
    close = base["Close"].astype("float64").to_numpy(copy=False)
    fund8h = base.get("Funding8h", base["FundingRate"]).astype("float64").to_numpy(copy=False)
    ret  = (close[max_w:max_w+T] - close[max_w-1:max_w+T-1]) / np.maximum(close[max_w-1:max_w+T-1], 1e-12)
    fund = fund8h[max_w:max_w+T] / 96.0
    ts   = pd.to_datetime(base.index[max_w:max_w+T], utc=True)
    print(f"[MTF] Combined obs shape: {combined.shape}")
    return dict(obs=combined, ret=ret.astype(np.float64), fund=fund.astype(np.float64), ts=ts, price=close[max_w:max_w+T].astype(np.float64))

# ===== Backtest core =====
@dataclass
class CostCfg:
    fee_bps: float = FEE_BPS
    slip_bps: float = SLIP_BPS
    min_dpos: float = EVAL_MIN_DPOS
    cooldown: int = EVAL_COOLDOWN
    smooth_a: float = EVAL_SMOOTH_A
    leverage: float = LEVERAGE

def _apply_action(a_raw: float, t: int, pos_target: float, pos_exec: float, cfg: CostCfg, last_change: int) -> Tuple[float, float, int]:
    if (t - last_change) < cfg.cooldown:
        a = pos_target
    else:
        a = (1 - cfg.smooth_a) * pos_target + cfg.smooth_a * float(np.clip(a_raw, -1.0, 1.0))
        delta = a - pos_target
        k = cfg.min_dpos
        if k > 0 and abs(delta) <= k:
            delta = delta * (abs(delta) / k)  # soft shrink
        a = pos_target + delta
        if abs(delta) > 1e-12:
            last_change = t
    dpos = a - pos_exec
    return a, dpos, last_change

def run_backtest(model: PPO, data: Dict[str, np.ndarray], cfg: CostCfg) -> Dict[str, any]:
    obs_mat = data["obs"]; rets = data["ret"]; funds = data["fund"]; ts = data["ts"]; price = data["price"]
    T = len(rets)
    pos_exec = 0.0; pos_target = 0.0; last_change = -10**9
    eq = START_CAP; eq_path = []

    # diagnostics
    fees=slips=funds_sum=0.0
    trades = 0
    turnover = 0.0
    raw_actions = np.zeros(T, dtype=float)
    pos_series  = np.zeros(T, dtype=float)
    dpos_series = np.zeros(T, dtype=float)

    det_flag = bool(EVAL_DET)

    for t in range(T):
        obs = obs_mat[t]
        a_raw, _ = model.predict(obs, deterministic=det_flag)
        a = float(a_raw[0]) * ACTION_GAIN
        raw_actions[t] = a
        pos_target, dpos, last_change = _apply_action(a, t, pos_target, pos_exec, cfg, last_change)
        fee  = cfg.fee_bps  * abs(dpos) * cfg.leverage
        slip = cfg.slip_bps * abs(dpos) * cfg.leverage
        pos_exec = pos_target

        pnl_ret = (pos_exec * cfg.leverage) * rets[t]
        fund_cost = pos_exec * funds[t]

        step_net = pnl_ret - fee - slip - fund_cost
        eq *= (1.0 + step_net)

        eq_path.append(eq); fees += fee; slips += slip; funds_sum += fund_cost
        pos_series[t]  = pos_exec
        dpos_series[t] = dpos
        if abs(dpos) > 1e-9:
            trades += 1
            turnover += abs(dpos)

    eq_arr = np.asarray(eq_path, dtype=float)
    total_ret = eq_arr[-1] / START_CAP - 1.0 if len(eq_arr) else 0.0
    step_net = np.diff(np.hstack(([START_CAP], eq_arr))) / np.maximum(np.hstack(([START_CAP], eq_arr[:-1])), 1e-12)
    mu = step_net.mean() if len(step_net) else 0.0
    sd = step_net.std(ddof=1) if len(step_net) > 1 else 0.0
    vol_ann = sd * math.sqrt(STEPS_PER_YEAR) if sd > 0 else 0.0
    sharpe  = (mu / sd * math.sqrt(STEPS_PER_YEAR)) if sd > 0 else 0.0
    if len(eq_arr):
        peak = np.maximum.accumulate(eq_arr); dd = eq_arr / np.maximum(peak, 1e-12) - 1.0; max_dd = dd.min()
    else:
        max_dd = 0.0

    # optional per-step action log
    if LOG_ACTION_CSV:
        df_log = pd.DataFrame({
            "ts": [str(x.to_pydatetime()) for x in ts],
            "price": data["price"].astype(float),
            "a_raw": raw_actions,
            "pos": pos_series,
            "dpos": dpos_series
        })
        df_log.to_csv(os.path.join(REPORT_DIR, "backtest_actions.csv"), index=False)

    # trade CSV (events only)
    if SAVE_CSV:
        rows = []
        for i in range(T):
            if abs(dpos_series[i]) > 1e-9:
                rows.append(dict(
                    ts=str(ts[i].to_pydatetime()),
                    price=float(price[i]),
                    a_raw=float(raw_actions[i]),
                    dpos=float(dpos_series[i]),
                    pos=float(pos_series[i])
                ))
        pd.DataFrame(rows).to_csv(os.path.join(REPORT_DIR, "backtest_trades.csv"), index=False)

    if SAVE_CHART and len(eq_arr):
        _save_chart(ts, eq_arr, os.path.join(REPORT_DIR, "backtest_chart.png"))

    # action stats
    abs_a = np.abs(raw_actions)
    abs_p = np.abs(pos_series)
    a_mean = float(abs_a.mean()) if len(abs_a) else 0.0
    a_p95  = float(np.percentile(abs_a, 95)) if len(abs_a) else 0.0
    p_mean = float(abs_p.mean()) if len(abs_p) else 0.0
    p_p95  = float(np.percentile(abs_p, 95)) if len(abs_p) else 0.0

    return dict(
        start=str(ts[0].to_pydatetime()) if len(ts) else "",
        end=str(ts[-1].to_pydatetime()) if len(ts) else "",
        bars=len(ts),
        final_capital=eq_arr[-1] if len(eq_arr) else START_CAP,
        total_return=total_ret, vol_annual=vol_ann, sharpe=sharpe, max_dd=max_dd,
        n_trades=trades, turnover=turnover,
        costs=dict(fees=fees, slippage=slips, funding=funds_sum),
        a_mean=a_mean, a_p95=a_p95, p_mean=p_mean, p_p95=p_p95,
        det=det_flag, gain=ACTION_GAIN, smooth=EVAL_SMOOTH_A
    )

# ===== Main =====
def main():
   warnings.filterwarnings("ignore")
   model_path = os.getenv("BACKTEST_MODEL") or _find_latest_model(MODEL_DIR)
   if not model_path:
       print(f"[error] No model .zip found in {MODEL_DIR}.")
       return

   print(f"[auto] Model: {model_path}")
   print(f"[auto] Split: {SPLIT}")

   try:
       model = PPO.load(model_path, device="cpu")
   except Exception:
       model = PPO.load(model_path, device="cpu", custom_objects={})

   feat_cols = _load_feature_list()

   mtf_ok = all(os.path.exists(os.path.join(PROC_DIR, f"fe_{SPLIT}_{tf}.parquet")) for tf in TIMEFRAMES)
   data = None; used = ""
   try:
       if mtf_ok:
           print("[auto] Building MTF dataset…")
           data = _build_data_mtf(SPLIT, feat_cols); used = "MTF"
       else:
           raise RuntimeError("MTF files missing")
   except Exception as e:
       print(f"[auto] MTF build failed ({e}). Falling back to STF.")
       data = _build_data_stf(SPLIT, feat_cols, WINDOW_STF); used = "STF"

   try:
       obs_dim = getattr(model.policy.observation_space, "shape", None)
       if obs_dim and len(obs_dim) == 1 and data["obs"].shape[1] != obs_dim[0]:
           print(f"[warn] Obs dim mismatch ({data['obs'].shape[1]} vs {obs_dim[0]}). Trying STF fallback…")
           data = _build_data_stf(SPLIT, feat_cols, WINDOW_STF); used = "STF"
   except Exception:
       pass

   print(f"[auto] Mode: {used} | Obs shape: {data['obs'].shape}")
   print(f"[cfg] DET={EVAL_DET} (0=stochastic), GAIN={ACTION_GAIN}, SMOOTH_A={EVAL_SMOOTH_A}, MIN_DPOS={EVAL_MIN_DPOS}, COOLDOWN={EVAL_COOLDOWN}")

   cfg = CostCfg(
       fee_bps=FEE_BPS, slip_bps=SLIP_BPS,
       min_dpos=EVAL_MIN_DPOS, cooldown=EVAL_COOLDOWN,
       smooth_a=EVAL_SMOOTH_A, leverage=LEVERAGE
   )

   print("[run] Backtest starting…")
   res = run_backtest(model, data, cfg)

   print("\n==================== BACKTEST SUMMARY ====================")
   print(f"기간: {res['start']} ~ {res['end']} | 바 수: {res['bars']:,}")
   print(f"초기 자본: ${START_CAP:,.0f} | 최종 자본: ${res['final_capital']:,.0f}")
   print(f"총 수익률: {res['total_return']*100:.2f}% | 연 변동성: {res['vol_annual']*100:.2f}% | 샤프: {res['sharpe']:.2f}")
   print(f"최대 손실: {res['max_dd']*100:.2f}%")
   fbps = res['costs']['fees']*1e4; sbps = res['costs']['slippage']*1e4; fundbps = res['costs']['funding']*1e4
   print(f"거래 수: {res['n_trades']:,} | 총 회전율(Σ|Δpos|): {res['turnover']*100:.2f}%")
   print(f"비용 → 수수료: {res['costs']['fees']*100:.4f}% ({fbps:.1f} bp) / "
         f"슬리피지: {res['costs']['slippage']*100:.4f}% ({sbps:.1f} bp) / "
         f"펀딩: {res['costs']['funding']*100:.4f}% ({fundbps:.1f} bp)")
   print(f"액션 통계 → mean|a|={res['a_mean']:.4f}, p95|a|={res['a_p95']:.4f}, mean|pos|={res['p_mean']:.4f}, p95|pos|={res['p_p95']:.4f}")
   
   # 저활동 감지 시 stochastic 모드 추가 실행
   LOW_ACTIVITY_THRESHOLD = 0.01  # 회전율 1% 미만
   LOW_TRADES_THRESHOLD = 200     # 거래 200건 미만
   
   if (res['turnover'] < LOW_ACTIVITY_THRESHOLD or 
       res['n_trades'] < LOW_TRADES_THRESHOLD or 
       res['a_mean'] < 0.02):
       
       print("\n[INFO] 저활동 감지 → stochastic 모드 3회 실행")
       
       # EVAL_DET 임시 변경
       original_eval_det = EVAL_DET
       globals()['EVAL_DET'] = False  # stochastic 모드
       
       try:
           stoch_results = []
           for i in range(3):
               stoch_res = run_backtest(model, data, cfg)
               stoch_results.append(stoch_res)
           
           # 평균 계산
           avg_return = sum(r['total_return'] for r in stoch_results) / 3
           avg_trades = sum(r['n_trades'] for r in stoch_results) / 3
           avg_turnover = sum(r['turnover'] for r in stoch_results) / 3
           avg_sharpe = sum(r['sharpe'] for r in stoch_results) / 3
           avg_fees = sum(r['costs']['fees'] for r in stoch_results) / 3
           avg_slips = sum(r['costs']['slippage'] for r in stoch_results) / 3
           avg_funding = sum(r['costs']['funding'] for r in stoch_results) / 3
           
           print("\n---------------- STOCHASTIC 모드 (MC=3) ----------------")
           print(f"평균 총 수익률: {avg_return*100:.2f}% | 평균 샤프: {avg_sharpe:.2f}")
           print(f"평균 거래 수: {avg_trades:.0f} | 평균 회전율: {avg_turnover*100:.2f}%")
           fbps = avg_fees*1e4; sbps = avg_slips*1e4; fundbps = avg_funding*1e4
           print(f"평균 비용 → 수수료: {avg_fees*100:.4f}% ({fbps:.1f} bp) / "
                 f"슬리피지: {avg_slips*100:.4f}% ({sbps:.1f} bp) / "
                 f"펀딩: {avg_funding*100:.4f}% ({fundbps:.1f} bp)")
           print("-------------------------------------------------------")
                 
       finally:
           globals()['EVAL_DET'] = original_eval_det  # 원복
   
   if SAVE_CSV:
       print(f"Trades CSV: {os.path.join(REPORT_DIR, 'backtest_trades.csv')}")
   if LOG_ACTION_CSV:
       print(f"Action CSV: {os.path.join(REPORT_DIR, 'backtest_actions.csv')}")
   if SAVE_CHART:
       print(f"Chart PNG : {os.path.join(REPORT_DIR, 'backtest_chart.png')}")
   print("==========================================================")

if __name__ == "__main__":
   main()