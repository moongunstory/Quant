# backtest_practitioner.py — Next-bar execution backtester (5m, crypto futures)
# - Uses processed features from fe.py: fe_{train|val|test}_5m.parquet + fe_feature_list_5m.json
# - Loads PPO model and replays deterministic policy
# - Costs ONLY when position changes (no double-count)
# - Funding applied per step: Funding8h/96 (fallback to FundingRate/96)
# - Outputs: summary to console, optional trades CSV and equity chart PNG
#
# Usage:
#   python ai_binance/train/backtest_practitioner.py \
#       --model ./ai_binance/data/model/ppo_practitioner_best.zip \
#       --split test --save-csv --save-chart

from __future__ import annotations
import os, json, math, argparse, warnings
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ===== Paths =====
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROC_DIR    = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
MODEL_DIR   = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "model"))
REPORT_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "reports"))
os.makedirs(REPORT_DIR, exist_ok=True)

# ===== Defaults (align with rl_practitioner.py) =====
WINDOW              = 48
FEE_BPS             = 0.0006     # taker fee per notional traded
SLIP_BPS            = 0.0003     # baseline slippage per turnover
MIN_DPOS            = 0.10       # ignore tiny Δpos
COOLDOWN            = 2          # bars to lock after a change
SMOOTH_ALPHA        = 0.25       # action smoothing toward target
LEVERAGE            = 1.0
TURN_BUDGET_DAILY   = 1.5        # turnover budget (per 288 steps)
START_CAPITAL       = 100_000.0

STEPS_PER_YEAR = 365 * 24 * 12   # 5m bars in 1y (24/7)

@dataclass
class CostConfig:
    fee_bps: float = FEE_BPS
    slip_bps: float = SLIP_BPS
    min_dpos: float = MIN_DPOS
    cooldown: int = COOLDOWN
    smooth_alpha: float = SMOOTH_ALPHA
    leverage: float = LEVERAGE

# ===== IO =====
def load_processed(split: str) -> Tuple[pd.DataFrame, List[str]]:
    X = pd.read_parquet(os.path.join(PROC_DIR, f"fe_{split}_5m.parquet"))
    with open(os.path.join(PROC_DIR, "fe_feature_list_5m.json"), "r", encoding="utf-8") as f:
        feat_cols = json.load(f)
    # sanity for required refs
    for c in ["close_ref", "FundingRate"]:
        if c not in X.columns:
            X[c] = 0.0
    if "Funding8h" not in X.columns:
        X["Funding8h"] = X["FundingRate"]
    X = X.sort_index()
    return X, feat_cols

def build_windows(df: pd.DataFrame, feat_cols: List[str], window: int) -> Dict[str, np.ndarray]:
    idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df.index, utc=True)
    F = len(feat_cols)
    N = len(df)
    if N <= window + 1:
        raise ValueError("Not enough rows for windowing")

    Xf = df[feat_cols].values.astype(np.float32)
    close = df["close_ref"].astype("float64").values
    fund8h = df["Funding8h"].astype("float64").values

    T = N - window
    obs = np.empty((T, window * F), dtype=np.float32)
    for t in range(T):
        obs[t] = Xf[t:t+window].reshape(-1)

    # next-bar return (obs at t → price move from t+window-1 → t+window)
    step_ret = (close[window:] - close[window-1:-1]) / np.maximum(close[window-1:-1], 1e-12)
    fund_step = fund8h[window:] / 96.0
    ts = pd.to_datetime(idx[window:], utc=True)
    px = close[window:]  # reference price for logging
    return dict(obs=obs, ret=step_ret.astype(np.float64), fund=fund_step.astype(np.float64),
                ts=ts, price=px.astype(np.float64))

# ===== Backtest Core =====
def _apply_action(a_raw: float, t: int, pos_target: float, pos_exec: float, cfg: CostConfig, last_change: int) -> Tuple[float, float, int]:
    # cooldown
    if (t - last_change) < cfg.cooldown:
        a = pos_target
    else:
        # smooth toward raw
        a = (1 - cfg.smooth_alpha) * pos_target + cfg.smooth_alpha * float(np.clip(a_raw, -1.0, 1.0))
        if abs(a - pos_target) < cfg.min_dpos:
            a = pos_target
        else:
            last_change = t
    dpos = a - pos_exec
    return a, dpos, last_change

def backtest(model: PPO, data: Dict[str, np.ndarray], cfg: CostConfig,
             window: int, save_csv: Optional[str] = None) -> Dict[str, any]:
    obs_mat = data["obs"]
    rets    = data["ret"]
    funds   = data["fund"]
    ts      = data["ts"]
    price   = data["price"]
    T = len(rets)

    pos_exec = 0.0
    pos_target = 0.0
    last_change = -10**9

    # logs
    eq = START_CAPITAL
    eq_path = []
    step_costs = []
    step_pnl = []
    step_pos = []
    step_ret = []
    step_fee = []
    step_slip = []
    step_fund = []
    trades = []  # only when position changes

    for t in range(T):
        obs = obs_mat[t]
        a_raw, _ = model.predict(obs, deterministic=True)

        # next-bar execution, costs on turnover
        pos_target, dpos, last_change = _apply_action(
            float(a_raw[0]), t, pos_target, pos_exec, cfg, last_change
        )
        fee = cfg.fee_bps  * abs(dpos) * cfg.leverage
        slip= cfg.slip_bps * abs(dpos) * cfg.leverage

        # execute now
        pos_exec = pos_target

        # pnl over this bar with executed pos
        pnl_ret = (pos_exec * cfg.leverage) * rets[t]
        fund_cost = pos_exec * funds[t]  # +rate costs longs, -rate costs shorts

        # equity update (λ 같은 학습용 벌점은 없음)
        step_net = pnl_ret - fee - slip - fund_cost
        eq *= (1.0 + step_net)

        # logs
        eq_path.append(eq)
        step_costs.append(fee + slip + fund_cost)
        step_pnl.append(pnl_ret)
        step_pos.append(pos_exec)
        step_ret.append(rets[t])
        step_fee.append(fee)
        step_slip.append(slip)
        step_fund.append(fund_cost)

        if abs(dpos) >= cfg.min_dpos:
            trades.append(dict(
                ts=str(ts[t].to_pydatetime()),
                price=float(price[t]),
                dpos=float(dpos),
                pos=float(pos_exec),
                fee=float(fee),
                slip=float(slip),
                fund=float(fund_cost),
                pnl_ret=float(pnl_ret),
                step_net=float(step_net)
            ))

    # metrics
    eq_arr = np.asarray(eq_path, dtype=float)
    rets_net = np.asarray(step_pnl, dtype=float) - (np.asarray(step_fee)+np.asarray(step_slip)+np.asarray(step_fund))
    # avoid nan
    rets_net = np.nan_to_num(rets_net, nan=0.0, posinf=0.0, neginf=0.0)

    total_ret = eq_arr[-1] / START_CAPITAL - 1.0 if len(eq_arr) else 0.0
    # per-step net return series for sharpe/vol
    step_net = rets_net
    mu = step_net.mean() if len(step_net) else 0.0
    sd = step_net.std(ddof=1) if len(step_net) > 1 else 0.0
    vol_annual = (sd * math.sqrt(STEPS_PER_YEAR)) if sd > 0 else 0.0
    sharpe = (mu / sd * math.sqrt(STEPS_PER_YEAR)) if sd > 0 else 0.0

    # max drawdown
    if len(eq_arr):
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr / np.maximum(peak, 1e-12)) - 1.0
        max_dd = dd.min()
    else:
        max_dd = 0.0

    # trade stats
    n_trades = len(trades)
    # per-trade PnL by aggregating steps until position flips to 0 or sign change
    hit, hold_bars = 0, []
    if n_trades:
        # reconstruct trade segments
        pos_series = np.asarray(step_pos)
        # mark trade boundaries when pos goes 0->nonzero, or sign flips
        boundaries = []
        prev = 0.0
        start_idx = None
        for i, p in enumerate(pos_series):
            if prev == 0.0 and p != 0.0:
                start_idx = i
            # close when back to 0 or sign flip
            if start_idx is not None:
                if (p == 0.0) or (np.sign(p) != np.sign(pos_series[start_idx])):
                    boundaries.append((start_idx, i))
                    start_idx = None
            prev = p
        if start_idx is not None:
            boundaries.append((start_idx, len(pos_series)-1))
        # compute per-trade net
        for s,e in boundaries:
            pnl_tr = (np.asarray(step_pnl[s:e+1]) - (np.asarray(step_fee[s:e+1]) +
                                                     np.asarray(step_slip[s:e+1]) +
                                                     np.asarray(step_fund[s:e+1]))).sum()
            hit += 1 if pnl_tr > 0 else 0
            hold_bars.append(e - s + 1)
    hit_rate = (hit / len(hold_bars)) if hold_bars else 0.0
    avg_hold = (np.mean(hold_bars) if hold_bars else 0.0)

    # CSV
    if save_csv:
        df_tr = pd.DataFrame(trades)
        df_tr.to_csv(save_csv, index=False)

    return dict(
        start=str(ts[0].to_pydatetime()) if len(ts) else "",
        end=str(ts[-1].to_pydatetime()) if len(ts) else "",
        bars=len(ts),
        final_capital=eq_arr[-1] if len(eq_arr) else START_CAPITAL,
        total_return=total_ret,
        vol_annual=vol_annual,
        sharpe=sharpe,
        max_dd=max_dd,
        n_trades=n_trades,
        hit_rate=hit_rate,
        avg_hold_bars=avg_hold,
        costs=dict(
            fees=float(np.sum(step_fee)),
            slippage=float(np.sum(step_slip)),
            funding=float(np.sum(step_fund))
        ),
        eq_path=eq_arr,
        ts=ts
    )

# ===== Plot =====
def save_chart(ts: pd.DatetimeIndex, eq: np.ndarray, out_path: str):
    if len(eq) == 0:
        return
    eq_norm = eq / eq[0]
    plt.figure(figsize=(10,4))
    plt.plot(ts, eq_norm, linewidth=1.0)
    plt.title("Equity Curve (normalized)")
    plt.xlabel("Time (UTC)")
    plt.ylabel("Equity (×)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

# ===== CLI =====
def parse_args():
    p = argparse.ArgumentParser(description="Practitioner-style backtest (5m, next-bar execution)")
    p.add_argument("--model", type=str, required=True, help="Path to PPO .zip")
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--window", type=int, default=WINDOW, help="Must match training window")
    p.add_argument("--fee-bps", type=float, default=FEE_BPS)
    p.add_argument("--slip-bps", type=float, default=SLIP_BPS)
    p.add_argument("--min-dpos", type=float, default=MIN_DPOS)
    p.add_argument("--cooldown", type=int, default=COOLDOWN)
    p.add_argument("--smooth-alpha", type=float, default=SMOOTH_ALPHA)
    p.add_argument("--leverage", type=float, default=LEVERAGE)
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--save-chart", action="store_true")
    return p.parse_args()

def main():
    warnings.filterwarnings("ignore")
    args = parse_args()

    # Load data
    print("RL Model Backtest Started!")
    print("Loading data…")
    df, feat_cols = load_processed(args.split)
    data = build_windows(df, feat_cols, args.window)

    # Load model (safe)
    print(f"Loading model: {args.model}")
    try:
        model = PPO.load(args.model, device="cpu")
    except Exception as e:
        # fallback without custom_objects in case of schedules
        model = PPO.load(args.model, device="cpu", custom_objects={})

    # Config
    cfg = CostConfig(
        fee_bps=args.fee_bps, slip_bps=args.slip_bps,
        min_dpos=args.min_dpos, cooldown=args.cooldown,
        smooth_alpha=args.smooth_alpha, leverage=args.leverage
    )

    # Backtest
    print("Running backtest simulation…")
    trades_csv = None
    if args.save_csv:
        trades_csv = os.path.join(REPORT_DIR, "backtest_trades.csv")
    res = backtest(model, data, cfg, args.window, save_csv=trades_csv)

    # Chart
    chart_path = None
    if args.save_chart:
        chart_path = os.path.join(REPORT_DIR, "backtest_chart.png")
        save_chart(res["ts"], res["eq_path"], chart_path)

    # Summary
    print("\n============================================================")
    print("BACKTEST RESULTS SUMMARY")
    print("============================================================")
    print(f"기간: {res['start']} ~ {res['end']} | 바 수: {res['bars']:,}")
    print(f"초기 자본: ${START_CAPITAL:,.0f}")
    print(f"최종 자본: ${res['final_capital']:,.0f}")
    print(f"총 수익률: {res['total_return']*100:.2f}%")
    print(f"연 변동성: {res['vol_annual']*100:.2f}%")
    print(f"샤프비율: {res['sharpe']:.2f}")
    print(f"최대 손실: {res['max_dd']*100:.2f}%")
    print("비용 합계:")
    print(f"  수수료: {res['costs']['fees']*100:.2f}% | 슬리피지: {res['costs']['slippage']*100:.2f}% | 펀딩: {res['costs']['funding']*100:.2f}%")
    print("거래 통계:")
    print(f"  거래 수: {res['n_trades']:,} | 승률: {res['hit_rate']*100:.2f}% | 평균 보유봉: {res['avg_hold_bars']:.1f}")
    if trades_csv:
        print(f"CSV 저장: {trades_csv}")
    if chart_path:
        print(f"차트 저장: {chart_path}")

if __name__ == "__main__":
    main()
