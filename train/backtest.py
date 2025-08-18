"""
backtest.py — Backtesting module (REV-1) for rl.py policy using fe.py outputs (ETHUSDT 5m)
- Fix: **No slippage double-count** → slippage only via execution price; costs exclude slippage.
- Align: **min_hold=24, cooldown=6, alpha_flip=3bp** with rl.py.
- Equity model: multiplicative; per-bar funding (8h → 96×5m).
- Outputs: summary, optional trades CSV & equity chart.

Usage:
    python ai_binance/train/backtest.py --model ./ai_binance/data/model/best_model.zip --split test --save-csv --save-chart
"""
from __future__ import annotations

import os
import math
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ===== Paths (aligned with fe.py & rl.py) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "model"))
REPORT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "reports"))
os.makedirs(REPORT_DIR, exist_ok=True)

INTERVAL = "5m"
TRAIN_P = os.path.join(PROC_DIR, f"fe_train_{INTERVAL}.parquet")
VAL_P   = os.path.join(PROC_DIR, f"fe_val_{INTERVAL}.parquet")
TEST_P  = os.path.join(PROC_DIR, f"fe_test_{INTERVAL}.parquet")
FEAT_P  = os.path.join(PROC_DIR, f"fe_feature_list_{INTERVAL}.json")

# ===== Trading parameters (mirror rl.py REV-1) =====
@dataclass
class Cfg:
    fee_rate: float = 0.0006          # taker per side (6bp)
    slip_bp: float = 0.0002           # slippage fraction (2bp) — applied to exec prices only
    min_hold: int = 24                # bars
    max_hold: int = 96                # bars (~8h)
    stop_pct: float = 0.010           # 1%
    stop_atr_mult: float = 2.0        # 2× ATR stop
    trail_mult: Optional[float] = 1.2 # trailing multiple of stop width (None to disable)
    leverage: float = 1.0
    funding_split: int = 96           # 8h → 96×5m
    daily_dd_limit: Optional[float] = None  # e.g., 0.03 for -3% daily limit; None to disable
    alpha_flip_bp: float = 0.0003     # extra penalty when changing position (3bp)
    cooldown_bars: int = 6            # wait after any trade


# ===== Helpers =====

def _ewm(arr: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return _ewm(tr, alpha=1/14)


def _max_drawdown(eqs: np.ndarray) -> float:
    peak = -np.inf
    mdd = 0.0
    for x in eqs:
        peak = max(peak, x)
        mdd = min(mdd, x/peak - 1.0)
    return mdd


def _annual_factor_5m() -> float:
    # 5m bars/year ≈ 365*24*12 = 105120
    return math.sqrt(365*24*12)


def load_split(split: str) -> pd.DataFrame:
    pmap = {"train": TRAIN_P, "val": VAL_P, "test": TEST_P}
    if split not in pmap:
        raise ValueError("split must be one of: train, val, test")
    p = pmap[split]
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing processed split: {p}")
    return pd.read_parquet(p)


def load_feature_list() -> List[str]:
    import json
    if not os.path.exists(FEAT_P):
        raise FileNotFoundError(f"Missing feature list: {FEAT_P}")
    with open(FEAT_P, "r", encoding="utf-8") as f:
        return list(json.load(f))


def a_to_pos(a: int) -> int:
    return (-1, 0, +1)[a]


# ===== Core backtester =====

def backtest(df: pd.DataFrame, feature_cols: List[str], model: PPO, cfg: Cfg, initial_equity: float = 100_000.0,
             timesteps: int = -1, deterministic: bool = True) -> Dict[str, Any]:
    # Refs & features
    for c in ["Open", "High", "Low", "Close", "FundingRate", "close_ref"]:
        if c not in df.columns:
            df[c] = 0.0
    close_ref = df["close_ref"].astype(float).to_numpy()
    close = df["Close"].astype(float).to_numpy()
    high = df["High"].astype(float).to_numpy()
    low = df["Low"].astype(float).to_numpy()
    fr = df["FundingRate"].astype(float).to_numpy()
    X = df[feature_cols].astype(float).to_numpy()

    atr = _atr14(high, low, close)

    n = len(df)
    start = 1
    end = n - 1
    if timesteps > 0:
        end = min(end, start + timesteps)

    equity = 1.0  # normalized
    eq_hist = [equity]
    pos_hist = []

    pos = 0
    entry_price = None
    hold_bars = 0
    peak_pnl = 0.0  # max (long) or min (short) of entry-based pnl during hold
    funding_accum = 0.0

    trades = []  # list of dicts
    entry_idx = None

    # Cooldown and daily DD tracking
    cooldown = 0
    idx = df.index
    cur_day = None
    day_peak = 1.0

    for i in range(start, end):
        price_t = float(close_ref[i]) or float(close[i])
        price_tp1 = float(close_ref[i+1]) or float(close[i+1])

        # Cooldown decay
        if cooldown > 0:
            cooldown -= 1

        # Policy action
        obs = X[i]
        action, _ = model.predict(obs, deterministic=deterministic)
        want_pos = a_to_pos(int(action))

        # Risk guards
        forced_exit = False
        if pos != 0 and entry_price is not None:
            ret_from_entry = (price_t / entry_price - 1.0) * (1 if pos == +1 else -1)
            stop_atr = cfg.stop_atr_mult * (atr[i] / price_t)
            stop_cut = min(cfg.stop_pct, stop_atr)
            # Stop
            if ret_from_entry <= -stop_cut:
                forced_exit = True
            # Trailing
            if (cfg.trail_mult is not None) and not forced_exit:
                dd = (peak_pnl - ret_from_entry) if pos == +1 else (ret_from_entry - peak_pnl)
                if dd >= cfg.trail_mult * stop_cut:
                    forced_exit = True
            # Max hold
            if hold_bars >= cfg.max_hold and not forced_exit:
                forced_exit = True

        want_flip = (pos != 0) and (want_pos == -pos)
        want_flat = (pos != 0) and (want_pos == 0)

        # Min-hold gate
        if pos != 0 and hold_bars < cfg.min_hold:
            want_flip = False
            want_flat = False

        # Execute exit
        if pos != 0 and (forced_exit or want_flat or want_flip):
            # Exit exec price & fees — slippage only via price
            exec_px = price_t * (1 - cfg.slip_bp) if pos == +1 else price_t * (1 + cfg.slip_bp)
            trade_ret = (exec_px / entry_price - 1.0) * pos * cfg.leverage
            exit_fee = cfg.fee_rate
            equity *= (1.0 + trade_ret) * (1.0 - exit_fee)

            # Save trade
            trades.append({
                "entry_time": idx[entry_idx],
                "exit_time": idx[i],
                "side": "LONG" if pos == +1 else "SHORT",
                "entry_price": float(entry_price),
                "exit_price": float(exec_px),
                "bars": int(hold_bars),
                "ret": float(trade_ret),
                "funding": float(funding_accum),
                "fee_entry": float(cfg.fee_rate),
                "fee_exit": float(exit_fee),
                "alpha_flip": float(cfg.alpha_flip_bp if True else 0.0),
            })

            # Flat out
            pos = 0
            entry_price = None
            hold_bars = 0
            peak_pnl = 0.0
            funding_accum = 0.0
            cooldown = cfg.cooldown_bars

        # Execute entry (respect cooldown)
        can_enter = (pos == 0) and (want_pos != 0) and (not forced_exit) and (cooldown == 0)
        if can_enter:
            pos = want_pos
            exec_px = price_t * (1 + cfg.slip_bp) if pos == +1 else price_t * (1 - cfg.slip_bp)
            entry_price = exec_px
            # Entry costs: fee + flip penalty; NO slippage here (in price)
            entry_cost = cfg.fee_rate + cfg.alpha_flip_bp
            equity *= (1.0 - entry_cost)
            hold_bars = 0
            peak_pnl = 0.0
            funding_accum = 0.0
            entry_idx = i
            cooldown = cfg.cooldown_bars

        # Running PnL for bar t→t+1 & funding
        bar_ret = (price_tp1 / price_t - 1.0) * pos * cfg.leverage
        funding_step = pos * (fr[i] / cfg.funding_split) * cfg.leverage
        equity *= (1.0 + bar_ret - funding_step)
        if pos != 0:
            funding_accum += -funding_step  # positive if we PAID

        # Peak pnl for trailing
        if pos != 0 and entry_price is not None:
            cur = (price_t / entry_price - 1.0) * (1 if pos == +1 else -1)
            if pos == +1:
                peak_pnl = max(peak_pnl, cur)
            else:
                peak_pnl = min(peak_pnl, cur)
            hold_bars += 1

        # Daily DD limit (optional)
        if cfg.daily_dd_limit is not None:
            d = idx[i].date() if hasattr(idx[i], 'date') else None
            if cur_day is None:
                cur_day = d
                day_peak = equity
            if d != cur_day:
                cur_day = d
                day_peak = equity
            dd = equity / (day_peak if day_peak > 0 else 1.0) - 1.0
            day_peak = max(day_peak, equity)
            if dd <= -cfg.daily_dd_limit and pos != 0:
                # force flat (sell at price_t)
                exec_px = price_t * (1 - cfg.slip_bp) if pos == +1 else price_t * (1 + cfg.slip_bp)
                trade_ret = (exec_px / entry_price - 1.0) * pos * cfg.leverage
                exit_fee = cfg.fee_rate
                equity *= (1.0 + trade_ret) * (1.0 - exit_fee)
                trades.append({
                    "entry_time": idx[entry_idx],
                    "exit_time": idx[i],
                    "side": "LONG" if pos == +1 else "SHORT",
                    "entry_price": float(entry_price),
                    "exit_price": float(exec_px),
                    "bars": int(hold_bars),
                    "ret": float(trade_ret),
                    "funding": float(funding_accum),
                    "fee_entry": float(cfg.fee_rate),
                    "fee_exit": float(exit_fee),
                    "alpha_flip": float(cfg.alpha_flip_bp),
                })
                pos = 0
                entry_price = None
                hold_bars = 0
                peak_pnl = 0.0
                funding_accum = 0.0
                cooldown = cfg.cooldown_bars

        eq_hist.append(equity)
        pos_hist.append(pos)

    eq = np.array(eq_hist)
    rets = np.diff(eq) / np.clip(eq[:-1], 1e-12, None)
    ann_factor = _annual_factor_5m()
    sharpe = (np.mean(rets) / (np.std(rets) + 1e-12)) * ann_factor if len(rets) > 1 else 0.0
    vol = np.std(rets) * ann_factor if len(rets) > 1 else 0.0
    mdd = _max_drawdown(eq)

    # Trade stats
    n_trades = len(trades)
    wins = sum(1 for t in trades if t["ret"] > 0)
    win_rate = (wins / n_trades) if n_trades > 0 else 0.0
    avg_hold = (sum(t["bars"] for t in trades) / n_trades) if n_trades > 0 else 0.0

    out = {
        "equity_curve": eq * (initial_equity),
        "final_equity": float(eq[-1] * initial_equity),
        "total_return": float(eq[-1] - 1.0),
        "volatility": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "n_trades": int(n_trades),
        "win_rate": float(win_rate),
        "avg_hold_bars": float(avg_hold),
        "pos_series": np.array(pos_hist, dtype=int),
        "trades": trades,
    }
    return out


# ===== Reporting =====

def save_report(res: Dict[str, Any], split: str, model_path: str, save_csv: bool, save_chart: bool) -> None:
    # Print summary
    print("============================================================")
    print("BACKTEST RESULTS SUMMARY")
    print("============================================================")
    print(f"모델: {os.path.basename(model_path)}")
    print(f"구간: {split.upper()} | 바 수: {len(res['equity_curve'])-1:,}")
    print("수익성:")
    print(f"   초기 자본: ${100_000:,.0f}")
    print(f"   최종 자본: ${res['final_equity']:,.0f}")
    print(f"   총 수익률: {res['total_return']*100:.2f}%")
    print("리스크:")
    print(f"   변동성(연): {res['volatility']*100:.2f}%")
    print(f"   샤프: {res['sharpe']:.2f}")
    print(f"   최대낙폭: {res['max_drawdown']*100:.2f}%")
    print("거래 통계:")
    print(f"   총 거래수: {res['n_trades']}")
    print(f"   승률: {res['win_rate']*100:.2f}%")
    print(f"   평균 보유: {res['avg_hold_bars']:.1f} 바")

    # Optional artifacts
    stem = f"backtest_{split}_{os.path.splitext(os.path.basename(model_path))[0]}"

    if save_csv:
        trades_df = pd.DataFrame(res["trades"]) if res["trades"] else pd.DataFrame(columns=[
            "entry_time","exit_time","side","entry_price","exit_price","bars","ret","funding","fee_entry","fee_exit","alpha_flip"
        ])
        trades_csv = os.path.join(REPORT_DIR, f"{stem}_trades.csv")
        trades_df.to_csv(trades_csv, index=False)
        print(f"거래 CSV 저장: {trades_csv}")

    if save_chart:
        eq = res["equity_curve"]
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(eq)
        ax.set_title("Equity Curve")
        ax.set_xlabel("Bars")
        ax.set_ylabel("Equity ($)")
        fig.tight_layout()
        chart_path = os.path.join(REPORT_DIR, f"{stem}_chart.png")
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"차트 저장: {chart_path}")


# ===== CLI =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.join(MODEL_DIR, "best_model.zip"))
    parser.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    parser.add_argument("--initial_capital", type=float, default=100_000.0)
    parser.add_argument("--timesteps", type=int, default=-1, help="-1 for full period")
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument("--save-chart", action="store_true")
    args = parser.parse_args()

    print("RL Model Backtest Started!")
    print("Loading data…")
    df = load_split(args.split)
    feature_cols = load_feature_list()

    print("Loading model:", args.model)
    model = PPO.load(args.model, device="cpu")

    print("Running backtest simulation…")
    res = backtest(df, feature_cols, model, cfg=Cfg(), initial_equity=args.initial_capital, timesteps=args.timesteps)

    print("Analyzing results…")
    save_report(res, args.split, args.model, save_csv=args.save_csv, save_chart=args.save_chart)


if __name__ == "__main__":
    main()
