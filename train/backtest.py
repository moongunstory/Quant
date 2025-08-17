# ai_binance/backtest.py
"""
Backtest for Trained RL Models — Minimal Causal Environment

핵심
- 학습 환경과 동일한 규칙(게이트/히스테리시스 없음)
  reward_t = prev_pos*log_return(t) − fee(체결시; per-side 0.05%) − funding(prev_pos)
  포지션 지시는 같은 틱에 내려도, 포지션은 다음 틱부터 유효
- 스위칭은 "청산+진입"으로 2사이드 비용
- 결과: trades / summary / 차트
"""

from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO

warnings.filterwarnings("ignore", category=UserWarning)

# ===== 경로 =====
PROC_DIR = "./ai_binance/data/processed"
RAW_DIR = "./ai_binance/data/raw"
MODEL_DIR = "./ai_binance/data/model"
REPORT_DIR = "./ai_binance/data/reports"

# ===== 상수 =====
INITIAL_CAPITAL = 100_000.0
COMMISSION_SIDE = 0.0005        # 0.05% / side
WINDOW = 48                     # 4h (5m * 48)

# ===== 결과 데이터클래스 =====
@dataclass
class BacktestResults:
    model_name: str
    start_date: str
    end_date: str
    total_days: int

    # 수익성
    initial_capital: float
    final_capital: float
    total_return: float           # %
    annual_return: float          # %

    # 리스크
    volatility: float             # 연율화 일간 변동성 %
    sharpe_ratio: float
    max_drawdown: float           # %

    # 거래 통계
    total_trades: int             # 라운드트립 개수(=open 수)
    winning_trades: int
    losing_trades: int
    win_rate: float               # %
    avg_trade_return: float       # %

    # 포지션 통계
    long_trades: int
    short_trades: int
    hold_ratio: float             # %

    # 비용(USD)
    total_commission: float
    commission_ratio: float       # % of initial

# ===== 데이터 로드 =====
def load_test_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = pd.read_parquet(os.path.join(PROC_DIR, "test_normalized.parquet"))
    feats = json.load(open(os.path.join(PROC_DIR, "feature_list.json"), "r"))
    X = X.reindex(columns=feats).dropna()

    df = pd.read_parquet(os.path.join(RAW_DIR, "fut_test_data_5m.parquet"))
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()

    close = df["Close"].astype(float).reindex(X.index).ffill().bfill()
    funding = df.get("FundingRate", pd.Series(0.0, index=df.index)).astype(float).reindex(X.index).ffill().bfill()

    print(f"Test data loaded: {len(X):,} rows")
    print(f"Period: {X.index[0]} ~ {X.index[-1]}")
    return X, close, funding

def load_model() -> PPO:
    best = os.path.join(MODEL_DIR, "best_model.zip")
    path = best if os.path.exists(best) else os.path.join(MODEL_DIR, "ppo_final_model.zip")
    if not os.path.exists(path):
        raise FileNotFoundError("모델 파일이 없습니다: best_model.zip / ppo_final_model.zip")
    print(f"Load model: {path}")
    return PPO.load(path, device="cpu")

# ===== 백테스트 엔진 =====
class BacktestEngine:
    """
    - 액션: {0=홀드/패스, 1=롱, 2=숏} → 다음 틱 포지션으로 적용
    - 스위칭은 2사이드 비용
    - 보상: prev_pos * log_ret(t) − fee − funding(prev_pos)
    """
    def __init__(self, model: PPO, X: pd.DataFrame, prices: pd.Series, funding: pd.Series):
        self.model = model
        self.X = X
        self.p = prices.reindex(X.index).ffill().bfill().astype(float)
        self.funding = funding.reindex(X.index).ffill().bfill().astype(float)
        self.nf = X.shape[1]

        # 프리컴퓨트
        self.logret = np.log(self.p / self.p.shift(1)).fillna(0.0)

        self.reset()

    # ----- 상태 관리 -----
    def reset(self):
        self.t = WINDOW
        self.pos = 0
        self.entry = None
        self.entry_time = None
        self.equity = 1.0

        self.open_count = 0
        self.commission_usd_sum = 0.0

        self.trades_sides: List[Dict] = []
        self.trades_round: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.prediction_log: List[Dict] = []

    # ----- 관측 구성: 윈도우 + 상태 피처 3개 -----
    def _state_vec(self, t: int) -> np.ndarray:
        time_in_pos = 0 if self.entry_time is None else (t - self.entry_time)
        time_in_pos_norm = min(time_in_pos, 1000) / 1000.0
        if self.entry is None or self.pos == 0:
            upnl_log = 0.0
        else:
            upnl_log = float(np.log(self.p.iloc[t] / self.entry)) * float(self.pos)
        return np.array([float(self.pos), float(time_in_pos_norm), float(upnl_log)], dtype=np.float32)

    def _obs(self, t: int) -> np.ndarray:
        w = self.X.iloc[t - WINDOW : t].values.astype(np.float32)
        return np.concatenate([w.reshape(-1), self._state_vec(t)], axis=0)

    # ----- 실행 -----
    def run(self) -> Tuple[BacktestResults, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("Starting backtest simulation...")
        self.reset()

        for t in range(self.t, len(self.X) - 1):  # -1 to read t+1 for logging
            self.t = t
            ts = self.X.index[t]
            price_prev = float(self.p.iloc[t - 1])
            price = float(self.p.iloc[t])
            price_next = float(self.p.iloc[t + 1])
            lr = float(np.log(price / price_prev))

            obs = self._obs(t)
            raw_action, _ = self.model.predict(obs, deterministic=True)

            # 액션 → 다음 바 목표 포지션
            target = {0: self.pos, 1: +1, 2: -1}.get(int(raw_action), self.pos)

            # 디버그(다음 바 수익)
            actual_future_lr = np.log(price_next / price)
            self.prediction_log.append({
                "timestamp": ts,
                "raw_action": int(raw_action),
                "gated_target": int(target),
                "actual_future_log_return": float(actual_future_lr),
            })

            # ▼ 보유 수익: 직전 포지션 기준
            prev_pos = self.pos
            reward = float(prev_pos * lr)

            # 체결/수수료 (per-side, switch=2 sides)
            sides = 0
            if prev_pos != target:
                if prev_pos == 0 or target == 0:
                    sides = 1
                else:
                    sides = 2
            fee_log = sides * COMMISSION_SIDE
            reward -= fee_log

            # USD 비용 집계(가치가치)
            if sides > 0:
                notion_usd = float(self.equity * INITIAL_CAPITAL)
                self.commission_usd_sum += notion_usd * COMMISSION_SIDE * sides
                if prev_pos != 0 and target == 0:
                    self.trades_sides.append({"time": ts, "side": "close", "price": price, "pos_after": 0})
                elif prev_pos == 0 and target != 0:
                    self.trades_sides.append({"time": ts, "side": "open", "price": price, "pos_after": target})
                    self.open_count += 1
                else:
                    # switch: close then open
                    self.trades_sides.append({"time": ts, "side": "close", "price": price, "pos_after": 0})
                    self.trades_sides.append({"time": ts, "side": "open", "price": price, "pos_after": target})
                    self.open_count += 1

            # 라운드트립 기록(청산 시점)
            if prev_pos != 0 and target != prev_pos:
                # 청산 발생
                if self.entry is not None:
                    pnl_pct = ((price / self.entry - 1.0) * 100.0) if prev_pos == 1 else ((self.entry / price - 1.0) * 100.0)
                    self.trades_round.append({
                        "entry_time": self.entry_time, "exit_time": ts,
                        "side": "LONG" if prev_pos == 1 else "SHORT",
                        "entry_price": self.entry, "exit_price": price,
                        "pnl_pct": pnl_pct
                    })

            # 펀딩비(직전 포지션 기준)
            fr = float(self.funding.iloc[t])
            is_funding_event = (getattr(ts, "minute", 0) == 0) and (getattr(ts, "hour", 0) % 8 == 0)
            if is_funding_event:
                reward -= (prev_pos * fr)

            # 에쿼티 업데이트
            self.equity *= float(np.exp(reward))

            # 포지션/엔트리 갱신
            if prev_pos != target:
                if target == 0:
                    self.entry = None
                    self.entry_time = None
                else:
                    self.entry = price
                    self.entry_time = self.t
            self.pos = target

            self.equity_curve.append(
                {"timestamp": ts, "price": price, "position": self.pos, "equity": self.equity * INITIAL_CAPITAL}
            )

        # ===== 디버그 로그 저장 =====
        os.makedirs(REPORT_DIR, exist_ok=True)
        pd.DataFrame(self.prediction_log).to_csv(os.path.join(REPORT_DIR, "prediction_log.csv"), index=False)

        # ===== 결과 집계 =====
        eq_df = pd.DataFrame(self.equity_curve).set_index("timestamp")
        start_date = str(eq_df.index[0].date())
        end_date = str(eq_df.index[-1].date())
        days = (eq_df.index[-1] - eq_df.index[0]).days or 1
        final_cap = float(eq_df["equity"].iloc[-1])

        daily = eq_df["equity"].resample("D").last().pct_change().dropna()
        vol = (daily.std() * np.sqrt(365) * 100.0) if len(daily) > 1 else 0.0
        total_ret = (final_cap / INITIAL_CAPITAL - 1.0) * 100.0
        annual_ret = total_ret * (365.0 / days)

        peak = eq_df["equity"].cummax()
        dd = (eq_df["equity"] - peak) / peak
        mdd = float(dd.min()) * -100.0

        rounds = pd.DataFrame(self.trades_round)
        if not rounds.empty:
            wins = int((rounds["pnl_pct"] > 0).sum())
            losses = int((rounds["pnl_pct"] <= 0).sum())
            win_rate = (wins / len(rounds)) * 100.0
            avg_tr = float(rounds["pnl_pct"].mean())
            long_tr = int((rounds["side"] == "LONG").sum())
            short_tr = int((rounds["side"] == "SHORT").sum())
        else:
            wins = losses = 0
            win_rate = 0.0
            avg_tr = 0.0
            long_tr = short_tr = 0

        hold_ratio = float((eq_df["position"] == 0).mean() * 100.0)
        sharpe = (annual_ret / vol) if vol > 0 else 0.0

        results = BacktestResults(
            model_name="PPO_Model",
            start_date=start_date,
            end_date=end_date,
            total_days=days,
            initial_capital=INITIAL_CAPITAL,
            final_capital=final_cap,
            total_return=total_ret,
            annual_return=annual_ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            max_drawdown=mdd,
            total_trades=int(len(rounds)),
            winning_trades=wins,
            losing_trades=losses,
            win_rate=win_rate,
            avg_trade_return=avg_tr,
            long_trades=long_tr,
            short_trades=short_tr,
            hold_ratio=hold_ratio,
            total_commission=float(self.commission_usd_sum),
            commission_ratio=float(self.commission_usd_sum / INITIAL_CAPITAL * 100.0),
        )
        return results, eq_df, rounds, pd.DataFrame(self.trades_sides)

# ===== 저장/시각화 =====
def save_outputs(eq_df: pd.DataFrame, rounds: pd.DataFrame, sides: pd.DataFrame, results: BacktestResults):
    os.makedirs(REPORT_DIR, exist_ok=True)

    try:
        plt.figure(figsize=(14, 8))
        (eq_df["equity"]).plot(linewidth=2)
        plt.axhline(INITIAL_CAPITAL, ls="--", alpha=0.6)
        plt.title("Backtest — Equity (TEST)")
        plt.ylabel("Capital ($)")
        plt.xlabel("Time")
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        fig_path = os.path.join(REPORT_DIR, "backtest_chart.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print("차트 저장:", fig_path)
    except Exception as e:
        print("[warn] chart failed:", e)

    print("CSV 저장은 기본 비활성화(주석 해제 시 활성).")

def print_summary(r: BacktestResults):
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"📅 기간: {r.start_date} ~ {r.end_date} ({r.total_days}일)")
    print(f"🤖 모델: {r.model_name}")
    print("\n💰 수익성:")
    print(f"   초기 자본: ${r.initial_capital:,.0f}")
    print(f"   최종 자본: ${r.final_capital:,.0f}")
    print(f"   총 수익률: {r.total_return:+.2f}%")
    print(f"   연간 수익률: {r.annual_return:+.2f}%")
    print("\n📉 리스크:")
    print(f"   변동성: {r.volatility:.2f}%")
    print(f"   샤프 비율: {r.sharpe_ratio:.2f}")
    print(f"   최대 손실: {r.max_drawdown:.2f}%")
    print("\n🔄 거래 통계:")
    print(f"   총 거래(라운드트립): {r.total_trades}회")
    print(f"   승리/패배: {r.winning_trades}/{r.losing_trades} (승률 {r.win_rate:.1f}%)")
    print(f"   평균 거래 수익률: {r.avg_trade_return:+.2f}%")
    print("\n📊 포지션 분석:")
    print(f"   롱/숏 거래: {r.long_trades}/{r.short_trades}")
    print(f"   홀드 비율: {r.hold_ratio:.1f}%")
    print("\n💸 비용(USD):")
    print(f"   총 수수료: ${r.total_commission:,.2f} ({r.commission_ratio:.2f}%)")
    print("=" * 60)

# ===== 메인 =====
def main():
    print("RL Model Backtest Started!")
    try:
        X, close, funding = load_test_data()
        model = load_model()

        engine = BacktestEngine(model, X, close, funding)
        results, eq_df, rounds, sides = engine.run()

        save_outputs(eq_df, rounds, sides, results)
        print_summary(results)

        print("\n✅ 백테스팅 완료!")
        print(f"📁 결과 파일: {REPORT_DIR}")
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
