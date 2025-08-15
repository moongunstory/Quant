# ai_binance/backtest.py
"""
Module 4 — Backtest for Trained RL Models (환경 일치 + 임계치 게이트 + 쿨다운)

핵심
- 학습 환경과 동일한 로그-보상 체계:
  reward = pos * log_return − (포지션 변경 시 수수료(사이드당 0.05%))
  equity *= exp(reward)
- 임계치(분위수) 기반 게이트 + 히스테리시스 + 쿨다운으로 과매매 억제
- 예측은 결정론(deterministic=True)로 재현성 보장
- 체결(사이드), 라운드트립(오픈 수), 수수료(달러) 현실적으로 집계
- 결과 저장: trades / summary / 차트

실행:
  python ai_binance/backtest.py

산출:
  - 거래 내역: ./ai_binance/data/reports/backtest_trades.csv
  - 성과 요약: ./ai_binance/data/reports/backtest_summary.csv
  - 에쿼티:   ./ai_binance/data/reports/backtest_equity.csv
  - 차트:     ./ai_binance/data/reports/backtest_chart.png
"""
from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

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

# ===== 백테스트 기본 =====
INITIAL_CAPITAL = 100_000.0
COMMISSION_SIDE = 0.0005      # 0.05% / side
WINDOW = 48                   # 4h context (5m * 48)

# ===== 임계치(스윕) 게이트 =====
ADAPTIVE_GATE = True          # 분위수 게이트 사용
ZWIN = 7 * 288                # 7일 롤링(5m=하루 288바)
Q_ENTER = 0.90                # 상위 10%만 진입 허용
H_EXIT = 0.6                  # 청산 임계 완화(히스테리시스)
COOLDOWN_BARS = 12            # 최근 전환 후 최소 대기(1h)

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
    avg_trade_return: float       # % (라운드트립 기준)

    # 포지션 통계
    long_trades: int
    short_trades: int
    hold_ratio: float             # %

    # 비용
    total_commission: float       # $
    commission_ratio: float       # % of initial

# ===== 데이터 로드 =====
def load_test_data() -> Tuple[pd.DataFrame, pd.Series]:
    X = pd.read_parquet(os.path.join(PROC_DIR, "test_normalized.parquet"))
    feats = json.load(open(os.path.join(PROC_DIR, "feature_list.json"), "r"))
    X = X.reindex(columns=feats).dropna()

    df = pd.read_parquet(os.path.join(RAW_DIR, "fut_test_data_5m.parquet"))
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    close = df.sort_index()["Close"].astype(float).reindex(X.index).ffill().bfill()

    print(f"Test data loaded: {len(X):,} rows")
    print(f"Period: {X.index[0]} ~ {X.index[-1]}")
    return X, close

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
    학습 환경과 동일한 로그-보상/수수료 규칙.
    수수료: 포지션 변경 시에만 사이드당 0.05%를 로그-보상에서 차감,
           달러 수수료는 당시 에쿼티×0.0005로 누적.
    임계치 게이트: z = |logret| / vol; 롤링 분위수(Q_ENTER) 이상만 진입/전환 허용, 청산은 완화(H_EXIT).
    쿨다운: 최근 전환 후 COOLDOWN_BARS 동안 전환 금지.
    """
    def __init__(self, model: PPO, X: pd.DataFrame, prices: pd.Series):
        self.model = model
        self.X = X
        self.p = prices.reindex(X.index).ffill().bfill().astype(float)
        self.nf = X.shape[1]

        # 프리컴퓨트
        self.logret = np.log(self.p / self.p.shift(1)).fillna(0.0)
        vol = self.logret.rolling(24, min_periods=1).std().fillna(0.0)  # ~2h
        self.z = (self.logret.abs() / (vol + 1e-12)).fillna(0.0)
        if ADAPTIVE_GATE:
            self.z_thr = (
                self.z.rolling(ZWIN, min_periods=max(50, ZWIN // 4))
                .apply(lambda x: np.quantile(x, Q_ENTER), raw=True)
                .shift(1)
                .bfill()
            )
        self.reset()

    def reset(self):
        self.t = WINDOW
        self.pos = 0                  # -1/0/+1
        self.entry = None
        self.entry_time = None
        self.equity = 1.0             # 로그 프레임(배율)
        self.last_switch_t = -10_000
        self.exec_sides = 0
        self.open_count = 0
        self.switch_count = 0
        self.fee_log_sum = 0.0
        self.fee_usd_sum = 0.0

        self.trades_sides: List[Dict] = []   # 사이드 로그(open/close)
        self.trades_round: List[Dict] = []   # 라운드트립(완결) 로그
        self.equity_curve: List[Dict] = []   # {timestamp, equity$, position, price}
        self.prediction_log: List[Dict] = [] # 디버깅용 예측/실제 수익률 로그

    def _obs(self, t: int) -> np.ndarray:
        if t >= WINDOW:
            w = self.X.iloc[t - WINDOW : t].values
        else:
            pad = np.tile(self.X.iloc[[0]].values, (WINDOW - t - 1, 1))
            w = np.vstack([pad, self.X.iloc[: t + 1].values])
        return w.reshape(-1).astype(np.float32)

    def _gate(self, target: int) -> int:
        # 쿨다운: 최근 전환 후 X바 이내면 전환 금지
        if target != self.pos and (self.t - self.last_switch_t) < COOLDOWN_BARS:
            return self.pos
        if target == self.pos:
            return target

        if ADAPTIVE_GATE:
            z_t = float(self.z.iloc[self.t])
            thr = float(self.z_thr.iloc[self.t])
            thr_enter = thr
            thr_exit = thr * H_EXIT
        else:
            z_t = 0.0
            thr_enter = -np.inf
            thr_exit = np.inf

        if self.pos == 0:  # 진입
            return target if z_t >= thr_enter else self.pos

        if target == 0:  # 청산
            return 0 if z_t >= thr_exit else self.pos

        # 전환(롱↔숏)은 진입과 동일 임계
        return target if z_t >= thr_enter else self.pos

    def _apply_switch(self, target: int, price: float, ts: pd.Timestamp) -> float:
        """
        포지션 변경 처리:
        - 기존 포지션 청산 시: 라운드트립 기록 + 수수료 1 side
        - 새 포지션 진입 시:   오픈 기록 + 수수료 1 side
        반환: 로그-수수료 합(fee_log) → reward에서 차감
        """
        fee_log = 0.0

        # 청산(현재 포지션이 있었다면)
        if target != self.pos and self.pos != 0:
            # 라운드트립 수익률 기록(슬리피지 0, 환경 일치)
            if self.entry is not None:
                if self.pos == 1:
                    pnl_pct = (price / self.entry - 1.0) * 100.0
                else:
                    pnl_pct = (self.entry / price - 1.0) * 100.0
                self.trades_round.append(
                    {
                        "entry_time": self.entry_time,
                        "exit_time": ts,
                        "side": "LONG" if self.pos == 1 else "SHORT",
                        "entry_price": self.entry,
                        "exit_price": price,
                        "pnl_pct": pnl_pct,
                    }
                )
            # 청산 수수료(사이드 1회)
            fee_log += COMMISSION_SIDE
            self.fee_log_sum += COMMISSION_SIDE
            self.fee_usd_sum += float(self.equity * INITIAL_CAPITAL * COMMISSION_SIDE)
            self.exec_sides += 1
            self.trades_sides.append({"time": ts, "side": "close", "price": price, "pos_after": 0 if target == 0 else target})
            self.entry = None
            self.entry_time = None

        # 진입(목표가 0이 아니면)
        if target != 0 and target != self.pos:
            # 전환이면 스위치 카운트
            if self.pos != 0:
                self.switch_count += 1
            # 진입 수수료(사이드 1회)
            fee_log += COMMISSION_SIDE
            self.fee_log_sum += COMMISSION_SIDE
            self.fee_usd_sum += float(self.equity * INITIAL_CAPITAL * COMMISSION_SIDE)
            self.exec_sides += 1
            self.trades_sides.append({"time": ts, "side": "open", "price": price, "pos_after": target})
            self.open_count += 1
            self.entry = price
            self.entry_time = ts

        # 상태 반영
        if target != self.pos:
            self.last_switch_t = self.t
            self.pos = target

        return fee_log

    def run(self) -> BacktestResults:
        print("Starting backtest simulation...")
        self.reset()

        for t in range(self.t, len(self.X) - 1): # -1 to allow looking at t+1
            self.t = t
            ts = self.X.index[t]
            price_prev = float(self.p.iloc[t - 1])
            price = float(self.p.iloc[t])
            price_next = float(self.p.iloc[t + 1]) # << Get next price for logging
            lr = float(np.log(price / price_prev))

            obs = self._obs(t)
            
            # 결정론 예측(재현성)
            action, _ = self.model.predict(obs, deterministic=True)
            target = 0 if action == 0 else (1 if action == 1 else -1)
            target = self._gate(target)

            # --- Prediction Logging ---
            actual_future_lr = np.log(price_next / price)
            self.prediction_log.append({
                "timestamp": ts,
                "predicted_action": target,
                "actual_future_log_return": actual_future_lr
            })
            # --------------------------

            # 보유 수익
            reward = float(self.pos * lr)
            # 포지션 변경 시 수수료(로그) 차감 + 거래 기록
            fee_log = self._apply_switch(target, price, ts)
            reward -= fee_log

            # 에쿼티 업데이트(로그 누적)
            self.equity *= float(np.exp(reward))
            self.equity_curve.append(
                {"timestamp": ts, "price": price, "position": self.pos, "equity": self.equity * INITIAL_CAPITAL}
            )

        # ===== 디버그 로그 저장 =====
        log_df = pd.DataFrame(self.prediction_log)
        log_path = os.path.join(REPORT_DIR, "prediction_log.csv")
        log_df.to_csv(log_path, index=False)
        print(f"Prediction log saved to: {log_path}")

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

        # 라운드트립 승/패/평균
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

        return BacktestResults(
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
            total_trades=int(len(rounds)),         # 라운드트립 개수(오픈 수)
            winning_trades=wins,
            losing_trades=losses,
            win_rate=win_rate,
            avg_trade_return=avg_tr,
            long_trades=long_tr,
            short_trades=short_tr,
            hold_ratio=hold_ratio,
            total_commission=float(self.fee_usd_sum),
            commission_ratio=float(self.fee_usd_sum / INITIAL_CAPITAL * 100.0),
        ), eq_df, rounds, pd.DataFrame(self.trades_sides)

# ===== 저장/시각화 =====
def save_outputs(eq_df: pd.DataFrame, rounds: pd.DataFrame, sides: pd.DataFrame, results: BacktestResults):
    os.makedirs(REPORT_DIR, exist_ok=True)

    # equity
    # eq_csv = os.path.join(REPORT_DIR, "backtest_equity.csv")
    # eq_df.to_csv(eq_csv)

    # trades (라운드트립)
    # tr_csv = os.path.join(REPORT_DIR, "backtest_trades.csv")
    # if not rounds.empty:
    #     rounds.to_csv(tr_csv, index=False)

    # side 로그(참고)
    # sides_csv = os.path.join(REPORT_DIR, "backtest_trades_sides.csv")
    # if not sides.empty:
    #     sides.to_csv(sides_csv, index=False)

    # summary
    # sm_csv = os.path.join(REPORT_DIR, "backtest_summary.csv")
    # pd.DataFrame([asdict(results)]).to_csv(sm_csv, index=False)

    # chart
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

    print("거래 라운드트립: (CSV 저장 비활성화)")
    print("거래 사이드: (CSV 저장 비활성화)")
    print("성과 요약: (CSV 저장 비활성화)")
    print("에쿼티 CSV: (CSV 저장 비활성화)")

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
    print("\n💸 비용:")
    print(f"   총 수수료: ${r.total_commission:,.2f} ({r.commission_ratio:.2f}%)")
    print("=" * 60)

# ===== 메인 =====
def main():
    print("RL Model Backtest Started!")
    try:
        X, close = load_test_data()
        model = load_model()

        engine = BacktestEngine(model, X, close)
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
