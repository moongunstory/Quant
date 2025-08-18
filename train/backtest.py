# ai_binance/train/backtest.py
"""
Backtest for Trained RL Models (VecNormalize compatible)
"""

from __future__ import annotations

import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings("ignore", category=UserWarning)

# ===== 경로 및 설정 (rl.py와 동기화) =====
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROC_DIR      = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "processed"))
MODEL_DIR     = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "models"))
REPORT_DIR    = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "reports"))

MODEL_NAME    = "ppo_mtf_features.zip"
ENV_STATS_NAME= "ppo_mtf_vecnorm.pkl"

INTERVAL      = "5m"
WINDOW        = 48
FEE_PER_SIDE  = 0.0005
SLIP_PER_SIDE = 0.0001
SEED          = 42
INITIAL_CAPITAL = 100_000.0

# ===== 결과 데이터클래스 (기존 backtest.py에서 가져옴) =====
@dataclass
class BacktestResults:
    model_name: str
    start_date: str
    end_date: str
    total_days: int
    initial_capital: float
    final_capital: float
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_trade_return: float
    long_trades: int
    short_trades: int
    hold_ratio: float
    total_commission: float
    commission_ratio: float

# ===== rl.py에서 가져온 환경 및 데이터 로더 =====
class FeatureStackedEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, X: np.ndarray, close: np.ndarray, funding_rate: np.ndarray,
                 window=48, fee_per_side=0.0005, slip_per_side=0.0001,
                 interval_min=5, random_start=True, seed: int = 42):
        assert X.ndim == 2, "X must be 2D [T, F]"
        T, F = X.shape
        assert len(close) == T and len(funding_rate) == T, "X/close/funding length mismatch"
        assert T >= window + 2, "데이터가 너무 짧음"
        self.X = X.astype(np.float32)
        self.F = F
        self.close = close.astype(np.float64)
        self.funding = funding_rate.astype(np.float64)
        self.window = int(window)
        self.cost_per_side = float(fee_per_side + slip_per_side)
        self.ret = np.diff(np.log(self.close))
        self.random_start = bool(random_start)
        self.interval_min = int(interval_min)
        self.fund_div = max(1, int(round(480 / max(1, self.interval_min))))
        self._rng = np.random.default_rng(seed)
        self.pos = 0
        self.bars_in_pos = 0
        self.t = self.window
        obs_dim = self.window * self.F + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def _obs(self, t):
        xw = self.X[t-self.window:t].reshape(-1)
        return np.concatenate([xw, np.array([float(self.pos), float(self.bars_in_pos)/100.0], dtype=np.float32)], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = 0
        self.bars_in_pos = 0
        self.t = self.window
        return self._obs(self.t), {}

    def step(self, action: int):
        prev_pos = self.pos
        sides = 0
        new_pos = self.pos
        if action == 0: new_pos = self.pos
        elif action == 1:
            if self.pos == 0: new_pos, sides = +1, 1
            elif self.pos == -1: new_pos, sides = +1, 2
        elif self.pos == +1 and action == 2:
            new_pos, sides = -1, 2
        elif action == 2:
            if self.pos == 0: new_pos, sides = -1, 1
        elif action == 3:
            if self.pos != 0: new_pos, sides = 0, 1
        r = self.ret[self.t]
        simple_ret = np.exp(r) - 1.0
        fund_step = (self.funding[self.t] / self.fund_div) * new_pos
        fee_step = sides * self.cost_per_side
        reward = (new_pos * simple_ret) - fee_step - fund_step
        self.pos = new_pos
        self.bars_in_pos = (self.bars_in_pos + 1) if self.pos != 0 else 0
        self.t += 1
        terminated = (self.t >= len(self.close) - 1)
        obs = self._obs(self.t) if not terminated else np.zeros_like(self._obs(self.t-1), dtype=np.float32)
        info = {
            "ret": float(simple_ret), 
            "reward": float(reward),
            "pos": int(self.pos), 
            "prev_pos": int(prev_pos),
            "sides": int(sides), 
            "fee_step": float(fee_step),
            "price": float(self.close[self.t-1])
        }
        return obs, float(reward), terminated, False, info

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex) and (df.index.name or "").lower() in ("open_time", "time"):
        df = df.reset_index()
    low = {c.lower(): c for c in df.columns}
    ren = {}
    if "open_time" in low: ren[low["open_time"]] = "time"
    if "time" in low: ren[low["time"]] = "time"
    if "close" in low: ren[low["close"]] = "close"
    elif "close_ref" in low: ren[low["close_ref"]] = "close"
    elif "close" not in low and "Close" in df.columns: ren["Close"] = "close"
    if "funding_rate" in low: ren[low["funding_rate"]] = "funding_rate"
    elif "fundingrate" in low: ren[low["fundingrate"]] = "funding_rate"
    elif "FundingRate" in df.columns: ren["FundingRate"] = "funding_rate"
    df = df.rename(columns=ren)
    if "close" not in df.columns: raise KeyError("close/close_ref/Close 칼럼이 필요합니다.")
    if "funding_rate" not in df.columns: df["funding_rate"] = 0.0
    if "time" in df.columns: df.set_index("time", inplace=True)
    return df

def load_processed(split: str):
    path = os.path.join(PROC_DIR, "fe_" + split + "_" + INTERVAL + ".parquet")
    df = pd.read_parquet(path)
    df = _normalize_df(df)
    feat_path = os.path.join(PROC_DIR, "fe_feature_list_" + INTERVAL + ".json")
    if os.path.exists(feat_path):
        with open(feat_path, "r", encoding="utf-8") as f: feature_cols: List[str] = json.load(f)
        exclude_low = {"time","open","high","low","close","volume","funding_rate","close_ref","FundingRate","Open","High","Low","Close","Volume"}
        feature_cols = [c for c in feature_cols if c not in exclude_low and c in df.columns]
    else:
        exclude_low = {"time","open","high","low","close","volume","funding_rate","close_ref","FundingRate","Open","High","Low","Close","Volume"}
        feature_cols = [c for c in df.columns if c not in exclude_low]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float64)
    funding = df["funding_rate"].to_numpy(dtype=np.float64)
    timestamps = df.index
    return X, close, funding, feature_cols, timestamps

class ZScaler:
    def __init__(self):
        self.mean, self.std = None, None
    def fit(self, X: np.ndarray):
        self.mean, self.std = X.mean(axis=0), X.std(axis=0); self.std[self.std==0] = 1.0
    def transform(self, X: np.ndarray):
        return (X - self.mean) / self.std

# ===== 리포팅 및 시각화 (기존 backtest.py에서 가져옴) =====
def print_summary(r: BacktestResults):
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"기간: {r.start_date} ~ {r.end_date} ({r.total_days}일)")
    print(f"모델: {r.model_name}")
    print("\n수익성:")
    print(f"   초기 자본: ${r.initial_capital:,.0f}")
    print(f"   최종 자본: ${r.final_capital:,.0f}")
    print(f"   총 수익률: {r.total_return:+.2f}%")
    print(f"   연간 수익률: {r.annual_return:+.2f}%")
    print("\n리스크:")
    print(f"   변동성: {r.volatility:.2f}%")
    print(f"   샤프 비율: {r.sharpe_ratio:.2f}")
    print(f"   최대 손실: {r.max_drawdown:.2f}%")
    print("\n거래 통계:")
    print(f"   총 거래(라운드트립): {r.total_trades}회")
    print(f"   승리/패배: {r.winning_trades}/{r.losing_trades} (승률 {r.win_rate:.1f}%)")
    print(f"   평균 거래 수익률: {r.avg_trade_return:+.2f}%")
    print("\n포지션 분석:")
    print(f"   롱/숏 거래: {r.long_trades}/{r.short_trades}")
    print(f"   관망 비율: {r.hold_ratio:.1f}%")
    print("\n비용(USD):")
    print(f"   총 수수료: ${r.total_commission:,.2f} ({r.commission_ratio:.2f}%)")
    print("=" * 60)

def save_plot(equity_curve: pd.DataFrame):
    os.makedirs(REPORT_DIR, exist_ok=True)
    fig_path = os.path.join(REPORT_DIR, "backtest_chart.png")
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax1 = plt.subplots(figsize=(16, 9))
        # Equity curve
        ax1.plot(equity_curve.index, equity_curve['equity'], label='Equity', color='blue', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Equity ($)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        # Price curve
        ax2 = ax1.twinx()
        ax2.plot(equity_curve.index, equity_curve['price'], label='Price', color='gray', alpha=0.5, linewidth=1)
        ax2.set_ylabel('Price', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        # Trades
        trades = equity_curve[equity_curve['trade_marker'] != '']
        long_entries = trades[(trades['trade_marker'] == 'Long Entry')]
        short_entries = trades[(trades['trade_marker'] == 'Short Entry')]
        exits = trades[(trades['trade_marker'] == 'Exit')]
        ax1.scatter(long_entries.index, long_entries['equity'], marker='^', color='green', s=100, label='Long Entry', zorder=5)
        ax1.scatter(short_entries.index, short_entries['equity'], marker='v', color='red', s=100, label='Short Entry', zorder=5)
        ax1.scatter(exits.index, exits['equity'], marker='x', color='black', s=100, label='Exit', zorder=5)
        
        fig.suptitle('Backtest Results', fontsize=16)
        fig.legend()
        fig.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"차트 저장: {fig_path}")
    except Exception as e:
        print(f"[warn] chart failed: {e}")

# ===== 새로운 백테스트 엔진 =====
def analyze_results(history: List[Dict], timestamps: pd.DatetimeIndex) -> Tuple[BacktestResults, pd.DataFrame]:
    equity = INITIAL_CAPITAL
    equity_history = []
    position_history = []
    price_history = []
    trade_markers = []
    
    round_trips = []
    current_trade = None

    for i, info in enumerate(history):
        equity *= (1 + info['reward'])
        equity_history.append(equity)
        position_history.append(info['pos'])
        price_history.append(info['price'])
        
        marker = ''
        is_trade = info['sides'] > 0
        if is_trade:
            # Round trip analysis
            if current_trade and info['pos'] != current_trade['side']:
                # Position closed or flipped
                current_trade['exit_price'] = info['price']
                current_trade['exit_time'] = timestamps[i]
                pnl_pct = ((current_trade['exit_price'] / current_trade['entry_price'] - 1) * 100) if current_trade['side'] == 1 else ((current_trade['entry_price'] / current_trade['exit_price'] - 1) * 100)
                current_trade['pnl_pct'] = pnl_pct
                round_trips.append(current_trade)
                current_trade = None
                marker = 'Exit'

            if not current_trade and info['pos'] != 0:
                # New position opened
                current_trade = {
                    'entry_price': info['price'],
                    'entry_time': timestamps[i],
                    'side': info['pos']
                }
                marker = 'Long Entry' if info['pos'] == 1 else 'Short Entry'
        trade_markers.append(marker)

    eq_df = pd.DataFrame({
        'equity': equity_history,
        'position': position_history,
        'price': price_history,
        'trade_marker': trade_markers
    }, index=timestamps[:len(history)])

    # --- 통계 계산 ---
    start_date, end_date = str(eq_df.index[0].date()), str(eq_df.index[-1].date())
    days = (eq_df.index[-1] - eq_df.index[0]).days or 1
    final_cap = eq_df["equity"].iloc[-1]
    total_ret = (final_cap / INITIAL_CAPITAL - 1.0) * 100.0
    annual_ret = total_ret * (365.0 / days)
    daily_ret = eq_df["equity"].resample("D").last().pct_change().dropna()
    vol = (daily_ret.std() * np.sqrt(365) * 100.0) if len(daily_ret) > 1 else 0.0
    sharpe = (annual_ret / vol) if vol > 0 else 0.0
    peak = eq_df["equity"].cummax()
    mdd = (((eq_df["equity"] - peak) / peak).min() * 100.0) if not peak.empty else 0.0
    
    rounds_df = pd.DataFrame(round_trips)
    if not rounds_df.empty:
        wins = int((rounds_df["pnl_pct"] > 0).sum())
        losses = len(rounds_df) - wins
        win_rate = (wins / len(rounds_df) * 100.0) if len(rounds_df) > 0 else 0.0
        avg_tr = rounds_df["pnl_pct"].mean()
        long_tr = int((rounds_df["side"] == 1).sum())
        short_tr = int((rounds_df["side"] == -1).sum())
    else:
        wins, losses, win_rate, avg_tr, long_tr, short_tr = 0,0,0,0,0,0

    total_commission = sum(info['fee_step'] * equity for info, equity in zip(history, equity_history))

    results = BacktestResults(
        model_name=MODEL_NAME,
        start_date=start_date, end_date=end_date, total_days=days,
        initial_capital=INITIAL_CAPITAL, final_capital=final_cap,
        total_return=total_ret, annual_return=annual_ret, volatility=vol,
        sharpe_ratio=sharpe, max_drawdown=abs(mdd),
        total_trades=len(rounds_df), winning_trades=wins, losing_trades=losses,
        win_rate=win_rate, avg_trade_return=avg_tr, long_trades=long_tr, short_trades=short_tr,
        hold_ratio=((eq_df["position"] == 0).mean() * 100.0),
        total_commission=total_commission,
        commission_ratio=(total_commission / INITIAL_CAPITAL * 100.0)
    )
    return results, eq_df

# ===== 메인 실행 함수 =====
def main():
    print("RL Model Backtest Started!")
    model_path = os.path.join(MODEL_DIR, MODEL_NAME)
    stats_path = os.path.join(MODEL_DIR, ENV_STATS_NAME)
    if not os.path.exists(model_path) or not os.path.exists(stats_path):
        raise FileNotFoundError(f"모델 또는 환경 통계 파일이 없습니다: {model_path}, {stats_path}")

    # 1. 데이터 로드 (학습/테스트)
    print("Loading data...")
    X_tr, _, _, _, _ = load_processed("train")
    X_te, c_te, f_te, _, ts_te = load_processed("test")

    # 2. 스케일러 학습 및 적용
    scaler = ZScaler(); scaler.fit(X_tr)
    X_te_s = scaler.transform(X_te)

    # 3. 환경 생성 및 정규화 통계 로드
    print("Creating environment and loading stats...")
    def make_env():
        return FeatureStackedEnv(X_te_s, c_te, f_te, window=WINDOW, random_start=False)
    
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(stats_path, env)
    env.training = False
    env.norm_reward = False

    # 4. 모델 로드
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path, device="cpu")

    # 5. 백테스트 실행
    print("Running backtest simulation...")
    obs = env.reset()
    history = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        history.append(info[0])

    # 6. 결과 분석 및 출력
    print("Analyzing results...")
    results, equity_curve = analyze_results(history, ts_te)
    print_summary(results)
    save_plot(equity_curve)

    print("\nBacktesting complete!")
    print(f"Result files saved in: {REPORT_DIR}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
