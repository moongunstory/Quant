# run.py
"""
BTC Jarvis Manager - Main Entry Point (Daemon Mode)

Modes:
1. Paper Trading: Virtual money trading (safe)
   - Resume: Continue from existing log
   - Restart: Start fresh (backup old log)
2. Live Trading: Real Binance Futures trading (REAL MONEY!)

자동 실행:
- 필수 데이터 파일이 없으면 → 누락된 모듈만 540일치 초기 수집
- 모든 파일이 있으면 → 각 데이터의 갱신 주기에 따라 필요한 항목만 '스마트 업데이트'
- 데이터 수집/가공 후 → Regression 모델 학습 + 예측
- 예측 기반 거래 실행 (Paper or Live)
- 1시간마다 반복
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Literal

from ingest.orchestrator import DataOrchestrator
from process.builder import build_all_features

# Regression models (NEW)
from model.regression.config import RegressionConfig
from model.regression.pipeline import train_all_horizons, generate_predictions

# Classification models (OLD - still available)
from model.daily.config import DailyConfig
from model.daily.pipeline import run_daily_cycle

# Trading components (NEW)
from trading.config import TradingConfig
from trading.strategy import SimpleStrategy
from trading.paper import PaperTrader
from trading.executor import BinanceExecutor
from trading.logger import TradeLogger


def check_data_status() -> Tuple[List[str], int]:
    """
    어떤 모듈의 필수 파일이 누락되었는지 확인하고, 가장 최신 데이터의 날짜를 찾습니다.
    (smart_update 도입 후 days_old는 사용되지 않지만, 초기 진단용으로 유지)

    Returns:
        (missing_modules_list, days_since_latest_date)
        missing_modules_list가 비어있으면 데이터가 완전하다는 의미입니다.
    """
    data_dir = Path("data/raw")
    # 센티먼트는 현재 파이프라인에서 사용하지 않으므로 제외
    all_modules = ['binance', 'macro', 'news', 'onchain', 'derivatives']
    if not data_dir.exists():
        return all_modules, 0

    # 모듈별 필수 파일 목록
    essential_files = {
        'binance': [
            'binance/ohlcv_futures_1h.parquet',
            'binance/ohlcv_spot_1h.parquet',
            'binance/oi_1h.parquet',
            'binance/ls_ratio_top_1h.parquet',
            'binance/funding_rate.parquet',
        ],
        'macro': [
            'macro/fred_dgs10.parquet',
            'macro/yahoo_gspc.parquet',
        ],
        'news': [
            'news/news_raw.parquet',
        ],
        'onchain': [
            'onchain/blockchain_com_n-transactions.parquet',
        ],
        'derivatives': [
            'derivatives/deribit_btc_dvol.parquet',
        ],
    }

    missing_modules = []
    for module, files in essential_files.items():
        for f in files:
            if not (data_dir / f).exists():
                print(f"⚠️ 필수 데이터 파일 누락: {f} ({module} 모듈 재수집 필요)")
                if module not in missing_modules:
                    missing_modules.append(module)
    
    if missing_modules:
        return missing_modules, 0

    # 모든 필수 파일이 존재하면, 가장 최신 날짜를 찾음
    latest_date = None
    for parquet_file in data_dir.rglob('*.parquet'):
        try:
            df = pd.read_parquet(parquet_file)
            date_col = 'timestamp' if 'timestamp' in df.columns else 'date'
            if date_col in df.columns and not df.empty:
                file_latest = pd.to_datetime(df[date_col]).max()
                if latest_date is None or file_latest > latest_date:
                    latest_date = file_latest
        except Exception:
            continue
    
    if latest_date is None:
        return all_modules, 0

    days_old = (pd.Timestamp.now().normalize() - latest_date.normalize()).days
    return [], days_old


def _print_today_predictions(daily_pred: pd.DataFrame) -> None:
    if daily_pred is None or daily_pred.empty:
        print("\n(오늘 생성된 예측 레코드가 없습니다.)")
        return

    daily_pred = daily_pred.sort_values("horizon_days")

    print("\n=== 오늘 BTC 방향 예측 요약 ===")
    as_of_ts = daily_pred["as_of_ts"].iloc[0]
    print(f"기준 시각 (as_of_ts): {as_of_ts}")

    label_str = {-1: "하락(-1)", 0: "중립(0)", 1: "상승(+1)"}

    for _, row in daily_pred.iterrows():
        lbl = row.pred_label
        lbl_txt = label_str.get(lbl, f"알 수 없음({lbl})")

        line = (
            f"- Horizon {row.horizon_days:>2}일 "
            f"→ 예측: {lbl_txt}, "
            f"P(하락)={row.proba_down:.3f}, "
            f"P(중립)={row.proba_flat:.3f}, "
            f"P(상승)={row.proba_up:.3f}"
        )

        # exp_return 컬럼이 있고 값이 있으면 퍼센트로 같이 출력
        if "exp_return" in daily_pred.columns and pd.notna(row.get("exp_return", np.nan)):
            # 비율(0.04) → 퍼센트(4.0)
            er_pct = row.exp_return * 100.0
            samples = int(row.get("exp_return_samples", 0) or 0)
            line += f", 예상 수익률≈{er_pct:.2f}%, (과거 샘플 {samples}개)"

        print(line)

def select_mode() -> Literal["paper", "live"]:
    """
    Prompt user to select trading mode (no CLI args).
    """
    print("=" * 60)
    print("🤖 BTC Jarvis Manager - Regression Trading System")
    print("=" * 60)
    print("\n모드 선택:")
    print("  1. Paper Trading (가상 자금)")
    print("  2. Live Trading (실제 자금 - 주의!)")
    print()

    while True:
        choice = input("선택 (1/2): ").strip()
        if choice == "1":
            return "paper"
        elif choice == "2":
            confirm = input("⚠️  실제 자금을 사용합니다. 'YES' 입력하여 확인: ").strip()
            if confirm == "YES":
                return "live"
            else:
                print("Live trading 취소. 다른 모드를 선택하세요.\n")
        else:
            print("잘못된 입력입니다. 1 또는 2를 선택하세요.\n")


def select_paper_mode() -> Literal["resume", "restart"]:
    """
    Prompt user to select paper trading mode: resume or restart.
    """
    print("\n가상 거래 모드:")
    print("  1. 이어서 하기 (기존 로그 유지)")
    print("  2. 처음부터 시작 (새 세션, 기존 로그는 백업)")
    print()

    while True:
        choice = input("선택 (1/2): ").strip()
        if choice == "1":
            return "resume"
        elif choice == "2":
            return "restart"
        else:
            print("잘못된 입력입니다. 1 또는 2를 선택하세요.\n")


def run_trading_daemon(mode: str, paper_mode: str = "resume"):
    """
    Main trading loop - runs continuously.

    Every 1 hour:
    - Update data
    - Rebuild features
    - Train models (once per day)
    - Generate predictions
    - Execute trades (if mode is paper or live)

    Args:
        mode: "paper" or "live"
        paper_mode: "resume" or "restart" (only for paper trading)
    """
    print(f"\n🚀 Trading daemon 시작 ({mode.upper()} 모드)...")
    print(f"⏰ 시작 시간: {datetime.now()}")
    print(f"📊 업데이트 주기: 1시간\n")

    # Initialize trading components
    trading_config = TradingConfig(mode=mode)

    if mode == "paper":
        # Handle restart: backup existing log and reset paper trader
        if paper_mode == "restart":
            log_path = Path(trading_config.paper_log_path)
            if log_path.exists():
                backup_path = log_path.parent / f"{log_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
                import shutil
                shutil.copy(log_path, backup_path)
                log_path.unlink()
                print(f"📦 기존 로그를 백업했습니다: {backup_path}")
            print("🔄 새로운 Paper Trading 세션을 시작합니다.\n")

        trader = PaperTrader(initial_capital=trading_config.paper_initial_capital)
        print(f"💰 Paper trading 초기 자본: ${trading_config.paper_initial_capital:,.0f}\n")
    elif mode == "live":
        # Load API keys from environment or config
        import os
        api_key = os.getenv("BINANCE_API_KEY", trading_config.api_key)
        api_secret = os.getenv("BINANCE_API_SECRET", trading_config.api_secret)

        if not api_key or not api_secret:
            print("❌ Binance API 키가 설정되지 않았습니다!")
            print("환경 변수 BINANCE_API_KEY, BINANCE_API_SECRET를 설정하세요.")
            return

        trader = BinanceExecutor(
            api_key=api_key,
            api_secret=api_secret,
            symbol=trading_config.symbol,
            leverage=trading_config.leverage
        )
        print(f"⚠️  LIVE TRADING ({trading_config.symbol}, {trading_config.leverage}x leverage)\n")

    strategy = SimpleStrategy(trading_config)
    trade_logger = TradeLogger(trading_config.get_log_path())

    # Data orchestrator
    orchestrator = DataOrchestrator()
    regression_config = RegressionConfig()

    models = None  # Will be loaded/trained
    iteration = 0

    while True:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"Iteration #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}\n")

        try:
            # Step 1: Update data
            print("📥 [1/5] 데이터 업데이트...")
            missing_modules, _ = check_data_status()
            if missing_modules:
                print(f"  누락 모듈: {', '.join(missing_modules)}")
                orchestrator.initial_collection(days=540, targets=missing_modules)
            else:
                orchestrator.smart_update()

            # Step 2: Build features
            print("⚙️  [2/5] 피처 빌드...")
            build_all_features()

            # Step 3: Load master data
            master_path = Path("data/processed/master_features_1h.parquet")
            if not master_path.exists():
                print("❌ 마스터 피처 파일이 없습니다. 다음 iteration에서 재시도...")
                time.sleep(300)
                continue

            df_master = pd.read_parquet(master_path)
            print(f"  Master data: {len(df_master)} rows")

            # Step 4: Train models (once per day at start, or every 24 iterations)
            print("🧠 [3/5] 모델 로딩/학습...")
            if models is None or iteration % 24 == 1:
                print("  새 모델 학습 중...")
                models = train_all_horizons(df_master, regression_config)
            else:
                print("  기존 모델 사용")

            # Step 5: Generate predictions
            print("🔮 [4/5] 예측 생성...")
            predictions = generate_predictions(df_master, models, regression_config)

            print("\n📊 예측 결과:")
            for horizon_hours, pred_return in predictions.items():
                horizon_days = horizon_hours // 24
                print(f"  {horizon_days}일: {pred_return:+.4f} ({pred_return*100:+.2f}%)")

            # Step 6: Execute trades
            print("\n💼 [5/5] 거래 로직 실행...")

            current_price = df_master.iloc[-1][regression_config.close_col]

            if mode == "paper":
                current_position = trader.position
            else:
                current_position = trader.get_position()

            # Generate signal
            action, size, stop_loss, take_profit = strategy.generate_signal(
                predictions=predictions,
                current_position=current_position
            )

            print(f"  신호: {action.upper()}")
            if action != "hold":
                print(f"  크기: {size:.2%}")
                print(f"  Stop Loss: {stop_loss:+.2%}")
                print(f"  Take Profit: {take_profit:+.2%}")

            # Execute
            result = trader.execute(
                action=action,
                size=size,
                current_price=current_price,
                timestamp=pd.Timestamp.now(),
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            if result.get('status') in ['executed', 'buy', 'sell']:
                print(f"  ✅ 거래 실행: {result}")
                trade_logger.log_trade(result)

            # Portfolio status
            if mode == "paper":
                equity = trader.get_equity(current_price)
                pnl = trader.get_pnl(current_price)
                ret = trader.get_return(current_price)
                print(f"\n💰 포트폴리오 상태:")
                print(f"  자본: ${equity:,.2f}")
                print(f"  P&L: ${pnl:+,.2f} ({ret:+.2%})")
                print(f"  포지션: {trader.position:.6f} BTC @ ${trader.entry_price:,.2f}")
            else:
                balance = trader.get_balance()
                position = trader.get_position()
                print(f"\n💰 포트폴리오 상태:")
                print(f"  잔액: ${balance:,.2f} USDT")
                print(f"  포지션: {position:.6f} BTC")

            # Wait for next hour
            print(f"\n⏳ 다음 업데이트까지 대기...")
            next_hour = (datetime.now() + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            wait_seconds = (next_hour - datetime.now()).total_seconds()

            if wait_seconds > 0:
                print(f"   다음 업데이트: {next_hour.strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(wait_seconds)

        except KeyboardInterrupt:
            print("\n\n🛑 종료 중...")

            if mode == "paper" and trader:
                print(f"\n📊 Paper Trading 최종 결과:")
                current_price = df_master.iloc[-1][regression_config.close_col]
                print(f"  초기 자본: ${trader.initial_capital:,.2f}")
                print(f"  최종 자본: ${trader.get_equity(current_price):,.2f}")
                print(f"  총 P&L: ${trader.get_pnl(current_price):+,.2f}")
                print(f"  수익률: {trader.get_return(current_price):+.2%}")

                summary = trader.get_trade_summary()
                print(f"  총 거래 수: {summary['total_trades']}")
                print(f"  승률: {summary['win_rate']:.1%}")

            break

        except Exception as e:
            print(f"\n❌ Iteration #{iteration} 오류: {e}")
            import traceback
            traceback.print_exc()
            print("5분 후 재시도...")
            time.sleep(300)  # Wait 5 minutes before retry


def main():
    """Entry point."""
    mode = select_mode()

    # If paper trading, check if log exists and ask resume/restart
    paper_mode = "resume"
    if mode == "paper":
        trading_config = TradingConfig(mode=mode)
        log_path = Path(trading_config.paper_log_path)
        if log_path.exists():
            paper_mode = select_paper_mode()
        else:
            print("\n🆕 새로운 Paper Trading 세션을 시작합니다.\n")

    run_trading_daemon(mode, paper_mode)


if __name__ == "__main__":
    main()
