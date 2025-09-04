# ai_binance/live/run.py
from __future__ import annotations
import os, sys, threading, csv
from datetime import datetime, timezone

# --- path shim ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # ai_binance/live
_AI_DIR   = os.path.abspath(os.path.join(_THIS_DIR, ".."))        # ai_binance
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # project root
for p in (_THIS_DIR, _AI_DIR, _ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from realtime_ingest import LiveIngest
from reporting import update_trade_log, generate_report, start_bot
from trader import (
    BinanceExchange, PublicBinanceData, PaperBroker,
    LiveTrader, PaperTrader, StepResult
)

# ===== 설정 =====
MODE = "live"                 # "live" 또는 "paper"
SYMBOL_ETH = "ETHUSDT"
SYMBOL_BTC = "BTCUSDT"
USE_TESTNET = False
NORM_REWARD_AT_TRAIN = True
LEVERAGE = 5
RISK_FRACTION = 1.0
PAPER_INITIAL_CAPITAL = 10_000.0

# 경로
DATA_DIR   = os.path.join(_AI_DIR, "data")
LOG_DIR    = os.path.join(DATA_DIR, "logs")
LOG_PATH   = os.path.join(LOG_DIR, "run_log.csv")
REPORT_MD  = os.path.join(LOG_DIR, "trading_report.md")

def _utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _load_api_keys() -> tuple[str, str]:
    """OS env 우선, 실패 시 ai_binance/.env 파싱(SECRET alias 지원)"""
    key = (os.getenv("BINANCE_API_KEY") or "").strip()
    sec = (os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_SECRET_KEY") or "").strip()
    if key and sec:
        return key, sec
    env_path = os.path.join(_AI_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            kv = {}
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    kv[k.strip()] = v.strip().strip('"').strip("'")
        key = (kv.get("BINANCE_API_KEY") or key).strip()
        sec = (kv.get("BINANCE_API_SECRET") or kv.get("BINANCE_SECRET_KEY") or sec).strip()
    return key, sec

def _calculate_stats_from_log(log_path: str) -> dict:
    stats = {
        "total_trades": 0, "win_rate": 0.0, "wins": 0,
        "long_trades": 0, "long_win_rate": 0.0, "long_wins": 0,
        "short_trades": 0, "short_win_rate": 0.0, "short_wins": 0,
        "hold_trades": 0,
    }
    if not os.path.exists(log_path):
        return stats

    trades = []
    with open(log_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ('Price', 'Profit'):
                if key in row and row[key] == '':
                    row[key] = '0'
            trades.append(row)

    if not trades:
        return stats

    current_pos = 'FLAT'
    for trade in trades:
        trade_type = trade.get('Type')
        if trade_type == 'ENTRY_LONG':
            current_pos = 'LONG'
        elif trade_type == 'ENTRY_SHORT':
            current_pos = 'SHORT'
        elif trade_type == 'CLOSE':
            if current_pos == 'FLAT':
                continue

            stats['total_trades'] += 1
            profit = float(trade.get('Profit', 0) or 0)
            if profit > 0:
                stats['wins'] += 1

            if current_pos == 'LONG':
                stats['long_trades'] += 1
                if profit > 0:
                    stats['long_wins'] += 1
            elif current_pos == 'SHORT':
                stats['short_trades'] += 1
                if profit > 0:
                    stats['short_wins'] += 1
            
            current_pos = 'FLAT'

    if stats['total_trades'] > 0:
        stats['win_rate'] = (stats['wins'] / stats['total_trades']) * 100
    if stats['long_trades'] > 0:
        stats['long_win_rate'] = (stats['long_wins'] / stats['long_trades']) * 100
    if stats['short_trades'] > 0:
        stats['short_win_rate'] = (stats['short_wins'] / stats['short_trades']) * 100
        
    return stats

def _start_bot_async():
    threading.Thread(target=start_bot, daemon=True).start()

def _build_stack():
    """MODE에 따라 인제스터/트레이더/초기자산 구성"""
    if MODE == "live":
        api_key, api_secret = _load_api_keys()
        if not (api_key and api_secret):
            raise SystemExit(
                "MODE='live'인데 키 없음. OS 환경변수 또는 ai_binance/.env에 설정하세요.\n"
                "예)\\n  BINANCE_API_KEY=...\\n  BINANCE_API_SECRET=...  (또는 BINANCE_SECRET_KEY=...)"
            )
        ex = BinanceExchange(
            symbol_eth=SYMBOL_ETH, symbol_btc=SYMBOL_BTC,
            api_key=api_key, api_secret=api_secret, use_testnet=USE_TESTNET
        )
        ex.set_leverage(LEVERAGE)
        ingest = LiveIngest(ex)
        trader = LiveTrader(
            exec_client=ex, norm_reward_at_train=NORM_REWARD_AT_TRAIN,
            symbol_eth=SYMBOL_ETH, leverage=LEVERAGE, risk_fraction=RISK_FRACTION,
        )
    else:
        data = PublicBinanceData(testnet=USE_TESTNET)
        ingest = LiveIngest(data)
        trader = PaperTrader(
            data_client=data, exec_client=PaperBroker(),
            norm_reward_at_train=NORM_REWARD_AT_TRAIN,
            symbol_eth=SYMBOL_ETH, leverage=LEVERAGE, risk_fraction=RISK_FRACTION,
            init_equity=PAPER_INITIAL_CAPITAL,
        )
    return ingest, trader

def main():
    print("--- RUN.PY MAIN FUNCTION STARTED ---")
    _start_bot_async()

    ingest, trader = _build_stack()
    
    report_data = _calculate_stats_from_log(LOG_PATH)
    report_data["session_start_time"] = _utcnow_str()
    report_data["initial_capital"] = trader.initial_equity
    
    report_data["total_equity"] = trader.equity if hasattr(trader, 'equity') else trader.initial_equity
    report_data["unrealized_pnl_amount"] = 0.0
    report_data["unrealized_pnl_percent"] = 0.0
    report_data["position"] = "FLAT"
    generate_report(REPORT_MD, report_data, is_new_session=True)

    position_type_for_stat = 'FLAT'

    while True:
        obs_s, ts = ingest.wait_next_5m_and_build(poll_sec=2.0, grace_sec=2.0)
        step: StepResult = trader.step(obs_s.to_numpy(), ts)

        new_trade_closed = False
        for row in step.logs:
            update_trade_log(LOG_PATH, row)
            if row.get('type', '').startswith('ENTRY_'):
                position_type_for_stat = row['type'].split('_')[1]
            elif row.get('type') == 'CLOSE':
                if position_type_for_stat == 'FLAT': continue
                new_trade_closed = True
                report_data['total_trades'] += 1
                profit = float(row.get('profit', 0) or 0)
                
                if profit > 0:
                    report_data['wins'] += 1

                if position_type_for_stat == 'LONG':
                    report_data['long_trades'] += 1
                    if profit > 0: report_data['long_wins'] += 1
                elif position_type_for_stat == 'SHORT':
                    report_data['short_trades'] += 1
                    if profit > 0: report_data['short_wins'] += 1
                
                position_type_for_stat = 'FLAT'

        if new_trade_closed:
            if report_data['total_trades'] > 0:
                report_data['win_rate'] = (report_data['wins'] / report_data['total_trades']) * 100
            if report_data['long_trades'] > 0:
                report_data['long_win_rate'] = (report_data['long_wins'] / report_data['long_trades']) * 100
            if report_data['short_trades'] > 0:
                report_data['short_win_rate'] = (report_data['short_wins'] / report_data['short_trades']) * 100

        report_data.update(step.report_snapshot)
        generate_report(REPORT_MD, report_data, is_new_session=False)

        s = step.summary
        filter_str = f" | filter={s['filter']}" if s.get('filter') else ""
        print(
            f"{ts.isoformat(timespec='seconds')} | mode={MODE} | pos={s['pos']} | "
            f"px={s['price']} | eq={s['equity']} | value={s['value']} | "
            f"prediction={s['prediction']}{filter_str}"
        )

if __name__ == "__main__":
    main()
