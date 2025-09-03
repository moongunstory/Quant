# ai_binance/live/run.py
from __future__ import annotations
import os, sys, threading
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
MODE = "paper"                 # "live" 또는 "paper"
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
REPORT_DIR = os.path.join(LOG_DIR, "reports")
LOG_PATH   = os.path.join(LOG_DIR, "run_log.csv")
REPORT_MD  = os.path.join(REPORT_DIR, "trading_report.md")

_ko_map = {
    'BULL': '상승', 'BEAR': '하락', 'SIDE': '횡보',
    'UP': '상승', 'DOWN': '하락', 'FLAT': '보합',
    'SQZ': '응축', '----': ' -- ', 'TRIG': '신호'
}

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

def _init_report(initial_equity: float) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)
    generate_report(
        REPORT_MD,
        {
            "session_start_time": _utcnow_str(),
            "initial_capital": initial_equity,
            "total_equity": initial_equity,
            "unrealized_pnl_amount": 0.0,
            "unrealized_pnl_percent": 0.0,
            "position": "FLAT",
            "total_trades": 0, "win_rate": 0.0,
            "long_trades": 0, "long_win_rate": 0.0,
            "short_trades": 0, "short_win_rate": 0.0,
            "hold_trades": 0,
        },
        is_new_session=True,
    )

def _compute_reason(obs_s) -> str:
    """영문 레짐/신호 → 한글 요약 문자열 생성"""
    get = obs_s.get
    btc_state = 'BULL' if get('f_btc1h_ret_1h_btc1h', 0) >  0.0005 else \
                'BEAR' if get('f_btc1h_ret_1h_btc1h', 0) < -0.0005 else 'SIDE'
    h4_dir    = get('f_4h_supertrend_dir_4h', 0)
    h4_state  = 'UP' if h4_dir > 0 else ('DOWN' if h4_dir < 0 else 'FLAT')
    h1_slope  = get('f_1h_ema50_slope_1h', 0)
    h1_state  = 'UP' if h1_slope >  0.0002 else ('DOWN' if h1_slope < -0.0002 else 'FLAT')
    m15_state = 'SQZ' if get('f_15m_squeeze_ratio_15m', 0) < -0.2 else '----'
    m5_state  = 'TRIG' if (get('f_5m_break_up_5m', 0) > 0 or get('f_5m_break_down_5m', 0) > 0) else '----'
    return "reason=[ " + " | ".join(_ko_map[s] for s in (btc_state, h4_state, h1_state, m15_state, m5_state)) + " ]"

def _start_bot_async():
    threading.Thread(target=start_bot, daemon=True).start()

def _build_stack():
    """MODE에 따라 인제스터/트레이더/초기자산 구성"""
    if MODE == "live":
        api_key, api_secret = _load_api_keys()
        if not (api_key and api_secret):
            raise SystemExit(
                "MODE='live'인데 키 없음. OS 환경변수 또는 ai_binance/.env에 설정하세요.\n"
                "예)\n  BINANCE_API_KEY=...\n  BINANCE_API_SECRET=...  (또는 BINANCE_SECRET_KEY=...)"
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
    _init_report(trader.initial_equity)

    while True:
        obs_s, ts = ingest.wait_next_5m_and_build(poll_sec=2.0, grace_sec=2.0)
        step: StepResult = trader.step(obs_s.to_numpy(), ts)

        for row in step.logs:
            update_trade_log(LOG_PATH, row)
        generate_report(REPORT_MD, step.report_snapshot, is_new_session=False)

        s = step.summary
        print(
            f"{ts.isoformat()} | mode={MODE} | action={s['action']} | "
            f"pos={s['pos']} | px={s['price']:.2f} | eq={s.get('equity','-')} | {_compute_reason(obs_s)}"
        )

if __name__ == "__main__":
    main()
