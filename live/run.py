# ai_binance/live/run.py
from __future__ import annotations
import os, sys
from datetime import datetime, timezone

# --- path shim ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))           # ai_binance/live
_AI_DIR   = os.path.abspath(os.path.join(_THIS_DIR, ".."))        # ai_binance
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))  # project root
for p in (_THIS_DIR, _AI_DIR, _ROOT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from realtime_ingest import LiveIngest
from reporting import update_trade_log, generate_report
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

def _utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ===== 간단 .env 로더 (OS env 우선, fallback: ai_binance/.env) =====
def _load_api_keys() -> tuple[str, str]:
    # 1) OS 환경변수
    key = (os.getenv("BINANCE_API_KEY") or "").strip()
    sec = (os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_SECRET_KEY") or "").strip()
    if key and sec:
        return key, sec
    # 2) ai_binance/.env
    env_path = os.path.join(_AI_DIR, ".env")
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            kv = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip().strip('"').strip("'")
            key = kv.get("BINANCE_API_KEY", key).strip()
            # SECRET alias 지원: BINANCE_API_SECRET, BINANCE_SECRET_KEY
            sec = (kv.get("BINANCE_API_SECRET") or kv.get("BINANCE_SECRET_KEY") or sec).strip()
    except FileNotFoundError:
        pass
    return key, sec

def main():
    if MODE == "live":
        api_key, api_secret = _load_api_keys()
        if not api_key or not api_secret:
            raise SystemExit(
                "MODE='live'인데 키 없음. OS 환경변수 또는 ai_binance/.env에 설정하세요.\n"
                "예)\n  BINANCE_API_KEY=...\n  BINANCE_API_SECRET=...  (또는 BINANCE_SECRET_KEY=...)"
            )
        ex = BinanceExchange(
            symbol_eth=SYMBOL_ETH, symbol_btc=SYMBOL_BTC,
            api_key=api_key, api_secret=api_secret,
            use_testnet=USE_TESTNET
        )
        ex.set_leverage(LEVERAGE)
        ingest = LiveIngest(ex)
        trader = LiveTrader(
            exec_client=ex,
            norm_reward_at_train=NORM_REWARD_AT_TRAIN,
            symbol_eth=SYMBOL_ETH,
            leverage=LEVERAGE,
            risk_fraction=RISK_FRACTION,
        )
        session_initial_equity = trader.initial_equity
    else:
        data = PublicBinanceData(testnet=USE_TESTNET)
        broker = PaperBroker()
        ingest = LiveIngest(data)
        trader = PaperTrader(
            data_client=data, exec_client=broker,
            norm_reward_at_train=NORM_REWARD_AT_TRAIN,
            symbol_eth=SYMBOL_ETH, leverage=LEVERAGE,
            risk_fraction=RISK_FRACTION,
            init_equity=PAPER_INITIAL_CAPITAL,
        )
        session_initial_equity = trader.initial_equity

    # 세션 헤더
    os.makedirs(REPORT_DIR, exist_ok=True)
    generate_report(
        REPORT_MD,
        {
            "session_start_time": _utcnow_str(),
            "initial_capital": session_initial_equity,
            "total_equity": session_initial_equity,
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

    # 메인 루프
    while True:
        obs_s, ts = ingest.wait_next_5m_and_build(poll_sec=2.0, grace_sec=2.0)
        step: StepResult = trader.step(obs_s.to_numpy(), ts)

        for row in step.logs:
            update_trade_log(LOG_PATH, row)
        generate_report(REPORT_MD, step.report_snapshot, is_new_session=False)

        s = step.summary

        # --- 판단 근거 문자열 생성 ---
        btc_state, h4_state, h1_state, m15_state, m5_state = 'SIDE', 'FLAT', 'FLAT', '----', '----'

        # 1. BTC 시장 (BULL/BEAR/SIDE)
        btc_ret = obs_s.get('f_btc1h_ret_1h_btc1h', 0)
        if btc_ret > 0.0005: btc_state = 'BULL'
        elif btc_ret < -0.0005: btc_state = 'BEAR'

        # 2. 4H 장기 추세 (UP/DOWN/FLAT)
        h4_supertrend = obs_s.get('f_4h_supertrend_dir_4h', 0)
        if h4_supertrend > 0: h4_state = 'UP'
        elif h4_supertrend < 0: h4_state = 'DOWN'

        # 3. 1H 중기 추세 (UP/DOWN/FLAT)
        h1_slope = obs_s.get('f_1h_ema50_slope_1h', 0)
        if h1_slope > 0.0002: h1_state = 'UP'
        elif h1_slope < -0.0002: h1_state = 'DOWN'

        # 4. 15M 변동성 (SQZ/----)
        m15_squeeze = obs_s.get('f_15m_squeeze_ratio_15m', 0)
        if m15_squeeze < -0.2: m15_state = 'SQZ'

        # 5. 5M 진입 신호 (TRIG/----)
        break_up = obs_s.get('f_5m_break_up_5m', 0)
        break_down = obs_s.get('f_5m_break_down_5m', 0)
        if break_up > 0 or break_down > 0: m5_state = 'TRIG'

        # 영문 키워드를 한글로 변환
        ko_map = {
            'BULL': '상승', 'BEAR': '하락', 'SIDE': '횡보',
            'UP':   '상승', 'DOWN': '하락', 'FLAT': '보합',
            'SQZ':  '응축', '----': ' -- ',
            'TRIG': '신호'
        }
        reason_str = (
            f"reason=[ {ko_map[btc_state]} | {ko_map[h4_state]} | {ko_map[h1_state]} | "
            f"{ko_map[m15_state]} | {ko_map[m5_state]} ]"
        )
        # --- 생성 끝 ---

        print(f"{ts.isoformat()} | mode={MODE} | action={s['action']} | pos={s['pos']} | px={s['price']:.2f} | eq={s.get('equity','-')} | {reason_str}")

if __name__ == "__main__":
    main()
