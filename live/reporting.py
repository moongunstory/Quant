# ai_binance/live/reporting.py
"""
Reporting + Telegram (merged)
- 파일: update_trade_log(), generate_report()
- 텔레그램 명령: /report, /log
- 푸시 유틸: notify_trade(), notify_snapshot()
.env:
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_ALLOWED_USER_IDS=123,456
  TELEGRAM_PUSH_CHAT_ID=123                # optional
  REPORT_PATH=./data/logs/trading_report.md
  TRADELOG_PATH=./data/logs/run_log.csv
"""
from __future__ import annotations
import os, csv, asyncio, logging
from collections import deque
from datetime import datetime, timezone
from typing import List, Dict, Any, Callable, Awaitable

# ── env ──
try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None  # type: ignore

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")
_cfg = dotenv_values(ENV_PATH) if dotenv_values else {}

TOKEN = (_cfg.get("TELEGRAM_BOT_TOKEN") or "").strip()
ALLOWED_IDS = {
    int(x.strip()) for x in (_cfg.get("TELEGRAM_ALLOWED_USER_IDS") or "").split(",") if x.strip()
}
REPORT_PATH   = os.path.abspath(_cfg.get("REPORT_PATH")   or os.path.join(BASE_DIR, "data", "logs", "trading_report.md"))
TRADELOG_PATH = os.path.abspath(_cfg.get("TRADELOG_PATH") or os.path.join(BASE_DIR, "data", "logs", "run_log.csv"))
PUSH_CHAT_ID  = int((_cfg.get("TELEGRAM_PUSH_CHAT_ID") or "0") or 0)

MAX_MSG = 4000
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
for noisy in ("httpx", "telegram.ext"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.info(f"Loaded ALLOWED_IDS: {ALLOWED_IDS}; REPORT_PATH={REPORT_PATH}; TRADELOG_PATH={TRADELOG_PATH}")

# ── utils ──
def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def _utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _chunk(text: str, n: int = MAX_MSG) -> List[str]:
    return [text[i:i+n] for i in range(0, len(text), n)]

def _tail_lines(path: str, n: int) -> List[str]:
    if not os.path.exists(path): return [f"[ERR] file not found: {path}"]
    dq = deque(maxlen=n)
    with open(path, "r", encoding="utf-8") as f:
        for line in f: dq.append(line.rstrip("\n"))
    return list(dq)

def _prefer_push_chat_id() -> int | None:
    return PUSH_CHAT_ID or (next(iter(ALLOWED_IDS)) if ALLOWED_IDS else None)

def _run_coro(coro: Awaitable[None]) -> None:
    try:
        asyncio.run(coro)
    except RuntimeError:  # 이미 루프가 있으면 백그라운드 태스크로
        loop = asyncio.get_event_loop()
        loop.create_task(coro)

# ── file reporting ──
def update_trade_log(log_path: str, trade_info: Dict[str, Any]) -> None:
    _ensure_dir(log_path)
    need_header = (not os.path.exists(log_path)) or (os.path.getsize(log_path) == 0)
    row = [trade_info.get(k, "") for k in ("timestamp", "type", "position", "price", "profit", "duration")]
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header: w.writerow(["Timestamp", "Type", "Position", "Price", "Profit", "Holding Duration"])
        w.writerow(row)

def generate_report(report_path: str, d: Dict[str, Any], is_new_session: bool) -> None:
    _ensure_dir(report_path)
    g = lambda k, dv=0.0: d.get(k, dv)
    content = (("\n\n" if is_new_session else "") + f"""
==================== TRADING REPORT ====================
매매 시작 시각 : {d.get('session_start_time', '')}
최근 갱신 시각 : {_utcnow_str()}

[ 현재 상태 ]
- 초기 자산: ${float(g('initial_capital')):,.2f}
- 총자산 (현재): ${float(g('total_equity')):,.2f}
- 미실현 손익: ${float(g('unrealized_pnl_amount')):,.2f} ({float(g('unrealized_pnl_percent')):.2f}%)
- 현재 포지션: {d.get('position', 'STANDBY')}

[ 매매 요약 (이번 세션) ]
- 전체 매매 횟수: {int(g('total_trades', 0))}회
- 전체 승률: {float(g('win_rate')):.1f}%

- 롱 포지션: {int(g('long_trades', 0))}회 (승률: {float(g('long_win_rate')):.1f}%)
- 숏 포지션: {int(g('short_trades', 0))}회 (승률: {float(g('short_win_rate')):.1f}%)
- 홀딩 횟수: {int(g('hold_trades', 0))}회

========================================================
""")
    with open(report_path, ("a" if is_new_session else "w"), encoding="utf-8") as f:
        f.write(content)

# ── Telegram push ──
async def _send_text_async(chat_id: int, text: str) -> None:
    if not (TOKEN and chat_id): return
    from telegram import Bot
    bot = Bot(token=TOKEN)
    for part in _chunk(text):
        await bot.send_message(chat_id=chat_id, text=part, disable_web_page_preview=True)

def notify_trade(tr: Dict[str, Any]) -> None:
    chat_id = _prefer_push_chat_id()
    if not chat_id: return
    msg = (
        f"💹 TRADE\n- time: {tr.get('timestamp','')}\n- type: {tr.get('type','')}\n"
        f"- pos : {tr.get('position','')}\n- price: {tr.get('price','')}\n"
    )
    if tr.get("profit") not in (None, ""): msg += f"- profit: {tr.get('profit')}\n"
    _run_coro(_send_text_async(chat_id, msg))

def notify_snapshot(s: Dict[str, Any]) -> None:
    chat_id = _prefer_push_chat_id();  
    if not chat_id: return
    msg = (
        f"📊 SNAPSHOT {_utcnow_str()}\n"
        f"- equity: ${float(s.get('total_equity',0.0)):.2f}\n"
        f"- uPnL : ${float(s.get('unrealized_pnl_amount',0.0)):.2f}\n"
        f"- pos  : {s.get('position','')}\n"
    )
    _run_coro(_send_text_async(chat_id, msg))

# ── Telegram bot (commands) ──
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes

def _guard(fn: Callable[[Update, ContextTypes.DEFAULT_TYPE], Awaitable[None]]):
    async def _w(update: Update, context: ContextTypes.DEFAULT_TYPE):
        uid = update.effective_user.id if update.effective_user else None
        if ALLOWED_IDS and uid not in ALLOWED_IDS: return
        return await fn(update, context)
    return _w

async def _reply(update: Update, text: str):
    for part in _chunk(text):
        await update.message.reply_text(part)

@_guard
async def reports(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(REPORT_PATH): return await _reply(update, f"[ERR] report not found: {REPORT_PATH}")
    with open(REPORT_PATH, "r", encoding="utf-8") as f: txt = f.read().strip() or "[INFO] report is empty."
    await _reply(update, txt)

@_guard
async def log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        n = int(context.args[0]) if context.args else 50
        n = max(1, min(n, 1000))
    except Exception:
        n = 50
    lines = _tail_lines(TRADELOG_PATH, n)
    if not lines:
        await update.message.reply_text("[INFO] Log file is empty.")
        return
    if lines[0].startswith("[ERR]"):
        await update.message.reply_text(lines[0])
        return
    
    txt = f"🧾 run_log.csv (tail {n})\n\n" + "\n".join(lines)
    if len(txt) > MAX_MSG:
        suffix = "\n... (message truncated)"
        txt = txt[:MAX_MSG - len(suffix)] + suffix
    await update.message.reply_text(txt)

def start_bot() -> None:
    """run.py에서 스레드로 호출"""
    print("--- REPORTING.PY START_BOT FUNCTION CALLED ---")
    loop = asyncio.new_event_loop();  asyncio.set_event_loop(loop)
    if not TOKEN:
        logging.error("TELEGRAM_BOT_TOKEN is not set in .env. Bot not starting.")
        return
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("report",    reports))
    app.add_handler(CommandHandler("log",       log))
    # 409 예방: 대기 업데이트 드랍 + 폴링 단일화
    app.run_polling(drop_pending_updates=True, stop_signals=())


if __name__ == "__main__":
    start_bot()
