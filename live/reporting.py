# ai_binance/live/reporting.py
"""
Reporting + Telegram (merged)
- 파일 리포팅: update_trade_log(), generate_report()
- 텔레그램: /start /whoami /reports /reportfile /log /logfile
- 푸시 유틸: notify_trade(), notify_snapshot()
.env 키:
  TELEGRAM_BOT_TOKEN=
  TELEGRAM_ALLOWED_USER_IDS=12345,67890
  TELEGRAM_PUSH_CHAT_ID=12345               # (옵션) 푸시 기본 채팅 ID
  REPORT_PATH=./data/logs/reports/trading_report.md
  TRADELOG_PATH=./data/logs/run_log.csv
"""

from __future__ import annotations
import os, csv, asyncio, logging
from collections import deque
from datetime import datetime, timezone
from typing import List, Dict, Any

# ───────── Env ─────────
try:
    from dotenv import dotenv_values
except Exception:  # dotenv 없어도 기본 경로만 사용
    dotenv_values = None  # type: ignore

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")
_cfg = dotenv_values(ENV_PATH) if dotenv_values else {}

TOKEN = (_cfg.get("TELEGRAM_BOT_TOKEN") or "").strip()
ALLOWED_IDS = {
    int(x.strip())
    for x in (_cfg.get("TELEGRAM_ALLOWED_USER_IDS") or "").split(",")
    if x.strip()
}
REPORT_PATH = os.path.join(BASE_DIR, "data", "logs", "reports", "trading_report.md")
TRADELOG_PATH = os.path.join(BASE_DIR, "data", "logs", "run_log.csv")
PUSH_CHAT_ID = int((_cfg.get("TELEGRAM_PUSH_CHAT_ID") or "0") or 0)

MAX_MSG = 4000  # Telegram 안전 길이
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger.info(f"Loaded ALLOWED_IDS: {ALLOWED_IDS}; REPORT_PATH={REPORT_PATH}; TRADELOG_PATH={TRADELOG_PATH}")

# ───────── 공통 유틸 ─────────
def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)

def _utcnow_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# ───────── File Reporting ─────────
def update_trade_log(log_path: str, trade_info: Dict[str, Any]) -> None:
    """
    매매 내역(run_log.csv) 업데이트 (비동기 없음 / 안전).
    기대 키: timestamp, type, position, price, profit(옵션), duration(옵션)
    """
    _ensure_dir(log_path)
    file_exists = os.path.exists(log_path)
    need_header = (not file_exists) or (os.path.getsize(log_path) == 0)

    row = [
        trade_info.get("timestamp", ""),
        trade_info.get("type", ""),
        trade_info.get("position", ""),
        trade_info.get("price", ""),
        trade_info.get("profit", ""),
        trade_info.get("duration", ""),
    ]
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["Timestamp", "Type", "Position", "Price", "Profit", "Holding Duration"])
        w.writerow(row)

def generate_report(report_path: str, report_data: Dict[str, Any], is_new_session: bool) -> None:
    """
    사람이 읽기 쉬운 리포트(trading_report.md) 생성/갱신.
    기대 키(없으면 기본값):
      session_start_time, initial_capital, position, total_equity,
      unrealized_pnl_amount, unrealized_pnl_percent,
      total_trades, win_rate, long_trades, long_win_rate,
      short_trades, short_win_rate, hold_trades
    """
    _ensure_dir(report_path)
    now_utc = _utcnow_str()
    g = lambda k, dv=0.0: report_data.get(k, dv)
    pos = report_data.get("position", "STANDBY")
    content = (
        "\n\n" if is_new_session else ""
    ) + f"""
==================== TRADING REPORT ====================
매매 시작 시각 : {report_data.get('session_start_time', '')}
최근 갱신 시각 : {now_utc}

[ 현재 상태 ]
- 초기 자산: ${float(g('initial_capital')):,.2f}
- 총자산 (현재): ${float(g('total_equity')):,.2f}
- 미실현 손익: ${float(g('unrealized_pnl_amount')):,.2f} ({float(g('unrealized_pnl_percent')):.2f}%)
- 현재 포지션: {pos}

[ 매매 요약 (이번 세션) ]
- 전체 매매 횟수: {int(report_data.get('total_trades', 0))}회
- 전체 승률: {float(g('win_rate')):.1f}%

- 롱 포지션: {int(report_data.get('long_trades', 0))}회 (승률: {float(g('long_win_rate')):.1f}%)
- 숏 포지션: {int(report_data.get('short_trades', 0))}회 (승률: {float(g('short_win_rate')):.1f}%)
- 홀딩 횟수: {int(report_data.get('hold_trades', 0))}회

========================================================
"""
    mode = "a" if is_new_session else "w"
    with open(report_path, mode, encoding="utf-8") as f:
        f.write(content)

# ───────── Telegram 공용(명령/푸시 공통) ─────────
def _chunk(text: str, n: int = MAX_MSG) -> List[str]:
    return [text[i:i+n] for i in range(0, len(text), n)]

def _tail_lines(path: str, n: int) -> List[str]:
    if not os.path.exists(path):
        return [f"[ERR] file not found: {path}"]
    dq = deque(maxlen=n)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            dq.append(line.rstrip("\n"))
    return list(dq)

def _prefer_push_chat_id() -> int | None:
    if PUSH_CHAT_ID:
        return PUSH_CHAT_ID
    if ALLOWED_IDS:
        return next(iter(ALLOWED_IDS))
    return None

# ───────── Telegram 푸시 (run에서 원할 때 호출) ─────────
async def _send_text_async(chat_id: int, text: str) -> None:
    if not TOKEN or not chat_id:
        return
    from telegram import Bot
    bot = Bot(token=TOKEN)
    for part in _chunk(text):
        await bot.send_message(chat_id=chat_id, text=part, disable_web_page_preview=True)

def notify_trade(trade_info: Dict[str, Any]) -> None:
    """
    즉시 거래 푸시(옵션): run에서 update_trade_log() 후 원하면 호출.
    .env의 TELEGRAM_PUSH_CHAT_ID 또는 ALLOWED_IDS 중 하나를 대상으로 전송.
    """
    chat_id = _prefer_push_chat_id()
    if not chat_id:
        return
    msg = (
        f"💹 TRADE\n"
        f"- time: {trade_info.get('timestamp','')}\n"
        f"- type: {trade_info.get('type','')}\n"
        f"- pos : {trade_info.get('position','')}\n"
        f"- price: {trade_info.get('price','')}\n"
    )
    if trade_info.get("profit") not in (None, ""):
        msg += f"- profit: {trade_info.get('profit')}\n"
    try:
        asyncio.run(_send_text_async(chat_id, msg))
    except RuntimeError:
        # 이미 다른 이벤트 루프가 돌고 있으면 백그라운드 태스크로
        loop = asyncio.get_event_loop()
        loop.create_task(_send_text_async(chat_id, msg))

def notify_snapshot(snap: Dict[str, Any]) -> None:
    """
    즉시 스냅샷 푸시(옵션): generate_report() 직후 한 줄 요약 보내고 싶을 때.
    기대 키: total_equity, unrealized_pnl_amount, position
    """
    chat_id = _prefer_push_chat_id()
    if not chat_id:
        return
    msg = (
        f"📊 SNAPSHOT { _utcnow_str() }\n"
        f"- equity: ${float(snap.get('total_equity',0.0)):.2f}\n"
        f"- uPnL : ${float(snap.get('unrealized_pnl_amount',0.0)):.2f}\n"
        f"- pos  : {snap.get('position','')}\n"
    )
    try:
        asyncio.run(_send_text_async(chat_id, msg))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.create_task(_send_text_async(chat_id, msg))

# ───────── Telegram 봇(명령형) ─────────
# 필요 시 이 파일을 단독 실행하면 폴링 봇이 뜬다.
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes

def _authorized(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    if not ALLOWED_IDS:
        return True
    return uid in ALLOWED_IDS

async def _send_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    for part in _chunk(text):
        await update.message.reply_text(part)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    msg = (
        "🤖 Trading Bot Ready.\n"
        "명령어:\n"
        "  /whoami       — 내 user id 확인\n"
        "  /reports      — trading_report.md 내용 보기\n"
        "  /reportfile   — trading_report.md 파일 전송\n"
        "  /log [N]      — run_log.csv 최근 N줄(기본 50)\n"
        "  /logfile      — run_log.csv 파일 전송\n"
        f"(allowed: {ALLOWED_IDS if ALLOWED_IDS else 'ALL'})"
    )
    await _send_text(update, context, msg)

async def whoami(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.effective_user else None
    await _send_text(update, context, f"👤 your id = `{uid}`")

async def reports(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    if not os.path.exists(REPORT_PATH):
        await _send_text(update, context, f"[ERR] report not found: {REPORT_PATH}")
        return
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    await _send_text(update, context, txt or "[INFO] report is empty.")

async def reportfile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    if not os.path.exists(REPORT_PATH):
        await _send_text(update, context, f"[ERR] report not found: {REPORT_PATH}")
        return
    await update.message.reply_document(InputFile(REPORT_PATH, filename=os.path.basename(REPORT_PATH)))

async def log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    try:
        n = int(context.args[0]) if context.args else 50
        n = max(1, min(n, 1000))
    except Exception:
        n = 50
    lines = _tail_lines(TRADELOG_PATH, n)
    if lines and not lines[0].startswith("[ERR]"):
        header = f"🧾 run_log.csv (tail {n})\n\n"
        body = "\n".join(lines)
        txt = header + body
        if len(txt) > MAX_MSG:
            await logfile(update, context)
            return
        await _send_text(update, context, txt)
    else:
        await _send_text(update, context, lines[0])

async def logfile(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    if not os.path.exists(TRADELOG_PATH):
        await _send_text(update, context, f"[ERR] log not found: {TRADELOG_PATH}")
        return
    await update.message.reply_document(InputFile(TRADELOG_PATH, filename=os.path.basename(TRADELOG_PATH)))

def start_bot() -> None:
    """텔레그램 봇을 시작합니다. run.py에서 스레드로 호출됩니다."""
    if not TOKEN:
        # raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")
        logger.error("TELEGRAM_BOT_TOKEN is not set in .env. Bot not starting.")
        return
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("reports", reports))
    app.add_handler(CommandHandler("reportfile", reportfile))
    app.add_handler(CommandHandler("log", log))
    app.add_handler(CommandHandler("logfile", logfile))
    app.run_polling()

if __name__ == "__main__":
    # 단독 테스트 실행용
    start_bot()