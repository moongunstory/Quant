import os, asyncio, logging
from collections import deque
from typing import List
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, ContextTypes
from dotenv import load_dotenv

# ====== .env 로드 ======
load_dotenv()

# ====== 로거 설정 ======
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)  # HTTPX 로거 레벨 조정
logger = logging.getLogger(__name__)

# ====== 설정 ======
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
ALLOWED_IDS = {
    int(x.strip()) for x in os.getenv("TELEGRAM_ALLOWED_USER_IDS", "").split(",") if x.strip()
}
logger.info(f"Loaded ALLOWED_IDS: {ALLOWED_IDS}")

REPORT_PATH = os.getenv("REPORT_PATH", "./data/reports/trading_report.md")
TRADELOG_PATH = os.getenv("TRADELOG_PATH", "./data/logs/run_log.csv")
MAX_MSG = 4000  # 텔레그램 메시지 길이 한계 안전선

# ====== 유틸 ======
def _authorized(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    logger.info(f"Checking authorization for user ID: {uid}")
    if not ALLOWED_IDS:
        logger.info("No ALLOWED_IDS set, authorizing.")
        return True
    is_auth = uid in ALLOWED_IDS
    logger.info(f"Is user authorized? {is_auth}")
    return is_auth

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

async def _send_text(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str):
    for part in _chunk(text):
        await update.message.reply_text(part)

# ====== 핸들러 ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _authorized(update):
        return
    msg = (
        "🤖 Trading Bot Ready.\n"
        "명령어:\n"
        "  /whoami  — 내 user id 확인\n"
        "  /reports — trading_report.md 내용 보기(길면 문서로 전송)\n"
        "  /reportfile — trading_report.md 파일 전송\n"
        "  /log [N] — run_log.csv 최근 N줄(기본 50줄)\n"
        "  /logfile — run_log.csv 파일 전송\n"
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
    size = os.path.getsize(REPORT_PATH)
    if size > 3000:  # 길면 파일로 전송
        await update.message.reply_document(InputFile(REPORT_PATH, filename=os.path.basename(REPORT_PATH)))
        return
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        await _send_text(update, context, "[INFO] report is empty.")
    else:
        await _send_text(update, context, txt)

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

# ====== 실행 ======
def main():
    if not TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in .env")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("whoami", whoami))
    app.add_handler(CommandHandler("reports", reports))
    app.add_handler(CommandHandler("reportfile", reportfile))
    app.add_handler(CommandHandler("log", log))
    app.add_handler(CommandHandler("logfile", logfile))
    app.run_polling(read_timeout=30, close_loop=False)

if __name__ == "__main__":
    main()
