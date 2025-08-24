# ai_binance/run.py
"""
Main Runner for Live Trading Stack (HRL/MTF compatible)
- RealtimeIngestMTF(5m 확정봉 + 15m/1h/4h 피처) → Dispatcher → Trader / OnlineLearner
- 온라인 업데이트(KL 가드) + 모델 자동 리로드(Worker/MaskablePPO 대응)
"""

from __future__ import annotations

import os
import time
import threading
import logging
import warnings
import csv
import subprocess
from queue import Queue
from typing import Optional
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# ===== SB3 안전 로더 (PPO 커스텀 스케줄 역직렬화 가드) =====
from stable_baselines3 import PPO
def _const_schedule(v: float):
    return lambda _progress_remaining: v
__ORIG_PPO_LOAD = PPO.load
def _ppo_load_safe(path, *args, **kwargs):
    try:
        return __ORIG_PPO_LOAD(path, *args, **kwargs)
    except Exception:
        co = dict(kwargs.get("custom_objects", {}))
        co.setdefault("lr_schedule", _const_schedule(3e-4))
        co.setdefault("clip_range", _const_schedule(0.2))
        kwargs["custom_objects"] = co
        return __ORIG_PPO_LOAD(path, *args, **kwargs)
PPO.load = _ppo_load_safe  # 전역 패치

# --- 내부 모듈 경로 ---
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# ⚠️ MTF 인제스터로 교체
from ai_binance.live.realtime_ingest import RealtimeIngest as RealtimeIngest
from ai_binance.live.trader import Trader                    # HRL/MaskablePPO 대응 리팩터
from ai_binance.live.learner import OnlineLearner     # 모듈명 수정(learner → online_learner)

# .env 로드 (프로젝트 루트 기준)
dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# =========================
# 설정
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent         # ~/ai_binance
MODEL_DIR = BASE_DIR / "data" / "model"
LOGS_DIR  = BASE_DIR / "data" / "logs"

ENABLE_TRADING          = True
TRADING_MODE            = os.getenv("TRADING_MODE", "paper")  # live / paper 
ENABLE_ONLINE_LEARNING  = True

# === 텔레그램 봇 실행 옵션 ===
ENABLE_TELEGRAM_BOT = True
TELEGRAM_BOT_PATH   = BASE_DIR / "telegram_bot.py"  # .env는 bot 내부에서 로드

# === 온라인 학습 트리거(보수형) ===
MIN_BUFFER_BARS     = 15_000   # ~52d @5m
TRIGGER_EVERY_BARS  = 5_000    # ~17.4d @5m
PROMO_COOLDOWN_BARS = 576      # 48h @5m

# OnlineLearner의 기본 저장명과 일치(Worker/MaskablePPO)
LIVE_OUT_NAME = os.getenv("LIVE_OUT_NAME", "worker_unified_live.zip")
LIVE_OUT_PATH = MODEL_DIR / LIVE_OUT_NAME

INGEST_QUEUE_MAX = 2
TRADER_QUEUE_MAX = 2
LEARN_QUEUE_MAX  = 2

BINANCE_API_KEY   = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY= os.getenv('BINANCE_SECRET_KEY')

# =========================
# CSV/콘솔 로거 설정
# =========================
class TradingLogFilter(logging.Filter):
    def filter(self, record):
        return record.threadName == 'Trader'

os.makedirs(LOGS_DIR, exist_ok=True)
log_file_path = os.path.join(LOGS_DIR, "run_log.csv")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

console_handler = logging.StreamHandler(sys.__stdout__)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

CSV_HEADER = ['timestamp', 'level', 'threadName', 'message']
csv_handler = None
if TRADING_MODE != "live":
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()

    class CsvFileHandler(logging.FileHandler):
        def __init__(self, filename, mode='a', encoding=None, delay=False):
            super().__init__(filename, mode, encoding, delay)
        def emit(self, record):
            msg = self.format(record)
            row = {
                'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'level': record.levelname,
                'threadName': record.threadName,
                'message': msg
            }
            with open(self.baseFilename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
                writer.writerow(row)

    csv_handler = CsvFileHandler(log_file_path)
    csv_handler.setFormatter(logging.Formatter('%(message)s'))
    csv_handler.addFilter(TradingLogFilter())
    logger.addHandler(csv_handler)

logging.captureWarnings(True)
pywarn = logging.getLogger("py.warnings")
pywarn.setLevel(logging.WARNING)
pywarn.handlers = []
pywarn.addHandler(console_handler)
if csv_handler is not None:
    pywarn.addHandler(csv_handler)

# === stdout/stderr → logger 리다이렉트 ===
class _StdToLog:
    def __init__(self, _logger, level):
        self.logger = _logger
        self.level = level
        self.buf = ""
        self.lock = threading.Lock()
    def write(self, msg: str):
        if not msg:
            return
        with self.lock:
            self.buf += msg
            while "\n" in self.buf:
                line, self.buf = self.buf.split("\n", 1)
                line = line.rstrip()
                if line:
                    self.logger.log(self.level, line)
    def flush(self):
        with self.lock:
            if self.buf.strip():
                self.logger.log(self.level, self.buf.strip())
                self.buf = ""

sys.stdout = _StdToLog(logger, logging.INFO)
sys.stderr = _StdToLog(logger, logging.ERROR)

# =========================
# 유틸
# =========================
def _mtime(path: str) -> float:
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return 0.0

# =========================
# 디스패처
# =========================
class Dispatcher(threading.Thread):
    daemon = True
    def __init__(self, ingest_q: Queue, trader_q: Queue, learn_q: Queue):
        super().__init__(name="Dispatcher")
        self.ingest_q = ingest_q
        self.trader_q = trader_q
        self.learn_q = learn_q
    def run(self):
        logger.info("디스패처 시작됨")
        while True:
            pkt = self.ingest_q.get()
            for q in (self.trader_q, self.learn_q):
                try:
                    if q.full():
                        q.get_nowait()
                    q.put_nowait(pkt)
                except Exception:
                    pass

# =========================
# 트레이더 리로더 (모델 클래스 보존)
# =========================
class TraderReloader(threading.Thread):
    daemon = True
    def __init__(self, trader: Trader, target_path: str, check_sec: float = 5.0):
        super().__init__(name="Reloader")
        self.trader = trader
        self.target_path = target_path
        self.check_sec = check_sec
        self._last_mtime = _mtime(target_path)
    def run(self):
        logger.info(f"리로더 감시 중: {self.target_path}")
        while True:
            time.sleep(self.check_sec)
            mt = _mtime(self.target_path)
            if mt > self._last_mtime:
                self._last_mtime = mt
                try:
                    loader = getattr(self.trader.model.__class__, "load", None) or PPO.load
                    new_model = loader(self.target_path, device="cpu")
                    self.trader.model = new_model
                    logger.info(f"모델 리로드 완료: {self.target_path}")
                except Exception as e:
                    logger.error(f"리로드 실패: {e}", exc_info=True)

# =========================
# 온라인 학습 워커
# =========================
class OnlineLearnWorker(threading.Thread):
    daemon = True
    def __init__(self, learn_q: Queue):
        super().__init__(name="Learner")
        self.learn_q = learn_q
        self.X_tail: Optional[pd.DataFrame] = None
        self.close_tail: Optional[pd.Series] = None
        self.funding_tail: Optional[pd.Series] = None
        self.bars_seen = 0
        self.last_trigger_bars = 0
        self.last_promo_bars = 0
        self.ol = OnlineLearner(out_model_name=LIVE_OUT_NAME)
    def run(self):
        logger.info("온라인 학습 워커 시작됨")
        while True:
            pkt = self.learn_q.get()

            # ---- (A) Trader → rollout 온라인 업데이트 ----
            if isinstance(pkt, dict) and {"obs", "actions", "rewards", "dones", "values", "log_probs"} <= set(pkt.keys()):
                try:
                    improved = self.ol.update_from_rollout(
                        pkt,
                        max_kl=0.02, epochs=1, batch_size=1024, lr=5e-5,
                        X_val=self.X_tail.iloc[-3000:] if isinstance(self.X_tail, pd.DataFrame) and len(self.X_tail) >= 3000 else None,
                        close_val=self.close_tail.iloc[-3000:] if isinstance(self.close_tail, pd.Series) and len(self.close_tail) >= 3000 else None,
                        funding_val=self.funding_tail.iloc[-3000:] if isinstance(self.funding_tail, pd.Series) and len(self.funding_tail) >= 3000 else None,
                        save_if_improved=True
                    )
                    if improved:
                        logger.info("✅ 온라인 업데이트 저장 완료")
                    else:
                        logger.info("ℹ️ 온라인 업데이트 미적용(KL 초과/조건 미충족)")
                except Exception as e:
                    logger.error(f"온라인 업데이트 실패: {e}", exc_info=True)
                continue

            # ---- (B) Ingest → 최근 꼬리 적재 및 주기적 미세조정 ----
            X_5m: pd.DataFrame = pkt.get("X_5m") or pkt.get("X")  # 호환
            close: pd.Series = pkt["close"]
            funding: Optional[pd.Series] = pkt.get("funding")

            # 5m 기준 꼬리만 보관(학습 분포 안정)
            want_tail = max(MIN_BUFFER_BARS + TRIGGER_EVERY_BARS + 5000, 30000)
            X_5m = (X_5m.get("X") or X_5m.get("5m")).iloc[-want_tail:].copy()
            close = close.reindex(X_5m.index).ffill().bfill()
            if funding is not None:
                funding = funding.reindex(X_5m.index).ffill().bfill()

            self.X_tail = X_5m
            self.close_tail = close
            self.funding_tail = funding
            self.bars_seen = len(X_5m)

            if (self.bars_seen >= MIN_BUFFER_BARS) and (self.bars_seen - self.last_trigger_bars >= TRIGGER_EVERY_BARS):
                if (self.bars_seen - self.last_promo_bars) < PROMO_COOLDOWN_BARS:
                    remain = PROMO_COOLDOWN_BARS - (self.bars_seen - self.last_promo_bars)
                    logger.info(f"쿨다운 중: {remain} bars 남음 — 미세학습 스킵")
                    continue

                self.last_trigger_bars = self.bars_seen
                try:
                    logger.info(f"미세조정 트리거 (bars={self.bars_seen:,})")
                    improved = self.ol.finetune_on_recent(self.X_tail, self.close_tail, funding=self.funding_tail)
                    if improved:
                        logger.info("✅ 모델 성능 향상 및 저장 완료")
                        self.last_promo_bars = self.bars_seen
                    else:
                        logger.info("❌ 성능 향상 없음")
                except Exception as e:
                    logger.error(f"미세조정 실패: {e}", exc_info=True)

# =========================
# 텔레그램 봇 러너
# =========================
class TelegramBotRunner(threading.Thread):
    daemon = True
    def __init__(self, bot_path: Path):
        super().__init__(name="TgBot")
        self.bot_path = str(bot_path)
        self.proc: Optional[subprocess.Popen] = None
    def run(self):
        logger.info(f"텔레그램 봇 감시 시작: {self.bot_path}")
        log_path = LOGS_DIR / "telegram_bot.log"
        while True:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"\n===== BOT RESTARTING AT {datetime.now().isoformat()} =====\n")
                    self.proc = subprocess.Popen(
                        [sys.executable, self.bot_path],
                        cwd=str(BASE_DIR),
                        stdout=subprocess.DEVNULL,
                        stderr=f
                    )
                    self.proc.wait()
            except Exception as e:
                logger.error(f"텔레그램 봇 프로세스 오류: {e}", exc_info=True)
            finally:
                logger.info("텔레그램 봇 재시작 대기 중...")
                time.sleep(3)

# =========================
# 메인
# =========================
def main():
    threading.current_thread().name = "MainThread"
    logger.info("스택 시작 중…")
    logger.info(f"트레이더 모드={TRADING_MODE} | 모델={LIVE_OUT_NAME}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    ingest_q = Queue(maxsize=INGEST_QUEUE_MAX)
    trader_q = Queue(maxsize=TRADER_QUEUE_MAX)
    learn_q  = Queue(maxsize=LEARN_QUEUE_MAX)

    ingest = RealtimeIngest(ingest_q)  # MTF 인제스터(5m 확정+MTF 피처)
    trader = None
    if ENABLE_TRADING:
        trader = Trader(
            mode=TRADING_MODE,
            q=trader_q,
            api_key=os.getenv('BINANCE_API_KEY'),
            secret_key=os.getenv('BINANCE_SECRET_KEY')
        )
        try:
            setattr(trader, "learn_q", learn_q)  # Trader → Learner rollout 전달
        except Exception:
            pass

    disp = Dispatcher(ingest_q, trader_q if ENABLE_TRADING else Queue(1),
                      learn_q if ENABLE_ONLINE_LEARNING else Queue(1))

    learn_worker = OnlineLearnWorker(learn_q) if ENABLE_ONLINE_LEARNING else None
    reloader = TraderReloader(trader, str(LIVE_OUT_PATH)) if (ENABLE_TRADING and ENABLE_ONLINE_LEARNING) else None

    threading.Thread(target=ingest.run, daemon=True, name="Ingest").start()
    disp.start()
    if ENABLE_TRADING:
        threading.Thread(target=trader.run, daemon=True, name="Trader").start()
    if ENABLE_ONLINE_LEARNING:
        learn_worker.start()
    if reloader:
        reloader.start()

    # 텔레그램 봇 (live 모드에서는 비활성 권장)
    tg_runner = None
    if ENABLE_TELEGRAM_BOT and TRADING_MODE != "live" and TELEGRAM_BOT_PATH.exists():
        if not os.getenv("TELEGRAM_BOT_TOKEN"):
            logger.warning("텔레그램 토큰(.env: TELEGRAM_BOT_TOKEN) 미설정 — 봇 미시작")
        else:
            tg_runner = TelegramBotRunner(TELEGRAM_BOT_PATH)
            tg_runner.start()
            logger.info(f"텔레그램 봇 시작: {TELEGRAM_BOT_PATH}")
    else:
        if TRADING_MODE == "live":
            logger.info("live 모드: 텔레그램/파일 로그 비활성화")
        else:
            logger.warning("텔레그램 봇 비활성 또는 파일 없음")

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        logger.info("사용자에 의해 중지됨")
    finally:
        try:
            if tg_runner and tg_runner.proc and tg_runner.proc.poll() is None:
                tg_runner.proc.terminate()
                try:
                    tg_runner.proc.wait(timeout=5)
                except Exception:
                    tg_runner.proc.kill()
        except Exception:
            pass
        logger.info("시스템 종료.")

if __name__ == "__main__":
    main()
