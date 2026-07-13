"""ledger — 라이브 사이클의 주문/체결/포지션을 JSONL 로 기록.

목적: 백테스트가 가정한 성과 대비 '실제(또는 페이퍼) 실행'이 어떻게 달랐는지
사후 추적할 근거를 남긴다. 한 줄 = 한 이벤트(JSON). logs/live-<date>.jsonl 에 append.
페이퍼와 실거래가 '동일 코드경로'를 타므로 같은 ledger 를 공유한다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config.backtest_settings import SETTINGS


def _ledger_path(today=None):
    today = today or datetime.now(timezone.utc).date()
    d = SETTINGS.logs_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"live-{today.isoformat()}.jsonl"


def record(event_type, payload, today=None):
    """한 이벤트 append. event_type: 'cycle'|'orders'|'target'|'fill'|'error' 등."""
    row = {"ts": datetime.now(timezone.utc).isoformat(),
           "type": event_type, **payload}
    path = _ledger_path(today)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return path
