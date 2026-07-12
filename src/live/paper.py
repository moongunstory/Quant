"""paper — 가상매매 실행/성과 추적.

핵심 원칙: 페이퍼는 실거래와 '동일 코드경로'를 탄다. 주문 생성(orders.generate_orders)
·기록(ledger)·목표계산(target_weights)은 모드와 무관하게 같고, 오직 '체결'만
시뮬레이션이다. orders.generate_orders(mode="paper") 가 이미 즉시·완전 체결을 가정해
positions.json 을 목표로 갱신하므로, 여기서는 그 위에 '페이퍼 자산곡선'만 얹는다.

mark_to_market: 저장된 페이퍼 포지션({coin: weight})을 그날 수익률에 곱해 하루 손익을
추정하고 equity 곡선(data/live/paper_equity.jsonl)에 append. 실거래 ledger 와 별개로,
'가상 계좌가 실제로 어떻게 움직였나'를 본다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config.backtest_settings import SETTINGS

EQUITY_PATH = SETTINGS.data_dir / "live" / "paper_equity.jsonl"


def mark_to_market(weights, day_returns, today=None):
    """weights: {coin: weight}(어제 목표=오늘 보유). day_returns: {coin: 당일수익률}.
    당일 페이퍼 손익 = Σ weight*return. equity 곡선에 append 후 (day_pnl, equity) 반환."""
    today = today or datetime.now(timezone.utc).date()
    day_pnl = sum(float(w) * float(day_returns.get(c, 0.0)) for c, w in weights.items())

    prev_equity = 0.0
    p = Path(EQUITY_PATH)
    if p.exists():
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            prev_equity = float(json.loads(lines[-1]).get("equity", 0.0))
    equity = prev_equity + day_pnl

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": today.isoformat(), "day_pnl": day_pnl,
                            "equity": equity}, ensure_ascii=False) + "\n")
    return day_pnl, equity
