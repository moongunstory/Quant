"""paper — 가상매매 실행/성과 추적.

핵심 원칙: 페이퍼는 실거래와 '동일 코드경로'를 탄다. 주문 생성(orders.generate_orders)
·기록(ledger)·목표계산(target_weights)은 모드와 무관하게 같고, 오직 '체결'만
시뮬레이션이다. orders.generate_orders(mode="paper") 가 이미 즉시·완전 체결을 가정해
positions.json 을 목표로 갱신하므로, 여기서는 그 위에 '페이퍼 자산곡선'만 얹는다.

mark_to_market: 저장된 페이퍼 포지션({coin: weight})을 그날 수익률에 곱해 하루 손익을
추정하고 equity 곡선(data/runtime/live/paper_equity.jsonl)에 append. 실거래 ledger 와 별개로,
'가상 계좌가 실제로 어떻게 움직였나'를 본다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config.backtest_settings import SETTINGS

EQUITY_PATH = SETTINGS.data_dir / "runtime" / "live" / "paper_equity.jsonl"


def mark_to_market(weights, day_returns, today=None, turnover=0.0):
    """weights: {coin: weight}(어제 목표=오늘 보유). day_returns: {coin: 당일수익률}.
    turnover: 이번 사이클 리밸런싱 회전율(=목표와 현재의 총 L1 드리프트, orders 의 'drift').

    당일 페이퍼 손익 = Σ weight*return − 매매비용. 매매비용 = turnover × (수수료+슬리피지).
    (예전엔 비용을 0 으로 뒀는데, '비용은 백테스트에 이미 반영됨' 가정이 라이브 회전율=
     백테스트 회전율일 때만 성립한다. 실제 라이브 회전율로 직접 차감해 곡선을 정직하게.)
    백테스트 engine 과 동일 공식: cost = turnover × (taker_fee_pct + slippage_pct)/100.
    equity 곡선에 append 후 (day_pnl, equity) 반환."""
    today = today or datetime.now(timezone.utc).date()
    gross_pnl = sum(float(w) * float(day_returns.get(c, 0.0)) for c, w in weights.items())
    cost_rate = (SETTINGS.taker_fee_pct + SETTINGS.slippage_pct) / 100.0
    trade_cost = float(turnover) * cost_rate
    day_pnl = gross_pnl - trade_cost

    prev_equity = 0.0
    p = Path(EQUITY_PATH)
    if p.exists():
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            last = json.loads(lines[-1])
            # 같은 날 두 번 실행(수동 + 스케줄)해도 하루 손익이 이중으로 쌓이지 않게,
            # 오늘 이미 기록이 있으면 기존 값을 그대로 반환(중복 append 방지).
            if last.get("date") == today.isoformat():
                return float(last.get("day_pnl", 0.0)), float(last.get("equity", 0.0))
            prev_equity = float(last.get("equity", 0.0))
    equity = prev_equity + day_pnl

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": today.isoformat(), "day_pnl": day_pnl,
                            "gross_pnl": gross_pnl, "trade_cost": trade_cost,
                            "turnover": float(turnover), "equity": equity},
                           ensure_ascii=False) + "\n")
    return day_pnl, equity
