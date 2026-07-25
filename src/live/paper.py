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


def mark_to_market(weights, day_returns, today=None, turnover=0.0, returns_rows=None):
    """weights: {coin: weight}(어제 목표=오늘 보유). day_returns: {coin: 당일수익률}.
    turnover: 이번 사이클 리밸런싱 회전율(=목표와 현재의 총 L1 드리프트, orders 의 'drift').

    returns_rows (2026-07-24): [{"date": "YYYY-MM-DD", "returns": {coin: ret}}, ...]
    target_weights 가 만든 '완결된 봉'의 일자별 수익률(오늘 부분봉 제외). 주어지면:
      - 마지막 기록 이후의 완결일들을 순서대로 전부 반영한다(스케줄이 하루 빠져도
        다음 사이클이 따라잡음 — 빠진 날 동안 포지션은 안 바뀌었으므로 같은 weights 로
        평가하는 게 정확하다).
      - 기록의 date = '수익이 난 날'(완결일). 같은 날 재실행하면 새 완결일이 없어
        자동으로 중복 방지된다.
    없으면(하위호환) 종전처럼 day_returns 한 행을 사이클 날짜로 기록한다.

    페이퍼 손익 = Σ weight*return − 매매비용. 매매비용 = turnover × (수수료+슬리피지)
    — 백테스트 engine 과 동일 공식. 비용은 이번 사이클 리밸런싱 1회분이므로 마지막
    행에만 차감한다. equity 곡선에 append 후 (day_pnl, equity) 반환."""
    today = today or datetime.now(timezone.utc).date()
    cost_rate = (SETTINGS.taker_fee_pct + SETTINGS.slippage_pct) / 100.0
    trade_cost_cycle = float(turnover) * cost_rate

    prev_equity = 0.0
    last_rec = None
    last_date = None
    p = Path(EQUITY_PATH)
    if p.exists():
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            last_rec = json.loads(lines[-1])
            prev_equity = float(last_rec.get("equity", 0.0))
            last_date = last_rec.get("date")

    if returns_rows:
        rows = [r for r in returns_rows
                if r.get("date") and (last_date is None or r["date"] > last_date)]
        if not rows:
            # 새 완결일 없음(같은 날 재실행 등) → 기존 마지막 기록 그대로(중복 방지).
            if last_rec is not None:
                return float(last_rec.get("day_pnl", 0.0)), float(last_rec.get("equity", 0.0))
            return 0.0, 0.0
        equity = prev_equity
        day_pnl = 0.0
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                rets = r["returns"]
                gross = sum(float(w) * float(rets.get(c, 0.0)) for c, w in weights.items())
                is_last = (i == len(rows) - 1)
                cost = trade_cost_cycle if is_last else 0.0
                day_pnl = gross - cost
                equity += day_pnl
                # btc_return: 대시보드의 'BTC 그냥 보유' 비교선용(같은 날짜 기준으로 정렬돼 정확)
                f.write(json.dumps({"date": r["date"], "day_pnl": day_pnl,
                                    "gross_pnl": gross, "trade_cost": cost,
                                    "turnover": float(turnover) if is_last else 0.0,
                                    "btc_return": float(rets.get("BTCUSDT", 0.0)),
                                    "equity": equity}, ensure_ascii=False) + "\n")
        return day_pnl, equity

    # ---- 하위호환 경로: day_returns 한 행, 사이클 날짜 기준 중복 방지 ----
    gross_pnl = sum(float(w) * float(day_returns.get(c, 0.0)) for c, w in weights.items())
    day_pnl = gross_pnl - trade_cost_cycle
    if last_rec is not None and last_date == today.isoformat():
        return float(last_rec.get("day_pnl", 0.0)), float(last_rec.get("equity", 0.0))
    equity = prev_equity + day_pnl

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": today.isoformat(), "day_pnl": day_pnl,
                            "gross_pnl": gross_pnl, "trade_cost": trade_cost_cycle,
                            "turnover": float(turnover),
                            "btc_return": float(day_returns.get("BTCUSDT", 0.0)),
                            "equity": equity},
                           ensure_ascii=False) + "\n")
    return day_pnl, equity
