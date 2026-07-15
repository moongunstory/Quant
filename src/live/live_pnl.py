"""live_pnl — 가상매매(paper)의 '실시간' 손익 계산.

왜 필요한가
-----------
기존 paper PnL(paper.py)은 '하루 단위'다. 하루 한 번 사이클에서 그날 수익률에 보유
비중을 곱해 equity 곡선(paper_equity.jsonl)에 한 칸씩 찍는다. 그래서 /잔고는 마지막에
찍힌 값(=전일/오늘 종가 기준)만 보여줄 수 있었다.

이 모듈은 그 사이(리밸런싱 이후 지금 이 순간까지)의 변동을 '현재가'로 채워준다.

원리
----
1) 리밸런싱할 때 그 순간의 마크가격을 코인별로 저장한다(진입가 스냅샷).
2) /잔고 요청이 오면 지금의 마크가격을 다시 불러와,
      리밸런싱 이후 실시간 변동 = Σ (비중 × (현재가 / 진입가 - 1))
   를 계산한다.
3) 실시간 가상 자산 = 기준 자산(마지막 종가 equity) + 리밸런싱 이후 실시간 변동.

포지션 키(coin)는 이미 바이낸스 심볼(예: BTCUSDT)이라 get_mark_prices 결과와 그대로
매칭된다. 실매매(real)는 바이낸스가 미실현 손익을 직접 계산하므로 이 모듈을 쓰지 않는다
(telegram_bot.cmd_balance 의 거래소 조회 경로).

전부 fail-open: 진입가 스냅샷이 없거나 현재가 조회가 실패하면 ok=False 를 돌려주고,
호출측(cmd_balance)은 저장된 일일 값으로 폴백한다.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from src.config.backtest_settings import SETTINGS

log = logging.getLogger("quant.live.live_pnl")

# 진입가 스냅샷 파일. runtime/live 아래라 remote_store 의 "runtime" prefix 로 R2 왕복된다.
ENTRY_PATH = SETTINGS.data_dir / "runtime" / "live" / "positions_entry.json"
EQUITY_PATH = SETTINGS.data_dir / "runtime" / "live" / "paper_equity.jsonl"


def _last_equity() -> float:
    """paper_equity.jsonl 마지막 줄의 누적 equity. 없으면 0.0."""
    p = Path(EQUITY_PATH)
    if not p.exists():
        return 0.0
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return 0.0
    try:
        return float(json.loads(lines[-1]).get("equity", 0.0))
    except Exception:
        return 0.0


def snapshot_entry_prices(weights: dict, mode: str = "paper", today=None) -> bool:
    """리밸런싱 직후 호출: 보유 코인들의 '지금 마크가격'을 진입가로 저장. 성공 여부 반환.

    weights: {coin(symbol): weight}. 마크가격 조회 실패 시 조용히 넘어간다(실시간 손익만
    비활성; 사이클은 계속). 저장 형식:
      {"date": "...", "base_equity": <스냅샷 시점 누적 equity>, "prices": {sym: price}}
    """
    if not weights:
        return False
    try:
        from src.live.exchange import exchange
        prices = exchange.get_mark_prices(mode="real")
    except Exception as e:
        log.warning("진입가 스냅샷용 마크가격 조회 실패(실시간 손익 이번엔 비활성): %s", e)
        return False

    entry_prices = {c: float(prices[c]) for c in weights if c in prices and prices[c]}
    if not entry_prices:
        log.warning("진입가로 저장할 코인의 마크가격이 하나도 없어 스냅샷을 건너뜁니다.")
        return False

    day = (today or datetime.now(timezone.utc).date())
    if isinstance(day, datetime):
        day = day.date()
    data = {
        "date": day.isoformat(),
        "base_equity": _last_equity(),
        "prices": entry_prices,
    }
    ENTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENTRY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("진입가 스냅샷 저장: %d종목 (기준 equity=%.6f)", len(entry_prices), data["base_equity"])
    return True


def _load_entry_snapshot() -> dict | None:
    p = Path(ENTRY_PATH)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def compute_live_pnl(mode: str = "paper") -> dict:
    """가상매매 실시간 손익 스냅샷을 계산해 반환.

    성공: {"ok": True, "base_equity", "intraday_return", "live_equity",
           "n_priced", "n_total", "missing", "as_of"}
    실패: {"ok": False, "reason": "..."}  (호출측은 저장된 일일 값으로 폴백)
    """
    from src.live import orders as OR

    weights = OR.load_positions()
    if not weights:
        return {"ok": False, "reason": "보유 포지션 없음"}

    entry = _load_entry_snapshot()
    if not entry or not entry.get("prices"):
        return {"ok": False, "reason": "진입가 스냅샷 없음(리밸런싱 1회 후 생성됨)"}

    try:
        from src.live.exchange import exchange
        cur = exchange.get_mark_prices(mode="real")
    except Exception as e:
        return {"ok": False, "reason": f"현재가 조회 실패: {e}"}

    entry_prices = entry.get("prices", {})
    base_equity = float(entry.get("base_equity", 0.0))

    intraday = 0.0
    n_priced = 0
    missing = []
    for coin, w in weights.items():
        ep = entry_prices.get(coin)
        cp = cur.get(coin)
        if ep and cp and float(ep) > 0:
            intraday += float(w) * (float(cp) / float(ep) - 1.0)
            n_priced += 1
        else:
            missing.append(coin)

    return {
        "ok": True,
        "base_equity": base_equity,
        "intraday_return": intraday,
        "live_equity": base_equity + intraday,
        "n_priced": n_priced,
        "n_total": len(weights),
        "missing": missing,
        "as_of": entry.get("date"),
    }
