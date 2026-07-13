"""freshness — 라이브 데이터 신선도 게이트 (coin live/freshness.py 이식).

왜 필요한가: 한 알파의 의존 데이터가 오늘까지 안 왔는데도 블렌드에 그 알파의
낡은/NaN 행이 섞여 들어가면, 스케줄된 실행이 조용히 북을 언더웨이트한다. 이 게이트는
그 수동적 위험을 '능동적 사전(pre-blend) 배제'로 바꾼다: 데이터가 안 온 알파는 결합
전에 아예 빼고 이름을 명시한다.

메커니즘: 각 알파의 수식에서 필요한 panel field(evaluate.required_fields)를 뽑고,
각 field 의 패널(data/market/panel/<field>.parquet)의 '마지막 유효(non-NaN) 날짜'를 확인한다.
그 날짜가 today 대비 max_staleness_days 보다 더 뒤처지면 그 알파를 STALE 로 표시.
"""
from __future__ import annotations

from datetime import date, timezone, datetime

import pandas as pd

from src.backtest import panel as P
from src.backtest.evaluate import required_fields

DEFAULT_MAX_STALENESS_DAYS = 2


def _panel_last_valid_date(field):
    """field 패널의 마지막 non-NaN 행 날짜(UTC date). 없으면 None."""
    try:
        panel = P.load_panel(field)
    except Exception:
        return None
    valid = panel.dropna(how="all")
    if valid.empty:
        return None
    ts = valid.index.max()
    return pd.Timestamp(ts).tz_convert("UTC").date() if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).date()


def check_alpha_freshness(spec, today=None, max_staleness_days=DEFAULT_MAX_STALENESS_DAYS):
    """한 알파의 신선도. -> {name, stale, worst_field, worst_date, staleness_days}."""
    today = today or datetime.now(timezone.utc).date()
    fields = [f for f in required_fields(spec.expression) if f in P.FIELD_SPECS or f == "close"]
    worst_field, worst_date, worst_days = None, None, -1
    for f in fields:
        d = _panel_last_valid_date(f)
        if d is None:
            worst_field, worst_date, worst_days = f, None, 10**9
            break
        days = (today - d).days
        if days > worst_days:
            worst_field, worst_date, worst_days = f, d, days
    stale = worst_days > max_staleness_days
    return {"name": spec.name, "stale": bool(stale), "worst_field": worst_field,
            "worst_date": worst_date.isoformat() if worst_date else None,
            "staleness_days": int(worst_days) if worst_days >= 0 else None}


def gate(specs, today=None, max_staleness_days=DEFAULT_MAX_STALENESS_DAYS):
    """알파 리스트 -> {fresh: [spec...], stale_names: [...], details: [...]}.
    fresh 만 결합에 넘긴다. 전부 stale 이면 fresh=[] (호출측이 all_stale 처리)."""
    today = today or datetime.now(timezone.utc).date()
    fresh, stale_names, details = [], [], []
    for s in specs:
        info = check_alpha_freshness(s, today=today, max_staleness_days=max_staleness_days)
        details.append(info)
        if info["stale"]:
            stale_names.append(s.name)
        else:
            fresh.append(s)
    return {"fresh": fresh, "stale_names": stale_names, "details": details,
            "all_stale": len(fresh) == 0, "today": today.isoformat()}
