"""target_weights — 승인 포트폴리오 config -> 오늘의 목표 가중치 행.

라이브 계층의 첫 조각. build_portfolio 를 그대로 재사용해 전체 히스토리로
combine+risk 를 돌리고, 결과 가중치 패널의 '마지막 행'을 오늘의 목표로 삼는다
(coin D69: 별도 스트리밍 리스크 엔진 없이 백테스트 함수 재사용).

freshness 게이트를 먼저 통과: 데이터가 안 온(stale) 알파는 결합 전에 제외.
전부 stale 이면 diagnostics.all_alphas_stale=True 로 표시하고 목표를 내지 않는다
(orders 가 이걸 보고 '청산 주문'이 아니라 SKIP 하도록 — fail-safe).
"""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

from src.backtest.spec import load_all
from src.portfolio import pipeline as PP
from src.live import freshness as FR

log = logging.getLogger("quant.live.target_weights")

_DUST = 1e-6


def _row_to_dict(panel):
    """가중치 패널의 '마지막 행'을 {coin: weight} 로 (NaN/dust 제외). 없으면 {}."""
    if panel is None or getattr(panel, "empty", True):
        return {}
    last = panel.iloc[-1]
    return {c: float(w) for c, w in last.items()
            if w == w and abs(float(w)) > _DUST}


def _select_specs(cfg, alphas_dir="data/strategy/alphas"):
    specs = load_all(alphas_dir)
    if cfg.alphas:
        by = {s.name: s for s in specs}
        specs = [by[n] for n in cfg.alphas if n in by]
    return specs


def compute_target_weights(cfg, today=None, max_staleness_days=FR.DEFAULT_MAX_STALENESS_DAYS,
                           rebuild=False, alphas_dir="data/strategy/alphas"):
    """cfg -> {date, weights:{coin:weight}, held_alphas, alpha_weights, diagnostics}."""
    today = today or datetime.now(timezone.utc).date()
    specs = _select_specs(cfg, alphas_dir=alphas_dir)

    if rebuild:
        from src.backtest import panel as P
        from src.backtest.evaluate import spec_required_fields
        from src.config.backtest_settings import SETTINGS

        # 파이프라인이 항상 로드하는 필드까지 포함해야 증분 후 캐시가 전부 최신이 된다.
        all_fields = {"close", "funding_rate", "quote_volume"}
        if SETTINGS.execution == "next_open":
            all_fields.add("open")   # 시가 체결 손익 계산용
        for s in specs:
            all_fields |= spec_required_fields(s.expression, s.neutralization)
        all_fields &= set(P.FIELD_SPECS) | set(P.DERIVED)

        log.info("패널 증분 업데이트 시작: %s", sorted(all_fields))
        for f in sorted(all_fields):
            try:
                P.update_panel(f)
            except Exception as e:
                log.warning("패널 증분 업데이트 실패 %r (전체 재빌드로 폴백): %s", f, e)
                P.load_panel(f, rebuild=True)
        P.update_funding_events()

        # 방금 캐시를 증분으로 최신화했으므로, 아래 build_portfolio 에 rebuild=True 를
        # 그대로 넘기면 전체 재빌드(~5분)를 또 해서 증분 업데이트가 무의미해진다.
        # 단, 증분 업데이트는 1d 패널만 다루므로 1d 외 bar 알파가 있으면 안전하게 유지.
        if all(getattr(s, "bar", "1d") == "1d" for s in specs):
            rebuild = False
        else:
            log.info("1d 외 bar 알파 존재 → 증분 캐시를 신뢰하지 않고 전체 재빌드 유지")

    g = FR.gate(specs, today=today, max_staleness_days=max_staleness_days)
    diagnostics = {"all_alphas_stale": g["all_stale"],
                   "stale_alphas": g["stale_names"],
                   "freshness": g["details"]}

    if g["all_stale"]:
        return {"date": today.isoformat(), "weights": {}, "held_alphas": [],
                "alpha_weights": {}, "diagnostics": diagnostics,
                "day_returns": {}, "day_returns_rows": []}

    fresh_names = [s.name for s in g["fresh"]]
    cfg_fresh = replace(cfg, alphas=fresh_names)
    out = PP.build_portfolio(cfg_fresh, rebuild=rebuild, alphas_dir=alphas_dir)

    final = out["weights"]
    if final.empty:
        return {"date": today.isoformat(), "weights": {}, "held_alphas": fresh_names,
                "alpha_weights": out["alpha_weights"], "diagnostics": diagnostics,
                "day_returns": {}, "day_returns_rows": [],
                "pre_risk_weights": {}, "alpha_contributions": {},
                "risk_stages": out.get("stages", [])}

    last_row = final.iloc[-1]
    weights = {c: float(w) for c, w in last_row.items()
               if w == w and abs(float(w)) > _DUST}  # NaN/dust 제외
    diagnostics["target_row_date"] = str(final.index[-1].date())

    # 당일 수익률 추출 (자산 가치 평가용)
    #
    # 주의(2026-07-24 수정): 패널의 마지막 행은 '오늘의 미완성(부분) 봉'인 경우가
    # 대부분이다 — 수집이 매일 자정 직후(예: UTC 00:10)에 돌기 때문에 오늘 행에는
    # 겨우 10~20분치 가격만 담겨 있다. 그 부분봉 수익률을 '하루 손익'으로 쓰면
    # 페이퍼 자산곡선이 매일 하루 수익의 대부분(자정 직후~다음날 자정)을 영구
    # 누락한다(실측: paper 일손익이 정상 일변동의 1/10 이하로 찍힘).
    # → 손익 평가는 '오늘(사이클 날짜) 이전의 완결된 행'만 쓴다.
    #   (목표 가중치는 그대로 마지막 행을 쓴다 — 포지션은 shift(delay)라 이미
    #    어제까지의 완결 데이터로 계산돼 있고, 부분봉 행이 있어야 신호가 하루
    #    늦지 않는다. 여기서 고치는 건 '손익 평가 행'뿐이다.)
    returns_df = out["result"].returns
    day_returns = {}
    day_returns_rows = []   # [{"date": "YYYY-MM-DD", "returns": {coin: ret}}, ...] 완결일만
    if not returns_df.empty:
        cutoff = pd.Timestamp(today)
        if returns_df.index.tz is not None:
            cutoff = cutoff.tz_localize(returns_df.index.tz)
        complete = returns_df.loc[returns_df.index < cutoff]
        # 최근 7일까지만: 스케줄이 며칠 빠졌어도 페이퍼 곡선이 따라잡을 수 있게(paper.py).
        for ts, row in complete.tail(7).iterrows():
            vals = {c: float(r) for c, r in row.items() if r == r}
            day_returns_rows.append({"date": pd.Timestamp(ts).date().isoformat(),
                                     "returns": vals})
        if day_returns_rows:
            day_returns = day_returns_rows[-1]["returns"]

    # ---- 텔레메트리(기여도 분석)용 상세 필드 ----
    # pre_risk_weights: 리스크 오버레이 '적용 전' 결합북의 오늘 행. 최종 weights 와
    # 비교하면 리스크 로직이 포지션을 얼마나 줄였는지(drag/benefit)를 역산할 수 있다.
    pre_risk_weights = _row_to_dict(out.get("combined_weights"))
    # alpha_contributions: 알파별 '결합북 내 기여분'의 오늘 행 (Σ = pre_risk_weights).
    # 다음 사이클의 day_returns 와 내적하면 알파별 당일 손익 기여가 나온다.
    alpha_contributions = {name: _row_to_dict(panel)
                           for name, panel in (out.get("contributions") or {}).items()}

    return {"date": today.isoformat(), "weights": weights, "held_alphas": fresh_names,
            "alpha_weights": out["alpha_weights"], "diagnostics": diagnostics,
            "day_returns": day_returns, "day_returns_rows": day_returns_rows,
            "pre_risk_weights": pre_risk_weights,
            "alpha_contributions": alpha_contributions,
            "risk_stages": out.get("stages", [])}
