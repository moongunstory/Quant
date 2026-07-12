"""target_weights — 승인 포트폴리오 config -> 오늘의 목표 가중치 행.

라이브 계층의 첫 조각. build_portfolio 를 그대로 재사용해 전체 히스토리로
combine+risk 를 돌리고, 결과 가중치 패널의 '마지막 행'을 오늘의 목표로 삼는다
(coin D69: 별도 스트리밍 리스크 엔진 없이 백테스트 함수 재사용).

freshness 게이트를 먼저 통과: 데이터가 안 온(stale) 알파는 결합 전에 제외.
전부 stale 이면 diagnostics.all_alphas_stale=True 로 표시하고 목표를 내지 않는다
(orders 가 이걸 보고 '청산 주문'이 아니라 SKIP 하도록 — fail-safe).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from src.backtest.spec import load_all
from src.portfolio import pipeline as PP
from src.live import freshness as FR

_DUST = 1e-6


def _select_specs(cfg, alphas_dir="data/alphas"):
    specs = load_all(alphas_dir)
    if cfg.alphas:
        by = {s.name: s for s in specs}
        specs = [by[n] for n in cfg.alphas if n in by]
    return specs


def compute_target_weights(cfg, today=None, max_staleness_days=FR.DEFAULT_MAX_STALENESS_DAYS,
                           rebuild=False, alphas_dir="data/alphas"):
    """cfg -> {date, weights:{coin:weight}, held_alphas, alpha_weights, diagnostics}."""
    today = today or datetime.now(timezone.utc).date()
    specs = _select_specs(cfg, alphas_dir=alphas_dir)

    g = FR.gate(specs, today=today, max_staleness_days=max_staleness_days)
    diagnostics = {"all_alphas_stale": g["all_stale"],
                   "stale_alphas": g["stale_names"],
                   "freshness": g["details"]}

    if g["all_stale"]:
        return {"date": today.isoformat(), "weights": {}, "held_alphas": [],
                "alpha_weights": {}, "diagnostics": diagnostics}

    fresh_names = [s.name for s in g["fresh"]]
    cfg_fresh = replace(cfg, alphas=fresh_names)
    out = PP.build_portfolio(cfg_fresh, rebuild=rebuild, alphas_dir=alphas_dir)

    final = out["weights"]
    if final.empty:
        return {"date": today.isoformat(), "weights": {}, "held_alphas": fresh_names,
                "alpha_weights": out["alpha_weights"], "diagnostics": diagnostics}

    last_row = final.iloc[-1]
    weights = {c: float(w) for c, w in last_row.items()
               if w == w and abs(float(w)) > _DUST}  # NaN/dust 제외
    diagnostics["target_row_date"] = str(final.index[-1].date())
    return {"date": today.isoformat(), "weights": weights, "held_alphas": fresh_names,
            "alpha_weights": out["alpha_weights"], "diagnostics": diagnostics}
