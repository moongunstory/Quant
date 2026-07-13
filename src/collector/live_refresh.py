"""live_refresh — 라이브용 '좁은' 데이터 최신화.

리서치/백테스트(full_collector)는 넓은 유니버스·넓은 필드로 수집하지만, 라이브는
이중으로 좁힌다(coin 의 '라이브 데이터 과부하' 실패를 구조적으로 차단):

  1) 종목: 현재 top-100 (최신 universe_snapshot).
  2) 필드: 채택된 알파들이 '실제로 쓰는 필드'의 합집합만. 그 필드가 속한 데이터셋만
     수집한다. (예: funding-only 알파만 있으면 klines 는 아예 최신화 안 함.)

compute_scope: 채택 알파 -> required_fields 합집합 -> 필요한 데이터셋 + top-100 심볼.
run: 그 스코프로 full_collector.run(datasets=...) 호출.

주의: full_collector.run 은 현재 심볼 단위 필터 인자가 없어(유니버스 전체 대상, gap-aware),
데이터셋 축소(필드 스코핑)는 지금 바로 적용되고, top-100 심볼 축소는 스코프로 계산·기록만
한다(full_collector 에 symbols= 인자를 추가하는 것이 다음 개선 — TODO 주석 참고).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from src.backtest.panel import FIELD_SPECS
from src.backtest.evaluate import required_fields
from src.backtest.spec import load_all
from src.config.backtest_settings import SETTINGS

log = logging.getLogger("quant.live_refresh")

# 라이브에서 항상 필요한 field(엔진이 close/funding 을 늘 씀).
_ALWAYS_FIELDS = ("close", "funding_rate")


def _fields_to_datasets(fields):
    """field 집합 -> 데이터셋 집합 (panel.FIELD_SPECS 의 첫 원소 = 데이터셋)."""
    datasets = set()
    for f in fields:
        spec = FIELD_SPECS.get(f)
        if spec:
            datasets.add(spec[0])
    return datasets


def _current_top100():
    """최신 universe_snapshot 의 멤버 = 현재 top-100. 없으면 빈 set."""
    snap_dir = SETTINGS.universe_snapshot_dir
    snaps = []
    for p in sorted(Path(snap_dir).glob("[0-9]*.json")):
        if p.stem.endswith("_diff"):
            continue
        snaps.append(p)
    if not snaps:
        return set()
    try:
        d = json.loads(snaps[-1].read_text(encoding="utf-8"))
    except Exception:
        return set()
    return set(d.get("members", []))


def compute_scope(cfg, alphas_dir="data/strategy/alphas"):
    """cfg -> {fields, datasets, symbols, alphas}. 라이브 최신화 스코프."""
    specs = load_all(alphas_dir)
    if cfg.alphas:
        by = {s.name: s for s in specs}
        specs = [by[n] for n in cfg.alphas if n in by]

    fields = set(_ALWAYS_FIELDS)
    for s in specs:
        fields |= set(required_fields(s.expression))
    fields = {f for f in fields if f in FIELD_SPECS or f in _ALWAYS_FIELDS}

    datasets = _fields_to_datasets(fields)
    symbols = _current_top100()
    return {"alphas": [s.name for s in specs], "fields": sorted(fields),
            "datasets": sorted(datasets), "symbols": sorted(symbols),
            "n_symbols": len(symbols)}


def run(cfg, alphas_dir="data/strategy/alphas", max_workers=10):
    """스코프 계산 후 필요한 데이터셋만 최신화."""
    scope = compute_scope(cfg, alphas_dir=alphas_dir)
    log.info("live_refresh 스코프: 알파=%s 필드=%s 데이터셋=%s top100=%d종목",
             scope["alphas"], scope["fields"], scope["datasets"], scope["n_symbols"])
    if not scope["datasets"]:
        log.info("최신화할 데이터셋 없음")
        return scope

    from src.collector import full_collector

    # 라이브는 데이터셋(필드 스코핑) + 종목(현재 top-100) 둘 다 좁힌다.
    # 스냅샷이 아예 없으면(cold-start) 경고만 하지 말고 유니버스 갱신 체인을 1회 돌려
    # 스냅샷을 '만든' 뒤 스코프를 다시 계산한다(사용자 요구: 없으면 생성).
    if not scope["symbols"]:
        log.warning("top-100 스냅샷 없음 → 유니버스 갱신 체인 자동 실행(cold-start 1회).")
        from src.collector import universe_maintenance
        universe_maintenance.run()
        scope = compute_scope(cfg, alphas_dir=alphas_dir)
        log.info("유니버스 갱신 후 스코프 재계산: top100=%d종목", scope["n_symbols"])

    if not scope["symbols"]:
        # 갱신 뒤에도 비면 데이터/설정 문제 → 전체 폴백(데이터 없는 것보단 낫다)하되 크게 경고.
        log.warning("갱신 후에도 스냅샷이 비어있음 → 전체 유니버스로 폴백. 데이터/설정 확인 필요.")

    full_collector.run(datasets=scope["datasets"], max_workers=max_workers,
                       symbols=(scope["symbols"] or None))
    return scope
