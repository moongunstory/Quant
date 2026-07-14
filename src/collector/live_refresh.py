"""live_refresh — 라이브용 '좁은' 데이터 최신화.

리서치/백테스트(full_collector)는 넓은 유니버스·넓은 필드로 수집하지만, 라이브는
이중으로 좁힌다(coin 의 '라이브 데이터 과부하' 실패를 구조적으로 차단):

  1) 종목: 현재 top-100 (최신 universe_snapshot).
  2) 필드: 채택된 알파들이 '실제로 쓰는 필드'의 합집합만. 그 필드가 속한 데이터셋만
     수집한다. (예: funding-only 알파만 있으면 klines 는 아예 최신화 안 함.)

compute_scope: 채택 알파 -> required_fields 합집합 -> 필요한 데이터셋 + top-100 심볼.
run: 그 스코프로 full_collector.run(datasets=...) 호출.

콜드스타트(데이터가 아예 없는 상태)도 이 모듈이 알아서 처리한다:
  - top-100 스냅샷이 없으면 universe_maintenance 체인을 1회 자동 실행해 만든다.
  - full_collector.run 은 심볼/데이터셋별로 이미 수집된 게 없으면(last_collected 없음)
    아카이브가 실제로 시작하는 시점(또는 상장일)부터 오늘까지를 자동으로 전부 채운다.
    즉 "필요한 만큼"이 아니라 "구할 수 있는 전체 히스토리"를 받아온다 — 알파들이
    쓰는 rolling window가 며칠 안에서만 필요해도 다년치를 다 받는다는 뜻인데,
    라이브에서 이렇게 하는 이유는 백테스트/재검증도 같은 원자료를 재사용하기 때문.

deadline(실행시간 제한 안전장치): 위 콜드스타트 백필은 심볼 수 x 데이터셋 수 x
연 단위 기간이라 시간이 오래 걸릴 수 있다. Lambda처럼 실행시간이 제한된 환경에서
중간에 강제종료(SIGKILL)당하면 정리가 안 되므로, deadline(마감 시각)을 받아
universe_maintenance/full_collector 아래까지 전달한다. 마감 전까지 처리한 만큼은
manifest에 정상 기록되므로, 다음 스케줄 실행이 그 지점부터 자연스럽게 이어서
계속한다(재수집 낭비 없음). CLI 등 시간제한 없는 환경에서는 deadline=None(기본값)
으로 끝까지 돈다.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

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


def run(cfg, alphas_dir="data/strategy/alphas", max_workers=10, deadline: Optional[float] = None):
    """스코프 계산 후 필요한 데이터셋만 최신화.

    deadline: time.monotonic() 기준 절대 마감 시각(초). None이면 무제한.
    """
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
        universe_maintenance.run(deadline=deadline)
        scope = compute_scope(cfg, alphas_dir=alphas_dir)
        log.info("유니버스 갱신 후 스코프 재계산: top100=%d종목", scope["n_symbols"])

    if not scope["symbols"]:
        # 갱신 뒤에도 비면 데이터/설정 문제 → 전체 폴백(데이터 없는 것보단 낫다)하되 크게 경고.
        log.warning("갱신 후에도 스냅샷이 비어있음 → 전체 유니버스로 폴백. 데이터/설정 확인 필요.")

    if deadline is not None and time.monotonic() >= deadline:
        log.warning("유니버스 갱신 체인에서 이미 시간 예산을 다 써서 원자료 수집(full_collector)은 이번 실행에서 스킵. 다음 실행에서 이어서 진행.")
        return scope

    full_collector.run(datasets=scope["datasets"], max_workers=max_workers,
                       symbols=(scope["symbols"] or None), deadline=deadline)
    return scope
