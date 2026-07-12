"""universe_maintenance — 월간 유니버스 갱신 체인(라이브 daily 사이클과 분리).

top-100 유니버스는 '월별' 개념(월말 리밸런싱 스냅샷)이라, 갱신도 월 1회가 맞다.
이 체인은 스냅샷을 최신으로 만드는 3단계를 한 번에 묶는다:

  1) symbol_universe : 거래 가능한 전체 심볼 목록 갱신(data/meta/symbol_list.json)
  2) universe_probe  : 경량 일봉으로 rolling_score 계산(data/scan/*.parquet) — 자기 데이터
                       를 직접 받으므로 선행 수집 불필요(gap-aware 증분)
  3) universe_builder: 그 점수로 월별 top-100 스냅샷 재구성(data/universe_snapshots/)

라이브 daily 사이클(live_refresh)은 이 결과(최신 스냅샷)를 '읽기만' 한다 — 무거운
재구성을 매일 돌리지 않는다. 월 1회 스케줄로 돌리거나, 스냅샷이 아예 없을 때
live_refresh 가 cold-start 로 1회 자동 호출한다.
"""
from __future__ import annotations

import logging

from src.collector import symbol_universe, universe_probe, universe_builder

log = logging.getLogger("quant.universe_maintenance")


def run() -> None:
    """유니버스 갱신 3단계를 순서대로 실행(순서 의존: 각 단계가 앞 단계 산출물을 씀)."""
    log.info("유니버스 갱신 1/3: 심볼 목록")
    symbol_universe.run()
    log.info("유니버스 갱신 2/3: 경량 스캔(rolling_score)")
    universe_probe.run()
    log.info("유니버스 갱신 3/3: 월별 top-100 스냅샷 재구성")
    universe_builder.run()
    log.info("유니버스 갱신 완료")
