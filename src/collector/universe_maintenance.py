"""universe_maintenance — 월간 유니버스 갱신 체인(라이브 daily 사이클과 분리).

top-100 유니버스는 '월별' 개념(월말 리밸런싱 스냅샷)이라, 갱신도 월 1회가 맞다.
이 체인은 스냅샷을 최신으로 만드는 3단계를 한 번에 묶는다:

  1) symbol_universe : 거래 가능한 전체 심볼 목록 갱신(data/strategy/meta/symbol_list.json)
  2) universe_probe  : 경량 일봉으로 rolling_score 계산(data/market/scan/*.parquet) — 자기 데이터
                       를 직접 받으므로 선행 수집 불필요(gap-aware 증분)
  3) universe_builder: 그 점수로 월별 top-100 스냅샷 재구성(data/market/universe/)

라이브 daily 사이클(live_refresh)은 이 결과(최신 스냅샷)를 '읽기만' 한다 — 무거운
재구성을 매일 돌리지 않는다. 월 1회 스케줄로 돌리거나, 스냅샷이 아예 없을 때
live_refresh 가 cold-start 로 1회 자동 호출한다.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.collector import symbol_universe, universe_probe, universe_builder

log = logging.getLogger("quant.universe_maintenance")


def run(deadline: Optional[float] = None) -> None:
    """유니버스 갱신 3단계를 순서대로 실행(순서 의존: 각 단계가 앞 단계 산출물을 씀).

    deadline: time.monotonic() 기준 절대 마감 시각(초). None이면 무제한.
    1단계(symbol_universe)와 3단계(universe_builder)는 네트워크 요청이 가볍거나
    없어서 빠르다 — 시간이 오래 걸릴 수 있는 건 2단계(universe_probe, 전체 유니버스
    x 수년치 히스토리 스캔)뿐이라 여기만 deadline을 전달한다.
    """
    log.info("유니버스 갱신 1/3: 심볼 목록")
    symbol_universe.run()
    log.info("유니버스 갱신 2/3: 경량 스캔(rolling_score)")
    universe_probe.run(deadline=deadline)
    log.info("유니버스 갱신 3/3: 월별 top-100 스냅샷 재구성")
    universe_builder.run()
    log.info("유니버스 갱신 완료")
