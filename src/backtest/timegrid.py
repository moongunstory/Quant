"""timegrid — 마스터 그리드 ↔ 알파별 bar 주기 정렬 (Phase 3).

배경: klines 는 1h 네이티브. 각 알파는 자기 데이터 성격에 맞는 bar(1h/4h/8h/1d)로
신호를 갱신한다. 서로 다른 주기의 알파를 한 포트폴리오로 결합하려면, 모두를 하나의
'마스터 그리드'(포트폴리오 알파들 중 가장 촘촘한 bar) 위에 올려야 한다.

핵심 규칙:
  - 알파 신호는 자기 bar 그리드에서 계산된다(ts_* 연산자의 창이 bar 단위가 되도록).
  - 계산된 포지션은 마스터 그리드로 reindex + ffill: '다음 리밸런싱 전까지 보유'.
  - 손익/결합/리스크는 전부 마스터 그리드에서 이뤄져 주기가 달라도 합쳐진다.
"""
from __future__ import annotations

import pandas as pd

# bar 문자열 -> (pandas offset, 시간(시간단위)) — 촘촘함 비교/리샘플에 사용.
_BAR_HOURS = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "8h": 8, "12h": 12, "1d": 24}


def bar_hours(bar: str) -> int:
    if bar not in _BAR_HOURS:
        raise ValueError(f"알 수 없는 bar {bar!r} (사용가능 {list(_BAR_HOURS)})")
    return _BAR_HOURS[bar]


def bar_to_offset(bar: str) -> str:
    """bar -> pandas resample offset ('1h'->'1h', '1d'->'1D')."""
    bar_hours(bar)  # 검증
    return "1D" if bar == "1d" else f"{_BAR_HOURS[bar]}h"


def finest_bar(bars) -> str:
    """여러 bar 중 가장 촘촘한(시간 작은) 것 = 마스터 그리드 bar."""
    bars = list(bars) or ["1d"]
    return min(bars, key=bar_hours)


def to_master(pos_bar: pd.DataFrame, master_index) -> pd.DataFrame:
    """알파 bar 그리드의 포지션을 마스터 그리드로 옮긴다.
    두 인덱스를 합쳐 ffill(다음 리밸런싱 전까지 보유) 후 마스터만 취한다."""
    union = pos_bar.index.union(master_index)
    return pos_bar.reindex(union).ffill().reindex(master_index)
