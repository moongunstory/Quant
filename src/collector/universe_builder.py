"""
universe_builder.py

universe_probe.py가 만든 rolling_score를 바탕으로, "매 리밸런싱 시점마다"의
유니버스 멤버십 스냅샷을 처음부터 끝까지 복원해서 쌓아둔다.

왜 "현재 시점"만 계산하면 안 되는가:
  백테스트를 할 때 2024년 3월 시점의 신호를 검증하려면 "2024년 3월 시점에
  실제로 유니버스에 있던 코인"만 대상으로 해야 한다. 지금 살아있는 코인
  기준으로 과거를 재구성하면 생존편향(survivorship bias)이 생긴다.
  그래서 이 모듈은 과거 모든 리밸런싱 시점(월별)에 대해 그 시점 기준
  스냅샷을 각각 만들어 파일로 남긴다.

이력밴드(hysteresis)가 있는 이유로 이전 달 멤버십을 알아야 이번 달을
계산할 수 있다. 따라서 반드시 시간순으로(오래된 달 -> 최신 달) 순차 처리한다.

초반 몇 달은 상장된 코인 수 자체가 적어서 유니버스가 사실상 무의미할 수
있다(예: 1~2개짜리 유니버스). 이를 막기 위해 "실제로 자격을 갖춘 심볼 수가
min_universe_symbols 이상인 첫 달"부터 리밸런싱을 시작한다.

만드는 것:
  data/universe_snapshots/{YYYY-MM}.json       : 그 달 멤버십 + 순위
  data/universe_snapshots/{YYYY-MM}_diff.json  : 전월 대비 entered/exited

이 모듈은 순위/멤버십 계산만 한다. 전체 필드 수집은 full_collector.py 몫이다.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.collector.shared import storage
from src.config.paths import UNIVERSE_RULES_PATH
from src.config.collection_rules import STALENESS_DAYS

logger = logging.getLogger(__name__)

CONFIG_PATH = UNIVERSE_RULES_PATH



def load_rules() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_symbol_meta() -> dict:
    payload = storage.load_json(category="meta", filename="symbol_list")
    if payload is None:
        raise RuntimeError("symbol_list.json이 없다. symbol_universe.run()을 먼저 실행해야 한다.")
    return payload["symbols"]


def load_all_scan_series() -> dict[str, pd.Series]:
    """
    scan 카테고리의 모든 심볼 parquet을 로드해 {symbol: rolling_score 시계열} 딕셔너리로 반환.
    인덱스는 datetime, 값은 rolling_score. asof 조회를 위해 정렬 상태로 둔다.
    """
    scan_dir = storage.PATHS["scan"]
    series_map: dict[str, pd.Series] = {}

    for path in sorted(scan_dir.glob("*.parquet")):
        symbol = path.stem
        df = storage.load_parquet(category="scan", filename=symbol)
        if df is None or "rolling_score" not in df.columns:
            continue
        s = df.set_index(pd.to_datetime(df["date"]))["rolling_score"].sort_index()
        series_map[symbol] = s

    return series_map


def _get_onboard_date(meta_entry: dict, scan_series: pd.Series) -> Optional[pd.Timestamp]:
    """상장일 판단. meta의 onboard_date 우선, 없으면(주로 DELISTED) scan 데이터의 최초 날짜로 근사."""
    onboard = meta_entry.get("onboard_date")
    if onboard:
        return pd.Timestamp(onboard)
    if len(scan_series) > 0:
        return scan_series.index[0]
    return None


def generate_rebalance_dates(
    scan_series_map: dict[str, pd.Series],
    symbol_meta: dict,
    rules: dict,
) -> list[pd.Timestamp]:
    """
    전체 scan 데이터 범위를 훑어서 월말(month-end) 리밸런싱 날짜 리스트를 만든다.

    시작점 결정 방식:
      1) "가장 이른 데이터 시작일 + min_listing_days"가 포함되는 달의 말일을 후보로 잡는다.
      2) 그 후보 시점부터 한 달씩 훑으면서, 실제로 자격을 갖춘(eligible) 심볼 수가
         min_universe_symbols 이상이 되는 첫 달을 진짜 시작점으로 삼는다.
         (초반 몇 달은 상장된 코인이 너무 적어 유니버스가 무의미할 수 있어서다.)
    끝점은 오늘을 넘지 않는 선에서 가장 최근 완결된 달의 말일까지.
    """
    if not scan_series_map:
        return []

    earliest = min(s.index[0] for s in scan_series_map.values() if len(s) > 0)
    today = pd.Timestamp(datetime.now(timezone.utc).date())

    first_possible = earliest + pd.Timedelta(days=rules["min_listing_days"])
    min_universe_symbols = rules["min_universe_symbols"]


    # MonthEnd(0)은 이미 월말이면 그대로, 아니면 다음 월말로 항상 앞으로만 굴려준다.
    cursor = first_possible + pd.offsets.MonthEnd(0)

    dates: list[pd.Timestamp] = []
    started = False

    while cursor <= today:
        if not started:
            eligible_count = len(_asof_eligible_scores(cursor, scan_series_map, symbol_meta, rules))
            if eligible_count < min_universe_symbols:
                cursor = cursor + pd.offsets.MonthEnd(1)
                continue
            started = True

        dates.append(cursor)
        cursor = cursor + pd.offsets.MonthEnd(1)

    if not started:
        logger.warning(
            "[universe_builder] 어떤 시점에도 자격을 갖춘 심볼 수가 min_universe_symbols(%d)에 "
            "도달하지 못했다. 리밸런싱을 시작할 수 없다.",
            min_universe_symbols,
        )

    return dates


def _asof_eligible_scores(
    date: pd.Timestamp,
    scan_series_map: dict[str, pd.Series],
    symbol_meta: dict,
    rules: dict,
) -> dict[str, float]:
    """
    주어진 리밸런싱 시점 기준, "그 시점에 존재했고 최소 상장기간을 채운" 심볼들의
    rolling_score를 반환한다. 상장폐지되어 데이터가 오래전에 끊긴 심볼은 제외한다
    (staleness 체크).
    """
    scores: dict[str, float] = {}

    for symbol, series in scan_series_map.items():
        meta_entry = symbol_meta.get(symbol, {})
        onboard = _get_onboard_date(meta_entry, series)
        if onboard is None or onboard > date:
            continue  # 아직 상장 전
        if (date - onboard).days < rules["min_listing_days"]:
            continue  # 최소 상장기간 미충족

        asof_idx = series.index.asof(date)
        if pd.isna(asof_idx):
            continue  # 이 시점 이전 데이터 자체가 없음

        if (date - asof_idx).days > STALENESS_DAYS:
            continue  # 데이터가 너무 오래된 값 -> 이 시점엔 이미 거래정지/상장폐지로 판단

        value = series.loc[asof_idx]
        if pd.isna(value):
            continue

        scores[symbol] = value

    return scores


def select_membership(
    prev_membership: set[str],
    scores: dict[str, float],
    rules: dict,
) -> tuple[set[str], dict[str, int]]:
    """
    이력밴드 규칙 적용:
      - 기존 멤버는 순위가 retain_rank 이내면 유지
      - 신규 진입은 순위가 entry_rank 이내여야 편입
    반환: (이번 시점 최종 멤버십, {symbol: rank})
    """
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    rank_map = {symbol: i + 1 for i, (symbol, _) in enumerate(ranked)}

    entry_rank = rules["entry_rank"]
    retain_rank = rules["retain_rank"]

    retained = {s for s in prev_membership if rank_map.get(s, float("inf")) <= retain_rank}
    new_entrants = {s for s, r in rank_map.items() if r <= entry_rank}

    membership = retained | new_entrants
    return membership, rank_map


def save_snapshot(date: pd.Timestamp, membership: set[str], rank_map: dict[str, int]) -> Path:
    ym = date.strftime("%Y-%m")
    payload = {
        "rebalance_date": date.strftime("%Y-%m-%d"),
        "member_count": len(membership),
        "members": sorted(membership),
        # membership의 모든 심볼은 select_membership 로직상 항상 rank_map에 존재한다
        # (retained는 rank_map.get(...) <= retain_rank를 통과해야 하므로).
        "ranks": {s: rank_map[s] for s in sorted(membership)},
    }
    return storage.save_json(payload, category="universe_snapshots", filename=ym)


def save_diff(date: pd.Timestamp, entered: set[str], exited: set[str]) -> Path:
    ym = date.strftime("%Y-%m")
    payload = {
        "rebalance_date": date.strftime("%Y-%m-%d"),
        "entered": sorted(entered),
        "exited": sorted(exited),
    }
    return storage.save_json(payload, category="universe_snapshots", filename=f"{ym}_diff")


def run() -> None:
    """bootstrap 이후, universe_probe 실행 뒤에 호출. 전체 히스토리를 처음부터 재계산한다."""
    storage.ensure_dirs()
    rules = load_rules()
    symbol_meta = load_symbol_meta()
    scan_series_map = load_all_scan_series()

    rebalance_dates = generate_rebalance_dates(scan_series_map, symbol_meta, rules)
    if not rebalance_dates:
        logger.warning("[universe_builder] 리밸런싱 가능한 데이터가 없다. universe_probe를 먼저 실행했는지 확인.")
        return

    prev_membership: set[str] = set()

    for date in rebalance_dates:
        scores = _asof_eligible_scores(date, scan_series_map, symbol_meta, rules)
        membership, rank_map = select_membership(prev_membership, scores, rules)

        if not membership:
            logger.warning(
                "[universe_builder] %s 시점 유니버스가 비어 있다 (eligible 심볼 %d개). "
                "설정값이나 데이터 공백을 확인해라.",
                date.strftime("%Y-%m"), len(scores),
            )

        entered = membership - prev_membership
        exited = prev_membership - membership

        save_snapshot(date, membership, rank_map)
        save_diff(date, entered, exited)

        logger.info(
            "[universe_builder] %s: 멤버 %d명 (신규 %d, 이탈 %d)",
            date.strftime("%Y-%m"), len(membership), len(entered), len(exited),
        )

        prev_membership = membership

    logger.info("[universe_builder] 완료. 총 %d개 리밸런싱 시점 처리", len(rebalance_dates))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run()