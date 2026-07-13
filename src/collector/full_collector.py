"""
full_collector.py

데이터셋마다 아카이브 존재 형태가 다르다는 게 실측으로 확인됨:
  - monthly_with_daily_fallback (klines, premiumIndexKlines):
      완결된 과거 달 -> monthly / 이번 달 -> daily, 그래도 없는 최근 며칠 -> REST
  - monthly_only (fundingRate):
      daily 아카이브 자체가 없음. 이번 달 -> 곧바로 REST
  - daily_only (metrics, bookDepth):
      monthly 아카이브 자체가 없음. 전체 역사를 daily 단위로만 수집.
      manifest도 월(YYYY-MM)이 아니라 날짜(YYYY-MM-DD) 단위로 추적.

주의: daily_only 데이터셋은 상장일부터 오늘까지 하루 단위 요청이 누적되므로
(예: 2020년 상장 심볼이면 2000회+) 심볼당 최초 백필이 상당히 느리다.
전체 유니버스로 돌리기 전 심볼 1~2개로 먼저 테스트 권장.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Optional

import pandas as pd

from src.collector.shared import archive_client, rest_client, storage
from src.collector.shared.archive_client import ArchiveError, ArchiveNotFoundError
from src.config.paths import UNIVERSE_RULES_PATH
from src.config.binance_api import DATASET_SPECS, DATASETS_DEFAULT

CONFIG_PATH = UNIVERSE_RULES_PATH

# 심볼 단위 병렬 수집 워커 수. 요청 간격 자체는 archive_client/rest_client 앞단의
# 전역 스로틀(shared/throttle.py)이 보장하므로, 워커를 늘려도 초당 요청 수는 안 늘어난다
# (다운로드/파싱 대기가 겹쳐지는 만큼만 빨라진다).
# 아카이브 간격 0.05초(20/s)를 실제로 채우려면 요청당 왕복 ~0.3초 기준 워커가 6개보다
# 많아야 해서 10으로 올림.
DEFAULT_MAX_WORKERS = 10

# daily 아카이브 404를 만났을 때, 오늘로부터 이 일수 이내면 "아직 발행 전"으로
# 간주하고 REST 폴백으로 넘어간다. 그보다 오래된 날짜의 404는 "원래 없는 날"로
# 확정하고 건너뛴다 (기존엔 무조건 break해서 중간 결측일 하나에 수집이 영구 정지했음).
GAP_RECENT_DAYS = 3

logger = logging.getLogger(__name__)


def _is_delisted(meta_entry: dict) -> bool:
    return meta_entry.get("status") == "DELISTED"


def _month_end_date(ym: str) -> date:
    y, m = map(int, ym.split("-"))
    return date(y + 1, 1, 1) - timedelta(days=1) if m == 12 else date(y, m + 1, 1) - timedelta(days=1)


def load_rules() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_symbol_meta() -> dict:
    payload = storage.load_json(category="meta", filename="symbol_list")
    if payload is None:
        raise RuntimeError("symbol_list.json이 없다. symbol_universe.run()을 먼저 실행해야 한다.")
    return payload["symbols"]


def collect_target_symbols() -> set[str]:
    snap_dir = storage.PATHS["universe_snapshots"]
    targets: set[str] = set()
    for path in sorted(snap_dir.glob("*.json")):
        if path.stem.endswith("_diff"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        targets.update(payload.get("members", []))
    return targets


# --- 월(YYYY-MM) 유틸 ---

def _month_range(start_ym: str, end_ym: str) -> list[str]:
    start = datetime.strptime(start_ym, "%Y-%m")
    end = datetime.strptime(end_ym, "%Y-%m")
    months = []
    cur = start
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        cur = cur.replace(year=cur.year + 1, month=1) if cur.month == 12 else cur.replace(month=cur.month + 1)
    return months


def _next_month(ym: str) -> str:
    y, m = map(int, ym.split("-"))
    return f"{y+1}-01" if m == 12 else f"{y}-{m+1:02d}"


def _prev_month(ym: str) -> Optional[str]:
    y, m = map(int, ym.split("-"))
    if m == 1:
        y, m = y - 1, 12
    else:
        m -= 1
    return f"{y}-{m:02d}"


# --- 일(YYYY-MM-DD) 유틸 ---

def _date_range(start_date: date, end_date: date) -> list[date]:
    days = []
    cur = start_date
    while cur <= end_date:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _determine_start_month(symbol: str, meta_entry: dict, dataset: str, interval: Optional[str]) -> Optional[str]:
    """
    상장일(onboard_date)과 "아카이브가 실제로 시작되는 달" 중 더 늦은 쪽을 쓴다.
    데이터셋마다 아카이브 시작 시점이 다르기 때문에(예: premiumIndexKlines는
    상장일보다 훨씬 늦게 발행되기 시작한 심볼이 많음) onboard_date만 믿으면
    첫 달부터 404 -> break -> manifest 미전진 -> 매 실행 헛수고 루프에 빠진다.
    """
    try:
        available = archive_client.list_available_periods(dataset=dataset, symbol=symbol, interval=interval, kind="monthly")
    except ArchiveError as e:
        logger.warning("%s[%s]: 아카이브 시작 달 조회 실패, 다음 실행에서 재시도: %s", symbol, dataset, e)
        return None
    archive_start = min(available) if available else None

    onboard_date = meta_entry.get("onboard_date")
    onboard_ym = onboard_date[:7] if onboard_date else None

    if archive_start and onboard_ym:
        return max(archive_start, onboard_ym)
    return archive_start or onboard_ym


def _determine_start_date(symbol: str, meta_entry: dict, dataset: str) -> Optional[date]:
    """
    daily_only 데이터셋의 시작일. onboard_date가 있어도 아카이브가 그보다 늦게
    시작하는 경우가 많다 (예: metrics 아카이브는 2021-12 무렵부터 존재하므로
    2020년 상장 심볼은 상장일 기준 첫 요청이 무조건 404).
    404 -> break -> REST 폴백(30일 제한) 400 -> 영원히 0건 수집되는 루프를 막기 위해
    실제 아카이브 최초 날짜와 onboard_date 중 더 늦은 쪽을 시작일로 쓴다.
    """
    try:
        available = archive_client.list_available_periods(dataset=dataset, symbol=symbol, kind="daily")
    except ArchiveError as e:
        logger.warning("%s[%s]: 아카이브 시작일 조회 실패, 다음 실행에서 재시도: %s", symbol, dataset, e)
        return None
    archive_start = datetime.strptime(min(available), "%Y-%m-%d").date() if available else None

    onboard_date = meta_entry.get("onboard_date")
    onboard = datetime.strptime(onboard_date[:10], "%Y-%m-%d").date() if onboard_date else None

    if archive_start and onboard:
        return max(archive_start, onboard)
    return archive_start or onboard


def _fetch_via_rest(dataset: str, symbol: str, interval: Optional[str], start_ms: int, end_ms: int) -> pd.DataFrame:
    if dataset in ("klines", "premiumIndexKlines"):
        return rest_client.fetch_klines_recent(symbol, interval, start_ms, end_ms)
    if dataset == "fundingRate":
        return rest_client.fetch_funding_rate_recent(symbol, start_ms, end_ms)
    if dataset == "metrics":
        return rest_client.fetch_metrics_recent(symbol, start_ms, end_ms)
    raise NotImplementedError(f"'{dataset}'는 REST 폴백이 정의되어 있지 않음 (bookDepth 등)")


def _collect_month_via_daily(symbol: str, dataset: str, ym: str, interval: Optional[str]):
    """
    monthly 아카이브가 없는 달을 daily 아카이브로 복원한다.
    반환: (DataFrame|None, ok)
      - ok=True: 모든 날을 확인함 (데이터 있는 날은 수집, 404인 날은 "원래 없음" 확정)
      - ok=False: 404가 아닌 오류(네트워크/5xx 등)를 만나 중단. "없음"으로 확정 불가,
        호출부는 manifest를 전진시키면 안 된다.
    """
    today = datetime.now(timezone.utc).date()
    y, m = map(int, ym.split("-"))
    cur = date(y, m, 1)
    end = min(_month_end_date(ym), today)

    frames: list[pd.DataFrame] = []
    while cur <= end:
        date_str = cur.strftime("%Y-%m-%d")
        try:
            frames.append(archive_client.fetch_archive_day(dataset, symbol, date_str, interval=interval))
        except ArchiveNotFoundError:
            pass  # 그 날 데이터가 원래 없음 (거래정지 등). 조용히 다음 날로.
        except ArchiveError as e:
            logger.warning("%s[%s] %s daily 폴백 실패(일시적일 수 있음): %s", symbol, dataset, date_str, e)
            return (pd.concat(frames, ignore_index=True) if frames else None), False        
        cur += timedelta(days=1)

    return (pd.concat(frames, ignore_index=True) if frames else None), True


# ---------------------------------------------------------------------------
# 분기 1: monthly_with_daily_fallback (klines, premiumIndexKlines)
# ---------------------------------------------------------------------------

def _collect_monthly_with_daily_fallback(symbol, meta_entry, rules, dataset, storage_key) -> Optional[Path]:
    spec = DATASET_SPECS[dataset]
    interval = rules["full_collect_interval"]

    last_collected = storage.get_last_collected(category=f"processed_{dataset}", symbol=storage_key)
    start_ym = _next_month(last_collected) if last_collected else _determine_start_month(symbol, meta_entry, dataset, interval)
    if start_ym is None:
        logger.warning("%s[%s]: 아카이브에 데이터 없음, 스킵", symbol, dataset)
        return None

    current_ym = datetime.now(timezone.utc).strftime("%Y-%m")
    end_ym = current_ym
    # 상장폐지 심볼은 아카이브상 마지막 존재 월까지만 수집한다.
    # 그 뒤 아카이브는 영원히 생기지 않으므로 매 실행 재시도하는 낭비를 막는다.
    if _is_delisted(meta_entry) and meta_entry.get("last_seen_month"):
        end_ym = min(end_ym, meta_entry["last_seen_month"])
    if start_ym > end_ym:
        return None

    months = _month_range(start_ym, end_ym)
    new_frames: list[pd.DataFrame] = []
    last_completed_month: Optional[str] = None

    for ym in months:
        if ym == current_ym:
            break  # 이번 달은 monthly 아카이브가 아직 없다. daily/REST로.
        try:
            frame = archive_client.fetch_archive_month(dataset, symbol, ym, interval=interval)
            new_frames.append(frame)
            last_completed_month = ym
        except ArchiveNotFoundError:
            # monthly가 없는 달은 daily 아카이브로 그 달을 복원 시도한다.
            #  - 오래전 달: daily 404까지 확인되면 "원래 없음"으로 확정, 건너뛰고 계속
            #    (기존엔 여기서 break -> 결측 월 하나에 이후 전체 수집이 영구 정지했음)
            #  - 직전 달: monthly 아카이브가 아직 발행 전일 수 있다. 데이터는 daily로
            #    채우되 manifest는 전진시키지 않아 다음 실행 때 monthly로 재확인한다.
            logger.info("%s[%s] %s: monthly 아카이브 없음, daily로 복원 시도", symbol, dataset, ym)
            frame, ok = _collect_month_via_daily(symbol, dataset, ym, interval)
            if frame is not None:
                new_frames.append(frame)
            if not ok:
                break  # 일시적 오류 섞임 -> 확정 불가, 다음 실행에서 이 달부터 재시도
            if ym < _prev_month(current_ym):
                last_completed_month = ym
                continue
            break  # 직전 달: 완료로 확정하지 않고 여기서 멈춤 (이번 달 daily/REST는 아래에서)
        except ArchiveError as e:
            logger.warning("%s[%s] %s 수집 실패: %s", symbol, dataset, ym, e)
            break
    if current_ym in months:
        now = datetime.now(timezone.utc)
        day = now.replace(day=1)
        rest_needed_from = None
        while day.date() <= now.date():
            date_str = day.strftime("%Y-%m-%d")
            try:
                frame = archive_client.fetch_archive_day(dataset, symbol, date_str, interval=interval)
                new_frames.append(frame)
            except ArchiveNotFoundError:
                rest_needed_from = day
                break
            except ArchiveError as e:
                logger.warning("%s[%s] %s daily 수집 실패: %s", symbol, dataset, date_str, e)
                rest_needed_from = day
                break            
            day += timedelta(days=1)

        # 상장폐지 심볼은 REST 엔드포인트에서 400이 나므로(심볼 자체가 무효) 시도하지 않는다.
        if rest_needed_from is not None and not _is_delisted(meta_entry):
            start_ms = int(rest_needed_from.timestamp() * 1000)
            end_ms = int(now.timestamp() * 1000)
            try:
                rest_frame = _fetch_via_rest(dataset, symbol, interval, start_ms, end_ms)
                if not rest_frame.empty:
                    new_frames.append(rest_frame)
            except ArchiveError as e:
                logger.warning("%s[%s] REST 폴백 실패: %s", symbol, dataset, e)

    if not new_frames:
        return None

    new_df = pd.concat(new_frames, ignore_index=True)
    path = storage.append_parquet(new_df, category=f"processed_{dataset}", filename=storage_key, dedup_key=spec["time_col"])
    if last_completed_month is not None:
        storage.update_last_collected(category=f"processed_{dataset}", symbol=storage_key, value=last_completed_month)
    return path


# ---------------------------------------------------------------------------
# 분기 2: monthly_only (fundingRate) — daily 아카이브가 없어 이번 달은 REST 직행
# ---------------------------------------------------------------------------

def _collect_monthly_only(symbol, meta_entry, rules, dataset, storage_key) -> Optional[Path]:
    spec = DATASET_SPECS[dataset]

    last_collected = storage.get_last_collected(category=f"processed_{dataset}", symbol=storage_key)
    start_ym = _next_month(last_collected) if last_collected else _determine_start_month(symbol, meta_entry, dataset, None)
    if start_ym is None:
        logger.warning("%s[%s]: 아카이브에 데이터 없음, 스킵", symbol, dataset)
        return None

    current_ym = datetime.now(timezone.utc).strftime("%Y-%m")
    end_ym = current_ym
    # 상장폐지 심볼은 아카이브상 마지막 존재 월까지만 (그 뒤는 영원히 안 생김)
    if _is_delisted(meta_entry) and meta_entry.get("last_seen_month"):
        end_ym = min(end_ym, meta_entry["last_seen_month"])
    if start_ym > end_ym:
        return None

    months = _month_range(start_ym, end_ym)
    new_frames: list[pd.DataFrame] = []
    last_completed_month: Optional[str] = None

    for ym in months:
        if ym == current_ym:
            break
        try:
            frame = archive_client.fetch_archive_month(dataset, symbol, ym)
            new_frames.append(frame)
            last_completed_month = ym
        except ArchiveNotFoundError:
            # fundingRate는 daily 아카이브가 없어서 daily 복원이 불가능하다.
            #  - 오래전 달: 404 = "원래 없음" 확정 (거래정지 등) -> 건너뛰고 계속
            #  - 직전 달: 아직 발행 전일 수 있음 -> 멈추고 다음 실행에서 재확인
            if ym < _prev_month(current_ym):
                logger.info("%s[%s] %s: monthly 아카이브 없음(과거 확정 결측), 건너뛰고 계속", symbol, dataset, ym)
                last_completed_month = ym
                continue
            logger.info("%s[%s] %s: monthly 아카이브 아직 없음(발행 전일 수 있음), 다음 실행에 재시도", symbol, dataset, ym)
            break
        except ArchiveError as e:
            logger.warning("%s[%s] %s 수집 실패: %s", symbol, dataset, ym, e)
            break
    # daily 아카이브가 없으므로 이번 달은 곧바로 REST (폐지 심볼은 REST가 400이므로 스킵)
    if current_ym in months and not _is_delisted(meta_entry):
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        try:
            rest_frame = _fetch_via_rest(
                dataset, symbol, None,
                int(month_start.timestamp() * 1000), int(now.timestamp() * 1000),
            )
            if not rest_frame.empty:
                new_frames.append(rest_frame)
        except ArchiveError as e:
            logger.warning("%s[%s] REST 폴백 실패: %s", symbol, dataset, e)

    if not new_frames:
        return None

    new_df = pd.concat(new_frames, ignore_index=True)
    path = storage.append_parquet(new_df, category=f"processed_{dataset}", filename=storage_key, dedup_key=spec["time_col"])
    if last_completed_month is not None:
        storage.update_last_collected(category=f"processed_{dataset}", symbol=storage_key, value=last_completed_month)
    return path


# ---------------------------------------------------------------------------
# 분기 3: daily_only (metrics, bookDepth) — 전체 역사를 하루 단위로 수집
# ---------------------------------------------------------------------------

def _collect_daily_only(symbol, meta_entry, rules, dataset, storage_key) -> Optional[Path]:
    spec = DATASET_SPECS[dataset]

    last_collected = storage.get_last_collected(category=f"processed_{dataset}", symbol=storage_key)
    if last_collected:
        start_date = datetime.strptime(last_collected, "%Y-%m-%d").date() + timedelta(days=1)
    else:
        start_date = _determine_start_date(symbol, meta_entry, dataset)
        if start_date is None:
            logger.warning("%s[%s]: 아카이브에 데이터 없음, 스킵", symbol, dataset)
            return None

    today = datetime.now(timezone.utc).date()
    end_date = today
    # 상장폐지 심볼은 아카이브상 마지막 존재 월의 말일까지만 수집한다.
    if _is_delisted(meta_entry) and meta_entry.get("last_seen_month"):
        end_date = min(today, _month_end_date(meta_entry["last_seen_month"]))
    if start_date > end_date:
        return None

    days = _date_range(start_date, end_date)
    new_frames: list[pd.DataFrame] = []
    last_completed_date: Optional[date] = None
    rest_needed_from: Optional[date] = None

    for d in days:
        date_str = d.strftime("%Y-%m-%d")
        try:
            frame = archive_client.fetch_archive_day(dataset, symbol, date_str)
            new_frames.append(frame)
            last_completed_date = d  # 오늘이 아니면(과거 확정일) 완료로 취급
        except ArchiveNotFoundError:
            if (today - d).days <= GAP_RECENT_DAYS:
                # 아직 업로드 안 된 최근 며칠 - 여기서부터 REST로
                rest_needed_from = d
                break
            # 과거의 결측일: 404 = "원래 없는 날"로 확정(거래정지 등), 건너뛰고 계속.
            # (기존엔 무조건 break -> 결측일 하나에 이후 수집이 영구 정지 + 매 실행 반복)
            logger.info("%s[%s] %s: daily 아카이브 없음(과거 확정 결측), 건너뜀", symbol, dataset, date_str)
            last_completed_date = d
            continue
        except ArchiveError as e:
            # 404가 아닌 오류(네트워크/5xx 등)는 "없음"으로 확정할 수 없다.
            # manifest를 전진시키지 않고 멈춰서 다음 실행 때 이 날부터 재시도.
            logger.warning("%s[%s] %s 수집 실패(일시적일 수 있음): %s", symbol, dataset, date_str, e)
            rest_needed_from = d
            break
    # 오늘 날짜는 daily 아카이브가 있었어도 '완료'로 기록하지 않는다 (이번 달과 동일한 이유)
    if last_completed_date == today:
        last_completed_date = today - timedelta(days=1)

    # 폐지 심볼은 REST가 400(무효 심볼)이므로 시도하지 않는다.
    if rest_needed_from is not None and not _is_delisted(meta_entry) and spec.get("rest_kind", True) is not None:
        start_ms = int(datetime.combine(rest_needed_from, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        try:
            rest_frame = _fetch_via_rest(dataset, symbol, None, start_ms, end_ms)
            if not rest_frame.empty:
                new_frames.append(rest_frame)
        except NotImplementedError:
            logger.info("%s[%s]: REST 폴백 미지원, 최근 구간은 다음 실행에 재시도", symbol, dataset)
        except ArchiveError as e:
            logger.warning("%s[%s] REST 폴백 실패: %s", symbol, dataset, e)

    if not new_frames:
        return None

    new_df = pd.concat(new_frames, ignore_index=True)
    path = storage.append_parquet(new_df, category=f"processed_{dataset}", filename=storage_key, dedup_key=spec["time_col"])
    if last_completed_date is not None:
        storage.update_last_collected(category=f"processed_{dataset}", symbol=storage_key, value=last_completed_date.strftime("%Y-%m-%d"))
    return path


# ---------------------------------------------------------------------------
# 디스패처
# ---------------------------------------------------------------------------

def collect_symbol_dataset(symbol: str, meta_entry: dict, rules: dict, dataset: str) -> Optional[Path]:
    spec = DATASET_SPECS[dataset]
    # 데이터셋별 하위 폴더(processed_{dataset})에 심볼명 그대로 저장한다.
    # 예전 방식({SYMBOL}__{dataset}.parquet 평면 저장)은 scripts/migrate_processed_layout.py로 이관.
    storage_key = symbol
    granularity = spec["archive_granularity"]

    if granularity == "monthly_with_daily_fallback":
        return _collect_monthly_with_daily_fallback(symbol, meta_entry, rules, dataset, storage_key)
    if granularity == "monthly_only":
        return _collect_monthly_only(symbol, meta_entry, rules, dataset, storage_key)
    if granularity == "daily_only":
        return _collect_daily_only(symbol, meta_entry, rules, dataset, storage_key)
    raise ValueError(f"알 수 없는 archive_granularity: {granularity}")


def run(datasets: Optional[list[str]] = None, max_workers: int = DEFAULT_MAX_WORKERS,
        symbols: Optional[list[str]] = None) -> None:
    storage.ensure_dirs()
    rules = load_rules()
    symbol_meta = load_symbol_meta()
    datasets = datasets or DATASETS_DEFAULT

    targets = collect_target_symbols()
    if not targets:
        logger.info("수집 대상 심볼이 없다. universe_builder를 먼저 실행했는지 확인.")
        return
    if symbols:
        # 라이브 최신화: 현재 top-100 등 지정 종목만 수집(백테스트는 symbols 생략 = 전체).
        want = set(symbols)
        targets = [s for s in targets if s in want]
        if not targets:
            logger.info("지정 심볼이 수집 대상에 하나도 없다(symbols=%d개). 확인 필요.", len(want))
            return

    total = len(targets)
    progress_lock = threading.Lock()
    processed_count = 0
    updated = 0

    def _run_one(symbol: str) -> None:
        nonlocal processed_count, updated
        meta_entry = symbol_meta.get(symbol, {})
        if symbol not in symbol_meta:
            logger.warning("%s: symbol_list.json에 메타 없음. onboard_date 없이 진행.", symbol)

        symbol_updated = False
        for dataset in datasets:
            result_path = collect_symbol_dataset(symbol, meta_entry, rules, dataset)
            if result_path is not None:
                symbol_updated = True

        with progress_lock:
            processed_count += 1
            if symbol_updated:
                updated += 1
            if processed_count % 20 == 0:
                logger.info("진행 %d/%d", processed_count, total)

    # 심볼 단위 병렬. 요청 속도는 전역 스로틀이 잡아주므로 여기선 워커 수만 관리한다.
    # manifest 갱신은 storage.update_last_collected 내부 락으로 보호된다.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, symbol): symbol for symbol in sorted(targets)}
        for future in as_completed(futures):
            future.result()  # 코드 버그성 예외는 숨기지 않고 그대로 전파

    logger.info("완료. 대상 %d개 중 %d개 심볼 갱신됨 (datasets=%s)", total, updated, datasets)