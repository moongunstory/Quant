"""
universe_probe.py

symbol_universe.py가 확보한 전체 심볼 목록을 대상으로,
유니버스 판단에 필요한 "가벼운" 필드만 뽑아 rolling 지표까지 계산해두는 모듈.
(예전 논의에서 light_scanner로 부르던 것과 동일한 역할, 이름만 변경)

무거운 1분봉 대신 1일봉(daily kline) 아카이브를 받아온다 -> 용량이 훨씬 작아서
"가벼운" 스캔이라는 이름값을 실제로 지킨다. 전체 필드(15개) 수집은 이 모듈의
역할이 아니고 나중에 만들 full_collector.py가 유니버스 스냅샷 합집합 심볼에
대해서만 수행한다.

이 모듈이 만드는 것:
  data/market/scan/{SYMBOL}.parquet : date, close, quote_volume, rolling_score
  data/market/scan/_manifest.json   : {symbol: "YYYY-MM"}  (gap-aware 재실행용)

rolling_score 계산 규칙은 전부 universe_rules.json에서 읽어온다.
이 파일에는 "몇 위 안에 들어야 편입되는지" 같은 순위/이력밴드 로직은 없다.
그건 universe_builder.py의 책임이다. 이 모듈은 순위를 매길 수 있는 재료
(rolling median quote_volume)까지만 만들어둔다.

예외 정책 (D-xx):
  archive_client가 정의한 ArchiveError 계열(404/HTTP/네트워크/타임아웃/zip 손상)만
  "예상 가능한 상황"으로 잡아서 스킵/폴백한다. 그 외 KeyError, TypeError 등은
  코드 버그일 가능성이 높으므로 여기서 잡지 않고 그대로 상위로 전파시켜 즉시
  발견되게 한다.

병렬 수집:
  심볼 단위로 ThreadPoolExecutor를 사용한다. Binance로 나가는 요청 간격은
  archive_client 앞단의 전역 스로틀(shared/throttle.py)이 프로세스 전체에
  강제한다 (워커 수를 늘려도 초당 요청 수는 늘지 않음). manifest 갱신의
  스레드 안전성은 storage.update_last_collected 내부 락이 보장한다.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.collector.shared import archive_client, storage
from src.config.paths import UNIVERSE_RULES_PATH

CONFIG_PATH = UNIVERSE_RULES_PATH

# 심볼 단위 병렬 수집 워커 수. 안정성 우선이라 넉넉하게 잡지 않는다.
DEFAULT_MAX_WORKERS = 6

logger = logging.getLogger(__name__)


class NoDataInMonthError(Exception):
    """월 전체(monthly + 일별 아카이브 전부)에 걸쳐 데이터를 찾지 못한 경우.

    archive_client의 개별 요청은 성공/실패했지만, 그 결과를 종합했을 때
    "이 달은 데이터가 없다"고 확정된 상태다. archive_client 레벨 오류는
    아니지만 universe_probe 입장에서는 동일하게 "예상 가능한 상황"이므로
    별도 예외로 분리해 상위에서 ArchiveError와 함께 잡을 수 있게 한다.

    주의: 이 예외는 "일별 폴백에서 만난 실패가 전부 404(ArchiveNotFoundError)"인
    경우에만 raise해야 한다. 네트워크/HTTP/타임아웃 등 다른 이유로 실패한 날이
    하나라도 섞여 있으면 TransientMonthFailure를 대신 raise해야 한다 (아래 참고).
    """


class TransientMonthFailure(Exception):
    """
    이 달에 데이터가 "원래 없다"고 확정할 수 없는 상태.

    일별 아카이브 폴백에서 유효한 데이터를 하나도 못 받았지만, 실패 사유 중에
    404(ArchiveNotFoundError)가 아닌 것(네트워크 오류/HTTP 5xx/타임아웃 등)이
    하나라도 섞여 있으면 이 예외를 raise한다.

    배경(버그 히스토리): 예전에는 이런 경우도 NoDataInMonthError와 동일하게
    "데이터 없음으로 확정, 다음 달 계속"으로 처리했다. 그 결과 실행 초반 일시적
    실패(레이트리밋 등으로 추정)를 겪은 상위 126개 심볼(BTCUSDT, ETHUSDT,
    XRPUSDT 등 2019~2022년 상장 코인)이 2020년대 전체 히스토리를 통째로
    "확정된 공백"으로 처리당하고 manifest(_manifest.json)가 최신 달까지
    전진해버려, 실제로는 하나도 못 받은 과거 데이터를 다시는 재수집하지 않는
    상태가 됐다. 이 예외는 그 상황과 "진짜로 상장 전이라 데이터가 없는 상황"을
    구분하기 위해 도입했다. 이 예외를 만나면 상위(probe_symbol)는 manifest를
    전진시키지 않고 그 자리에서 멈춰서, 다음 실행 때 같은 달부터 재시도한다.
    """


# 요청 속도 제한은 archive_client 앞단의 전역 스로틀(shared/throttle.py)이,
# manifest 동시 갱신 보호는 storage.update_last_collected 내부 락이 담당한다.
# (예전엔 이 모듈이 자체 _RateLimiter와 manifest 락을 갖고 있었는데,
#  full_collector도 병렬화되면서 공유 계층으로 옮겼다.)


def load_rules() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _month_range(start_ym: str, end_ym: str) -> list[str]:
    """start_ym부터 end_ym까지 (둘 다 포함) 연월 문자열 리스트. "YYYY-MM" 형식."""
    start = datetime.strptime(start_ym, "%Y-%m")
    end = datetime.strptime(end_ym, "%Y-%m")
    months = []
    cur = start
    while cur <= end:
        months.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months


def _determine_start_month(symbol: str, meta_entry: dict) -> Optional[str]:
    """
    이 심볼의 스캔을 어느 달부터 시작해야 하는지 결정.
    onboard_date가 있으면 그 달부터, 없으면 (delisted 등) 아카이브에서 직접 확인.
    아카이브에 데이터가 전혀 없으면 None 반환 (스캔 대상에서 제외).

    archive_start_month가 meta_entry에 이미 캐싱돼 있으면 재조회하지 않는다.
    캐시가 없는 경우에만 조회하고, 조회에 성공(값이 존재)하면 meta_entry에
    기록한다 (호출부인 run()이 최종적으로 symbol_list.json에 저장).
    """
    onboard_date = meta_entry.get("onboard_date")
    onboard_ym = onboard_date[:7] if onboard_date else None  # "YYYY-MM-DD" -> "YYYY-MM"

    cached_archive_start = meta_entry.get("archive_start_month")
    if cached_archive_start is not None:
        archive_start_ym = cached_archive_start
    else:
        try:
            available_months = archive_client.list_available_months(symbol, market="um")
        except archive_client.ArchiveError as e:
            logger.warning(f"{symbol}: archive_start_month 조회 실패, 다음 실행에서 재시도: {e}")
            available_months = []

        archive_start_ym = available_months[0] if available_months else None
        if archive_start_ym is not None:
            meta_entry["archive_start_month"] = archive_start_ym

    if onboard_ym and archive_start_ym:
        return max(onboard_ym, archive_start_ym)
    if archive_start_ym:
        return archive_start_ym
    return onboard_ym


def _fetch_month_quote_volume(symbol: str, year_month: str, interval: str) -> pd.DataFrame:
    """월간 daily kline을 받아 date/close/quote_volume만 남긴 DataFrame으로 축약."""
    df = archive_client.fetch_kline_month(symbol=symbol, interval=interval, year_month=year_month, market="um")
    df = df[["open_time", "close", "quote_volume"]].rename(columns={"open_time": "date"})
    df["date"] = df["date"].dt.date.astype(str)
    return df


def _days_in_month(year_month: str) -> list[str]:
    """해당 연월의 날짜(YYYY-MM-DD) 목록. 오늘 이후는 제외."""
    year, month = map(int, year_month.split("-"))
    start = date(year, month, 1)
    next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    today = datetime.now(timezone.utc).date()

    days = []
    cur = start
    while cur < next_month and cur <= today:
        days.append(cur.isoformat())
        cur += timedelta(days=1)
    return days


def _fetch_month_quote_volume_via_daily(symbol: str, year_month: str, interval: str) -> pd.DataFrame:
    """
    monthly 아카이브가 없는 달을 위한 폴백.
    해당 월의 daily zip을 내부 스레드풀을 이용해 병렬로 받아와 이어붙인다.
    상장 전 날짜처럼 데이터가 아예 없는 날은 개별적으로 조용히 건너뛴다.
    """
    days = _days_in_month(year_month)
    frames = [None] * len(days)  # 날짜 순서를 정확히 보장하기 위해 리스트 미리 할당
    had_non_404_failure = [False] * len(days)  # 404가 아닌 이유로 실패한 날 표시

    def _fetch_single_day(idx: int, day: str):
        try:
            df = archive_client.fetch_kline_day(symbol=symbol, interval=interval, date=day, market="um")
            df = df[["open_time", "close", "quote_volume"]].rename(columns={"open_time": "date"})
            df["date"] = df["date"].dt.date.astype(str)
            return idx, df, False
        except archive_client.ArchiveNotFoundError:
            # 상장 전 날짜 등 "원래 없는 게 정상"인 경우. 조용히 다음 날로.
            return idx, None, False
        except archive_client.ArchiveError as e:
            # HTTP/네트워크/타임아웃 등. 404가 아니므로 "원래 없다"고 확정할 수 없다.
            # 이 날 하루만 포기하고 계속 진행하되, 나중에 이 달 전체를 "확정된 공백"으로
            # 처리하면 안 된다는 신호로 남긴다.
            logger.debug(f"{symbol} {day} 일별 아카이브 조회 실패(404 아님, 일시적일 수 있음), 건너뜀: {e}")
            return idx, None, True

    # 하루씩 동기적으로 받던 것을 스레드풀을 이용해 한 달 치를 병렬로 빠르게 수집 (기존 대비 수십 배 속도 향상)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_fetch_single_day, i, day) for i, day in enumerate(days)]
        for future in as_completed(futures):
            idx, df, is_non_404 = future.result()
            if df is not None:
                frames[idx] = df
            if is_non_404:
                had_non_404_failure[idx] = True

    # None이 아닌 정상 수집된 DataFrame만 필터링 (원래 순서가 유지됨)
    valid_frames = [f for f in frames if f is not None]

    if not valid_frames:
        if any(had_non_404_failure):
            # 404가 아닌 오류가 하나라도 섞여 있으면 "이 달은 원래 데이터가 없다"고
            # 확정할 근거가 없다. NoDataInMonthError로 뭉뚱그리면 상위에서 manifest를
            # 이 달까지 전진시켜버려서 진짜 데이터를 영영 재수집 안 하게 되므로 구분한다.
            raise TransientMonthFailure(
                f"{symbol} {year_month}: 일부 날짜가 404가 아닌 오류로 실패, 데이터 없음으로 확정 불가"
            )
        raise NoDataInMonthError(f"{symbol} {year_month}: 일별 아카이브 폴백에서도 데이터를 찾지 못함")
    return pd.concat(valid_frames, ignore_index=True)


def _compute_rolling_score(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """quote_volume에 rolling median(또는 rules 지정 방식)을 적용해 rolling_score 컬럼 추가."""
    df = df.sort_values("date").reset_index(drop=True)

    window = rules["rolling_window_days"]
    min_periods = rules.get("min_rolling_periods", window)
    aggregation = rules["aggregation"]

    roller = df["quote_volume"].rolling(window=window, min_periods=min_periods)
    if aggregation == "median":
        df["rolling_score"] = roller.median()
    elif aggregation == "mean":
        df["rolling_score"] = roller.mean()
    else:
        raise ValueError(f"지원하지 않는 aggregation: {aggregation}")

    return df


def probe_symbol(symbol: str, meta_entry: dict, rules: dict, deadline: Optional[float] = None) -> Optional[Path]:
    """단일 심볼 gap-aware 스캔. 새로 받아올 게 없으면 None 반환.

    deadline: time.monotonic() 기준 절대 마감 시각(초). None이면 무제한.
    """
    last_collected = storage.get_last_collected(category="scan", symbol=symbol)

    if last_collected:
        start_ym = _month_range(last_collected, last_collected)[0]
        y, m = map(int, start_ym.split("-"))
        start_ym = f"{y+1}-01" if m == 12 else f"{y}-{m+1:02d}"
    else:
        start_ym = _determine_start_month(symbol, meta_entry)
        if start_ym is None:
            logger.info(f"{symbol}: 아카이브에 데이터 없음, 스킵")
            return None

    end_ym = datetime.now(timezone.utc).strftime("%Y-%m")

    last_seen_month = meta_entry.get("last_seen_month")
    if meta_entry.get("status") == "DELISTED" and last_seen_month:
        end_ym = min(end_ym, last_seen_month)

    if start_ym > end_ym:
        return None

    months = _month_range(start_ym, end_ym)
    current_ym = datetime.now(timezone.utc).strftime("%Y-%m")

    new_frames = []
    last_processed_ym: Optional[str] = None

    for idx, ym in enumerate(months):
        if deadline is not None and time.monotonic() >= deadline:
            logger.info(f"{symbol} {ym}: 시간 예산 초과, 여기서 멈추고 다음 실행에서 이어서")
            break
        is_last_month = (idx == len(months) - 1)
        is_in_progress_month = (ym == current_ym)
        try:
            frame = _fetch_month_quote_volume(symbol, ym, interval=rules["scan_interval"])
            new_frames.append(frame)
            last_processed_ym = ym
        except archive_client.ArchiveError as e:
            if is_in_progress_month:
                logger.info(f"{symbol} {ym} 진행 중인 달, daily로 폴백: {e}")
                try:
                    frame = _fetch_month_quote_volume_via_daily(symbol, ym, interval=rules["scan_interval"])
                    new_frames.append(frame)
                except TransientMonthFailure as e2:
                    logger.warning(f"{symbol} {ym} 일시적 오류로 확정 불가, 다음 실행에서 재시도: {e2}")
                except (archive_client.ArchiveError, NoDataInMonthError) as e2:
                    logger.info(f"{symbol} {ym} 아직 데이터 없음 (다음 실행에 재시도): {e2}")
                break

            if is_last_month:
                logger.info(f"{symbol} {ym} 수집 실패 (스킵): {e}")
                break

            logger.info(f"{symbol} {ym} 월간 아카이브 없음, 일별 아카이브로 재시도: {e}")
            try:
                frame = _fetch_month_quote_volume_via_daily(symbol, ym, interval=rules["scan_interval"])
                new_frames.append(frame)
                last_processed_ym = ym
            except TransientMonthFailure as e2:
                # 404가 아닌 오류가 섞여 있어 "데이터 없음"으로 확정할 수 없다.
                # last_processed_ym을 이 달로 전진시키지 않고 여기서 멈춰서,
                # 다음 실행이 정확히 이 달부터 다시 시도하게 한다.
                logger.warning(f"{symbol} {ym} 일시적 오류로 확정 불가, 여기서 멈추고 다음 실행에서 재시도: {e2}")
                break
            except (archive_client.ArchiveError, NoDataInMonthError) as e2:
                logger.info(f"{symbol} {ym} 데이터 없음으로 확정, 건너뛰고 다음 달 계속: {e2}")
                last_processed_ym = ym
                continue

    if not new_frames:
        if last_processed_ym is not None:
            storage.update_last_collected(category="scan", symbol=symbol, value=last_processed_ym)
        return None

    new_df = pd.concat(new_frames, ignore_index=True)[["date", "close", "quote_volume"]]

    existing_df = storage.load_parquet(category="scan", filename=symbol)
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    else:
        combined = new_df.sort_values("date").reset_index(drop=True)

    full_df = _compute_rolling_score(combined, rules)
    path = storage.save_parquet(full_df, category="scan", filename=symbol)

    if last_processed_ym is not None:
        storage.update_last_collected(category="scan", symbol=symbol, value=last_processed_ym)

    return path


def run(max_workers: int = DEFAULT_MAX_WORKERS, deadline: Optional[float] = None) -> None:
    """bootstrap 이후 정기 실행되는 진입점. main.py에서 호출.

    deadline: time.monotonic() 기준 절대 마감 시각(초). None이면 무제한(기본값, CLI용).
    Lambda 콜드스타트처럼 실행시간이 제한된 환경에서, 전체 유니버스(수백 심볼) x
    수년치 히스토리 스캔이 타임아웃으로 강제 종료되는 걸 막기 위한 안전장치.
    마감이 지나면 남은 심볼은 아예 시작하지 않고, 이미 시작한 심볼도 진행 중이던
    달에서 멈춘다. manifest는 완료분만큼만 전진해 있으므로 다음 실행이 정확히
    이어서 계속한다.
    """
    storage.ensure_dirs()
    rules = load_rules()

    symbol_list_payload = storage.load_json(category="meta", filename="symbol_list")
    if symbol_list_payload is None:
        raise RuntimeError("symbol_list.json이 없다. symbol_universe.run()을 먼저 실행해야 한다.")

    symbols = symbol_list_payload["symbols"]
    total = len(symbols)
    updated = 0

    progress_lock = threading.Lock()
    processed_count = 0

    def _run_one(symbol: str, meta_entry: dict) -> Optional[Path]:
        nonlocal processed_count
        result = probe_symbol(symbol, meta_entry, rules, deadline=deadline)
        with progress_lock:
            processed_count += 1
            if processed_count % 50 == 0:
                logger.info(f"진행 {processed_count}/{total}")
        return result

    skipped_symbols = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for symbol, meta_entry in symbols.items():
            if deadline is not None and time.monotonic() >= deadline:
                skipped_symbols += 1
                continue
            futures[executor.submit(_run_one, symbol, meta_entry)] = symbol
        for future in as_completed(futures):
            symbol = futures[future]
            result_path = future.result()
            if result_path is not None:
                updated += 1

    # archive_start_month 캐시 등 meta_entry 갱신분은 여기서만 저장된다.
    # 마감 초과로 중간에 멈췄어도 이미 처리한 심볼의 캐시는 보존해서 다음 실행에 재활용한다.
    storage.save_json(symbol_list_payload, category="meta", filename="symbol_list")

    if skipped_symbols:
        logger.warning(f"시간 예산 초과로 {skipped_symbols}개 심볼은 이번 실행에서 아예 시작 못함(다음 실행에서 이어서 진행)")

    logger.info(f"완료. {total}개 중 {updated}개 심볼 갱신됨")