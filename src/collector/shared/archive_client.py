"""
archive_client.py

바이낸스 공개 아카이브(data.binance.vision)와 통신하는 저수준 공용 모듈.
역할은 딱 세 가지로 한정한다:
  1. S3 디렉토리 리스팅
  2. zip 파일 다운로드
  3. zip 안의 csv를 파싱해서 DataFrame으로 변환

klines 외 데이터셋(premiumIndexKlines/metrics/fundingRate/bookDepth)도
동일한 3단계를 거치므로, 데이터셋별 분기는 DATASET_SPECS(binance_api.py)로
설정만 바꿔서 처리한다. "이 데이터로 뭘 계산할지"는 여전히 이 모듈의 관심사가 아니다.

premiumIndexKlines/metrics/fundingRate/bookDepth 관련 경로/컬럼/time_format은
2026-07-09 실측 검증 완료 (DATASET_SPECS 정의부인 binance_api.py 주석 참고).
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import requests
import pandas as pd

from src.collector.shared import throttle
from src.config.binance_api import (
    S3_LIST_ENDPOINT, ARCHIVE_BASE, S3_NS, DATASET_SPECS,
)

logger = logging.getLogger(__name__)

# 재시도 대상: 레이트리밋(429), IP 밴(418), 서버 오류(5xx)
RETRYABLE_STATUS = (418, 429, 500, 502, 503, 504)
MAX_RETRIES = 4
BACKOFF_CAP_SEC = 120  # Retry-After가 이보다 길면(장기 밴) 기다리는 대신 실패시킨다


# ---------------------------------------------------------------------------
# 커스텀 예외 계층 (기존과 동일)
# ---------------------------------------------------------------------------

class ArchiveError(Exception):
    """archive_client 관련 "예상 가능한" 예외의 공통 기반 클래스."""


class ArchiveNotFoundError(ArchiveError):
    """요청한 아카이브 파일이 존재하지 않음 (HTTP 404)."""


class ArchiveHTTPError(ArchiveError):
    """404가 아닌 HTTP 오류 (5xx, 403 등)."""


class ArchiveNetworkError(ArchiveError):
    """연결 실패, DNS 오류 등 네트워크 레벨 문제."""


class ArchiveTimeoutError(ArchiveError):
    """요청 타임아웃."""


class ArchiveCorruptedError(ArchiveError):
    """다운로드는 됐으나 zip/csv 파싱에 실패한 경우."""


# ---------------------------------------------------------------------------
# 공용 HTTP 요청 래퍼 (기존과 동일)
# ---------------------------------------------------------------------------

def _request_get(url: str, params: Optional[dict] = None, timeout: int = 30) -> requests.Response:
    """
    전역 속도 제한(throttle) + 재시도 백오프가 적용된 GET.
      - 429/418/5xx: Retry-After 헤더(있으면)와 지수 백오프 중 큰 값만큼 기다렸다 재시도
      - 타임아웃/연결 실패: 지수 백오프 후 재시도
      - 404: 재시도 의미 없음, 즉시 ArchiveNotFoundError
    MAX_RETRIES를 소진하면 마지막 오류를 그대로 던진다.
    """
    last_error: Optional[ArchiveError] = None

    for attempt in range(MAX_RETRIES + 1):
        throttle.wait(url)  # 아카이브/REST를 호스트로 구분해 각자의 간격 적용
        try:
            resp = requests.get(url, params=params, timeout=timeout)
        except requests.exceptions.Timeout:
            last_error = ArchiveTimeoutError(f"요청 타임아웃: {url}")
        except requests.exceptions.ConnectionError:
            last_error = ArchiveNetworkError(f"네트워크 연결 실패: {url}")
        except requests.exceptions.RequestException as e:
            raise ArchiveNetworkError(f"요청 실패: {url} ({e})") from e
        else:
            if resp.status_code == 404:
                raise ArchiveNotFoundError(f"아카이브 파일 없음 (404): {url}")

            if resp.status_code in RETRYABLE_STATUS:
                retry_after = resp.headers.get("Retry-After")
                wait_sec = max(float(retry_after) if retry_after else 0.0, float(2 ** attempt))
                last_error = ArchiveHTTPError(f"HTTP 오류 {resp.status_code}: {url}")
                if wait_sec > BACKOFF_CAP_SEC:
                    # 장기 밴(418 등). 기다려봐야 이번 실행 안에 안 풀린다.
                    logger.error("HTTP %d, Retry-After=%.0fs (한도 %ds 초과) -> 중단: %s",
                                 resp.status_code, wait_sec, BACKOFF_CAP_SEC, url)
                    raise last_error
                if attempt < MAX_RETRIES:
                    logger.warning("HTTP %d, %.0f초 후 재시도 (%d/%d): %s",
                                   resp.status_code, wait_sec, attempt + 1, MAX_RETRIES, url)
                    time.sleep(wait_sec)
                continue

            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                raise ArchiveHTTPError(f"HTTP 오류 {resp.status_code}: {url}") from e
            return resp

        # 타임아웃/연결 실패 경로: 지수 백오프 후 재시도
        if attempt < MAX_RETRIES:
            wait_sec = float(2 ** attempt)
            logger.warning("%s, %.0f초 후 재시도 (%d/%d)", last_error, wait_sec, attempt + 1, MAX_RETRIES)
            time.sleep(wait_sec)

    raise last_error


# ---------------------------------------------------------------------------
# S3 디렉토리 리스팅
# ---------------------------------------------------------------------------

@dataclass
class ListingPage:
    prefixes: list[str]
    keys: list[str]
    is_truncated: bool
    next_marker: Optional[str]


def _list_s3_page(prefix: str, delimiter: str = "/", marker: str = "") -> ListingPage:
    params = {"prefix": prefix, "delimiter": delimiter}
    if marker:
        params["marker"] = marker

    resp = _request_get(S3_LIST_ENDPOINT, params=params, timeout=30)
    root = ET.fromstring(resp.content)

    prefixes = [el.find(f"{S3_NS}Prefix").text for el in root.findall(f"{S3_NS}CommonPrefixes")]
    keys = [el.find(f"{S3_NS}Key").text for el in root.findall(f"{S3_NS}Contents")]
    is_truncated_el = root.find(f"{S3_NS}IsTruncated")
    is_truncated = (is_truncated_el is not None and is_truncated_el.text == "true")

    next_marker = None
    if is_truncated:
        nm_el = root.find(f"{S3_NS}NextMarker")
        if nm_el is not None:
            next_marker = nm_el.text
        elif prefixes:
            next_marker = prefixes[-1]
        elif keys:
            next_marker = keys[-1]

    return ListingPage(prefixes=prefixes, keys=keys, is_truncated=is_truncated, next_marker=next_marker)


def list_symbols_with_archive(market: str = "um", interval_kind: str = "monthly") -> list[str]:
    """kline 기준 아카이브에 존재하는 전체 심볼 목록 (기존 동일)."""
    prefix = f"data/futures/{market}/{interval_kind}/klines/"
    symbols: list[str] = []
    marker = ""
    while True:
        page = _list_s3_page(prefix=prefix, delimiter="/", marker=marker)
        for p in page.prefixes:
            symbols.append(p.rstrip("/").split("/")[-1])
        if not page.is_truncated or not page.next_marker:
            break
        marker = page.next_marker
    return sorted(set(symbols))


def _archive_prefix(dataset: str, symbol: str, kind: str, market: str, interval: Optional[str]) -> str:
    spec = DATASET_SPECS[dataset]
    segment = spec["archive_segment"]
    if spec["has_interval_folder"]:
        if interval is None:
            raise ValueError(f"'{dataset}'는 interval이 반드시 필요합니다")
        return f"data/futures/{market}/{kind}/{segment}/{symbol}/{interval}/"
    return f"data/futures/{market}/{kind}/{segment}/{symbol}/"


def list_available_periods(
    dataset: str, symbol: str, market: str = "um",
    interval: Optional[str] = None, kind: str = "monthly",
) -> list[str]:
    """
    특정 데이터셋/심볼에 대해 아카이브가 존재하는 기간 목록.
    kind="monthly"면 "YYYY-MM", kind="daily"면 "YYYY-MM-DD" 형태로 반환.
    """
    prefix = _archive_prefix(dataset, symbol, kind, market, interval)
    periods: list[str] = []
    marker = ""
    while True:
        page = _list_s3_page(prefix=prefix, delimiter="/", marker=marker)
        for key in page.keys:
            fname = key.split("/")[-1]
            if not fname.endswith(".zip"):
                continue
            stem = fname[: -len(".zip")]
            # 예: "BTCUSDT-1h-2023-05" -> "2023-05" / "BTCUSDT-fundingRate-2023-05-01" -> "2023-05-01"
            parts = stem.split("-")
            if kind == "monthly":
                period = "-".join(parts[-2:])
            else:  # daily
                period = "-".join(parts[-3:])
            periods.append(period)
        if not page.is_truncated or not page.next_marker:
            break
        marker = page.next_marker
    return sorted(set(periods))


# 하위호환: 기존 호출부(collect_target_symbols 등)가 쓰던 이름 유지
def list_available_months(symbol: str, market: str = "um", interval: str = "1d") -> list[str]:
    return list_available_periods(dataset="klines", symbol=symbol, market=market, interval=interval, kind="monthly")


# ---------------------------------------------------------------------------
# zip 다운로드 + 파싱 (제네릭)
# ---------------------------------------------------------------------------

def download_archive_zip(
    dataset: str, symbol: str, market: str = "um",
    interval: Optional[str] = None,
    year_month: Optional[str] = None, date: Optional[str] = None,
) -> bytes:
    """
    dataset(klines/premiumIndexKlines/metrics/fundingRate/bookDepth) 공용 zip 다운로드.
    year_month 지정 시 monthly, date 지정 시 daily.
    """
    spec = DATASET_SPECS[dataset]
    segment = spec["archive_segment"]

    if spec["has_interval_folder"] and interval is None:
        raise ValueError(f"'{dataset}'는 interval이 반드시 필요합니다")

    interval_part = f"/{interval}" if spec["has_interval_folder"] else ""
    label_part = f"-{interval}" if spec["has_interval_folder"] else ""
    filename_suffix = spec.get("archive_filename_suffix", "")

    if year_month:
        url = (
            f"{ARCHIVE_BASE}/futures/{market}/monthly/"
            f"{segment}/{symbol}{interval_part}/"
            f"{symbol}{label_part}{filename_suffix}-{year_month}.zip"
        )
    elif date:
        url = (
            f"{ARCHIVE_BASE}/futures/{market}/daily/"
            f"{segment}/{symbol}{interval_part}/"
            f"{symbol}{label_part}{filename_suffix}-{date}.zip"
        )
    else:
        raise ValueError("year_month 또는 date 중 하나는 반드시 지정해야 한다")

    resp = _request_get(url, timeout=60)
    return resp.content


def parse_archive_zip(dataset: str, zip_bytes: bytes) -> pd.DataFrame:
    """dataset 스펙에 맞춰 zip 바이트를 DataFrame으로 변환."""
    spec = DATASET_SPECS[dataset]
    columns = spec["columns"]

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = zf.namelist()
            if not names:
                raise ArchiveCorruptedError("zip 안에 파일이 없음 (빈 zip)")
            with zf.open(names[0]) as f:
                raw_bytes = f.read()

            first_line = raw_bytes.split(b"\n", 1)[0].decode("utf-8", errors="ignore").strip()
            # 첫 컬럼명이 실제 스펙 첫 컬럼명과 같으면 헤더가 있는 것으로 판단
            has_header = first_line.split(",")[0].strip().lower() == columns[0].lower()

            df = pd.read_csv(
                io.BytesIO(raw_bytes),
                header=0 if has_header else None,
                names=columns,
            )
    except zipfile.BadZipFile as e:
        raise ArchiveCorruptedError(f"zip 압축 해제 실패: {e}") from e
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise ArchiveCorruptedError(f"csv 파싱 실패: {e}") from e

    time_col = spec["time_col"]
    time_format = spec["time_format"]

    if time_format == "ms":
        df[time_col] = _parse_ms_or_datetime(df[time_col])
    elif time_format == "datetime":
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
    else:
        raise ValueError(f"Unknown time_format: {time_format}")

    if "close_time" in df.columns:
        df["close_time"] = _parse_ms_or_datetime(df["close_time"])

    return df


def _parse_ms_or_datetime(series: pd.Series) -> pd.Series:
    """time_format="ms"로 선언된 컬럼을 파싱한다. 실측(2026-07-09) 결과 klines/
    premiumIndexKlines/fundingRate는 실제로 ms epoch가 맞다. 다만 DATASET_SPECS의
    time_format 선언이 실측과 어긋나는 경우 pd.to_datetime(unit="ms")가 바로 죽는
    대신 datetime 문자열로 안전하게 재시도하는 방어 로직으로 남겨둔다
    (스펙이 또 틀렸을 때 전체 수집이 죽는 대신 로그만 남기고 넘어가게 하기 위함)."""
    sample = series.dropna().iloc[0] if not series.dropna().empty else None
    if sample is not None and isinstance(sample, str) and not sample.strip().lstrip("-").isdigit():
        return pd.to_datetime(series, utc=True)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() and not series.isna().any():
        # 숫자로 변환 안 되는 값(=datetime 문자열)이 섞여 있음 -> datetime으로 파싱
        return pd.to_datetime(series, utc=True)

    return pd.to_datetime(numeric, unit="ms", utc=True)


def fetch_archive_month(dataset: str, symbol: str, year_month: str, market: str = "um", interval: Optional[str] = None) -> pd.DataFrame:
    raw = download_archive_zip(dataset=dataset, symbol=symbol, market=market, interval=interval, year_month=year_month)
    return parse_archive_zip(dataset, raw)


def fetch_archive_day(dataset: str, symbol: str, date: str, market: str = "um", interval: Optional[str] = None) -> pd.DataFrame:
    raw = download_archive_zip(dataset=dataset, symbol=symbol, market=market, interval=interval, date=date)
    return parse_archive_zip(dataset, raw)


# --- 하위호환 래퍼 (기존 호출부가 그대로 동작하도록) ---

def fetch_kline_month(symbol: str, interval: str, year_month: str, market: str = "um") -> pd.DataFrame:
    return fetch_archive_month("klines", symbol=symbol, year_month=year_month, market=market, interval=interval)


def fetch_kline_day(symbol: str, interval: str, date: str, market: str = "um") -> pd.DataFrame:
    return fetch_archive_day("klines", symbol=symbol, date=date, market=market, interval=interval)