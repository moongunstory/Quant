"""
rest_client.py

data.binance.vision 아카이브가 아직 커버 못 하는 "최근 구간"을 REST API로
직접 채우기 위한 모듈. archive_client.py와 역할을 명확히 나눈다:
  - archive_client: 확정된 과거 구간, 대량/저비용
  - rest_client: 최근 구간(아카이브 미발행분), 소량/실시간성

예외 정책은 archive_client와 동일하게 ArchiveError 계층을 재사용한다
(호출부 입장에서 "출처가 아카이브냐 API냐"는 몰라도 되게 하기 위함).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from src.config.binance_api import FAPI_BASE, REST_ENDPOINTS, KLINE_COLUMNS
from src.collector.shared.archive_client import _request_get as _archive_request_get

logger = logging.getLogger(__name__)

REST_LIMIT_MAX = 1000  # /fapi/v1/* 엔드포인트 상한 (klines는 1500까지지만 1000으로 통일)

# /futures/data/* 엔드포인트(openInterestHist 등 metrics 5종)는 규격이 다르다:
#   - limit 최대 500 (초과 시 HTTP 400 — 실제로 이것 때문에 모든 metrics 폴백이 실패했었음)
#   - 최근 30일치 데이터만 제공 (그보다 오래된 구간은 REST로 채울 수 없음)
FUTURES_DATA_LIMIT_MAX = 500
FUTURES_DATA_LOOKBACK_DAYS = 30


def _rest_get(path: str, params: dict) -> list:
    """전역 속도 제한 + 429/418/5xx 백오프가 적용된 REST GET (archive_client와 공유)."""
    url = f"{FAPI_BASE}{path}"
    resp = _archive_request_get(url, params=params, timeout=30)
    return resp.json()


def _paginate(
    path: str, symbol: str, start_ms: int, end_ms: int,
    extra_params: Optional[dict] = None, limit: int = REST_LIMIT_MAX,
) -> list:
    """startTime/endTime/limit 페이징 공통 로직. limit은 엔드포인트별 상한에 맞춰야 한다."""
    extra_params = extra_params or {}
    rows: list = []
    cursor = start_ms

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
            **extra_params,
        }
        page = _rest_get(path, params)
        if not page:
            break
        rows.extend(page)

        last_ts = page[-1][0] if isinstance(page[-1], list) else page[-1].get("fundingTime") or page[-1].get("timestamp")
        if last_ts is None or last_ts <= cursor:
            break
        cursor = last_ts + 1

        if len(page) < limit:
            break

    return rows


def fetch_klines_recent(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """klines REST 폴백. 아카이브에 아직 없는 최신 구간(오늘~며칠 전)을 채울 때 사용."""
    rows = _paginate(REST_ENDPOINTS["klines"], symbol, start_ms, end_ms, extra_params={"interval": interval})
    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    numeric_cols = [c for c in df.columns if c not in ("open_time", "close_time")]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return df


def fetch_funding_rate_recent(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """fundingRate REST 폴백."""
    rows = _paginate(REST_ENDPOINTS["fundingRate"], symbol, start_ms, end_ms)
    if not rows:
        return pd.DataFrame(columns=["calc_time", "funding_interval_hours", "last_funding_rate"])
    df = pd.DataFrame(rows)
    # REST 응답 필드명(fundingTime, fundingRate)을 아카이브 컬럼명에 맞춤
    df = df.rename(columns={"fundingTime": "calc_time", "fundingRate": "last_funding_rate"})
    df["calc_time"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
    if "funding_interval_hours" not in df.columns:
        # REST 응답엔 이 필드가 없다 (아카이브 전용 필드). None을 넣으면 all-NA object
        # 컬럼이 돼서 아카이브 데이터(실제 값 4/8 등)와 concat할 때 FutureWarning과
        # dtype 오염이 생기므로 float NaN으로 맞춘다.
        df["funding_interval_hours"] = float("nan")
    return df[["calc_time", "funding_interval_hours", "last_funding_rate"]]


def fetch_metrics_recent(symbol: str, start_ms: int, end_ms: int, period: str = "5m") -> pd.DataFrame:
    """
    metrics REST 폴백. 아카이브 metrics 한 파일은 실제로 5개의 서로 다른 REST
    엔드포인트를 합성한 것임 (2026-07-09 fapi.binance.com 직접 조회로 필드명 실측 확인):
      - open_interest_hist              -> sum_open_interest, sum_open_interest_value
      - top_long_short_account_ratio    -> count_toptrader_long_short_ratio (= longShortRatio)
      - top_long_short_position_ratio   -> sum_toptrader_long_short_ratio (= longShortRatio)
      - global_long_short_account_ratio -> count_long_short_ratio (= longShortRatio)
      - taker_long_short_ratio          -> sum_taker_long_short_vol_ratio (= buySellRatio, symbol 필드 없음)
    5개 전부 timestamp(ms, period 단위로 정렬됨) 기준 outer merge.
    """
    columns_out = [
        "create_time", "symbol", "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
    ]

    endpoint_specs = {
        "oi": (REST_ENDPOINTS["open_interest_hist"], {
            "sumOpenInterest": "sum_open_interest",
            "sumOpenInterestValue": "sum_open_interest_value",
        }),
        "top_acct": (REST_ENDPOINTS["top_long_short_account_ratio"], {
            "longShortRatio": "count_toptrader_long_short_ratio",
        }),
        "top_pos": (REST_ENDPOINTS["top_long_short_position_ratio"], {
            "longShortRatio": "sum_toptrader_long_short_ratio",
        }),
        "global_acct": (REST_ENDPOINTS["global_long_short_account_ratio"], {
            "longShortRatio": "count_long_short_ratio",
        }),
        "taker": (REST_ENDPOINTS["taker_long_short_ratio"], {
            "buySellRatio": "sum_taker_long_short_vol_ratio",
        }),
    }

    # /futures/data/*는 최근 30일치만 제공한다. 그보다 오래된 startTime을 그대로
    # 보내는 건 의미가 없으므로 클램프한다 (오래된 구간은 아카이브만이 유일한 소스).
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    earliest_ms = now_ms - FUTURES_DATA_LOOKBACK_DAYS * 86_400_000
    if start_ms < earliest_ms:
        logger.info(
            "%s metrics REST: 요청 시작점이 30일 제한보다 오래됨 -> 최근 30일로 클램프 "
            "(그 이전 구간은 REST로 채울 수 없음)", symbol,
        )
        start_ms = earliest_ms
    if start_ms >= end_ms:
        return pd.DataFrame(columns=columns_out)

    frames: dict[str, pd.DataFrame] = {}
    for key, (path, rename_map) in endpoint_specs.items():
        rows = _paginate(
            path, symbol, start_ms, end_ms,
            extra_params={"period": period}, limit=FUTURES_DATA_LIMIT_MAX,
        )
        value_cols = list(rename_map.values())
        if not rows:
            frames[key] = pd.DataFrame(columns=["timestamp", *value_cols])
            continue
        df = pd.DataFrame(rows).rename(columns=rename_map)[["timestamp", *value_cols]]
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames[key] = df

    if frames["oi"].empty:
        return pd.DataFrame(columns=columns_out)

    merged = frames["oi"]
    for key in ("top_acct", "top_pos", "global_acct", "taker"):
        if not frames[key].empty:
            merged = merged.merge(frames[key], on="timestamp", how="outer")

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged = merged.rename(columns={"timestamp": "create_time"})
    merged["create_time"] = pd.to_datetime(merged["create_time"], unit="ms", utc=True)
    merged["symbol"] = symbol

    for col in columns_out:
        if col not in merged.columns:
            # 결측 지표 컬럼은 float NaN으로 (pd.NA는 object 컬럼이 돼 concat 시 dtype 문제 유발)
            merged[col] = float("nan")

    return merged[columns_out]
