"""
binance_api.py

바이낸스 "외부 API 자체"에 관한 상수 (엔드포인트, 응답 스키마).
어떤 수집기가 쓰는지와 무관하게, 바이낸스 쪽 계약이 바뀌면 여기만 고치면 된다.
수집 정책(필터링/속도 제한 등)은 collection_rules.py로 분리한다.

2026-07-09 실측 검증 완료 (BTCUSDT / 0GUSDT 샘플 다운로드 기준).
premiumIndexKlines/metrics/fundingRate/bookDepth 스펙은 전부 추측값이었고
아래 두 가지가 실제와 달라서 수집 실패/스킵을 유발하고 있었다:
  - metrics: create_time이 ms epoch가 아니라 'YYYY-MM-DD HH:MM:SS' 문자열이었음
    (columns 자체는 정확했음)
  - fundingRate: calc_time이 datetime 문자열이 아니라 ms epoch였음. 게다가
    실제 파일명이 "{symbol}-fundingRate-{YYYY-MM}.zip"인데 filename_suffix가
    비어있어서 "{symbol}-{YYYY-MM}.zip"으로 요청 -> 항상 404 -> 모든 심볼의
    fundingRate가 처음부터 전부 스킵되고 있었음.
premiumIndexKlines/bookDepth는 기존 추측값이 실측과 일치함 (변경 없음).
"""

# --- REST API 베이스 ---
FAPI_BASE = "https://fapi.binance.com"
FAPI_EXCHANGE_INFO_URL = f"{FAPI_BASE}/fapi/v1/exchangeInfo"

# --- 공개 아카이브 (data.binance.vision) ---
S3_LIST_ENDPOINT = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
ARCHIVE_BASE = "https://data.binance.vision/data"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

# ---------------------------------------------------------------------------
# 데이터셋별 컬럼 스키마
# ---------------------------------------------------------------------------

# USDS-M 무기한 선물 kline 컬럼 (바이낸스 공식 스펙, 헤더 없는 csv)
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

# premiumIndexKlines: kline과 동일한 12컬럼 구조 (open/high/low/close가 premium index 값)
PREMIUM_INDEX_KLINE_COLUMNS = KLINE_COLUMNS

# metrics: 실측 검증 완료 (2026-07-09, BTCUSDT daily 샘플)
METRICS_COLUMNS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

# fundingRate: 실측 검증 완료 (2026-07-09, 0GUSDT monthly 샘플). 컬럼명은 맞았음.
FUNDING_RATE_COLUMNS = [
    "calc_time",
    "funding_interval_hours",
    "last_funding_rate",
]

# bookDepth: 실측 검증 완료 (2026-07-09, BTCUSDT daily 샘플)
BOOK_DEPTH_COLUMNS = [
    "timestamp",
    "percentage",
    "depth",
    "notional",
]

# ---------------------------------------------------------------------------
# 데이터셋 스펙: 아카이브 경로 구성 + 시간 컬럼 + REST 폴백 정보를 한곳에서 관리
# ---------------------------------------------------------------------------

DATASET_SPECS = {
    "klines": {
        "archive_segment": "klines",
        "has_interval_folder": True,
        "archive_granularity": "monthly_with_daily_fallback",
        "columns": KLINE_COLUMNS,
        "time_col": "open_time",
        "time_format": "ms",
    },

    "premiumIndexKlines": {
        "archive_segment": "premiumIndexKlines",
        "has_interval_folder": True,
        "archive_granularity": "monthly_with_daily_fallback",
        "columns": PREMIUM_INDEX_KLINE_COLUMNS,
        "time_col": "open_time",
        "time_format": "ms",
    },

    "metrics": {
        "archive_segment": "metrics",
        "archive_filename_suffix": "-metrics",
        "has_interval_folder": False,
        "archive_granularity": "daily_only",
        "columns": METRICS_COLUMNS,
        "time_col": "create_time",
        "time_format": "datetime",  # 실측: 'YYYY-MM-DD HH:MM:SS' 문자열, ms 아님
    },

    "fundingRate": {
        "archive_segment": "fundingRate",
        "archive_filename_suffix": "-fundingRate",  # 실측: 파일명에 세그먼트명이 들어감 (없으면 항상 404)
        "has_interval_folder": False,
        "archive_granularity": "monthly_only",
        "columns": FUNDING_RATE_COLUMNS,
        "time_col": "calc_time",
        "time_format": "ms",  # 실측: ms epoch, datetime 문자열 아님
    },

    "bookDepth": {
        "archive_segment": "bookDepth",
        "archive_filename_suffix": "-bookDepth",  # 실측: 파일명에 세그먼트명이 들어감 (없으면 항상 404). metrics/fundingRate와 동일 패턴.
        "has_interval_folder": False,
        "archive_granularity": "daily_only",
        "columns": BOOK_DEPTH_COLUMNS,
        "time_col": "timestamp",
        "time_format": "datetime",
    },
}

DATASETS_DEFAULT = ["klines", "premiumIndexKlines", "metrics", "fundingRate"]
DATASETS_OPTIONAL = ["bookDepth"]  # "spot"은 market이 다르므로 별도 모듈에서 처리 권장

# --- REST 엔드포인트 (최신 구간 폴백용) ---
REST_ENDPOINTS = {
    "klines": "/fapi/v1/klines",
    "premiumIndexKlines": "/fapi/v1/premiumIndexKlines",
    "fundingRate": "/fapi/v1/fundingRate",
    "open_interest_hist": "/futures/data/openInterestHist",
    "top_long_short_account_ratio": "/futures/data/topLongShortAccountRatio",
    "top_long_short_position_ratio": "/futures/data/topLongShortPositionRatio",
    "global_long_short_account_ratio": "/futures/data/globalLongShortAccountRatio",
    "taker_long_short_ratio": "/futures/data/takerlongshortRatio",
}

