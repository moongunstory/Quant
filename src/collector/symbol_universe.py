"""
symbol_universe.py

전체 USDS-M 무기한 선물 심볼 목록을 확보한다.
"현재 상장 중인 심볼"과 "과거에 상장됐다가 폐지된 심볼"을 합쳐서
data/strategy/meta/symbol_list.json 하나로 저장한다.

이 결과물은 시세 데이터가 아니라 메타데이터이기 때문에
scan/universe_snapshots/processed 와는 별도 경로(meta)에 둔다.

실행 성격: 자주 돌릴 필요 없음. 아카이브에 신규 상장/폐지가 반영된 걸
확인하고 싶을 때 수동/저빈도로 재실행하면 된다 (예: 월 1회, universe_builder 직전).

정보의 한계:
  - REST exchangeInfo는 "현재" 상태만 준다 (상장폐지된 심볼은 응답에서 아예 빠짐).
  - 아카이브 S3 리스팅은 "한 번이라도 데이터가 쌓인 심볼 전체"를 준다 (생존/폐지 무관).
  - 따라서 "아카이브에는 있는데 현재 exchangeInfo에는 없는 심볼" = 상장폐지로 간주한다.
  - 정확한 상장폐지 "일자"는 이 REST 응답만으로는 알 수 없고,
    아카이브에 존재하는 마지막 연월(last_seen_month)로 근사한다.
    엄밀한 상장폐지일이 필요해지면 나중에 별도 소스를 찾아야 한다 (지금은 근사치로 충분).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from src.collector.shared import archive_client, storage
from src.config.binance_api import FAPI_EXCHANGE_INFO_URL
from src.config.collection_rules import CONTRACT_TYPE_FILTER, QUOTE_ASSET_FILTER
from src.config.symbol_patterns import QUARTERLY_PATTERN, SETTLED_SUFFIX

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# delivery(분기별 만기) 계약 판별
# ---------------------------------------------------------------------------
#
# exchangeInfo가 주는 contractType 필드가 "진짜" 판별 기준이다 (하드코딩이 아니라
# 바이낸스 API 응답 자체). 문제는 이 필드가 "현재 상장 중인" 심볼에만 존재한다는
# 것 - 상장폐지된(만기 지난) delivery 계약은 exchangeInfo에서 아예 사라지므로,
# S3 아카이브 리스팅에서 발견한 과거 심볼에 대해서는 contractType을 조회할 방법이
# 없다. 그래서 "심볼명이 {pair}_{YYMMDD} 형태"라는 바이낸스의 명명 규칙에 기대어
# 추론할 수밖에 없다.
#
# 이 추론이 "그냥 하드코딩된 정규식 찍어맞추기"가 되지 않도록, 매 실행마다
# *현재* exchangeInfo에 남아있는 delivery 계약(contractType != PERPETUAL)을 대상으로
# "심볼명 == {pair}_{실제 deliveryDate를 YYMMDD로 변환한 값}"이 100% 성립하는지
# 라이브 데이터로 검증한다 (_verify_delivery_naming_convention). 하나라도 어긋나면
# 바이낸스가 명명 규칙을 바꿨다는 뜻이므로, 조용히 잘못 분류하는 대신 그 자리에서
# 즉시 예외를 던진다. 즉 이 패턴은 "믿고 쓰는 가정"이 아니라 "매번 재검증되는 가정"이다.
# 패턴 상수는 src.config.symbol_patterns에 정의되어 있다.


def _is_delivery_contract(symbol: str) -> bool:
    return bool(QUARTERLY_PATTERN.search(symbol))


def _verify_delivery_naming_convention(raw_symbols: list[dict]) -> None:
    """
    "delivery 계약 심볼명은 {pair}_{deliveryDate를 YYMMDD로 표현한 값}이다"라는
    가정을, 현재 exchangeInfo에 남아있는 delivery 계약 전체를 대상으로 검증한다.
    상장폐지된 delivery 계약을 걸러낼 때 쓰는 _is_delivery_contract 정규식이 아직도
    유효한 가정인지 매 실행마다 실측하는 안전장치. 하나라도 안 맞으면 침묵하지 않고
    즉시 실패시켜서 (바이낸스가 규칙을 바꿨다는 신호이므로) 코드 버그로 조기 발견되게 한다.
    """
    checked = 0
    for s in raw_symbols:
        contract_type = s.get("contractType")
        delivery_ms = s.get("deliveryDate")
        
        # 1. contractType이 PERPETUAL(무기한) 계열이면 무조건 패스
        if contract_type in ["PERPETUAL", "TRADIFI_PERPETUAL"]:
            continue
            
        # 2. 안전장치로 deliveryDate가 없는 것도 패스
        if not delivery_ms:
            continue

        symbol = s["symbol"]
        pair = s.get("pair", symbol)
        expected_suffix = datetime.fromtimestamp(delivery_ms / 1000, tz=timezone.utc).strftime("%y%m%d")
        expected_symbol = f"{pair}_{expected_suffix}"

        if symbol != expected_symbol or not _is_delivery_contract(symbol):
            raise RuntimeError(
                f"delivery 계약 명명 규칙 검증 실패: symbol={symbol!r}, contractType={contract_type!r}, "
                f"기대한 형태={expected_symbol!r}. 바이낸스가 명명 규칙을 바꿨을 수 있으니 "
                f"_is_delivery_contract/QUARTERLY_PATTERN(src.config.symbol_patterns)을 다시 점검해야 한다."
            )
        checked += 1

    if checked == 0:
        logger.warning(
            "현재 exchangeInfo에 검증 가능한 delivery 계약이 하나도 없음 "
            "(과거 아카이브 심볼 필터링의 전제를 이번 실행에서는 재검증하지 못함)"
        )
    else:
        logger.info(f"delivery 계약 명명 규칙 검증 통과 ({checked}개 계약으로 확인)")


def fetch_exchange_info_raw() -> list[dict]:
    """exchangeInfo 원본 symbols 배열 그대로 반환 (필터링 없음, 검증/파생 용도)."""
    resp = requests.get(FAPI_EXCHANGE_INFO_URL, timeout=30)
    resp.raise_for_status()
    return resp.json().get("symbols", [])


def fetch_current_symbols(raw_symbols: list[dict]) -> dict[str, dict]:
    """
    현재 거래 가능한(또는 일시중단 포함) USDT 무기한 심볼과 상태를 가져온다.
    반환: {symbol: {"status": str, "onboard_date": str|None}}
    """
    result: dict[str, dict] = {}
    for s in raw_symbols:
        if s.get("contractType") != CONTRACT_TYPE_FILTER:
            continue
        if s.get("quoteAsset") != QUOTE_ASSET_FILTER:
            continue

        symbol = s["symbol"]
        onboard_ms = s.get("onboardDate")
        onboard_date = None
        if onboard_ms:
            onboard_date = datetime.fromtimestamp(onboard_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        result[symbol] = {
            "status": s.get("status"),  # 예: TRADING, PENDING_TRADING, BREAK
            "onboard_date": onboard_date,
        }

    return result


def fetch_archive_symbols() -> list[str]:
    """
    아카이브에 kline 데이터가 한 번이라도 존재하는 전체 심볼 중
    'USDT 무기한'에 해당하는 것만 남긴다.

    분류 규칙:
      1) 분기 계약(_YYMMDD 접미사)은 제외.
      2) SETTLED/_SETTLED 접미사는 폐지 후 정산 표시일 뿐이므로 떼어내고
         원래 quote asset을 판단한다 (예: AERGOUSDTSETTLED -> AERGOUSDT).
      3) 위 처리 후 이름이 USDT로 끝나는 것만 채택.

    분류가 안 되는 심볼(BUSD/USDC/기타)이 나오면 그건 정상 -
    다른 마진 자산이거나 예외적인 심볼(ETHBTC, BTCUSD1 등)이므로 걸러낸다.
    단, 이 카테고리에 예상 못한 게 섞여 있을 수 있으니 로그로 남겨서
    매 실행마다 사람이 확인할 수 있게 한다.
    """
    all_symbols = archive_client.list_symbols_with_archive(market="um", interval_kind="monthly")

    usdt_perp = []
    excluded_other = []  # BUSD/USDC/분기 계약이 아닌데도 USDT로 안 끝나는 것들

    for s in all_symbols:
        if _is_delivery_contract(s):
            continue

        base = SETTLED_SUFFIX.sub("", s)

        if base.endswith("USDT"):
            usdt_perp.append(s)
        elif base.endswith("BUSD") or base.endswith("USDC"):
            continue  # 정상적으로 걸러지는 케이스
        else:
            excluded_other.append(s)

    if excluded_other:
        logger.warning(
            f"USDT/BUSD/USDC/분기 어디에도 안 걸리는 심볼 {len(excluded_other)}개 발견 "
            f"(수동 확인 필요): {excluded_other}"
        )

    return usdt_perp


def build_symbol_universe() -> dict[str, dict]:
    """
    현재 심볼 + 아카이브 심볼을 합쳐 최종 메타데이터를 만든다.

    각 심볼 레코드:
      {
        "status": "TRADING" | "DELISTED" | "PENDING_TRADING" | "BREAK" ...
        "onboard_date": "YYYY-MM-DD" | None,   # exchangeInfo 기준, 없으면 미상
        "last_seen_month": "YYYY-MM" | None,   # DELISTED인 경우 아카이브상 마지막 존재 월
        "in_current_exchange_info": bool,
        "in_archive": bool,
      }
    """
    raw_symbols = fetch_exchange_info_raw()
    _verify_delivery_naming_convention(raw_symbols)

    current = fetch_current_symbols(raw_symbols)
    archive_symbols = set(fetch_archive_symbols())

    universe: dict[str, dict] = {}

    # 1) 현재 exchangeInfo에 있는 심볼부터 채운다.
    for symbol, meta in current.items():
        universe[symbol] = {
            "status": meta["status"],
            "onboard_date": meta["onboard_date"],
            "last_seen_month": None,
            "in_current_exchange_info": True,
            "in_archive": symbol in archive_symbols,
        }

    # 2) 아카이브에는 있지만 현재 exchangeInfo에 없는 심볼 = 상장폐지로 간주.
    delisted = archive_symbols - set(current.keys())
    for symbol in delisted:
        last_months = archive_client.list_available_months(symbol, market="um")
        universe[symbol] = {
            "status": "DELISTED",
            "onboard_date": None,  # exchangeInfo에 없어 onboard_date 확보 불가. 필요 시 최초 월로 근사 가능.
            "last_seen_month": last_months[-1] if last_months else None,
            "in_current_exchange_info": False,
            "in_archive": True,
        }

    return universe


def run() -> None:
    """bootstrap 진입점. main.py에서 이 함수를 호출한다."""
    storage.ensure_dirs()
    universe = build_symbol_universe()

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_symbols": len(universe),
        "symbols": universe,
    }

    path = storage.save_json(payload, category="meta", filename="symbol_list")
    print(f"[symbol_universe] {len(universe)}개 심볼 저장 완료 -> {path}")


