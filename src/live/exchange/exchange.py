"""exchange — 실거래(real) 모드에서 쓰는 거래소 조회/주문 래퍼.

trade/client.py(클라이언트 생성) + trade/order.py(시장가 주문) 위에, orders.py 가
필요로 하는 '현재 포지션 조회 / 마크가격 / 수량 반올림 / 주문 전송'을 얹는다.

주의: 페이퍼 모드에서는 이 모듈을 전혀 호출하지 않는다(orders.py 가 positions.json
을 현재 포지션으로 씀). 실거래 조회 메서드는 Binance USDT-M 선물 SDK 응답 형태에
의존하므로, 실계좌/테스트넷 연결 후 실제 응답으로 한 번 검증이 필요하다(가상매매로
충분히 검증한 뒤 실거래 전환 — PLAN Phase 2).
"""
from __future__ import annotations

import logging
from decimal import Decimal, ROUND_DOWN

from src.config.backtest_settings import SETTINGS
from src.live.exchange.client import get_client
from src.live.exchange.order import place_market_order as _place_market_order

log = logging.getLogger("quant.live.exchange")

# 심볼별 LOT step 캐시(거래소 exchangeInfo 에서 1회 조회). 미조회 시 기본 step.
_LOT_STEP_CACHE: dict[str, float] = {}
_DEFAULT_QTY_STEP = 0.001
_LOT_STEP_CACHE_LOADED = False


def _ensure_lot_step_cache(mode="real"):
    """exchangeInfo 를 1회 조회해 심볼별 LOT_SIZE stepSize 로 _LOT_STEP_CACHE 를 채운다.
    이전엔 이 캐시가 절대 채워지지 않아 모든 심볼이 기본 step(0.001)으로 반올림됐고,
    그 결과 정수 단위(step=1)만 허용하는 심볼(예: 1000BONKUSDT)에 소수점 수량을 보내
    바이낸스가 -1111 'Precision is over the maximum defined for this asset.' 로 거부했다."""
    global _LOT_STEP_CACHE_LOADED
    if _LOT_STEP_CACHE_LOADED:
        return
    client = get_client(mode)
    resp = client.rest_api.exchange_information()
    data = resp.data() if hasattr(resp, "data") else resp
    n = 0
    for s in getattr(data, "symbols", None) or []:
        for f in getattr(s, "filters", None) or []:
            if getattr(f, "filter_type", None) == "LOT_SIZE":
                try:
                    _LOT_STEP_CACHE[s.symbol] = float(f.step_size)
                    n += 1
                except (TypeError, ValueError):
                    continue
                break
    log.info("exchangeInfo LOT_SIZE step 캐시 로드: %d개 심볼", n)
    _LOT_STEP_CACHE_LOADED = True


def _raw_position_information(mode="real"):
    """position_information_v2 원본 리스트(dict) 그대로. get_open_positions 와
    detect_position_mode 가 공유(같은 호출 1회 재사용은 호출자 몫)."""
    client = get_client(mode)
    resp = client.rest_api.position_information_v2()
    data = resp.data() if hasattr(resp, "data") else resp
    log.debug("position_information_v2 raw type=%s len=%s",
              type(data), (len(data) if hasattr(data, "__len__") else "?"))
    return data or []


def get_account_equity(mode="real"):
    """실제/테스트넷 계좌의 현재 총 자산(USD 환산, 미실현손익 포함)을 조회.

    실계좌·테스트넷 모두 같은 엔드포인트를 쓰므로 그대로 호환된다. orders.py 가
    이 값을 그날의 book_aum_usd(주문 수량 계산 기준 총액) 로 써서, '가상의 10만불'
    같은 고정값이 실제 계좌 크기와 안 맞아 증거금 부족(-2019)이 나던 문제를 없앤다.
    margin_balance(총 지갑잔고+미실현손익)를 우선 쓰고, 없으면 wallet_balance로 폴백."""
    client = get_client(mode)
    resp = client.rest_api.account_information_v3()
    data = resp.data() if hasattr(resp, "data") else resp
    for attr in ("total_margin_balance", "total_wallet_balance"):
        val = getattr(data, attr, None)
        if val is not None:
            try:
                equity = float(val)
                log.info("get_account_equity(mode=%s): %s=%s", mode, attr, equity)
                return equity
            except (TypeError, ValueError):
                continue
    raise RuntimeError("account_information_v3 응답에서 총자산 필드를 찾을 수 없음")


def get_open_positions(mode="real"):
    """{symbol: signed_qty} (롱 +, 숏 -). 포지션 없으면 {}."""
    out = {}
    for p in _raw_position_information(mode):
        amt = float(p.position_amt)
        if amt != 0.0:
            out[p.symbol] = amt
    return out


def detect_position_mode(mode="real"):
    """계정의 포지션 모드를 '이미 쓰는' position_information_v2 응답으로 판별.
    (새 SDK 메서드 없이 판별 — positionSide 필드 기준)
      원웨이(one-way) : 각 심볼 한 줄, positionSide == "BOTH"
      헷지(hedge)     : 심볼당 LONG/SHORT 두 줄
    LONG/SHORT 가 하나라도 보이면 'hedge', 아니면 'one_way'. 응답이 비면 'unknown'.
    """
    rows = _raw_position_information(mode)
    if not rows:
        return "unknown"
    sides = {str(getattr(p, "position_side", "") or "").upper() for p in rows}
    if "LONG" in sides or "SHORT" in sides:
        return "hedge"
    return "one_way"


def assert_one_way_mode(mode="real"):
    """실매매 프리플라이트: 원웨이 모드가 아니면 명확한 에러로 주문을 막는다(fail-closed).
    이 프로그램 주문은 positionSide="BOTH" 라 헷지 모드에선 전량 실패하므로, 사고 전에
    사람이 알아채게 한다. 'unknown'(응답 비어 판별 불가)도 안전하게 막는다."""
    pm = detect_position_mode(mode)
    log.info("assert_one_way_mode: detected=%s", pm)
    if pm != "one_way":
        raise RuntimeError(
            f"계정 포지션 모드가 '{pm}' 입니다. 이 프로그램은 원웨이(One-way) 모드 전용"
            " (주문 positionSide='BOTH')입니다. 바이낸스 선물 설정에서 'One-way Mode'로"
            " 바꾼 뒤(보유 포지션 없을 때만 변경 가능) 다시 실행하세요."
            " (검증 없이 진행하면 모든 주문이 거부됩니다.)"
        )


def get_mark_prices(mode="real"):
    """{symbol: mark_price}."""
    client = get_client(mode)
    resp = client.rest_api.mark_price()
    data = resp.data() if hasattr(resp, "data") else resp
    if hasattr(data, "actual_instance"):
        # symbol 없이 호출 시 mark_price() 는 oneOf 래퍼(MarkPriceResponse)를 돌려준다.
        # 실제 리스트는 .actual_instance 안에 있음(안 풀면 BaseModel.__iter__ 가
        # (필드명, 값) 튜플을 내놔서 전부 조용히 스킵됨).
        data = data.actual_instance
    log.debug("mark_price raw type=%s (after unwrap) len=%s",
              type(data), (len(data) if hasattr(data, "__len__") else "?"))
    out = {}
    skipped = 0
    for row in (data or []):
        try:
            out[row.symbol] = float(row.mark_price)
        except (AttributeError, TypeError, ValueError) as exc:
            skipped += 1
            log.debug("mark_price row skipped (%s): %r", exc, row)
            continue
    log.info("get_mark_prices: %d개 파싱, %d개 스킵", len(out), skipped)
    return out


def round_quantity(symbol, qty, mode="real"):
    """심볼 LOT step 으로 내림 반올림. exchangeInfo 캐시가 비어 있으면 먼저 채운다
    (전엔 캐시가 절대 안 채워져서 모든 심볼에 기본 step=0.001 을 써 정수 전용 심볼에서
    -1111 Precision 에러가 났다). Decimal 로 계산해 float 이진오차로 인한 미세 초과
    (예: step=0.1 인데 결과가 1.2000000002 로 나와 다시 정밀도 초과되는 것)를 막는다."""
    if not _LOT_STEP_CACHE_LOADED:
        try:
            _ensure_lot_step_cache(mode)
        except Exception:
            log.exception("exchangeInfo LOT_SIZE 캐시 로드 실패 -- 기본 step 사용")
    step = _LOT_STEP_CACHE.get(symbol, _DEFAULT_QTY_STEP)
    if step <= 0:
        return float(qty)
    step_d = Decimal(str(step))
    qty_d = Decimal(str(abs(qty)))
    floored = (qty_d / step_d).to_integral_value(rounding=ROUND_DOWN) * step_d
    result = float(floored)
    return result if qty >= 0 else -result


def set_leverage(symbol, leverage=None, mode="real"):
    """진입 전 심볼 레버리지를 target_leverage(기본 1배)로 강제. 멱등.

    변경: 예전엔 실패해도 조용히 넘어갔지만(swallow), 그러면 거래소 심볼 기본값
    (10~20배 등)으로 그대로 진입하는 사고가 날 수 있다. 이제 실패하면 예외를
    올려서(fail-closed) 호출자(orders._send_real_orders)가 그 심볼 주문을 전송하지
    않게 한다 = '원하는 레버리지를 확인 못 하면 거래 안 함'."""
    leverage = SETTINGS.target_leverage if leverage is None else leverage
    client = get_client(mode)
    return client.rest_api.change_initial_leverage(symbol=symbol, leverage=int(leverage))


def place_market_order(symbol, side, quantity, mode="real", reduce_only=False):
    """시장가 주문 전송(orders.py 가 호출). trade/order.place_market_order 래핑."""
    client = get_client(mode)
    log.info("place_market_order 전송: symbol=%s side=%s qty=%s reduce_only=%s",
              symbol, side, quantity, reduce_only)
    result = _place_market_order(client, symbol, side, quantity, reduce_only=reduce_only)
    log.info("place_market_order 응답: %r", result)
    return result
