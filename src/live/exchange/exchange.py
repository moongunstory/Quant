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

# 심볼별 거래소 필터 캐시(exchangeInfo 에서 1회 조회). 미조회 시 기본값.
#   _LOT_STEP_CACHE      : LOT_SIZE stepSize (수량 반올림 단위)
#   _MIN_NOTIONAL_CACHE  : MIN_NOTIONAL notional (주문 최소 명목가, USDT)
_LOT_STEP_CACHE: dict[str, float] = {}
_MIN_NOTIONAL_CACHE: dict[str, float] = {}
_DEFAULT_QTY_STEP = 0.001
_DEFAULT_MIN_NOTIONAL = 5.0   # 바이낸스 USDT-M 선물 통상 최소 명목가(폴백)
_LOT_STEP_CACHE_LOADED = False

# 이번 실행에서 이미 레버리지를 세팅한 심볼(심볼당 1회만 호출 → API 낭비/속도 개선).
# 컨테이너 재사용 시에도 멱등이라 유지되어 무방(값이 안 바뀌면 재호출 불필요).
_LEVERAGE_SET_CACHE: set[str] = set()


def _ensure_lot_step_cache(mode="real"):
    """exchangeInfo 를 1회 조회해 심볼별 LOT_SIZE stepSize + MIN_NOTIONAL 을 캐시한다.
    이전엔 이 캐시가 절대 채워지지 않아 모든 심볼이 기본 step(0.001)으로 반올림됐고,
    그 결과 정수 단위(step=1)만 허용하는 심볼(예: 1000BONKUSDT)에 소수점 수량을 보내
    바이낸스가 -1111 'Precision is over the maximum defined for this asset.' 로 거부했다.
    MIN_NOTIONAL 도 같은 조회에서 채워, 작은 주문이 -4164(최소 명목가 미만)로 거부되기
    전에 로컬에서 걸러낸다(orders.py)."""
    global _LOT_STEP_CACHE_LOADED
    if _LOT_STEP_CACHE_LOADED:
        return
    client = get_client(mode)
    resp = client.rest_api.exchange_information()
    data = resp.data() if hasattr(resp, "data") else resp
    n = 0
    n_notional = 0
    for s in getattr(data, "symbols", None) or []:
        got_lot = False
        for f in getattr(s, "filters", None) or []:
            ftype = getattr(f, "filter_type", None)
            if ftype == "LOT_SIZE" and not got_lot:
                try:
                    _LOT_STEP_CACHE[s.symbol] = float(f.step_size)
                    n += 1
                    got_lot = True
                except (TypeError, ValueError):
                    continue
            elif ftype == "MIN_NOTIONAL":
                # 필드명은 SDK/버전에 따라 notional 또는 min_notional 로 올 수 있어 둘 다 시도.
                raw = getattr(f, "notional", None)
                if raw is None:
                    raw = getattr(f, "min_notional", None)
                try:
                    if raw is not None:
                        _MIN_NOTIONAL_CACHE[s.symbol] = float(raw)
                        n_notional += 1
                except (TypeError, ValueError):
                    pass
    log.info("exchangeInfo 캐시 로드: LOT_SIZE %d개, MIN_NOTIONAL %d개 심볼", n, n_notional)
    _LOT_STEP_CACHE_LOADED = True


def min_notional(symbol, mode="real"):
    """심볼의 최소 주문 명목가(USDT). 캐시 미로드면 먼저 채운다. 없으면 기본값."""
    if not _LOT_STEP_CACHE_LOADED:
        try:
            _ensure_lot_step_cache(mode)
        except Exception:
            log.exception("exchangeInfo 캐시 로드 실패 -- 기본 min_notional 사용")
    return _MIN_NOTIONAL_CACHE.get(symbol, _DEFAULT_MIN_NOTIONAL)


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


def detect_margin_mode(mode="real", symbols=None):
    """마진모드를 position_information_v2 응답의 margin_type 필드로 판별.
      크로스  : margin_type == "cross"   (또는 "crossed")
      독립    : margin_type == "isolated"
    바이낸스에서 마진모드는 '심볼별' 설정이므로, symbols 를 주면 그 심볼들만 본다
    (오늘 매매할 종목만 검사 → 안 쓰는 심볼 하나가 isolated 라고 전체를 막지 않음).
    대상 중 하나라도 isolated 면 'isolated'(안전한 쪽), 전부 cross 면 'cross',
    판별 불가/대상 없음이면 'unknown'.

    (테스트넷 실응답 확인 완료: 필드명 'margin_type', 값 'cross'/'isolated'.)"""
    rows = _raw_position_information(mode)
    if not rows:
        return "unknown"
    want = {s.upper() for s in symbols} if symbols else None
    seen = set()
    for p in rows:
        if want is not None and str(getattr(p, "symbol", "")).upper() not in want:
            continue
        mt = getattr(p, "margin_type", None)
        if mt is None:
            mt = getattr(p, "marginType", None)
        if mt is None:
            continue
        seen.add(str(mt).strip().lower())
    if not seen:
        return "unknown"
    if "isolated" in seen:
        return "isolated"
    if seen & {"cross", "crossed"}:
        return "cross"
    return "unknown"


def assert_cross_margin(mode="real", symbols=None):
    """실매매 프리플라이트: (매매할 심볼들이) 크로스 마진이 아니면 막는다(fail-closed).
    이 전략은 롱숏이 서로 증거금을 받쳐주는 크로스 마진 전제이므로, 독립(isolated)이면
    한 코인만 개별 청산되는 사고가 날 수 있어 사람이 먼저 바꾸게 한다. symbols 를 주면
    그 종목만 검사한다(안 쓰는 심볼 때문에 전체가 막히는 과잉차단 방지). 'unknown'도
    안전하게 막는다."""
    mm = detect_margin_mode(mode, symbols=symbols)
    log.info("assert_cross_margin(symbols=%s): detected=%s",
             (list(symbols)[:5] if symbols else "ALL"), mm)
    if mm != "cross":
        raise RuntimeError(
            f"매매 대상 심볼의 마진모드가 '{mm}' 입니다. 이 전략은 크로스(Cross) 마진 전용입니다"
            " (롱숏이 서로 증거금을 받쳐주는 구조 가정). 바이낸스 선물 설정에서 해당 심볼들을"
            " 'Cross Margin'으로 바꾼 뒤 다시 실행하세요. (독립 모드면 한 코인만 개별 청산되어"
            " 시장중립 가정이 깨질 수 있어 fail-closed 로 막습니다.)"
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
    # 심볼당 이번 실행에서 이미 세팅했으면 재호출 생략(멱등, API 낭비 방지).
    if symbol in _LEVERAGE_SET_CACHE:
        return None
    client = get_client(mode)
    result = client.rest_api.change_initial_leverage(symbol=symbol, leverage=int(leverage))
    _LEVERAGE_SET_CACHE.add(symbol)
    return result


def place_market_order(symbol, side, quantity, mode="real", reduce_only=False,
                       client_order_id=None):
    """시장가 주문 전송(orders.py 가 호출). trade/order.place_market_order 래핑.

    client_order_id: 주면 거래소에 newClientOrderId 로 실려, 같은 id 의 주문이 두 번
    가면 바이낸스가 중복으로 거부한다 → 재실행/재시도 시 이중 진입 방지(멱등)."""
    client = get_client(mode)
    log.info("place_market_order 전송: symbol=%s side=%s qty=%s reduce_only=%s coid=%s",
              symbol, side, quantity, reduce_only, client_order_id)
    result = _place_market_order(client, symbol, side, quantity, reduce_only=reduce_only,
                                 client_order_id=client_order_id)
    log.info("place_market_order 응답: %r", result)
    return result
