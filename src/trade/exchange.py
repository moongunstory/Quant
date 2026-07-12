"""exchange — 실거래(real) 모드에서 쓰는 거래소 조회/주문 래퍼.

trade/client.py(클라이언트 생성) + trade/order.py(시장가 주문) 위에, orders.py 가
필요로 하는 '현재 포지션 조회 / 마크가격 / 수량 반올림 / 주문 전송'을 얹는다.

주의: 페이퍼 모드에서는 이 모듈을 전혀 호출하지 않는다(orders.py 가 positions.json
을 현재 포지션으로 씀). 실거래 조회 메서드는 Binance USDT-M 선물 SDK 응답 형태에
의존하므로, 실계좌/테스트넷 연결 후 실제 응답으로 한 번 검증이 필요하다(가상매매로
충분히 검증한 뒤 실거래 전환 — PLAN Phase 2).
"""
from __future__ import annotations

import math

from src.config.backtest_settings import SETTINGS
from src.trade.client import get_client
from src.trade.order import place_market_order as _place_market_order

# 심볼별 LOT step 캐시(거래소 exchangeInfo 에서 1회 조회). 미조회 시 기본 step.
_LOT_STEP_CACHE: dict[str, float] = {}
_DEFAULT_QTY_STEP = 0.001


def get_open_positions(mode="real"):
    """{symbol: signed_qty} (롱 +, 숏 -). 포지션 없으면 {}."""
    client = get_client(mode)
    resp = client.rest_api.position_information_v2()
    data = resp.data() if hasattr(resp, "data") else resp
    out = {}
    for p in (data or []):
        amt = float(p.get("positionAmt", 0.0))
        if amt != 0.0:
            out[p["symbol"]] = amt
    return out


def get_mark_prices(mode="real"):
    """{symbol: mark_price}."""
    client = get_client(mode)
    resp = client.rest_api.mark_price()
    data = resp.data() if hasattr(resp, "data") else resp
    out = {}
    for row in (data or []):
        try:
            out[row["symbol"]] = float(row["markPrice"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def round_quantity(symbol, qty):
    """심볼 LOT step 으로 내림 반올림. step 미상이면 기본 step."""
    step = _LOT_STEP_CACHE.get(symbol, _DEFAULT_QTY_STEP)
    if step <= 0:
        return float(qty)
    return math.floor(abs(qty) / step) * step * (1 if qty >= 0 else -1)


def set_leverage(symbol, leverage=1, mode="real"):
    """레버리지 설정(멱등, 저렴). 실패해도 조용히 넘어감(주문 전 방어)."""
    try:
        client = get_client(mode)
        client.rest_api.change_initial_leverage(symbol=symbol, leverage=int(leverage))
    except Exception:
        pass


def place_market_order(symbol, side, quantity, mode="real", reduce_only=False):
    """시장가 주문 전송(orders.py 가 호출). trade/order.place_market_order 래핑."""
    client = get_client(mode)
    return _place_market_order(client, symbol, side, quantity, reduce_only=reduce_only)
