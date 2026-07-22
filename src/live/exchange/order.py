from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)


def place_market_order(
    client: DerivativesTradingUsdsFutures,
    symbol: str,
    side: str,
    quantity: float,
    position_side: str = "BOTH",
    reduce_only: bool = False,
    client_order_id: str | None = None,
):
    """시장가 주문을 실행한다. reduce_only=True면 포지션 축소/청산 전용.

    client_order_id: 주면 newClientOrderId 로 실려 같은 id 재전송을 거래소가 중복 거부한다
    (멱등). 필드명은 SDK 버전에 따라 new_client_order_id 일 수 있어, 없으면 생략한다."""
    kwargs = dict(
        symbol=symbol,
        side=side,
        type="MARKET",
        quantity=quantity,
        position_side=position_side,
        reduce_only=reduce_only,
    )
    if client_order_id is not None:
        kwargs["new_client_order_id"] = client_order_id
    return client.rest_api.new_order(**kwargs)