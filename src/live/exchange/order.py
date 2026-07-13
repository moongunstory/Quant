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
):
    """시장가 주문을 실행한다. reduce_only=True면 포지션 축소/청산 전용."""
    return client.rest_api.new_order(
        symbol=symbol,
        side=side,
        type="MARKET",
        quantity=quantity,
        position_side=position_side,
        reduce_only=reduce_only,
    )