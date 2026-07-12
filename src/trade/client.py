import os

from binance_common.configuration import ConfigurationRestAPI
from binance_common.constants import (
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL,
    DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL,
)
from binance_sdk_derivatives_trading_usds_futures.derivatives_trading_usds_futures import (
    DerivativesTradingUsdsFutures,
)

from src.config import credentials


def get_client(mode: str | None = None) -> DerivativesTradingUsdsFutures:
    """
    mode="live" 또는 "testnet"을 넘기면 그 모드로, 생략하면 .env의 TRADING_MODE로 클라이언트를 만든다.
    """
    mode = mode or os.getenv("TRADING_MODE")
    is_live = mode == "live"

    api_key, secret_key = (
        (credentials.BINANCE_API_KEY, credentials.BINANCE_SECRET_KEY)
        if is_live
        else (credentials.TESTNET_API_KEY, credentials.TESTNET_SECRET_KEY)
    )

    base_path = (
        DERIVATIVES_TRADING_USDS_FUTURES_REST_API_PROD_URL
        if is_live
        else DERIVATIVES_TRADING_USDS_FUTURES_REST_API_TESTNET_URL
    )

    config = ConfigurationRestAPI(api_key=api_key, api_secret=secret_key, base_path=base_path)
    return DerivativesTradingUsdsFutures(config_rest_api=config)
