import logging
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

log = logging.getLogger("quant.live.exchange.client")


def get_client(mode: str | None = None) -> DerivativesTradingUsdsFutures:
    """
    mode="live" 또는 "testnet"을 넘기면 그 모드로, 생략하면 .env의 TRADING_MODE로 클라이언트를 만든다.

    주의: orders.py/exchange.py 는 실거래 코드 경로를 항상 mode="real" 문자열로 부른다
    ("live" 가 아님). 여기서 is_live 판정은 문자 그대로 "live" 하고만 비교하므로
    mode="real" 이 들어오면 is_live=False -> TESTNET 키/URL 을 쓴다. 지금 단계에선 의도된
    동작(실계좌 대신 테스트넷 검증)이지만 헷갈리기 쉬워 매번 로그로 남긴다.
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

    log.info("get_client(mode=%r) -> is_live=%s base_path=%s api_key_set=%s",
              mode, is_live, base_path, bool(api_key))

    config = ConfigurationRestAPI(api_key=api_key, api_secret=secret_key, base_path=base_path)
    return DerivativesTradingUsdsFutures(config_rest_api=config)
