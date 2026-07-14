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
    실계좌(live) / 테스트넷(testnet) 선택은 오직 .env 의 TRADING_MODE 로만 결정한다.

    mode 인자("real"/"paper" 등)는 호출 경로 표시일 뿐 계좌 선택에는 쓰지 않는다.
    (예전엔 mode=="live" 문자열 비교였는데, exchange.py 가 항상 "real" 을 넘겨서
    env 가 무시되고 코드 수정 없이는 실계좌 전환이 불가능한 구조였다.)

    TRADING_MODE=live  -> 실계좌 키/URL (돈이 나가는 모드!)
    그 외/미설정       -> 테스트넷 키/URL (안전 기본값, fail-safe)
    """
    trading_mode = (os.getenv("TRADING_MODE") or "testnet").strip().lower()
    is_live = trading_mode == "live"

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

    log.info("get_client(mode=%r, TRADING_MODE=%r) -> is_live=%s base_path=%s api_key_set=%s",
              mode, trading_mode, is_live, base_path, bool(api_key))

    config = ConfigurationRestAPI(api_key=api_key, api_secret=secret_key, base_path=base_path)
    return DerivativesTradingUsdsFutures(config_rest_api=config)
