"""
throttle.py

바이낸스로 나가는 모든 HTTP 요청의 전역 속도 제한기.

제한기를 HTTP 계층(archive_client/rest_client) 바로 앞에 두고 모든 수집 모듈이
공유한다. 워커 스레드를 몇 개로 늘리든 바이낸스 입장에서 보는 요청 간격은
설정값 밑으로 내려가지 않는다.

대상별로 제한기를 분리한다:
  - 아카이브(data.binance.vision): CDN이라 관대함 -> 빠른 간격
  - REST(fapi.binance.com): 실제 레이트리밋 존재 -> 보수적 간격
"""

from __future__ import annotations

import threading
import time

from src.config.collection_rules import ARCHIVE_REQUEST_INTERVAL_SEC, REST_REQUEST_INTERVAL_SEC


class RateLimiter:
    """여러 스레드가 공유하는 최소 요청 간격 제한기."""

    def __init__(self, interval_sec: float):
        self._interval = interval_sec
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            remaining = self._interval - (now - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()


ARCHIVE_LIMITER = RateLimiter(ARCHIVE_REQUEST_INTERVAL_SEC)
REST_LIMITER = RateLimiter(REST_REQUEST_INTERVAL_SEC)

# 하위호환: 예전 코드가 GLOBAL_LIMITER/wait()를 참조할 수 있음 (보수적인 쪽으로)
GLOBAL_LIMITER = REST_LIMITER


def wait(url: str = "") -> None:
    """url의 호스트를 보고 알맞은 제한기를 적용한다. 모르면 보수적인 REST 쪽."""
    if "data.binance.vision" in url or "s3-ap-northeast-1.amazonaws.com" in url:
        ARCHIVE_LIMITER.wait()
    else:
        REST_LIMITER.wait()
