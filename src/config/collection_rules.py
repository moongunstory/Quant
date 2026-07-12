"""
collection_rules.py

"무엇을/어떻게 수집할지"에 관한 정책 상수. 바이낸스 API 자체의 스펙(binance_api.py)과는
구분한다 - 이쪽은 우리 수집기가 임의로 정한 필터/속도 제한 값이다.
"""

# USDT 마진 무기한 계약만 대상으로 한다 (COIN-M, 만기 있는 delivery 계약은 제외).
# 이 필터가 excluded_symbols.txt 문제(주식/ETF 심볼 혼입)의 재발을 막아준다.
CONTRACT_TYPE_FILTER = "PERPETUAL"
QUOTE_ASSET_FILTER = "USDT"

# 요청 사이 최소 대기시간(초). 대상별로 분리한다:
#  - 아카이브(data.binance.vision): S3/CDN이라 fapi 같은 레이트리밋이 없다.
#    0.05초(초당 20요청)도 보수적인 편. 429/418 백오프가 있어 문제 시 자동 감속.
#  - REST(fapi.binance.com): 진짜 레이트리밋(가중치/분)이 있는 곳. 보수적으로 유지.
ARCHIVE_REQUEST_INTERVAL_SEC = 0.05
REST_REQUEST_INTERVAL_SEC = 0.3

# 하위호환 (예전 코드가 참조할 수 있음)
REQUEST_INTERVAL_SEC = REST_REQUEST_INTERVAL_SEC

# scan은 1일봉 기준이라, 이 정도 공백이면 "그 시점에 이미 상장폐지/거래중단됐다"고
# 판단한다. 30일 rolling window보다 살짝 여유를 둬서 데이터 지연 등 오탐을 줄인다.
STALENESS_DAYS = 35

