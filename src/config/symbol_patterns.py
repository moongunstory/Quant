"""
symbol_patterns.py

심볼명 패턴 판별에 쓰는 정규식 상수.
상장폐지된 계약의 심볼 필터링, 정산 표시 처리 등에 활용된다.
"""

import re

# 분기별 만기 계약(delivery) 판별: {pair}_YYMMDD 형태
# 예: BTCUSD_260625, ETHUSD_260926
QUARTERLY_PATTERN = re.compile(r"_\d{6}$")

# 상장폐지 후 정산 표시 접미사 판별: _SETTLED 또는 SETTLED
# 예: AERGOUSDTSETTLED, PAIRUSDT_SETTLED
SETTLED_SUFFIX = re.compile(r"_?SETTLED$")
