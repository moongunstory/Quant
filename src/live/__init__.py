"""live — 승인된 알파를 실제(또는 가상) 주문으로 바꾸는 라이브 계층.

흐름(handler.run_cycle):
  freshness  알파별 데이터 신선도 게이트(stale 알파는 블렌드에서 제외).
  target_weights  승인 config -> 전체 재계산 -> 오늘 목표가중 행({coin: weight}).
  orders     목표 - 현재 diff -> 주문 리스트(dust 무시).
  paper/exchange  가상 즉시체결 or 실거래 전송(동일 코드경로, 체결부만 교체).
  ledger     주문/체결/포지션 기록(JSONL) -> 백테스트 대비 실측 슬리피지 추적.

라이브도 백테스트 함수를 그대로 재사용한다(coin D69): 매 사이클 전체 히스토리를
재계산하고 결과의 '마지막 행'을 오늘 결정으로 쓴다 -> 별도 스트리밍 리스크 엔진 불필요.
"""
