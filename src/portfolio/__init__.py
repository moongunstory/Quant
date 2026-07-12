"""portfolio — 여러 승인 알파를 하나의 포트폴리오 가중치로 결합.

combine.py : 알파별 가중치 패널(dict) -> 결합 가중치 패널(일별 L1=1).
             가중방식(WEIGHTING_REGISTRY): equal / inverse_vol.
config.py  : portfolio.json(알파 선택 + 가중 + risk_pipeline) 로더/스키마.
pipeline.py: 승인 config -> 알파별 백테스트 -> combine -> risk 오버레이 -> 리포트.

coin research/portfolio 이식(Phase 1 스코프). hedge/low_correlation 선택 등
고급 기능은 Phase 3. PLAN_quant_upgrade.md 참고.
"""
