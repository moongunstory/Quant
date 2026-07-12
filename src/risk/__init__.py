"""risk — 결합된 포트폴리오 가중치에 씌우는 안전 오버레이 (Phase 1 이식 예정).

지금은 stage 리포트(report.py)만 존재한다: 각 리스크 모듈 적용 전/후의
북을 샤프/MDD/수익률/자산변동률로 추적하는 인터페이스. RISK_REGISTRY 파이프라인
(position_cap/gross_cap/vol_target/mdd_killswitch/...)은 Phase 1에서 coin
research/risk/risk.py 를 이식해 채운다. PLAN_quant_upgrade.md 참고.
"""
