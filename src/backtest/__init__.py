"""Quant 백테스트 패키지.

coin 폴더의 WorldQuant식 알파 엔진을 Quant(top-100 유니버스) 데이터 레이아웃에
맞게 이식한 것. 자세한 설명은 src/backtest/README.md 참고.

레이어(데이터는 한 방향으로만 흐른다):
    panel  ->  engine  ->  metrics / validation
    (데이터를 date×coin 표로)  (알파 -> 포지션 -> 순손익)  (성과 채점 / 과최적화 검증)
"""
