"""A-2: 방향성(directional) ON vs OFF — walkforward OOS A/B.

'5:5 강제가 손해냐'의 실측. 방향성 정책을 켠 경우(현재 프로덕션)와 끈 경우로
알파 북을 각각 재구성해 walkforward OOS 를 비교한다. 자격 알파(carry/orderbook
패밀리의 부호 있는 신호, 예: funding_carry_signed)만 방향 베팅을 하므로, ON/OFF
차이가 곧 '방향 베팅의 순기여'다.

    python scripts/directional_ab.py

주의: 방향성 상태는 '알파 북 자체'에 배는 값이라(neutralization 승격) 각 케이스마다
데이터를 다시 로드한다 — 두 번 로드라 조금 느리다(정상).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.backtest.directional as D
from src.portfolio.config import load_portfolio_config
from src.portfolio.pipeline import _load_families
from src.backtest import walkforward as WF, metrics as M

BASE = "data/portfolio_4alpha.json"   # 프로덕션(inverse_vol)


def run(flag, cfg, families):
    # collect_alpha_books 는 내부에서 D.is_enabled(policy) 로 방향성 여부를 정한다.
    # 모듈 속성을 덮어써서 ON/OFF 를 강제(정책 파일은 그대로 둠).
    D.is_enabled = lambda *a, **k: flag
    series, pos, mp, fe = WF.collect_alpha_books(families=families)
    wf = WF.run_walkforward_portfolio(series, pos, mp, fe, cfg, families=families)
    return wf["oos_pnl"]


def main():
    families = _load_families()
    cfg = load_portfolio_config(BASE)
    _orig = D.is_enabled  # 복원용

    hdr = f"{'방향성':<12}{'OOS샤프':>9}{'mdd':>8}{'연수익':>9}{'2023':>8}{'2025':>8}"
    print("\n=== A-2: directional ON vs OFF (walkforward OOS) ===")
    print(hdr)
    print("-" * len(hdr))
    try:
        for label, flag in [("ON(현재)", True), ("OFF(전부5:5)", False)]:
            print(f"데이터 로드 중 — {label}…")
            oos = run(flag, cfg, families)
            ys = WF.yearly_sharpe(oos)
            print(f"{label:<12}{M.sharpe(oos):>9.2f}{M.max_drawdown(oos):>8.3f}"
                  f"{M.ann_return(oos):>+9.3f}{ys.get(2023, float('nan')):>8.2f}"
                  f"{ys.get(2025, float('nan')):>8.2f}")
    finally:
        D.is_enabled = _orig


if __name__ == "__main__":
    main()
