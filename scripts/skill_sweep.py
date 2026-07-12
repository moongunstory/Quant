"""skill 가중 파라미터 스윕 — walkforward(OOS)로 half_life/floor/power 비교.

한 번 실행하면 데이터를 '한 번만' 로드하고(느린 부분), 여러 파라미터 조합의
OOS 샤프/mdd/연수익 + 약점 해(2023·2025) 샤프를 한 표로 찍는다. inverse_vol
기준행도 같이 넣어 skill 튜닝이 그보다 나은지 바로 비교.

    python scripts/skill_sweep.py

주의: 선택(selection)·리스크 스택은 base config(data/portfolio_4alpha_skill.json)
그대로 고정하고 '가중 파라미터만' 바꾼다 → 가중 효과만 격리해서 본다.
"""
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.portfolio.config import load_portfolio_config
from src.portfolio.pipeline import _load_families
from src.backtest import walkforward as WF, metrics as M

BASE = "data/portfolio_4alpha_skill.json"

# (라벨, weighting dict) — 위에서부터 baseline, 그다음 튜닝 후보들.
RUNS = [
    ("inverse_vol",      {"method": "inverse_vol", "params": {"max_weight": 0.35}}),
    ("skill hl90 f0 p1", {"method": "skill", "params": {"half_life": 90,  "floor": 0.0, "power": 1.0, "max_weight": 0.5}}),
    ("skill hl45 f0 p1", {"method": "skill", "params": {"half_life": 45,  "floor": 0.0, "power": 1.0, "max_weight": 0.5}}),
    ("skill hl180 f0 p1",{"method": "skill", "params": {"half_life": 180, "floor": 0.0, "power": 1.0, "max_weight": 0.5}}),
    ("skill hl90 f0.5 p1",{"method": "skill","params": {"half_life": 90,  "floor": 0.5, "power": 1.0, "max_weight": 0.5}}),
    ("skill hl90 f0 p2", {"method": "skill", "params": {"half_life": 90,  "floor": 0.0, "power": 2.0, "max_weight": 0.5}}),
    ("skill hl45 f0.5 p1.5",{"method":"skill","params":{"half_life": 45,  "floor": 0.5, "power": 1.5, "max_weight": 0.5}}),
]


def main():
    families = _load_families()
    cfg = load_portfolio_config(BASE)

    print("데이터 로드 중(1회)…")
    series, pos_panels, master_panels, funding_events = WF.collect_alpha_books(
        families=families)

    hdr = (f"{'가중방식':<20}{'OOS샤프':>9}{'mdd':>8}{'연수익':>9}"
           f"{'2023':>8}{'2025':>8}")
    print("\n=== skill 가중 파라미터 스윕 (walkforward OOS) ===")
    print(hdr)
    print("-" * len(hdr))
    for label, wdict in RUNS:
        c = replace(cfg, weighting=wdict)
        wf = WF.run_walkforward_portfolio(
            series, pos_panels, master_panels, funding_events, c,
            families=families)
        oos = wf["oos_pnl"]
        ys = WF.yearly_sharpe(oos)
        print(f"{label:<20}{M.sharpe(oos):>9.2f}{M.max_drawdown(oos):>8.3f}"
              f"{M.ann_return(oos):>+9.3f}{ys.get(2023, float('nan')):>8.2f}"
              f"{ys.get(2025, float('nan')):>8.2f}")


if __name__ == "__main__":
    main()
