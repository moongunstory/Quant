"""A-3: 거래비용 스트레스 테스트 — walkforward OOS 가 비용에 얼마나 민감한가.

엔진은 이미 '일별 회전율 × (taker 0.04% + 슬리피지 0.05% = 0.09%)'를 매일 차감한다.
두 가지 우려를 한 번에 본다:
  (1) walkforward 가 폴드 경계의 로스터 교체 '전환비용'을 안 물린다(미부과분).
  (2) 알트에선 실제 슬리피지가 0.05% 보다 클 수 있다(낙관 가정).
둘 다 '실질 비용이 가정보다 높다'는 방향이므로, 비용률을 1×~4× 로 올려도 OOS 가
버티면 2.42 는 안전, 무너지면 비용이 진짜 위협이다.

    python scripts/cost_stress.py

방식: 데이터를 1회 로드하고, 매 반복마다 SETTINGS 의 비용률만 바꿔(엔진이 최종
손익을 그 비용으로 재계산) walkforward 를 다시 돌린다. 선택 점수화는 기본비용으로
고정된 series 를 쓰므로 로스터는 거의 불변 — 순수 '비용→손익' 민감도만 본다.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.backtest_settings import SETTINGS
from src.portfolio.config import load_portfolio_config
from src.portfolio.pipeline import _load_families
from src.backtest import walkforward as WF, metrics as M

BASE = "data/portfolio_4alpha.json"
BASE_COST = 0.09   # taker 0.04 + slippage 0.05 (%)


def set_total_cost(pct):
    # frozen dataclass 라 object.__setattr__ 로 우회. 합만 중요하므로 taker=0, slippage=합.
    object.__setattr__(SETTINGS, "taker_fee_pct", 0.0)
    object.__setattr__(SETTINGS, "slippage_pct", float(pct))


def main():
    families = _load_families()
    cfg = load_portfolio_config(BASE)

    print("데이터 로드 중(1회)…")
    series, pos, mp, fe = WF.collect_alpha_books(families=families)

    hdr = (f"{'비용':<14}{'OOS샤프':>9}{'mdd':>8}{'연수익':>9}{'2023':>8}{'2025':>8}")
    print("\n=== A-3: 거래비용 스트레스 (walkforward OOS) ===")
    print(hdr)
    print("-" * len(hdr))
    for mult in [1, 2, 3, 4]:
        set_total_cost(BASE_COST * mult)
        wf = WF.run_walkforward_portfolio(series, pos, mp, fe, cfg, families=families)
        oos = wf["oos_pnl"]
        ys = WF.yearly_sharpe(oos)
        label = f"{mult}x ({BASE_COST*mult:.2f}%)"
        print(f"{label:<14}{M.sharpe(oos):>9.2f}{M.max_drawdown(oos):>8.3f}"
              f"{M.ann_return(oos):>+9.3f}{ys.get(2023, float('nan')):>8.2f}"
              f"{ys.get(2025, float('nan')):>8.2f}")
    print("\n1x = 현재 가정(0.09%). 2~4x = 전환비용 미부과 + 알트 슬리피지까지 감안한 보수 시나리오.")


if __name__ == "__main__":
    main()
