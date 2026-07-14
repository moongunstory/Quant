"""백테스트 설정 — 모든 숫자(수수료/슬리피지/경로)를 한 곳에.

매직넘버 금지: 비용 가정을 바꾸려면 여기 한 줄만 고친다.
환경변수로 덮어쓸 수 있게 해서, 나중에 클라우드/라이브로 옮겨도 코드 변경 없이 동작.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Quant/src/config/backtest_settings.py -> Quant/ 가 루트
_ROOT = Path(__file__).resolve().parents[2]


def _f(env, default):
    return float(os.environ.get(env, default))


@dataclass(frozen=True)
class Settings:
    # 데이터 경로 (Quant/data/...)
    data_dir: Path = Path(os.environ.get("QUANT_DATA_DIR", _ROOT / "data"))

    # 비용 가정 (%) — Binance USDT-M 선물 기준.
    # taker 0.04% + 보수적 슬리피지 0.05%. 회전율에 곱해져 순손익에서 차감된다.
    taker_fee_pct: float = _f("QUANT_TAKER_FEE_PCT", 0.04)
    slippage_pct: float = _f("QUANT_SLIPPAGE_PCT", 0.05)

    # 코인은 365일 거래 -> 샤프 연율화에 252가 아니라 365 사용.
    trading_days_per_year: int = int(os.environ.get("QUANT_TRADING_DAYS", 365))

    # 체결가 가정. 백테스트 손익을 어느 가격에 체결됐다고 볼지:
    #  "close"     (기본) 신호를 만든 그 봉 종가에 즉시 체결됐다고 가정(close→close).
    #              구현이 단순하고 슬리피지로 보정하지만, '결정=체결' 순간이 겹쳐 약간
    #              낙관적.
    #  "next_open" 신호는 봉 종가에 나지만 체결은 '다음 봉 시가'라고 가정(open→open).
    #              실전(종가 보고 다음 봉에 주문)과 더 정합적 = 백테스트 신뢰도↑.
    #              대신 종가→다음시가 갭 움직임은 포기. open 패널이 필요.
    # 기본은 "close"라 기존 숫자/동작은 그대로. 신뢰도 점검 땐 QUANT_EXECUTION=next_open
    # 으로 돌려 두 결과를 비교한다(엣지가 체결가정에 얼마나 의존하는지 측정).
    execution: str = os.environ.get("QUANT_EXECUTION", "close")

    # ---- 포트폴리오 결합(selection) ----
    # low_correlation 선택 시 이미 뽑은 알파와 |net-PnL 상관|이 이 값 이상이면
    # 제외(greedy dedup). coin D-값 이식(기본 0.5).
    combine_corr_threshold: float = _f("QUANT_COMBINE_CORR_THRESHOLD", 0.5)

    # 재현성(순열검정 등)용 시드.
    random_seed: int = int(os.environ.get("QUANT_SEED", 42))

    # ---- 리스크 오버레이 기본값 (coin research/risk 이식, config로 덮어씀) ----
    # 단일 코인 최대 가중치(±). 한 코인 폭발 방지.
    position_cap: float = _f("QUANT_POSITION_CAP", 0.05)
    # 시장중립 허용 순노출(|Σw|) 한계.
    net_exposure_limit: float = _f("QUANT_NET_EXPOSURE_LIMIT", 0.20)
    # vol_target 목표 연변동성.
    target_annual_vol: float = _f("QUANT_TARGET_ANNUAL_VOL", 0.15)
    # 킬스위치: 낙폭 한계/재진입/램프업.
    mdd_kill: float = _f("QUANT_MDD_KILL", 0.15)
    mdd_reentry: float = _f("QUANT_MDD_REENTRY", 0.075)
    rampup_days: int = int(os.environ.get("QUANT_RAMPUP_DAYS", 5))
    # 거래 최소 단위(가중치 반올림). 0 이면 반올림 안 함.
    lot_step: float = _f("QUANT_LOT_STEP", 0.001)
    # participation_cap: 코인 자기 ADV 대비 최대 보유비율 + 가정 북 규모(USD).
    participation_rate: float = _f("QUANT_PARTICIPATION_RATE", 0.05)
    book_aum_usd: float = _f("QUANT_BOOK_AUM_USD", 100_000.0)

    # ---- 실매매(real) 거래소 안전설정 ----
    # 진입 전 각 심볼 레버리지를 이 배수로 강제 설정한다. 기본 1배(현물처럼, 청산위험
    # 최소). 거래소의 심볼 기본값(10~20배 등)으로 잘못 진입하는 사고를 막는다.
    # 이 값 설정에 실패하면 그 심볼 주문은 전송하지 않는다(fail-closed, orders.py).
    target_leverage: int = int(os.environ.get("QUANT_TARGET_LEVERAGE", 1))
    # 실매매 전 계정이 '원웨이 모드'인지 확인하고, 헷지 모드면 주문을 막을지 여부.
    # 이 프로그램의 주문은 positionSide="BOTH"(원웨이 전용)라 헷지 모드에선 실패한다.
    require_one_way_mode: bool = os.environ.get("QUANT_REQUIRE_ONE_WAY", "1") not in ("0", "false", "False")

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "market" / "processed"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "runtime" / "logs"

    @property
    def panel_dir(self) -> Path:
        return self.data_dir / "market" / "panel"

    @property
    def universe_snapshot_dir(self) -> Path:
        return self.data_dir / "market" / "universe"

    @property
    def alphas_dir(self) -> Path:
        return self.data_dir / "strategy" / "alphas"


SETTINGS = Settings()
