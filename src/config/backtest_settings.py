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
