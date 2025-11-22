from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ----------------------------------------------------
# 1) 프로젝트 루트 경로 설정 & PYTHONPATH 추가
#    현재 파일: model/daily/hpo/run.py 기준으로 계산
# ----------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent        # .../model/daily/hpo
PROJECT_ROOT = CURRENT_DIR.parents[3]                # .../ (프로젝트 루트)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------
# 2) 내부 모듈 임포트
# ----------------------------------------------------
from model.daily.config import DailyConfig
from model.daily.hpo.core.engine import run_hpo_for_all_horizons


# ----------------------------------------------------
# 3) 도우미: 마스터 피처 파일 찾기
# ----------------------------------------------------
def _find_master_features_path() -> Path:
    """
    HPO에 사용할 마스터 피처 파일 경로를 찾는다.
    우선 1d → 없으면 1h 순으로 시도.
    """
    candidates: List[Path] = [
        PROJECT_ROOT / "data" / "processed" / "master_features_1d.parquet",
        PROJECT_ROOT / "data" / "processed" / "master_features_1h.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "마스터 피처 파일을 찾지 못했습니다.\n"
        "다음 경로 중 하나가 존재해야 합니다:\n"
        + "\n".join(str(p) for p in candidates)
    )


# ----------------------------------------------------
# 4) 메인 HPO 실행 함수
# ----------------------------------------------------
def run_hpo(as_of_ts: Optional[str] = None, horizons: Optional[List[int]] = None) -> None:
    """
    일간(daily) 모델용 HPO를 한 번 실행한다.

    - as_of_ts: 기준 시각 (예: '2025-11-21 00:00:00')
      None 이면 마스터 피처의 마지막 시각을 기준으로 사용
    - horizons: [3, 7, 30, 90] 형태로 지정 가능
      None 이면 DailyConfig.horizons_days 사용
    """
    cfg = DailyConfig()

    master_path = _find_master_features_path()
    print(f"[HPO] 마스터 피처 로드: {master_path}")
    df_master = pd.read_parquet(master_path)

    if as_of_ts is not None:
        from pandas import to_datetime

        as_of_ts_parsed = to_datetime(as_of_ts, utc=True)
    else:
        as_of_ts_parsed = None

    if horizons is None:
        horizons_to_use = cfg.horizons_days
    else:
        horizons_to_use = horizons

    print(f"[HPO] 대상 horizon(days): {list(horizons_to_use)}")
    print(f"[HPO] 기준 시각(as_of_ts): {as_of_ts_parsed}")

    run_hpo_for_all_horizons(
        df_master=df_master,
        cfg=cfg,
        horizons=horizons_to_use,
        as_of_ts=as_of_ts_parsed,
    )

    print("[HPO] 완료: data/hpo/daily/trials/*, best/* 갱신됨")


# ----------------------------------------------------
# 5) 스크립트로 직접 실행할 때 진입점
# ----------------------------------------------------
if __name__ == "__main__":
    # 지금은 옵션 없이 기본값으로만 실행.
    # 필요하면 나중에 argparse 붙여서
    # --as-of, --horizons 옵션 추가하면 됨.
    run_hpo()
