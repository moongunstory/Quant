"""
paths.py

프로젝트 전역에서 쓰는 파일시스템 경로/파일명 상수를 모아둔다.
"어디에 저장하는지"에 관한 것은 전부 여기로 온다 (수집기 모듈이 몇 개든 상관없이).

DATA_ROOT는 프로젝트 루트 기준 상대경로다.
실행 위치가 달라지면 깨지므로, main.py에서 항상 프로젝트 루트를 cwd로 잡거나
여기서 절대경로를 주입하는 방식을 나중에 고려해야 한다.
"""

import os
from pathlib import Path

# DATA_ROOT는 기본값이 프로젝트 루트 기준 상대경로 "data" 지만,
# 환경변수 QUANT_DATA_DIR 로 덮어쓸 수 있다. (backtest_settings.Settings.data_dir 와 동일한 변수)
# Lambda 처럼 파일시스템이 /tmp 빼고 읽기전용인 환경에서는 QUANT_DATA_DIR=/tmp/quant-data
# 로 지정해, 수집기(full_collector/live_refresh)의 저장 경로도 쓰기 가능한 위치로 옮긴다.
# 로컬에서는 이 변수를 안 주므로 기존과 동일하게 "data" 를 쓴다.
DATA_ROOT = Path(os.environ.get("QUANT_DATA_DIR", "data"))

# ── 디렉토리 구조 ────────────────────────────────────────────────────────────
#
#  data/
#   ├── market/       시장 데이터 수집 파이프라인 (수집·가공·캐시)
#   │   ├── processed/     full_collector 원본 데이터 (심볼별 parquet)
#   │   ├── scan/          universe_probe 경량 스캔 결과
#   │   ├── universe/      월별 top-N 유니버스 스냅샷 + 룰
#   │   └── panel/         백테스트용 date×coin 피벗 캐시
#   ├── strategy/     전략 설계·정의 파일
#   │   ├── alphas/        알파 시그널 정의 (JSON)
#   │   ├── portfolio/     포트폴리오 구성 설정 (config.json)
#   │   └── meta/          심볼 마스터, 알파 패밀리, 방향성 정책
#   ├── runtime/      실시간 운영 상태
#   │   ├── live/          실전/모의 포지션·주문 이력
#   │   └── logs/          리밸런싱 로그·포트폴리오 백업
#   └── experiments/  백테스트 실험 결과 (루트 직속)
#
# ─────────────────────────────────────────────────────────────────────────────

# full_collector가 다루는 데이터셋 목록. 데이터셋마다 하위 폴더와 별도 manifest를 갖는다.
# (binance_api.DATASET_SPECS의 키와 반드시 일치해야 한다. 새 데이터셋 추가 시 양쪽 모두 수정.)
PROCESSED_DATASETS = ["klines", "premiumIndexKlines", "metrics", "fundingRate", "bookDepth"]

PATHS = {
    # ── market/ ──────────────────────────────────────────────────────────────
    "processed": DATA_ROOT / "market" / "processed",       # full_collector 결과 루트
    # 데이터셋별 저장 경로: data/market/processed/{dataset}/{SYMBOL}.parquet + _manifest.json
    **{f"processed_{ds}": DATA_ROOT / "market" / "processed" / ds for ds in PROCESSED_DATASETS},
    "scan": DATA_ROOT / "market" / "scan",                 # universe_probe 경량 스캔
    "universe_snapshots": DATA_ROOT / "market" / "universe",  # 월별 top-N 스냅샷 + diff
    "universe_rules": DATA_ROOT / "market" / "universe" / "rules",  # 유니버스 판단 규칙
    "panel": DATA_ROOT / "market" / "panel",               # date×coin 피벗 캐시

    # ── strategy/ ────────────────────────────────────────────────────────────
    "alphas": DATA_ROOT / "strategy" / "alphas",           # 알파 시그널 정의
    "portfolio": DATA_ROOT / "strategy" / "portfolio",     # 포트폴리오 구성 설정 (config.json 등)
    "meta": DATA_ROOT / "strategy" / "meta",               # 심볼 마스터, 알파 패밀리, 방향성 정책

    # ── runtime/ ─────────────────────────────────────────────────────────────
    "live": DATA_ROOT / "runtime" / "live",                # 실전/모의 포지션·주문 이력
    "logs": DATA_ROOT / "runtime" / "logs",                # 리밸런싱 로그·포트폴리오 백업

    # ── research/ ────────────────────────────────────────────────────────────
    "experiments": DATA_ROOT / "experiments",               # 백테스트 실험 결과 (루트 직속)
}

MANIFEST_FILENAME = "_manifest.json"
UNIVERSE_RULES_FILENAME = "universe_rules.json"
UNIVERSE_RULES_PATH = PATHS["universe_rules"] / UNIVERSE_RULES_FILENAME

