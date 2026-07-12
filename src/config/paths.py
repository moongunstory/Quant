"""
paths.py

프로젝트 전역에서 쓰는 파일시스템 경로/파일명 상수를 모아둔다.
"어디에 저장하는지"에 관한 것은 전부 여기로 온다 (수집기 모듈이 몇 개든 상관없이).

DATA_ROOT는 프로젝트 루트 기준 상대경로다.
실행 위치가 달라지면 깨지므로, main.py에서 항상 프로젝트 루트를 cwd로 잡거나
여기서 절대경로를 주입하는 방식을 나중에 고려해야 한다.
"""

from pathlib import Path

DATA_ROOT = Path("data")

# full_collector가 다루는 데이터셋 목록. 데이터셋마다 하위 폴더와 별도 manifest를 갖는다.
# (binance_api.DATASET_SPECS의 키와 반드시 일치해야 한다. 새 데이터셋 추가 시 양쪽 모두 수정.)
PROCESSED_DATASETS = ["klines", "premiumIndexKlines", "metrics", "fundingRate", "bookDepth"]

PATHS = {
    "meta": DATA_ROOT / "meta",                       # symbol_list.json 등 메타데이터
    "raw": DATA_ROOT / "raw",                          # (필요 시) 원본 보관
    "scan": DATA_ROOT / "scan",                        # light_scanner 결과
    "universe_snapshots": DATA_ROOT / "universe_snapshots",  # 월별 top-N 스냅샷 + diff
    "universe_rules": DATA_ROOT / "universe_snapshots" / "rules",  # universe_probe/builder 판단 규칙(json)
    "processed": DATA_ROOT / "processed",              # full_collector 결과 루트 (하위는 데이터셋별)
    "live_log": DATA_ROOT / "live_log",                # 실전 매매 로그 (ingest.py 관련, 연구 데이터와 분리)
    # 데이터셋별 저장 경로: data/processed/{dataset}/{SYMBOL}.parquet + 데이터셋별 _manifest.json
    # (예전엔 processed 평면에 {SYMBOL}__{dataset}.parquet로 섞여 있었다.
    #  scripts/migrate_processed_layout.py로 1회 마이그레이션.)
    **{f"processed_{ds}": DATA_ROOT / "processed" / ds for ds in PROCESSED_DATASETS},
}

MANIFEST_FILENAME = "_manifest.json"
UNIVERSE_RULES_FILENAME = "universe_rules.json"
UNIVERSE_RULES_PATH = PATHS["universe_rules"] / UNIVERSE_RULES_FILENAME

