"""
storage.py

로컬 디스크 저장/조회를 담당하는 공용 모듈.
archive_client.py가 "외부에서 받아온 파싱된 데이터"를 넘겨주면,
그걸 어디에 어떤 포맷으로 저장하고 다시 불러올지는 전부 여기서 결정한다.

이 모듈이 갖는 두 가지 책임:
  1. parquet 파일 read/write (데이터 종류별 표준 경로 규칙 포함)
  2. manifest 관리 — 심볼별 "마지막으로 수집 완료한 지점"을 기록
     -> universe_probe, full_collector 등 gap-aware 수집 모듈들이 이 manifest를 보고 동작한다.
     category(scan/processed 등)마다 manifest가 따로 있다 (paths.py 참고).

주의: DATA_ROOT는 프로젝트 루트 기준 상대경로다 (src/config/paths.py 참고).
실행 위치가 달라지면 깨지므로, main.py에서 항상 프로젝트 루트를 cwd로 잡거나
나중에 절대경로를 주입하는 방식을 고려해야 한다.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config.paths import PATHS, MANIFEST_FILENAME

# manifest는 여러 워커 스레드가 공유하는 파일이라 read-modify-write 구간을
# 락으로 보호해야 갱신 유실이 없다 (universe_probe/full_collector 병렬 수집용).
# 읽기(load_manifest)도 같은 락을 쓰므로 재진입 가능한 RLock이어야 한다
# (update_last_collected가 락을 쥔 채 load_manifest를 호출한다).
_manifest_lock = threading.RLock()


def ensure_dirs() -> None:
    """최초 실행 시 필요한 디렉토리를 전부 생성. main.py 시작 지점에서 한 번 호출."""
    for p in PATHS.values():
        p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# parquet R/W
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, category: str, filename: str) -> Path:
    """
    category: PATHS의 키 (예: "scan", "processed")
    filename: 확장자 없이 (예: "BTCUSDT") -> 내부에서 .parquet 붙임
    """
    if category not in PATHS:
        raise ValueError(f"알 수 없는 category: {category}. PATHS 키 중 하나여야 함: {list(PATHS)}")

    path = PATHS[category] / f"{filename}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _coerce_mixed_numeric_columns(df)
    df.to_parquet(path, index=False)
    return path


def _coerce_mixed_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """object dtype 컬럼에 숫자 문자열과 float이 섞여 있으면(예: 아카이브 CSV와
    REST 폴백 데이터를 concat한 경우) pyarrow가 to_parquet에서 터진다.
    실제로 전부 숫자로 변환 가능한 object 컬럼은 미리 float으로 캐스팅해준다."""
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            # 원래 결측이 아니었는데 변환 후 NaN이 됐다면(=진짜 문자열 데이터) 건드리지 않음
            if converted.notna().equals(df[col].notna()):
                df[col] = converted
    return df


def load_parquet(category: str, filename: str) -> Optional[pd.DataFrame]:
    path = PATHS[category] / f"{filename}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def append_parquet(df: pd.DataFrame, category: str, filename: str, dedup_key: str = "open_time") -> Path:
    """
    기존 parquet에 새 데이터를 이어붙인다. gap-aware 증분 수집에서 사용.
    dedup_key 기준으로 중복 행은 제거하고 정렬해서 저장.

    concat 전에 (1) 빈 프레임은 제외하고 (2) 한쪽에서 all-NA인 컬럼은 반대쪽 dtype으로
    미리 캐스팅한다. 안 그러면 pandas FutureWarning(empty/all-NA concat 동작 변경 예정)이
    뜨고, 차기 pandas에서 결과 dtype이 조용히 달라질 수 있다.
    """
    existing = load_parquet(category, filename)

    if existing is None or existing.empty:
        combined = df.sort_values(dedup_key).reset_index(drop=True)
        return save_parquet(combined, category, filename)

    if df.empty:
        # 붙일 게 없으면 기존 파일을 다시 쓸 필요도 없다
        return PATHS[category] / f"{filename}.parquet"

    for col in existing.columns.intersection(df.columns):
        if existing[col].dtype == df[col].dtype:
            continue
        try:
            if df[col].isna().all():
                df[col] = df[col].astype(existing[col].dtype)
            elif existing[col].isna().all():
                existing[col] = existing[col].astype(df[col].dtype)
        except (TypeError, ValueError):
            pass  # 캐스팅 불가능한 조합이면 pandas 기본 동작에 맡긴다

    combined = pd.concat([existing, df], ignore_index=True)
    combined = combined.drop_duplicates(subset=[dedup_key]).sort_values(dedup_key).reset_index(drop=True)
    return save_parquet(combined, category, filename)


# ---------------------------------------------------------------------------
# JSON 범용 R/W (심볼 목록, 유니버스 스냅샷 등)
# ---------------------------------------------------------------------------

def save_json(obj, category: str, filename: str) -> Path:
    path = PATHS[category] / f"{filename}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load_json(category: str, filename: str):
    path = PATHS[category] / f"{filename}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# manifest: 심볼별 마지막 수집 완료 지점 (gap-aware 수집 모듈들이 사용)
#
# category별로 별도 manifest를 둔다. universe_probe(scan)와 full_collector
# (processed)는 서로 다른 데이터를 다른 주기로 수집하므로 진행 상황도 따로
# 추적해야 한다. 하나로 합치면 "scan은 다 됐는데 processed는 안 된" 상태를
# 구분할 수 없다.
# ---------------------------------------------------------------------------

def _manifest_path(category: str) -> Path:
    if category not in PATHS:
        raise ValueError(f"알 수 없는 category: {category}. PATHS 키 중 하나여야 함: {list(PATHS)}")
    return PATHS[category] / MANIFEST_FILENAME


def load_manifest(category: str) -> dict:
    with _manifest_lock:
        path = _manifest_path(category)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def get_last_collected(category: str, symbol: str) -> Optional[str]:
    """symbol의 마지막 수집 완료 지점(YYYY-MM-DD 또는 YYYY-MM). 기록 없으면 None -> 전체 백필 대상."""
    manifest = load_manifest(category)
    return manifest.get(symbol)


def update_last_collected(category: str, symbol: str, value: str) -> None:
    """수집 성공 후 반드시 호출. 이 값이 없으면 다음 실행 때 전체를 다시 백필하게 된다.
    스레드 안전: 락으로 갱신 유실을 막고, temp 파일 + os.replace(원자적 교체)로
    쓰는 도중 죽거나 다른 프로세스가 읽어도 반쯤 쓰인 파일이 보이지 않게 한다."""
    with _manifest_lock:
        manifest = load_manifest(category)
        manifest[symbol] = value
        path = _manifest_path(category)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)