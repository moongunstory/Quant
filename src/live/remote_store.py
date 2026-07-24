"""remote_store — S3 호환 오브젝트 스토리지(Cloudflare R2 / Oracle Cloud / Backblaze B2 등)
동기화 모듈.

왜 필요한가
-----------
Lambda 컨테이너의 파일시스템은 /tmp 를 빼면 전부 읽기전용이고, /tmp 는 실행이 끝나면
사라진다. 즉 라이브 매매에 필요한 데이터(패널·유니버스)와 매매가 남기는 상태(포지션·원장)를
Lambda '바깥'의 영속 저장소에 둬야 한다. 이 데이터는 본질적으로 parquet/JSON '파일'이라
S3 호환 오브젝트 스토리지가 가장 잘 맞는다.

AWS S3 대신 Cloudflare R2 / Oracle Cloud 등을 쓸 수 있는 이유
-----------------------------------------------------------
이 회사들은 전부 'S3 호환 API' 를 제공한다. boto3(파이썬 AWS SDK)에 접속주소(endpoint_url)와
키만 그 회사 것으로 바꿔주면 코드 한 줄 안 고치고 그대로 붙는다. 그래서 아래 설정은 전부
provider 중립적인 이름(S3_*)을 쓴다. R2 를 Oracle 로 바꾸고 싶으면 env 4개만 갈아끼우면 된다.

로컬 vs Lambda
--------------
- 로컬(REMOTE_STORE 미설정): 이 모듈은 완전히 비활성. import 해도 아무 동작 안 함. 기존 로컬
  parquet 흐름 그대로.
- Lambda(REMOTE_STORE=1): lambda_handler 가 시작할 때 sync_down 으로 R2 → /tmp 다운로드,
  매매 후 sync_up 으로 /tmp → R2 업로드.
- 초기 데이터 주입: 로컬에서 `python -m src.live.remote_store push` 를 돌리면 로컬 data/ 의
  패널·유니버스를 R2 로 올린다. Lambda 첫 실행 전에 한 번 해두면 콜드스타트 백필이 필요 없다.

환경변수
--------
  REMOTE_STORE          "1"/"true"/"r2"/"s3" 중 하나면 활성 (기본 미설정 = 비활성)
  S3_ENDPOINT_URL       예) https://<계정ID>.r2.cloudflarestorage.com
  S3_BUCKET             버킷 이름
  S3_ACCESS_KEY_ID      액세스 키
  S3_SECRET_ACCESS_KEY  시크릿 키
  S3_REGION             기본 "auto" (R2 는 auto, Oracle 등은 해당 리전)
  QUANT_RETENTION_DAYS  (선택) 지정 시 이 일수보다 오래된 날짜별 런타임 파일을 정리
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

from src.config.backtest_settings import SETTINGS

log = logging.getLogger("quant.remote_store")

# Lambda 가 시작할 때 R2 에서 내려받고(sync_down), 끝날 때 다시 올리는(sync_up) 데이터 하위경로.
# data_dir(=/tmp/quant-data) 기준 상대경로이며, 그대로 오브젝트 키가 된다.
# Lambda 가 '직접 수집'하므로 원본(processed)까지 왕복 보존해야, 매일 몇 년치를 다시 받지 않고
# 어제까지의 데이터에 오늘치만 이어붙이는 증분 수집이 된다.
#  - market/processed : full_collector 원본(심볼별 parquet). 증분 수집이 여기에 이어붙는다.
#  - market/panel     : 알파가 읽는 date×coin 패널(원본으로부터 재빌드됨).
#  - market/universe  : 시점별 top-100 스냅샷(패널 마스킹/유니버스 판단).
#  - market/scan      : universe_probe 경량 스캔(유니버스 재구성 시 필요).
#  - runtime          : 포지션/원장/텔레메트리 등 매매가 '남기는' 상태.
PERSIST_PREFIXES = ("market/processed", "market/panel", "market/universe", "market/scan", "runtime")
DOWN_PREFIXES = PERSIST_PREFIXES
UP_PREFIXES = PERSIST_PREFIXES


# ---------------------------------------------------------------------------
# 활성 여부 / 클라이언트
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """REMOTE_STORE 환경변수가 켜져 있으면 True."""
    return os.environ.get("REMOTE_STORE", "").strip().lower() in ("1", "true", "yes", "r2", "s3", "on")


def _bucket() -> str:
    b = os.environ.get("S3_BUCKET")
    if not b:
        raise RuntimeError("S3_BUCKET 환경변수가 없습니다.")
    return b


def _client():
    """boto3 S3 호환 클라이언트. boto3 는 무거우니 필요할 때만 import."""
    import boto3
    from botocore.config import Config

    endpoint = os.environ.get("S3_ENDPOINT_URL")
    if not endpoint:
        raise RuntimeError("S3_ENDPOINT_URL 환경변수가 없습니다.")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("S3_REGION", "auto"),
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )


def _data_dir() -> Path:
    return Path(SETTINGS.data_dir)


# ---------------------------------------------------------------------------
# 다운로드 / 업로드
# ---------------------------------------------------------------------------

def _iter_remote_keys(client, prefix: str) -> Iterable[str]:
    """버킷에서 prefix 로 시작하는 모든 오브젝트 키를 순회(페이지네이션 처리)."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):  # 디렉토리 표식은 건너뜀
                yield key


def sync_down(prefixes: Iterable[str] = DOWN_PREFIXES) -> int:
    """R2 → 로컬(data_dir). prefix 하위 오브젝트를 내려받는다. 받은 파일 수 반환.

    (2026-07-24) 로컬에 '같은 크기'로 이미 존재하는 파일은 건너뛴다 — 원자료가 ~8GB 라
    매 실행 전체 다운로드는 Lambda 시간/네트워크 낭비다. 따뜻한(재사용) 컨테이너에서는
    /tmp 에 어제 데이터가 남아 있어 대부분 스킵되고, 콜드스타트(빈 /tmp)에서는 전과
    동일하게 전체를 받는다(스킵 대상이 없으므로 동작 변화 없음)."""
    if not is_enabled():
        return 0
    client = _client()
    root = _data_dir()

    keys_to_download = []
    skipped = 0
    for prefix in prefixes:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                local = root / key
                if local.exists() and local.stat().st_size == obj["Size"]:
                    skipped += 1
                    continue
                keys_to_download.append(key)
    if skipped:
        log.info("R2 다운로드 스킵: %d개 파일(로컬과 크기 동일 — 변경 없음)", skipped)

    n = len(keys_to_download)
    if n > 0:
        from concurrent.futures import ThreadPoolExecutor
        def download_one(key):
            dest = root / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(_bucket(), key, str(dest))
            
        with ThreadPoolExecutor(max_workers=20) as executor:
            executor.map(download_one, keys_to_download)
            
    log.info("R2 다운로드 완료: %d개 파일 (prefixes=%s)", n, list(prefixes))
    return n


def sync_down_files(keys: Iterable[str]) -> int:
    """R2 → 로컬. '특정 오브젝트 키'만 골라 내려받는다(웹훅 조회 명령용, 빠름).

    prefix 전체(수백 개)를 훑는 sync_down 과 달리, config/positions/equity 처럼 소수의
    파일만 콕 집어 받으므로 텔레그램 웹훅이 몇 초 안에 응답할 수 있다. 원격에 없는 키는
    조용히 건너뛴다(예: 진입가 스냅샷이 아직 없는 초기 상태)."""
    if not is_enabled():
        return 0
    client = _client()
    root = _data_dir()
    keys_list = list(keys)
    n = 0
    if keys_list:
        from concurrent.futures import ThreadPoolExecutor
        def download_one(key):
            dest = root / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                client.download_file(_bucket(), key, str(dest))
                return 1
            except Exception as e:
                log.debug("sync_down_files: %s 없음/실패(건너뜀): %s", key, e)
                return 0
                
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(download_one, keys_list)
            n = sum(results)
            
    log.info("R2 타깃 다운로드: %d개 파일", n)
    return n


def _iter_local_files(prefix: str) -> Iterable[Path]:
    base = _data_dir() / prefix
    if not base.exists():
        return
    for p in base.rglob("*"):
        if p.is_file():
            yield p


# 크기 비교로 '안 바뀐 파일' 업로드를 건너뛰는 prefix (대용량 원자료·캐시).
# parquet 은 내용이 바뀌면 압축 결과 크기가 사실상 항상 달라지므로 크기 비교로 충분하다.
# runtime(포지션/원장)과 universe(스냅샷 JSON)는 작지만 절대 유실되면 안 되므로
# 비교 없이 무조건 업로드한다.
_SIZE_SKIP_PREFIXES = ("market/processed", "market/panel", "market/scan")


def _remote_size_map(client, prefix: str) -> dict:
    """{key: size} — prefix 하위 원격 오브젝트 크기 맵(증분 업로드 판단용)."""
    sizes = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for obj in page.get("Contents", []):
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def sync_up(prefixes: Iterable[str] = UP_PREFIXES) -> int:
    """로컬(data_dir) → R2. prefix 하위 파일을 업로드. 올린 파일 수 반환.

    (2026-07-24) 대용량 prefix(market/processed·panel·scan)는 원격과 '크기가 같은'
    파일을 건너뛴다 — 예전엔 매 실행 수백 개의 몇 년치 원본 parquet 를 통째로 재업로드해
    Lambda 실행시간을 태웠다(그만큼 수집/매매 예산이 줄었다). 하루에 실제로 바뀌는 건
    증분 수집이 이어붙인 파일들뿐이다. runtime/universe 는 무조건 업로드(유실 방지)."""
    if not is_enabled():
        return 0
    client = _client()
    root = _data_dir()

    files_to_upload = []
    skipped = 0
    for prefix in prefixes:
        size_check = any(prefix.startswith(p) or p.startswith(prefix)
                         for p in _SIZE_SKIP_PREFIXES)
        remote_sizes = _remote_size_map(client, prefix) if size_check else {}
        for path in _iter_local_files(prefix):
            key = path.relative_to(root).as_posix()
            if size_check and key.startswith(_SIZE_SKIP_PREFIXES) \
                    and remote_sizes.get(key) == path.stat().st_size:
                skipped += 1
                continue
            files_to_upload.append(path)
    if skipped:
        log.info("R2 업로드 스킵: %d개 파일(원격과 크기 동일 — 변경 없음)", skipped)

    n = len(files_to_upload)
    if n > 0:
        from concurrent.futures import ThreadPoolExecutor
        def upload_one(path):
            key = path.relative_to(root).as_posix()
            client.upload_file(str(path), _bucket(), key)
            
        with ThreadPoolExecutor(max_workers=20) as executor:
            executor.map(upload_one, files_to_upload)
            
    log.info("R2 업로드 완료: %d개 파일 (prefixes=%s)", n, list(prefixes))
    return n


# ---------------------------------------------------------------------------
# 보존기간 정리 (오래된 데이터 자동 삭제)
# ---------------------------------------------------------------------------

def prune_old_runtime(retention_days: Optional[int] = None) -> int:
    """runtime 하위의 '날짜별' 파일 중 retention_days 보다 오래된 것을 로컬에서 삭제.

    orders_YYYY-MM-DD.json, telemetry-YYYY-MM-DD.json 처럼 날짜가 파일명에 든 것만 대상.
    삭제 후 sync_up 하면 R2 에서도 사라진다(다음 다운로드 대상에서 빠짐).
    retention_days=None 이거나 QUANT_RETENTION_DAYS 미설정이면 아무것도 안 함.

    참고: 패널/유니버스의 '오래된 행' 정리는 데이터를 만드는 로컬에서 관리한다. Lambda 는
    로컬이 올린 걸 그대로 읽을 뿐이라, 여기서는 무한히 쌓이는 날짜별 런타임 파일만 정리한다."""
    if retention_days is None:
        env = os.environ.get("QUANT_RETENTION_DAYS")
        retention_days = int(env) if env and env.strip() else None
    if not retention_days or retention_days <= 0:
        return 0

    cutoff = datetime.now(timezone.utc).date() - timedelta(days=retention_days)
    runtime = _data_dir() / "runtime"
    if not runtime.exists():
        return 0

    removed = 0
    for path in runtime.rglob("*"):
        if not path.is_file():
            continue
        d = _date_in_name(path.name)
        if d is not None and d < cutoff:
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        log.info("보존기간(%d일) 초과 런타임 파일 %d개 정리", retention_days, removed)
    return removed


def _date_in_name(name: str) -> Optional[date]:
    """파일명에 든 YYYY-MM-DD 를 date 로. 없으면 None."""
    import re
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 초기 주입(로컬 → R2): 매매에 필요한 '스코프'만 올린다
#
# 로컬 data/ 에는 전체 유니버스 원본(7.7GB, metrics 만 6.4GB)이 있지만, 라이브 매매엔
# 채택 알파가 실제로 쓰는 데이터셋 × 현재 top-100 심볼만 필요하다(약 1.6GB). 전체를 올리면
# R2 무료 한도(10GB)를 낭비하므로, live_refresh.compute_scope 와 동일한 기준으로 좁혀서 올린다.
# 이후 Lambda 가 매일 이 스코프에 오늘치만 증분 수집한다.
# ---------------------------------------------------------------------------

def _latest_universe_members() -> set:
    """최신 universe_snapshot 의 멤버(현재 top-100). 없으면 빈 set."""
    import json
    d = _data_dir() / "market" / "universe"
    snaps = [p for p in sorted(d.glob("[0-9]*.json")) if not p.stem.endswith("_diff")]
    if not snaps:
        return set()
    try:
        return set(json.loads(snaps[-1].read_text(encoding="utf-8")).get("members", []))
    except Exception:
        return set()


def _scope_datasets() -> list:
    """채택 알파들이 실제로 쓰는 데이터셋 목록(live_refresh 와 동일 기준)."""
    from src.portfolio.spec import load_portfolio_spec
    from src.collector.live_refresh import compute_scope
    cfg_path = _data_dir() / "strategy" / "portfolio" / "config.json"
    cfg = load_portfolio_spec(str(cfg_path))
    return compute_scope(cfg)["datasets"]


def push_initial() -> int:
    """로컬 → R2 초기 주입. panel/universe/scan 전체 + 스코프 processed(데이터셋×top100)만."""
    client = _client()
    root = _data_dir()
    
    files_to_upload = []

    # 1) 패널·유니버스·스캔은 통째로 (작음)
    for prefix in ("market/panel", "market/universe", "market/scan"):
        for path in _iter_local_files(prefix):
            files_to_upload.append(path)

    # 2) 원본(processed)은 스코프 데이터셋 × top-100 심볼 + 각 데이터셋 manifest 만
    members = _latest_universe_members()
    datasets = _scope_datasets()
    for ds in datasets:
        dsdir = root / "market" / "processed" / ds
        if not dsdir.exists():
            continue
        mani = dsdir / "_manifest.json"
        if mani.exists():
            files_to_upload.append(mani)
        for sym in members:
            f = dsdir / f"{sym}.parquet"
            if f.exists():
                files_to_upload.append(f)

    n = len(files_to_upload)
    if n > 0:
        from concurrent.futures import ThreadPoolExecutor
        def upload_one(path):
            client.upload_file(str(path), _bucket(), path.relative_to(root).as_posix())
            
        with ThreadPoolExecutor(max_workers=20) as executor:
            executor.map(upload_one, files_to_upload)

    log.info("초기 주입 완료: %d개 파일 (datasets=%s, top100=%d종목)", n, datasets, len(members))
    return n


# ---------------------------------------------------------------------------
# 로컬용 CLI: 초기 데이터 주입 / 확인
# ---------------------------------------------------------------------------

def _cli(argv=None):
    """python -m src.live.remote_store {push|pull|list}

    push : 로컬 data/ 의 패널·유니버스를 R2 로 업로드(초기 주입/갱신).
    pull : R2 의 매매 데이터를 로컬로 내려받음(점검용).
    list : R2 에 올라간 오브젝트 목록/개수 확인.

    로컬에서 실행할 때도 REMOTE_STORE=1 및 S3_* 환경변수가 필요하다(.env 에 넣어두면 됨).
    """
    import argparse
    from dotenv import load_dotenv

    # 로컬 .env 로드(REMOTE_STORE, S3_* 를 여기서 읽음)
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    p = argparse.ArgumentParser(description="R2/S3 호환 스토리지 동기화")
    p.add_argument("action", choices=["push", "pull", "list"])
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not is_enabled():
        raise SystemExit("REMOTE_STORE 가 꺼져 있습니다. .env 에 REMOTE_STORE=1 과 S3_* 를 설정하세요.")

    if args.action == "push":
        n = push_initial()
        print(f"초기 주입 완료: {n}개 파일 → 버킷 {_bucket()}")
    elif args.action == "pull":
        n = sync_down(DOWN_PREFIXES)
        print(f"다운로드 완료: {n}개 파일 ← 버킷 {_bucket()}")
    elif args.action == "list":
        client = _client()
        total = 0
        for prefix in DOWN_PREFIXES:
            cnt = sum(1 for _ in _iter_remote_keys(client, prefix))
            print(f"  {prefix}: {cnt}개")
            total += cnt
        print(f"합계: {total}개 오브젝트 (버킷 {_bucket()})")


if __name__ == "__main__":
    _cli()
