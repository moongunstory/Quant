"""reset — /초기화 명령의 범위별 삭제 로직.

왜 이 모듈이 필요한가
--------------------
예전 /초기화 는 positions.json 하나만 비웠다. 그런데 사용자가 "내역"으로 느끼는
것들은 전부 다른 파일에 있다:

  - paper_equity.jsonl   : 누적 수익곡선 (대시보드 총자산·누적%·/잔고)
  - positions_entry.json : 진입가 스냅샷 (/잔고 실시간 손익)
  - telemetry-<date>.json: 사이클 기록 (/기록·/이유)
  - orders_<date>.json   : 주문 기록
  - logs/live-<date>.jsonl: 이벤트 로그

그래서 초기화해도 총자산/수익률/매매기록이 그대로 보였다. 이 모듈은 범위를
골라서 지울 수 있게 한다.

범위(scope)
-----------
  포지션 : 보유 포지션 + 진입가 스냅샷만 비움 (수익 기록은 유지)
  잔고   : 포지션 + 누적 수익곡선(paper_equity.jsonl)까지 리셋
  전체   : 잔고 + 날짜별 기록(텔레메트리/주문/이벤트로그) 전부 삭제

'빈 파일 덮어쓰기' vs 'R2 삭제 + 마커'
--------------------------------------
positions/entry/equity 같은 단일 상태 파일은 **빈 내용으로 덮어쓴다**.
remote_store 의 sync 는 업로드/다운로드만 있고 원격 삭제가 없으므로, 빈 파일을
R2 에 올려두면 어느 컨테이너가 받아도 초기화 상태가 그대로 전파된다(안전).

반면 telemetry-<date>.json 처럼 날짜별로 쌓이는 파일은 덮어쓸 대상이 아니라서
R2 에서 **직접 삭제**해야 한다. 문제는 warm Lambda 컨테이너의 /tmp 에 옛 사본이
남아 있으면 다음 사이클의 sync_up 이 도로 올려 기록이 '부활'할 수 있다는 것.
그래서 초기화 시각을 last_reset.json 마커로 남기고(이 마커도 runtime 이라 R2 로
전파됨), 매 사이클 시작 때 enforce_reset_marker() 가 마커 날짜보다 오래된
날짜별 파일을 로컬에서 지운다. (마커 당일 파일은 초기화 이후 새로 생긴 것일 수
있어 건드리지 않는다 — 최악의 잔존은 '초기화 당일 하루치'뿐.)

전부 표준 라이브러리만 사용(Lambda 호환).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

from src.config.backtest_settings import SETTINGS
from src.live.ledger import TELEMETRY_DIR

log = logging.getLogger("quant.live.reset")

LIVE_DIR = SETTINGS.data_dir / "runtime" / "live"
POSITIONS_PATH = LIVE_DIR / "positions.json"
ENTRY_PATH = LIVE_DIR / "positions_entry.json"
EQUITY_PATH = LIVE_DIR / "paper_equity.jsonl"
MARKER_PATH = LIVE_DIR / "last_reset.json"

# 범위 이름 → 짧은 설명 (도움말/미리보기 공용)
SCOPES = {
    "포지션": "보유 포지션 + 진입가 스냅샷만 비움 (수익 기록 유지)",
    "잔고": "포지션 + 누적 수익곡선(총자산·수익률) 리셋",
    "전체": "잔고 + 매매기록(/기록·/이유·텔레메트리·주문·로그) 전부 삭제",
}


def _overwrite(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _dated_local_files() -> list[Path]:
    """날짜별로 쌓이는 기록 파일들(로컬). 전체 초기화 때 삭제 대상."""
    out: list[Path] = []
    tel = Path(TELEMETRY_DIR)
    if tel.exists():
        out += list(tel.glob("telemetry-????-??-??.json"))
        out += list((tel / "bundles").glob("*.zip"))
    if LIVE_DIR.exists():
        out += list(LIVE_DIR.glob("orders_????-??-??.json"))
    logs = SETTINGS.logs_dir
    if logs.exists():
        out += list(logs.glob("live-????-??-??.jsonl"))
    return out


def _dated_remote_prefixes() -> list[str]:
    """전체 초기화 때 R2 에서 지울 prefix 들(오브젝트 키 기준)."""
    return [
        "runtime/live/telemetry/",
        "runtime/logs/",
    ]


def preview(scope: str) -> list[str]:
    """이 범위로 초기화하면 뭐가 지워지는지 사람이 읽는 설명 목록."""
    lines = ["• 보유 포지션(positions.json) → 빈 상태",
             "• 진입가 스냅샷(실시간 손익 기준점) → 빈 상태"]
    if scope in ("잔고", "전체"):
        lines.append("• 누적 수익곡선(총자산/수익률/자산곡선) → 0부터 다시 시작")
    if scope == "전체":
        n = len(_dated_local_files())
        lines.append(f"• 매매기록·텔레메트리·주문기록·이벤트로그 삭제 (로컬 {n}개 파일 + R2)")
        lines.append("  └ /기록·/이유 명령의 과거 데이터도 사라져요")
    return lines


def apply_reset(scope: str, today=None) -> dict:
    """초기화 실행. 반환: {"scope", "cleared": [...], "deleted_local": n, "deleted_remote": n}.

    R2 삭제는 remote_store 가 켜져 있을 때만(로컬 개발 환경에서는 로컬만).
    """
    if scope not in SCOPES:
        raise ValueError(f"알 수 없는 초기화 범위: {scope}")
    now = datetime.now(timezone.utc)
    cleared: list[str] = []

    # 1) 단일 상태 파일: 빈 내용으로 덮어쓰기(→ sync_up 이 R2 에 전파)
    _overwrite(POSITIONS_PATH, "{}")
    _overwrite(ENTRY_PATH, "{}")
    cleared += ["포지션", "진입가 스냅샷"]

    if scope in ("잔고", "전체"):
        # 자산 곡선의 시작점(0.0%)을 마커 당일 날짜로 추가하여, 렌더링 시 0에서 시작해서 위/아래로 움직이도록 함.
        reset_date_str = now.date().isoformat()
        start_rec = {
            "date": reset_date_str,
            "day_pnl": 0.0,
            "gross_pnl": 0.0,
            "trade_cost": 0.0,
            "turnover": 0.0,
            "btc_return": 0.0,
            "equity": 0.0
        }
        _overwrite(EQUITY_PATH, json.dumps(start_rec, ensure_ascii=False) + "\n")
        cleared.append("누적 수익곡선")

    deleted_local = 0
    deleted_remote = 0
    if scope == "전체":
        # 2) 날짜별 기록: 로컬 삭제
        for p in _dated_local_files():
            try:
                p.unlink()
                deleted_local += 1
            except OSError:
                pass

        # 3) R2 에서도 삭제(안 지우면 다음 sync_down 때 되살아난다)
        try:
            from src.live import remote_store as RS
            if RS.is_enabled():
                keys: list[str] = []
                for prefix in _dated_remote_prefixes():
                    keys += RS.list_keys(prefix)
                # runtime/live/ 바로 아래의 orders_*.json
                keys += [k for k in RS.list_keys("runtime/live/orders_")]
                deleted_remote = RS.delete_keys(keys)
        except Exception as e:
            log.error("R2 기록 삭제 실패(로컬은 삭제됨): %s", e, exc_info=True)

        cleared.append("매매기록(텔레메트리/주문/로그)")

    # 4) 초기화 마커: warm 컨테이너의 옛 사본이 sync_up 으로 부활하는 것을
    #    enforce_reset_marker() 가 막을 수 있도록 시각을 남긴다.
    _overwrite(MARKER_PATH, json.dumps(
        {"ts": now.isoformat(), "scope": scope}, ensure_ascii=False, indent=2))

    log.info("초기화 완료: scope=%s local=%d remote=%d", scope, deleted_local, deleted_remote)
    return {"scope": scope, "cleared": cleared,
            "deleted_local": deleted_local, "deleted_remote": deleted_remote}


def enforce_reset_marker() -> int:
    """매 사이클 시작(sync_down 직후) 호출: '전체' 초기화 마커보다 오래된 날짜별
    기록 파일이 로컬(/tmp)에 남아 있으면 지운다(부활 방지). 반환: 지운 파일 수."""
    p = Path(MARKER_PATH)
    if not p.exists():
        return 0
    try:
        marker = json.loads(p.read_text(encoding="utf-8"))
        if marker.get("scope") != "전체":
            return 0
        reset_day = date.fromisoformat(marker["ts"][:10])
    except Exception:
        return 0

    import re
    removed = 0
    for f in _dated_local_files():
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if not m:
            continue
        try:
            d = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if d < reset_day:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        log.info("초기화 마커(%s) 이전 잔존 기록 %d개 정리(부활 방지)", reset_day, removed)
    return removed
