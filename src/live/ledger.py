"""ledger — 라이브 사이클의 주문/체결/포지션 + 상세 텔레메트리를 기록.

목적: 백테스트가 가정한 성과 대비 '실제(또는 페이퍼) 실행'이 어떻게 달랐는지
사후 추적(audit)할 근거를 남긴다. 두 갈래의 기록을 담당한다.

  1) 이벤트 로그(record): logs/live-<date>.jsonl 에 한 줄=한 이벤트로 append.
  2) 텔레메트리 스냅샷(record_telemetry): 실행 시점의 결합/리스크/목표 가중치와
     실현수익률을 통째로 telemetry-<date>.json 으로 남겨, 백테스트를 다시 돌리지
     않고도 알파 기여도·리스크 오버레이 drag/benefit 을 역추적할 수 있게 한다.
     build_telemetry_bundle 로 최근 N일을 zip 으로 묶어 텔레그램 전송에 쓴다.

페이퍼와 실거래가 '동일 코드경로'를 타므로 같은 ledger 를 공유한다.
표준 라이브러리만 사용(Lambda 등 경량 환경 호환).
"""
from __future__ import annotations

import json
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.config.backtest_settings import SETTINGS

# 텔레메트리 스냅샷 저장 위치(하루 한 파일). 이 값/스키마가 바뀌면 attribution 파서도 확인.
TELEMETRY_DIR = SETTINGS.data_dir / "runtime" / "live" / "telemetry"
TELEMETRY_SCHEMA_VERSION = 1


def _ledger_path(today=None):
    today = today or datetime.now(timezone.utc).date()
    d = SETTINGS.logs_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"live-{today.isoformat()}.jsonl"


def record(event_type, payload, today=None):
    """한 이벤트 append. event_type: 'cycle'|'orders'|'target'|'fill'|'error' 등."""
    row = {"ts": datetime.now(timezone.utc).isoformat(),
           "type": event_type, **payload}
    path = _ledger_path(today)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return path


# ---------------------------------------------------------------- 텔레메트리
def telemetry_path(day, telemetry_dir=None) -> Path:
    """해당 날짜의 telemetry-<date>.json 경로."""
    d = Path(telemetry_dir or TELEMETRY_DIR)
    return d / f"telemetry-{day.isoformat()}.json"


def record_telemetry(target, order_record, prev_positions=None, today=None,
                     mode="paper", telemetry_dir=None) -> Path | None:
    """한 사이클의 텔레메트리 스냅샷을 telemetry-<date>.json 으로 저장.

    target       : target_weights.compute_target_weights() 결과 dict.
    order_record : orders.generate_orders() 결과 dict.
    prev_positions : 이번 사이클 '직전' 보유 포지션({coin: weight}). 이 포지션이
                     당일 day_returns 를 벌었으므로 사후 손익 재구성(lag-1)에 쓰인다.
    """
    today = today or datetime.now(timezone.utc).date()
    diag = target.get("diagnostics", {}) or {}

    snapshot = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "date": target.get("date", today.isoformat()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        # --- 알파 라인업 / 신선도 ---
        "held_alphas": target.get("held_alphas", []),
        "stale_alphas": diag.get("stale_alphas", []),
        "all_alphas_stale": bool(diag.get("all_alphas_stale", False)),
        "target_row_date": diag.get("target_row_date"),
        # --- 결합 비중(블렌드) : {alpha -> scalar} ---
        "alpha_weights": target.get("alpha_weights", {}),
        # --- 가중치 3종 (기여도/리스크 분석의 핵심) ---
        "pre_risk_weights": target.get("pre_risk_weights", {}),   # 리스크 前 결합북(Σ|w|≈1)
        "target_weights": target.get("weights", {}),              # 리스크 後 최종 목표
        "prev_positions": prev_positions or {},                   # 당일 수익을 실제 번 포지션
        "alpha_contributions": target.get("alpha_contributions", {}),  # Σ = pre_risk_weights
        # --- 실현 수익률(당일, MARK 기준) : {coin -> return} ---
        "day_returns": target.get("day_returns", {}),
        # --- 리스크 파이프라인 stage 성과(sharpe/mdd/...) ---
        "risk_stages": target.get("risk_stages", []),
        # --- 주문 요약(현실 체결 audit) ---
        "orders": {
            "mode": order_record.get("mode", mode),
            "skipped": order_record.get("skipped", False),
            "skip_reason": order_record.get("skip_reason"),
            "n_orders": order_record.get("n_orders", 0),
            "drift": order_record.get("drift"),
            "rebalance_band": order_record.get("rebalance_band"),
            "orders": order_record.get("orders", []),
        },
    }

    path = telemetry_path(today, telemetry_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return path


def list_recent_telemetry(days=30, today=None, telemetry_dir=None):
    """최근 `days` 일 범위 telemetry 파일 경로 리스트(날짜 오름차순)."""
    today = today or datetime.now(timezone.utc).date()
    d = Path(telemetry_dir or TELEMETRY_DIR)
    if not d.exists():
        return []
    cutoff = today - timedelta(days=days - 1)
    files = []
    for p in d.glob("telemetry-????-??-??.json"):
        try:
            file_day = date.fromisoformat(p.stem.replace("telemetry-", ""))
        except ValueError:
            continue
        if cutoff <= file_day <= today:
            files.append((file_day, p))
    return [p for _, p in sorted(files)]


def build_telemetry_bundle(days=30, today=None, telemetry_dir=None, out_dir=None):
    """최근 `days` 일 telemetry 파일을 하나의 zip 으로 묶어 경로 반환(없으면 None)."""
    today = today or datetime.now(timezone.utc).date()
    files = list_recent_telemetry(days=days, today=today, telemetry_dir=telemetry_dir)
    if not files:
        return None
    start = files[0].stem.replace("telemetry-", "")
    end = files[-1].stem.replace("telemetry-", "")
    out_dir = Path(out_dir or (Path(telemetry_dir or TELEMETRY_DIR) / "bundles"))
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"telemetry-bundle-{start}_{end}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)
    return zip_path
