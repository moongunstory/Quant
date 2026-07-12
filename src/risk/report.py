"""report — 리스크 파이프라인의 stage별 성과 추적.

각 리스크 모듈을 적용한 '직후'의 북 순손익을 4지표로 요약한다:

    sharpe       위험대비수익 (연환산)
    mdd          최대낙폭 (작을수록 안전)
    ann_return   연환산 수익률 (1단위 북)
    vol          자산변동률 = 순손익의 연환산 실현변동성

목적: `input -> position_cap -> gross_cap -> vol_target -> mdd_killswitch -> ...`
각 단계에서 위 4지표가 어떻게 변하는지 한 표로 보여, "어느 모듈이 도움/손해인지"
즉시 판단. Phase 1 에서 risk.run_risk_pipeline() 이 각 stage 의 net_pnl 을 넘겨
이 리포트를 만든다.

인터페이스(파이프라인 아직 없어도 지금 확정):
    report_from_stage_pnls([(stage_name, net_pnl_series), ...]) -> {"rows": [...]}
    format_table(report)  -> str            (콘솔 출력용)
    save(report, tag)     -> (json_path, csv_path)   (logs/ 에 저장)
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.backtest import metrics as M
from src.config.backtest_settings import SETTINGS

# stage 마다 남길 지표(순서 = 표 컬럼 순서). 확장 시 여기만 바꾼다.
STAGE_METRIC_KEYS = ("sharpe", "mdd", "ann_return", "vol")


def stage_metrics(net_pnl) -> dict:
    """한 stage 의 순손익 시리즈 -> 4지표 dict."""
    s = pd.Series(net_pnl)
    return {
        "sharpe": M.sharpe(s),
        "mdd": M.max_drawdown(s),
        "ann_return": M.ann_return(s),
        "vol": M.realized_vol(s),
    }


def report_from_stage_pnls(stage_pnls) -> dict:
    """[(stage_name, net_pnl_series), ...] -> {"rows": [{stage, **metrics, delta_*}]}.

    각 행에 이전 stage 대비 변화량(delta_sharpe 등)도 넣어, 모듈 하나가
    지표를 얼마나 움직였는지 바로 읽히게 한다. net_pnl 이 None(비활성 stage)이면
    지표는 None."""
    rows = []
    prev = None
    for name, pnl in stage_pnls:
        if pnl is None:
            rows.append({"stage": name, "enabled": False,
                         **{k: None for k in STAGE_METRIC_KEYS}})
            continue
        m = stage_metrics(pnl)
        row = {"stage": name, "enabled": True, **m}
        if prev is not None:
            for k in STAGE_METRIC_KEYS:
                row[f"delta_{k}"] = m[k] - prev[k]
        rows.append(row)
        prev = m
    return {"rows": rows}


def report_from_pipeline_stages(stages) -> dict:
    """risk.run_risk_pipeline() 의 stages 리스트(각 dict 에 sharpe/mdd/
    ann_return/vol 이미 포함)를 그대로 리포트 rows 로 변환하고, 이전 stage 대비
    delta 를 붙인다."""
    rows = []
    prev = None
    for st in stages:
        row = {"stage": st["stage"], "enabled": st.get("enabled", True)}
        for k in STAGE_METRIC_KEYS:
            row[k] = st.get(k)
        if st.get("enabled", True) and prev is not None:
            for k in STAGE_METRIC_KEYS:
                if row[k] is not None and prev.get(k) is not None:
                    row[f"delta_{k}"] = row[k] - prev[k]
        if st.get("enabled", True):
            prev = {k: st.get(k) for k in STAGE_METRIC_KEYS}
        rows.append(row)
    return {"rows": rows}


def _verdict(d_sharpe, d_mdd, d_vol=None) -> str:
    """한 stage 가 도움인지 손해인지 한 단어로. 리스크 모듈은 보통 낙폭(mdd)이나
    변동성(vol)을 줄이려고 샤프를 조금 깎는다 — 그래서 샤프만 보면 오판한다.
    낙폭·변동성 변화까지 같이 본다(둘 다 작을수록 안전 → 음수 = 위험 감소)."""
    if d_sharpe is None:
        return "기준"
    if d_sharpe > 0.02:
        return "도움(샤프↑)"
    if d_sharpe < -0.02:
        # 샤프는 깎였지만 낙폭 또는 변동성을 줄였으면 '의도된 맞바꿈', 아니면 순손해.
        cut_risk = (d_mdd is not None and d_mdd < -0.005) or \
                   (d_vol is not None and d_vol < -0.005)
        return "위험↓(샤프댓가)" if cut_risk else "손해"
    return "중립"


def format_table(report) -> str:
    """report -> 콘솔용 정렬 표 문자열. 각 stage 의 절대값 + 직전 대비 변화(Δ)와
    도움/손해 판정을 함께 보여, '이 리스크 모듈이 실제로 도움이 되는지' 바로 읽힌다."""
    header = (f"{'stage':<22}{'sharpe':>9}{'Δsharpe':>9}"
              f"{'mdd':>8}{'Δmdd':>8}{'ann_ret':>9}{'vol':>8}  판정")
    lines = [header, "-" * 82]
    for r in report["rows"]:
        if not r.get("enabled", True):
            lines.append(f"{r['stage']:<22}{'(disabled)':>25}")
            continue
        sh, mdd = r.get("sharpe"), r.get("mdd")
        ar, vol = r.get("ann_return"), r.get("vol")
        d_sh, d_mdd, d_vol = r.get("delta_sharpe"), r.get("delta_mdd"), r.get("delta_vol")
        c_sh = f"{sh:>9.3f}" if isinstance(sh, (int, float)) else f"{'-':>9}"
        c_dsh = f"{d_sh:>+9.3f}" if isinstance(d_sh, (int, float)) else f"{'-':>9}"
        c_mdd = f"{mdd:>8.3f}" if isinstance(mdd, (int, float)) else f"{'-':>8}"
        c_dmdd = f"{d_mdd:>+8.3f}" if isinstance(d_mdd, (int, float)) else f"{'-':>8}"
        c_ar = f"{ar:>9.3f}" if isinstance(ar, (int, float)) else f"{'-':>9}"
        c_vol = f"{vol:>8.3f}" if isinstance(vol, (int, float)) else f"{'-':>8}"
        verdict = _verdict(d_sh, d_mdd, d_vol)
        lines.append(f"{r['stage']:<22}{c_sh}{c_dsh}{c_mdd}{c_dmdd}{c_ar}{c_vol}  {verdict}")
    return "\n".join(lines)


def _logs_dir() -> Path:
    d = SETTINGS.data_dir.parent / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(report, tag="risk_stages") -> tuple[Path, Path]:
    """report 를 logs/<tag>-<UTC타임스탬프>.{json,csv} 로 저장."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    d = _logs_dir()
    json_path = d / f"{tag}-{ts}.json"
    csv_path = d / f"{tag}-{ts}.csv"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                         encoding="utf-8")

    rows = report["rows"]
    fieldnames = ["stage", "enabled"] + list(STAGE_METRIC_KEYS) + \
                 [f"delta_{k}" for k in STAGE_METRIC_KEYS]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return json_path, csv_path
