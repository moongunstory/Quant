from __future__ import annotations

import os
from typing import List

import pandas as pd
from sklearn.metrics import accuracy_score

from .config import DailyConfig


def print_and_save_report(
    df_log: pd.DataFrame,
    cfg: DailyConfig,
    as_of_date: pd.Timestamp,
) -> None:
    """간단한 한국어 성능 리포트 출력 + txt 저장."""
    lines: List[str] = []
    lines.append("=== 일일 성능 리포트 ===")
    lines.append(f"기준 날짜: {as_of_date.date()}")
    lines.append("")

    for d in cfg.horizons_days:
        sub = df_log[df_log["horizon_days"] == d]
        sub = sub.dropna(subset=["realized_label"])
        if len(sub) == 0:
            lines.append(f"- {d}일 후 수익률: 아직 평가 가능한 샘플 없음")
            continue

        acc = accuracy_score(sub["realized_label"], sub["pred_label"])
        lines.append(
            f"- {d}일 후 수익률: 샘플 {len(sub)}개, 방향 정확도 {acc:.3f}"
        )

    text = "\n".join(lines)
    print(text)

    if cfg.save_report:
        os.makedirs(cfg.report_dir, exist_ok=True)
        fname = f"daily_report_{as_of_date.date()}.txt"
        fpath = os.path.join(cfg.report_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text + "\n")
