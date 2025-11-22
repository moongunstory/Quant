from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_trials_for_horizon(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"trials 파일이 없습니다: {path}")
    return pd.read_parquet(path)


def select_best_config_from_trials(
    df_trials: pd.DataFrame,
    horizon_days: int,
    metric_col: str = "val_score",
    mode: str = "max",
) -> Optional[Dict]:
    """
    trials DataFrame에서 주어진 horizon에 대해
    metric 기준으로 가장 좋은 설정 1개를 뽑는다.
    """
    sub = df_trials[df_trials["horizon_days"] == horizon_days].copy()
    sub = sub[sub["status"] == "ok"]
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=[metric_col])

    if sub.empty:
        return None

    if mode == "max":
        idx = sub[metric_col].idxmax()
    else:
        idx = sub[metric_col].idxmin()

    row = sub.loc[idx]

    # lgbm_ prefix 붙은 컬럼만 따로 모아 dict로 변환
    lgbm_params: Dict[str, float] = {}
    for col in sub.columns:
        if col.startswith("lgbm_"):
            key = col[len("lgbm_") :]
            lgbm_params[key] = row[col]

    best_cfg: Dict = {
        "horizon_days": int(row["horizon_days"]),
        "window_days": int(row["window_days"]),
        "threshold": float(row["threshold"]),
        "feature_group": row.get("feature_group", "all"),
        "metric": metric_col,
        "val_score": float(row[metric_col]),
        "lgbm_params": lgbm_params,
    }
    return best_cfg


def save_best_configs(
    best_dir: Path,
    best_map: Dict[int, Dict],
    filename: str = "",
) -> None:
    """
    best_dir 아래에 horizon별 best 설정을 json으로 저장.

    - filename을 지정하지 않으면 horizon별 개별 파일:
      best_config_{3d,7d,...}.json
    - filename을 지정하면 (예: best_config_daily.json)
      전체를 한 파일에 horizon_days 키로 묶어서 저장.
    """
    best_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        # 전체를 묶어서 하나의 json 파일로 저장
        out_path = best_dir / filename
        # 키를 문자열로 바꿔서 저장 (json 호환성)
        payload = {str(k): v for k, v in best_map.items()}
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return

    # horizon별 개별 파일로 저장
    for horizon_days, cfg in best_map.items():
        out_path = best_dir / f"best_config_{int(horizon_days)}d.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
