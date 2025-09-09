import warnings
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Optional, Tuple

from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def zscore(s: pd.Series, win: Optional[int] = None) -> pd.Series:
    if win is None:
        mu, sd = s.mean(), s.std()
        return ((s - mu) / (sd or 1e-9)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    mu = s.rolling(win, min_periods=win).mean()
    sd = s.rolling(win, min_periods=win).std().replace(0, np.nan)
    return ((s - mu) / sd).replace([np.inf, -np.inf], 0.0).fillna(0.0)

def safe_log(s: pd.Series) -> pd.Series:
    return (np.sign(s) * np.log1p(np.abs(s))).replace([np.inf, -np.inf], 0.0).fillna(0.0)

def clip_series(s: pd.Series, min_val=-10, max_val=10) -> pd.Series:
    return s.clip(lower=min_val, upper=max_val)

# === 개선된 HPO 방식: 피처 조합 정의 ===
def get_feature_specs_for_tf(tf: str) -> List[Tuple[str, str]]:
    numeric_cols = {
        "5m": ["Close", "Volume", "rsi_14", "macd", "ema_20", "adx_14", "bb_mid"],
        "15m": ["Close", "Volume", "rsi_14", "macd", "ema_20", "adx_14", "bb_mid"],
        "1h": ["Close", "Volume", "rsi_14", "macd", "ema_60", "adx_14", "atr_14"],
        "4h": ["Close", "Volume", "rsi_14", "macd", "ema_120", "adx_14", "atr_14"]
    }.get(tf, [])

    specs = []
    for col in numeric_cols:
        specs.extend([
            ("sq", col),
            ("log", col),
            ("zscore", col),
            ("cond_gt0", col),
            ("cond_lt0", col),
            ("cond_gt70", col),
            ("cond_lt30", col),
        ])

    for col1, col2 in combinations(numeric_cols, 2):
        specs.extend([
            ("diff", f"{col1}__{col2}"),
            ("ratio", f"{col1}__{col2}"),
            ("mul", f"{col1}__{col2}"),
        ])
    return specs


def generate_feature(df: pd.DataFrame, spec: Tuple[str, str]) -> Tuple[str, pd.Series]:
    kind, arg = spec

    if kind in ["diff", "ratio", "mul"]:
        col1, col2 = arg.split("__")
        s1, s2 = df[col1], df[col2]
        if kind == "diff":
            name = f"f_diff_{col1}_{col2}"
            return name, (s1 - s2).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        elif kind == "ratio":
            name = f"f_ratio_{col1}_{col2}"
            ratio = s1 / (s2 + 1e-9)
            return name, clip_series(ratio.replace([np.inf, -np.inf], 0.0).fillna(0.0))
        elif kind == "mul":
            name = f"f_mul_{col1}_{col2}"
            return name, (s1 * s2).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    else:
        col = arg
        s = df[col]
        if kind == "sq":
            name = f"f_sq_{col}"
            return name, clip_series((s ** 2).replace([np.inf, -np.inf], 0.0).fillna(0.0))
        elif kind == "log":
            name = f"f_log_{col}"
            return name, clip_series(safe_log(s))
        elif kind == "zscore":
            name = f"f_zscore_{col}"
            return name, zscore(s)
        elif kind == "cond_gt0":
            name = f"f_cond_{col}_gt0"
            return name, (s > 0).astype(float)
        elif kind == "cond_lt0":
            name = f"f_cond_{col}_lt0"
            return name, (s < 0).astype(float)
        elif kind == "cond_gt70":
            name = f"f_cond_{col}_gt70"
            return name, (s > 70).astype(float)
        elif kind == "cond_lt30":
            name = f"f_cond_{col}_lt30"
            return name, (s < 30).astype(float)

    raise ValueError(f"Unknown feature spec: {spec}")


# === 선택된 피처 필터링 ===
def filter_features(df: pd.DataFrame, target: pd.Series, top_k: int = 300, vif_thresh: float = 10.0) -> pd.DataFrame:
    df = df.copy()
    features = [c for c in df.columns if c.startswith("f_")]
    X = df[features].fillna(0.0)
    y = target.loc[X.index]

    mi = mutual_info_classif(X, y, discrete_features=False)
    top_idx = np.argsort(mi)[::-1][:top_k]
    selected = X.columns[top_idx].tolist()
    X = X[selected]

    X_const = add_constant(X)
    keep_cols = X.columns.tolist()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        while True:
            vifs = pd.Series([variance_inflation_factor(X_const.values, i)
                              for i in range(X_const.shape[1])],
                             index=X_const.columns)
            vifs = vifs.drop("const", errors="ignore")
            high_vif = vifs[vifs > vif_thresh]
            if high_vif.empty:
                break
            drop_col = high_vif.idxmax()
            keep_cols.remove(drop_col)
            X_const = X_const.drop(columns=[drop_col])

    return df[keep_cols]
