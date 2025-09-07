# train/prepare/feature_engineering.py

import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Optional
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

def generate_feature_combinations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_feats = {}

    numeric_cols = df.select_dtypes(include=["float", "float32", "float64", "int"]).columns.tolist()

    for col1, col2 in combinations(numeric_cols, 2):
        s1, s2 = df[col1], df[col2]

        new_feats[f"f_diff_{col1}_{col2}"] = (s1 - s2).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        new_feats[f"f_ratio_{col1}_{col2}"] = (s1 / (s2 + 1e-9)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        new_feats[f"f_mul_{col1}_{col2}"] = (s1 * s2).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    for col in numeric_cols:
        s = df[col]
        new_feats[f"f_sq_{col}"] = (s ** 2).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        new_feats[f"f_log_{col}"] = np.log1p(np.abs(s)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        new_feats[f"f_zscore_{col}"] = zscore(s)
        if s.max() > 70:
            new_feats[f"f_cond_{col}_gt70"] = (s > 70).astype(float)
        if s.min() < 30:
            new_feats[f"f_cond_{col}_lt30"] = (s < 30).astype(float)
        new_feats[f"f_cond_{col}_gt0"] = (s > 0).astype(float)
        new_feats[f"f_cond_{col}_lt0"] = (s < 0).astype(float)

    new_df = pd.concat([df] + [pd.Series(v, name=k, index=df.index) for k, v in new_feats.items()], axis=1)
    return new_df

def filter_features(df: pd.DataFrame, target: pd.Series, top_k: int = 300, vif_thresh: float = 10.0) -> pd.DataFrame:
    df = df.copy()
    features = [c for c in df.columns if c.startswith("f_")]
    X = df[features].fillna(0.0)
    y = target.loc[X.index]

    # Mutual Information 기반 필터링
    mi = mutual_info_classif(X, y, discrete_features=False)
    top_idx = np.argsort(mi)[::-1][:top_k]
    selected = X.columns[top_idx].tolist()
    X = X[selected]

    # VIF 기반 다중공선성 제거
    X_const = add_constant(X)
    keep_cols = X.columns.tolist()
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
