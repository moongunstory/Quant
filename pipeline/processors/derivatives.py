# pipeline/processors/derivatives.py

from __future__ import annotations
import pandas as pd

from .utils import ensure_sorted_date


def build_derivatives_features(df_dvol: pd.DataFrame) -> pd.DataFrame:
    """
    Deribit DVOL DataFrame → 일단위 피처.

    df_dvol: 최소 ['date', 'value']
    """
    if df_dvol is None or df_dvol.empty:
        raise RuntimeError("Empty DVOL dataframe given to build_derivatives_features")

    df = ensure_sorted_date(df_dvol, "date")
    df = df.rename(columns={"value": "dvol_level"})

    df["dvol_delta_1d"] = df["dvol_level"].diff(1)
    df["dvol_delta_7d"] = df["dvol_level"].diff(7)

    roll = df["dvol_level"].rolling(180)
    df["dvol_z_180d"] = (df["dvol_level"] - roll.mean()) / roll.std()

    return df[["date", "dvol_level", "dvol_delta_1d", "dvol_delta_7d", "dvol_z_180d"]]
