# model/regression/pipeline.py
"""
Main regression training pipeline.
"""

from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from .config import RegressionConfig
from .dataset import (
    ensure_datetime_index,
    build_supervised_for_horizon,
    make_sample_weight,
)
from .lgbm.model import create_lgbm_regressor
from .lgbm.trainer import train_lgbm_regressor
from .xgboost.model import create_xgboost_regressor
from .xgboost.trainer import train_xgboost_regressor
from .ensemble.blend import SimpleEnsemble


def train_horizon_ensemble(
    df_master: pd.DataFrame,
    horizon_hours: int,
    cfg: RegressionConfig,
) -> Tuple[SimpleEnsemble, Dict]:
    """
    Train ensemble of models for a specific horizon.

    Args:
        df_master: Master features DataFrame
        horizon_hours: Prediction horizon in hours
        cfg: RegressionConfig

    Returns:
        ensemble: Trained ensemble model
        metadata: Training metadata (RMSE, feature names, etc.)
    """
    # Ensure datetime index
    df_master = ensure_datetime_index(df_master, cfg)
    as_of_ts = df_master.index.max()

    print(f"  Training models for {horizon_hours}h horizon...")

    # Build supervised dataset
    X, y, feature_names = build_supervised_for_horizon(
        df=df_master,
        as_of_ts=as_of_ts,
        horizon_hours=horizon_hours,
        cfg=cfg,
    )

    n_samples = X.shape[0]
    print(f"    Samples: {n_samples}")

    # Train/val split (time-based)
    split_idx = int(n_samples * (1 - cfg.val_ratio))
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    # Sample weights (recent data weighting)
    sample_weight_all = make_sample_weight(y, cfg)
    sw_train = sample_weight_all[:split_idx]
    sw_val = sample_weight_all[split_idx:]

    # Train LightGBM
    print("    Training LightGBM...")
    lgbm_params = cfg.get_lgbm_params_for(horizon_hours)
    lgbm_model = create_lgbm_regressor(lgbm_params)
    lgbm_model = train_lgbm_regressor(
        model=lgbm_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        sample_weight_train=sw_train,
        sample_weight_val=sw_val,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )

    # Validation metrics for LightGBM
    lgbm_pred_val = lgbm_model.predict(X_val)
    lgbm_rmse = np.sqrt(np.mean((y_val - lgbm_pred_val) ** 2))
    lgbm_dir_acc = np.mean(np.sign(y_val) == np.sign(lgbm_pred_val))

    print(f"      LGBM RMSE: {lgbm_rmse:.6f}, Dir Acc: {lgbm_dir_acc:.3f}")

    # Train XGBoost
    print("    Training XGBoost...")
    xgb_params = cfg.get_xgboost_params_for(horizon_hours)
    xgb_model = create_xgboost_regressor(xgb_params)
    xgb_model = train_xgboost_regressor(
        model=xgb_model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        sample_weight_train=sw_train,
        sample_weight_val=sw_val,
        early_stopping_rounds=cfg.early_stopping_rounds,
    )

    # Validation metrics for XGBoost
    xgb_pred_val = xgb_model.predict(X_val)
    xgb_rmse = np.sqrt(np.mean((y_val - xgb_pred_val) ** 2))
    xgb_dir_acc = np.mean(np.sign(y_val) == np.sign(xgb_pred_val))

    print(f"      XGB RMSE: {xgb_rmse:.6f}, Dir Acc: {xgb_dir_acc:.3f}")

    # Create ensemble (weighted by inverse RMSE)
    # Better models (lower RMSE) get higher weight
    lgbm_inv_rmse = 1.0 / lgbm_rmse if lgbm_rmse > 0 else 1.0
    xgb_inv_rmse = 1.0 / xgb_rmse if xgb_rmse > 0 else 1.0
    total = lgbm_inv_rmse + xgb_inv_rmse

    weights = {
        'lgbm': lgbm_inv_rmse / total,
        'xgb': xgb_inv_rmse / total,
    }

    ensemble = SimpleEnsemble(
        models={'lgbm': lgbm_model, 'xgb': xgb_model},
        weights=weights
    )

    # Ensemble validation metrics
    ensemble_pred_val = ensemble.predict(X_val)
    ensemble_rmse = np.sqrt(np.mean((y_val - ensemble_pred_val) ** 2))
    ensemble_dir_acc = np.mean(np.sign(y_val) == np.sign(ensemble_pred_val))

    print(f"      Ensemble RMSE: {ensemble_rmse:.6f}, Dir Acc: {ensemble_dir_acc:.3f}")
    print(f"      Weights: LGBM={weights['lgbm']:.3f}, XGB={weights['xgb']:.3f}")

    metadata = {
        'horizon_hours': horizon_hours,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'n_train': split_idx,
        'n_val': len(y_val),
        'lgbm_rmse': lgbm_rmse,
        'lgbm_dir_acc': lgbm_dir_acc,
        'xgb_rmse': xgb_rmse,
        'xgb_dir_acc': xgb_dir_acc,
        'ensemble_rmse': ensemble_rmse,
        'ensemble_dir_acc': ensemble_dir_acc,
        'weights': weights,
        'trained_at': datetime.now().isoformat(),
    }

    return ensemble, metadata


def train_all_horizons(
    df_master: pd.DataFrame,
    cfg: RegressionConfig = None,
) -> Dict[int, SimpleEnsemble]:
    """
    Train ensemble models for all configured horizons.

    Args:
        df_master: Master features DataFrame
        cfg: RegressionConfig (default: use defaults)

    Returns:
        Dict mapping horizon_hours to trained ensemble
    """
    if cfg is None:
        cfg = RegressionConfig()

    print("=" * 60)
    print("Training Regression Models (Ensemble)")
    print("=" * 60)
    print(f"Horizons: {cfg.horizons_hours}")
    print(f"Master data: {len(df_master)} rows")
    print()

    models = {}

    for horizon_hours in cfg.horizons_hours:
        try:
            ensemble, metadata = train_horizon_ensemble(
                df_master=df_master,
                horizon_hours=horizon_hours,
                cfg=cfg,
            )

            models[horizon_hours] = ensemble

            # Save model if configured
            if cfg.save_models:
                save_ensemble(ensemble, metadata, horizon_hours, cfg)

        except Exception as e:
            print(f"  [ERROR] Failed to train {horizon_hours}h: {e}")
            continue

    print()
    print(f" Trained {len(models)}/{len(cfg.horizons_hours)} horizons")
    return models


def save_ensemble(
    ensemble: SimpleEnsemble,
    metadata: Dict,
    horizon_hours: int,
    cfg: RegressionConfig,
) -> None:
    """Save ensemble model and metadata."""
    model_dir = Path(cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d")
    model_path = model_dir / f"ensemble_{horizon_hours}h_{timestamp}.pkl"

    # Save ensemble + metadata
    with open(model_path, 'wb') as f:
        pickle.dump({'ensemble': ensemble, 'metadata': metadata}, f)

    print(f"    Saved: {model_path}")


def load_ensemble(horizon_hours: int, cfg: RegressionConfig) -> Tuple[SimpleEnsemble, Dict]:
    """Load latest ensemble model for a horizon."""
    model_dir = Path(cfg.model_dir)

    # Find latest model file
    pattern = f"ensemble_{horizon_hours}h_*.pkl"
    model_files = list(model_dir.glob(pattern))

    if not model_files:
        raise FileNotFoundError(f"No model found for {horizon_hours}h in {model_dir}")

    latest_model = max(model_files, key=os.path.getmtime)

    with open(latest_model, 'rb') as f:
        data = pickle.load(f)

    return data['ensemble'], data['metadata']


def generate_predictions(
    df_master: pd.DataFrame,
    models: Dict[int, SimpleEnsemble],
    cfg: RegressionConfig = None,
) -> Dict[int, float]:
    """
    Generate predictions for all horizons.

    Args:
        df_master: Master features DataFrame
        models: Dict mapping horizon_hours to trained ensemble
        cfg: RegressionConfig

    Returns:
        Dict mapping horizon_hours to predicted return
    """
    if cfg is None:
        cfg = RegressionConfig()

    df_master = ensure_datetime_index(df_master, cfg)
    predictions = {}

    for horizon_hours, model in models.items():
        try:
            # Get latest features (last row)
            X_latest = df_master.iloc[-1:].copy()

            # Drop close price and any targets
            drop_cols = [cfg.close_col] + [f"ret_{h}h" for h in cfg.horizons_hours]
            X_latest = X_latest.select_dtypes(include=[np.number])
            X_latest = X_latest.drop(columns=drop_cols, errors='ignore')

            # Predict
            pred_return = model.predict(X_latest.values)[0]
            predictions[horizon_hours] = pred_return

        except Exception as e:
            print(f"[ERROR] Failed to predict {horizon_hours}h: {e}")
            predictions[horizon_hours] = 0.0

    return predictions
