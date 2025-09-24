"""Configuration objects for SAC LSTM training and environment modules."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Shared hyper-parameters used across SAC/LSTM modules."""

    log_std_min: float = -1.0
    log_std_max: float = 0.0
    alpha_min: float = 0.05
    alpha_max: float = 0.3
    grad_clip_norm: float = 5.0
    reward_scale: float = 1.0
    target_entropy_scale: float = -0.5
    initial_alpha: float = 0.2
    fixed_alpha: float = 0.10


@dataclass(frozen=True)
class EnvConfig:
    """Environment specific thresholds and trade management settings."""

    action_threshold_open: float = 0.30
    action_threshold_close: float = 0.15
    action_threshold_flip: float = 0.45
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.01
