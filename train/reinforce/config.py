"""Configuration objects for SAC LSTM training and environment modules."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainingConfig:
    """Shared hyper-parameters used across SAC/LSTM modules."""

    log_std_min: float = -1.0
    log_std_max: float = 0.0
    alpha_min: float = 0.10
    alpha_max: float = 0.8
    grad_clip_norm: float = 2.0
    reward_scale: float = 1.0
    target_entropy_scale: float = -0.5
    initial_alpha: float = 0.2
    fixed_alpha: float = 0.10
    hpo_batch_size: int = 256
    learning_starts_fraction: float = 0.2
    learning_starts_min_steps: int = 10_000
    train_split_ratio: float = 0.8
    risk_free_rate: float = 0.0
    evaluation_mdd_penalty: float = 1.0
    evaluation_sharpe_weight: float = 0.7
    evaluation_calmar_weight: float = 0.3
    periods_per_year: int = 252 * 24 * 12
    min_trades_per_1k: float = 5.0
    max_trades_per_1k: float = 150.0

    def compute_learning_starts(self, total_steps: int) -> int:
        """Return the number of warm-up steps before the first update."""

        warmup_from_fraction = int(total_steps * self.learning_starts_fraction)
        return max(self.learning_starts_min_steps, warmup_from_fraction)


@dataclass(frozen=True)
class EnvConfig:
    """Environment specific thresholds and trade management settings."""

    action_threshold_open: float = 0.30
    action_threshold_close: float = 0.25
    action_threshold_flip: float = 0.45
    take_profit_pct: float = 0.02
    stop_loss_pct: float = 0.01
    min_hold_bars: int = 10
    flip_penalty: float = 0.0014
    ohlcv_close_idx: int = 3
    ohlcv_high_idx: int = 1
    ohlcv_low_idx: int = 2
    bar_interval_minutes: int = 5
    seq_lens: dict[str, int] = field(
        default_factory=lambda: {"ohlcv": 48, "index": 48, "funding": 7, "dune": 7, "other": 48}
    )
