# train/reinforce/networks/lstm_actor_critic.py

from __future__ import annotations

import torch
import torch.nn as nn

from ai_binance.train.reinforce.config import TrainingConfig


def _build_lstm_modules(input_dims, hidden_dim, lstm_layers):
    return nn.ModuleDict(
        {
            name: nn.LSTM(
                input_size=dim,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                batch_first=True,
            )
            for name, dim in input_dims.items()
        }
    )


def _extract_last_hidden(lstm_modules, state_seq_dict):
    outputs = []
    for name, lstm in lstm_modules.items():
        x = state_seq_dict[name]
        out, _ = lstm(x)
        outputs.append(out[:, -1, :])
    return outputs


def _shared_head(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.1),
    )


class LSTMActor(nn.Module):
    def __init__(
        self,
        input_dims,
        action_dim,
        hidden_dim=128,
        lstm_layers=1,
        *,
        training_config: TrainingConfig | None = None,
        log_std_min: float | None = None,
        log_std_max: float | None = None,
    ):
        """
        input_dims: {"ohlcv": 8, "funding": 4, "dune": 3}
        """
        super().__init__()
        self.lstm_modules = _build_lstm_modules(input_dims, hidden_dim, lstm_layers)
        self.hidden_dim = hidden_dim

        config = training_config or TrainingConfig()

        # log-std 범위(런타임에 조절 가능)
        self.log_std_min = float(log_std_min if log_std_min is not None else config.log_std_min)
        self.log_std_max = float(log_std_max if log_std_max is not None else config.log_std_max)

        combined_dim = hidden_dim * len(input_dims)

        # 비선형 레이어 추가
        self.fc_shared = _shared_head(combined_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)

        # 합리적 초기 σ (≈ e^{-0.5} ≈ 0.6)
        nn.init.constant_(self.fc_log_std.bias, -0.3)

    @torch.no_grad()
    def set_log_std_bounds(self, min_v: float, max_v: float):
        """학습 중 분산 범위를 더 조이거나 풀어줄 수 있게 함."""
        self.log_std_min = float(min_v)
        self.log_std_max = float(max_v)

    def forward(self, state_seq_dict):
        """
        state_seq_dict: {"ohlcv": (B, T1, D1), "funding": (B, T2, D2), ...}
        """
        lstm_outputs = _extract_last_hidden(self.lstm_modules, state_seq_dict)
        combined_features = torch.cat(lstm_outputs, dim=-1)

        shared_features = self.fc_shared(combined_features)
        mu = self.fc_mu(shared_features)
        log_std_raw = self.fc_log_std(shared_features)

        # Smooth bounds: map raw -> [log_std_min, log_std_max] via tanh (better gradients than clamp)
        log_std = torch.tanh(log_std_raw)
        log_std = self.log_std_min + 0.5 * (log_std + 1.0) * (self.log_std_max - self.log_std_min)

        return mu, log_std, combined_features


class LSTMCritic(nn.Module):
    def __init__(self, input_dims, action_dim, hidden_dim=128, lstm_layers=1):
        super().__init__()
        self.lstm_modules = _build_lstm_modules(input_dims, hidden_dim, lstm_layers)
        self.hidden_dim = hidden_dim

        combined_dim = hidden_dim * len(input_dims) + action_dim

        # 비선형 레이어 추가
        self.fc_shared = _shared_head(combined_dim, hidden_dim)
        self.fc_q = nn.Linear(hidden_dim, 1)

    def forward(self, state_seq_dict, action):
        """
        state_seq_dict: {"ohlcv": (B, T1, D1), "funding": (B, T2, D2), ...}
        action: (B, action_dim)
        """
        lstm_outputs = _extract_last_hidden(self.lstm_modules, state_seq_dict)
        combined_features = torch.cat(lstm_outputs + [action], dim=-1)
        shared_features = self.fc_shared(combined_features)
        q_value = self.fc_q(shared_features)
        return q_value, shared_features
