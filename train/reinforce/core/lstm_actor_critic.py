# train/reinforce/networks/lstm_actor_critic.py

import torch
import torch.nn as nn


class LSTMActor(nn.Module):
    def __init__(self, input_dims: dict, action_dim, hidden_dim=128):
        """
        input_dims: {"ohlcv": 8, "funding": 4, "dune": 3}
        """
        super().__init__()
        self.lstm_modules = nn.ModuleDict()
        self.hidden_dim = hidden_dim

        for name, dim in input_dims.items():
            self.lstm_modules[name] = nn.LSTM(dim, hidden_dim, batch_first=True)

        combined_dim = hidden_dim * len(input_dims)
        self.fc_mu = nn.Linear(combined_dim, action_dim)
        self.fc_log_std = nn.Linear(combined_dim, action_dim)

    def forward(self, state_seq_dict):  # action 파라미터 제거
        """
        state_seq_dict: {"ohlcv": (B, T1, D1), "funding": (B, T2, D2), ...}
        """
        lstm_outputs = []

        for name, lstm in self.lstm_modules.items():  # group_lstms → lstm_modules
            x = state_seq_dict[name]  # shape: (B, T_k, D_k)
            out, _ = lstm(x)
            h_last = out[:, -1, :]  # 마지막 시점만 추출
            lstm_outputs.append(h_last)

        # 모든 그룹의 마지막 시점 출력을 결합
        combined_features = torch.cat(lstm_outputs, dim=-1)
        
        mu = self.fc_mu(combined_features)
        log_std = self.fc_log_std(combined_features)
        
        return mu, log_std, combined_features


class LSTMCritic(nn.Module):
    def __init__(self, input_dims: dict, action_dim, hidden_dim=128):
        super().__init__()
        self.lstm_modules = nn.ModuleDict()
        self.hidden_dim = hidden_dim

        for name, dim in input_dims.items():
            self.lstm_modules[name] = nn.LSTM(dim, hidden_dim, batch_first=True)

        combined_dim = hidden_dim * len(input_dims) + action_dim
        self.fc_q = nn.Linear(combined_dim, 1)

    def forward(self, state_seq_dict, action):
        """
        state_seq_dict: {"ohlcv": (B, T1, D1), "funding": (B, T2, D2), ...}
        action: (B, action_dim)
        """
        lstm_outputs = []
        
        for name, lstm in self.lstm_modules.items():
            x = state_seq_dict[name]  # shape: (B, T_k, D_k)
            out, _ = lstm(x)
            h_last = out[:, -1, :]  # 마지막 시점만 추출
            lstm_outputs.append(h_last)

        # 모든 그룹의 마지막 시점 출력 + action 결합
        combined_features = torch.cat(lstm_outputs + [action], dim=-1)
        q_value = self.fc_q(combined_features)
        
        return q_value, combined_features