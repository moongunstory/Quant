import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOPolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 2):
        super().__init__()
        
        # LSTM으로 시계열 정보 처리
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)

        # 정책망 (action 확률 분포)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # action_dim = 2 (Long vs Hold)
        )

        # 가치망 (LayerNorm 포함)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        """
        x: 상태 시퀀스 (batch_size, seq_len, input_dim)
        반환: 정책 logits, 상태 가치
        """
        lstm_out, _ = self.lstm(x)                   # (batch, seq, hidden_dim)
        last_hidden = lstm_out[:, -1, :]             # 마지막 시점의 hidden state

        logits = self.policy_head(last_hidden)       # (batch, action_dim)
        value = self.value_head(last_hidden).squeeze(-1)  # (batch,)
        return logits, value

    def get_action(self, x):
        """
        행동 샘플링: 정책 분포에서 하나 선택 + log_prob + 가치 예측 + 확신도 벡터 반환
        """
        logits, value = self.forward(x)
        # Reorder probabilities so index 0 corresponds to "enter" and 1 to "hold"
        probs = torch.softmax(logits, dim=-1)[:, [1, 0]]
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, probs

    def evaluate_action(self, x, action):
        """
        PPO 학습 시: 행동의 log_prob, entropy, value 예측
        """
        logits, value = self.forward(x)
        # Use same [enter, hold] ordering during evaluation
        logits = logits[:, [1, 0]]
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()