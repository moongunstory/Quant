import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLSTMPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        seq_len: int,
        action_dim: int,
        trend_dim: int = 3,
        aux_coeff: float = 0.1,
        aux_lr: float = 1e-3,
        freeze_backbone_for_aux: bool = True,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 1,
        mlp_hidden_dims: tuple[int, int] = (128, 64),
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.obs_dim = obs_dim
        self.seq_len = seq_len
        self.action_dim = action_dim
        self.aux_coeff = aux_coeff
        self.freeze_backbone_for_aux = freeze_backbone_for_aux

        # 1. input projection → LSTM input
        self.input_proj = nn.Linear(obs_dim, lstm_hidden_dim)

        # 2. LSTM encoder
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # 3. MLP for policy
        self.policy_net = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], action_dim),
        )

        # 4. MLP for value function
        self.value_net = nn.Sequential(
            nn.Linear(lstm_hidden_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], 1),
        )

        # 5. Trend head (auxiliary task)
        self.trend_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, trend_dim),
        )

        self.aux_optimizer = th.optim.Adam(self.trend_head.parameters(), lr=aux_lr)

        self.to(self.device)

    def extract_lstm_features(self, obs_seq: th.Tensor) -> th.Tensor:
        # obs_seq: (B, T, D)
        xf = self.input_proj(obs_seq)  # (B, T, H)
        _, (h_n, _) = self.lstm(xf)
        return h_n[-1]  # (B, H)

    def forward(self, obs_seq: th.Tensor):
        z = self.extract_lstm_features(obs_seq)  # (B, H)
        logits = self.policy_net(z)              # (B, action_dim)
        value = self.value_net(z).squeeze(-1)    # (B,)
        return logits, value

    def get_action(self, obs_seq: th.Tensor, deterministic=False, action_mask: th.Tensor | None = None):
        logits, value = self.forward(obs_seq)  # (B, action_dim), (B,)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        dist = th.distributions.Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def compute_trend_logits(self, obs_seq: th.Tensor) -> th.Tensor:
        with th.no_grad():
            z = self.extract_lstm_features(obs_seq)
            return self.trend_head(z)

    def aux_loss(self, obs_seq: th.Tensor, labels: th.Tensor) -> th.Tensor:
        z = self.extract_lstm_features(obs_seq)
        if self.freeze_backbone_for_aux:
            z = z.detach()
        logits = self.trend_head(z)
        return F.cross_entropy(logits, labels)

    def aux_train_step(self, obs_seq: th.Tensor, labels: th.Tensor,
                       coeff: float | None = None, max_grad_norm: float = 1.0) -> float:
        self.trend_head.train(True)
        loss = self.aux_loss(obs_seq, labels)
        scale = self.aux_coeff if coeff is None else coeff
        self.aux_optimizer.zero_grad(set_to_none=True)
        (scale * loss).backward()
        if max_grad_norm:
            nn.utils.clip_grad_norm_(self.trend_head.parameters(), max_grad_norm)
        self.aux_optimizer.step()
        return float((scale * loss).detach().item())
