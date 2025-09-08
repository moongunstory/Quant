import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class MultiTimeframeLSTMPolicy(nn.Module):
    def __init__(
        self,
        obs_dims: Dict[str, int],
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
        self.aux_coeff = aux_coeff
        self.freeze_backbone_for_aux = freeze_backbone_for_aux

        self.timeframes = list(obs_dims.keys())
        self.lstm_hidden_dim = lstm_hidden_dim

        # Projections + LSTMs per timeframe
        self.projs = nn.ModuleDict({
            tf: nn.Linear(obs_dims[tf], lstm_hidden_dim) for tf in self.timeframes
        })
        self.lstms = nn.ModuleDict({
            tf: nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
            for tf in self.timeframes
        })

        fusion_dim = lstm_hidden_dim * len(self.timeframes)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], action_dim),
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], 1),
        )

        # Trend head uses only 1h and 4h features
        self.trend_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, trend_dim),
        )

        self.aux_optimizer = th.optim.Adam(self.trend_head.parameters(), lr=aux_lr)

        self.to(self.device)

    def encode(self, x: th.Tensor, tf: str) -> th.Tensor:
        x_proj = self.projs[tf](x)
        _, (h_n, _) = self.lstms[tf](x_proj)
        return h_n[-1]  # (B, H)

    def forward(self, obs_5m, obs_15m, obs_1h, obs_4h):
        # Encode all timeframes
        feats = {
            tf: self.encode(eval(f"obs_{tf}"), tf)
            for tf in self.timeframes
        }

        # Combine all features for policy/value
        z = th.cat([feats[tf] for tf in self.timeframes], dim=-1)  # (B, H*4)
        logits = self.policy_net(z)
        value = self.value_net(z).squeeze(-1)
        return logits, value

    def get_action(self, obs_5m, obs_15m, obs_1h, obs_4h, deterministic=False, action_mask=None):
        logits, value = self.forward(obs_5m, obs_15m, obs_1h, obs_4h)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        dist = th.distributions.Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def compute_trend_logits(self, obs_1h, obs_4h):
        with th.no_grad():
            z_1h = self.encode(obs_1h, "1h")
            z_4h = self.encode(obs_4h, "4h")
            z = th.cat([z_1h, z_4h], dim=-1)
            return self.trend_head(z)

    def aux_loss(self, obs_1h, obs_4h, labels):
        z_1h = self.encode(obs_1h, "1h")
        z_4h = self.encode(obs_4h, "4h")
        z = th.cat([z_1h, z_4h], dim=-1)
        if self.freeze_backbone_for_aux:
            z = z.detach()
        logits = self.trend_head(z)
        return F.cross_entropy(logits, labels)

    def aux_train_step(self, obs_1h, obs_4h, labels, coeff=None, max_grad_norm=1.0):
        self.trend_head.train(True)
        loss = self.aux_loss(obs_1h, obs_4h, labels)
        scale = self.aux_coeff if coeff is None else coeff
        self.aux_optimizer.zero_grad(set_to_none=True)
        (scale * loss).backward()
        if max_grad_norm:
            nn.utils.clip_grad_norm_(self.trend_head.parameters(), max_grad_norm)
        self.aux_optimizer.step()
        return float((scale * loss).detach().item())
