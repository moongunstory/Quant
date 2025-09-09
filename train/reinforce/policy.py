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

        self.projs = nn.ModuleDict({
            tf: nn.Linear(obs_dims[tf], lstm_hidden_dim) for tf in self.timeframes
        })
        self.lstms = nn.ModuleDict({
            tf: nn.LSTM(lstm_hidden_dim, lstm_hidden_dim, num_layers=num_lstm_layers, batch_first=True)
            for tf in self.timeframes
        })

        fusion_dim = lstm_hidden_dim * len(self.timeframes)

        self.policy_net = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], action_dim),
        )

        self.value_net = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[0], mlp_hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dims[1], 1),
        )

        self.trend_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, trend_dim),
        )

        self.aux_optimizer = th.optim.Adam(self.trend_head.parameters(), lr=aux_lr)

        self._init_weights()  # ✅ 추가된 가중치 초기화 호출
        self.to(self.device)

    def _init_weights(self):
        for tf in self.projs:
            nn.init.xavier_uniform_(self.projs[tf].weight)
            if self.projs[tf].bias is not None:
                nn.init.zeros_(self.projs[tf].bias)

        for tf in self.lstms:
            for name, param in self.lstms[tf].named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def encode(self, x: th.Tensor, tf: str) -> th.Tensor:
        # 입력 확인
        if th.isnan(x).any() or th.isinf(x).any():
            print(f"[ERROR] NaN or Inf in input to encode({tf})")

        print(f"[{tf}] input stats — mean: {x.mean().item():.6f}, std: {x.std().item():.6f}, min: {x.min().item():.6f}, max: {x.max().item():.6f}")

        # weight/bias 확인
        w = self.projs[tf].weight
        print(f"[{tf}] weight stats — mean: {w.mean().item():.6f}, std: {w.std().item():.6f}")

        b = self.projs[tf].bias
        if b is not None:
            print(f"[{tf}] bias stats — mean: {b.mean().item():.6f}, std: {b.std().item():.6f}")
        else:
            print(f"[{tf}] bias is None")

        # 프로젝션
        x_proj = self.projs[tf](x)
        if th.isnan(x_proj).any() or th.isinf(x_proj).any():
            print(f"[ERROR] NaN after projection in {tf} — input shape: {x.shape}, x_proj shape: {x_proj.shape}")
            print("x_proj sample:", x_proj[0, :5])

        # LSTM
        _, (h_n, _) = self.lstms[tf](x_proj)
        if th.isnan(h_n).any() or th.isinf(h_n).any():
            print(f"[ERROR] NaN in LSTM output h_n in {tf} — h_n shape: {h_n.shape}")
            print("h_n sample:", h_n[:, :5])

        return h_n[-1]  # (B, H)

    def forward(self, obs_5m, obs_15m, obs_1h, obs_4h):
        inputs = {"5m": obs_5m, "15m": obs_15m, "1h": obs_1h, "4h": obs_4h}
        for tf, x in inputs.items():
            if th.isnan(x).any() or th.isinf(x).any():
                print(f"[ERROR] NaN or Inf in input {tf}: min={x.min().item():.4f}, max={x.max().item():.4f}")

        feats = {tf: self.encode(x, tf) for tf, x in inputs.items()}
        for tf, f in feats.items():
            if th.isnan(f).any() or th.isinf(f).any():
                print(f"[ERROR] NaN or Inf in encoded feature {tf}: min={f.min().item():.4f}, max={f.max().item():.4f}")

        z = th.cat([feats[tf] for tf in self.timeframes], dim=-1)
        if th.isnan(z).any() or th.isinf(z).any():
            print("[ERROR] NaN or Inf in concatenated features z")

        logits = self.policy_net(z)
        if th.isnan(logits).any() or th.isinf(logits).any():
            print("[ERROR] NaN or Inf in final logits (forward)")

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
