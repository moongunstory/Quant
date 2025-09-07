from __future__ import annotations
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

class MultiHeadLSTMPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args,
                 trend_dim: int = 3,
                 aux_coeff: float = 0.1,
                 aux_lr: float = 1e-3,
                 freeze_backbone_for_aux: bool = True,
                 lstm_hidden_dim: int = 128,
                 num_lstm_layers: int = 1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_coeff = aux_coeff
        self.freeze_backbone_for_aux = freeze_backbone_for_aux

        d_feat = int(self.features_extractor.features_dim)
        d_pi = int(self.mlp_extractor.latent_dim_pi)
        d_vf = int(self.mlp_extractor.latent_dim_vf)

        # LSTM 추가
        self.lstm = nn.LSTM(input_size=d_feat,
                            hidden_size=lstm_hidden_dim,
                            num_layers=num_lstm_layers,
                            batch_first=True)

        self.trend_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, trend_dim),
        )
        self.action_net = nn.Linear(d_pi, self.action_space.n)
        self.value_net = nn.Linear(d_vf, 1)

        if hasattr(self, "_init_weights"):
            self._init_weights(self.action_net)
            self._init_weights(self.value_net)
            self.trend_head.apply(self._init_weights)

        self.aux_optimizer = th.optim.Adam(self.trend_head.parameters(), lr=aux_lr)

    def extract_lstm_features(self, obs_seq: th.Tensor) -> th.Tensor:
        B, T, _ = obs_seq.shape
        xf = self.extract_features(obs_seq.view(-1, obs_seq.shape[-1]))
        xf = xf.view(B, T, -1)  # (batch, seq_len, feat)
        _, (h_n, _) = self.lstm(xf)
        return h_n[-1]  # 마지막 레이어의 마지막 히든 상태

    @th.no_grad()
    def compute_trend_logits(self, obs_seq: th.Tensor) -> th.Tensor:
        z = self.extract_lstm_features(obs_seq)
        return self.trend_head(z)

    def aux_loss(self, obs_seq: th.Tensor, labels_4h: th.Tensor) -> th.Tensor:
        z = self.extract_lstm_features(obs_seq)
        if self.freeze_backbone_for_aux:
            z = z.detach()
        logits = self.trend_head(z)
        return F.cross_entropy(logits, labels_4h)

    def aux_train_step(self, obs_seq: th.Tensor, labels_4h: th.Tensor,
                       coeff: float | None = None, max_grad_norm: float = 1.0) -> float:
        self.trend_head.train(True)
        loss = self.aux_loss(obs_seq, labels_4h)
        scale = self.aux_coeff if coeff is None else coeff
        self.aux_optimizer.zero_grad(set_to_none=True)
        (scale * loss).backward()
        if max_grad_norm:
            nn.utils.clip_grad_norm_(self.trend_head.parameters(), max_grad_norm)
        self.aux_optimizer.step()
        return float((scale * loss).detach().item())

    def forward(self, obs: th.Tensor, deterministic=False, action_masks=None):
        # 입력: (batch, seq_len, feat_dim) 형태 가정
        features = self.extract_lstm_features(obs)
        policy_latent, value_latent = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(policy_latent, action_masks=action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        value = self.value_net(value_latent)
        return actions, value, log_prob
