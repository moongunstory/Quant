from __future__ import annotations
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

class MultiHeadPolicy(MaskableActorCriticPolicy):
    """
    MaskablePPO 호환 정책:
      - 베이스의 mlp_extractor가 만든 latent_pi/vf 차원에 자동 맞춤
      - TrendHead: 보조손실(CE)만; PPO 그라디언트에는 미포함
      - TimingHead: 기본 action_net/value_net (베이스 forward 사용)
    """
    def __init__(self, *args, trend_dim: int = 3, aux_coeff: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.aux_coeff = aux_coeff

        # 베이스가 이미 features_extractor와 mlp_extractor를 만들어둠
        d_feat = self.features_extractor.features_dim
        d_pi   = self.mlp_extractor.latent_dim_pi
        d_vf   = self.mlp_extractor.latent_dim_vf

        # 보조 Trend Head: features_extractor 출력 기반
        self.trend_head = nn.Sequential(
            nn.Linear(d_feat, 128), nn.ReLU(),
            nn.Linear(128, trend_dim)  # logits: [short, neutral, long]
        )

        # Timing Head: 베이스 forward가 latent_pi/vf를 바로 action_net/value_net에 넣음
        self.action_net = nn.Linear(d_pi, self.action_space.n)
        self.value_net  = nn.Linear(d_vf, 1)

        # ⚠ _build 호출 금지: 알고리즘이 내부에서 optimizer 셋업함

    # ===== Aux(Trend) 유틸 =====
    @th.no_grad()
    def compute_trend_logits(self, obs: th.Tensor) -> th.Tensor:
        z = self.extract_features(obs)          # features_extractor 출력
        return self.trend_head(z)

    def aux_loss(self, obs: th.Tensor, labels_4h: th.Tensor) -> th.Tensor:
        z = self.extract_features(obs)
        logits = self.trend_head(z)
        return F.cross_entropy(logits, labels_4h)
