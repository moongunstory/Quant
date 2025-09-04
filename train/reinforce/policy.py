from __future__ import annotations
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

class MultiHeadPolicy(MaskableActorCriticPolicy):
    """
    MaskablePPO 호환 정책:
      - 베이스의 features_extractor / mlp_extractor를 그대로 사용
      - TimingHead: 기본 action_net / value_net (정책, 가치)
      - TrendHead: 보조 분류(head) + 별도 옵티마이저로 학습(기본: backbone 고정)
    """

    def __init__(
        self,
        *args,
        trend_dim: int = 3,
        aux_coeff: float = 0.1,
        aux_lr: float = 1e-3,
        freeze_backbone_for_aux: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aux_coeff = float(aux_coeff)
        self.freeze_backbone_for_aux = bool(freeze_backbone_for_aux)

        # 베이스가 이미 features_extractor/ mlp_extractor를 구성해둠
        d_feat = int(self.features_extractor.features_dim)
        d_pi   = int(self.mlp_extractor.latent_dim_pi)
        d_vf   = int(self.mlp_extractor.latent_dim_vf)

        # ----- Trend Aux Head (features_extractor 출력 기반) -----
        self.trend_head = nn.Sequential(
            nn.Linear(d_feat, 128),
            nn.ReLU(),
            nn.Linear(128, int(trend_dim)),  # logits: e.g. [short, neutral, long]
        )

        # ----- Timing Head (정책/가치) : 베이스 latent에 맞춰 재선언 -----
        #  (super().__init__)에서 이미 만들어두지만, 명시적으로 우리 규격으로 재지정
        self.action_net = nn.Linear(d_pi, self.action_space.n)
        self.value_net  = nn.Linear(d_vf, 1)

        # SB3 표준 초기화 적용(오버라이드한 레이어만)
        if hasattr(self, "_init_weights"):
            self._init_weights(self.action_net)
            self._init_weights(self.value_net)
            self.trend_head.apply(self._init_weights)

        # ----- Aux 전용 옵티마이저 (트렌드 헤드만) -----
        self.aux_optimizer = th.optim.Adam(self.trend_head.parameters(), lr=float(aux_lr))

    # ===== Aux(Trend) 유틸 =====
    @th.no_grad()
    def compute_trend_logits(self, obs: th.Tensor) -> th.Tensor:
        """
        추론용: backbone + trend_head 로짓 (no grad)
        """
        z = self.extract_features(obs)          # [B, d_feat]
        return self.trend_head(z)               # [B, trend_dim]

    def aux_loss(self, obs: th.Tensor, labels_4h: th.Tensor) -> th.Tensor:
        """
        보조 손실만 계산 (기본: backbone 고정).
        - freeze_backbone_for_aux=True면 feature를 detach하여 backbone 그라디언트 차단
        - 반환값: CE loss (미스케일)
        """
        z = self.extract_features(obs)          # [B, d_feat]
        if self.freeze_backbone_for_aux:
            z = z.detach()                      # ✅ backbone 고정
        logits = self.trend_head(z)             # [B, trend_dim]
        return F.cross_entropy(logits, labels_4h)

    def aux_train_step(
        self,
        obs: th.Tensor,
        labels_4h: th.Tensor,
        coeff: float | None = None,
        max_grad_norm: float = 1.0,
    ) -> float:
        """
        콜백에서 호출하기 위한 단일 스텝 학습 헬퍼.
        - trend_head만 업데이트
        - coeff가 주어지면 스케일 적용(없으면 self.aux_coeff)
        - 반환: 스케일 적용 후 loss 값(float)
        """
        self.trend_head.train(True)
        loss = self.aux_loss(obs, labels_4h)
        scale = float(self.aux_coeff if coeff is None else coeff)

        self.aux_optimizer.zero_grad(set_to_none=True)
        (scale * loss).backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.trend_head.parameters(), max_grad_norm)
        self.aux_optimizer.step()

        return float((scale * loss).detach().item())
