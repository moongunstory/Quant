import os
import torch
import torch.optim as optim
import logging
from modules.ppo_runtime.rollout_updater import RolloutBuffer
from modules.training.ppo.core.model import PPOPolicyNetwork
from modules.training.ppo.core.core import (
    compute_ppo_loss, compute_value_loss, compute_explained_variance
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_ppo_live(
    direction: str,
    buffer_path: str,
    imitation_model_path: str,
    value_model_path: str,
    save_path: str,
    total_epochs: int = 5,
    batch_size: int = 64,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    lr: float = 2.5e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    logger.info(f"🚀 PPO 실전 학습 시작: {direction.upper()}")
    buffer = RolloutBuffer.load(buffer_path)
    input_dim = buffer.observations[0].shape[-1]

    model = PPOPolicyNetwork(input_dim=input_dim, hidden_dim=256).to(device)
    model.load_model(imitation_model_path)
    logger.info("📦 모방 학습 모델 로드 완료")

    # 가치망 초기화
    logger.info("🎯 value head 랜덤 초기화")
    model.value_head.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None)

    buffer.compute_returns_and_advantages(last_value=0.0, gamma=gamma, lam=lam)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(total_epochs):
        model.train()
        for obs_batch, action_batch, return_batch, adv_batch, old_logprob_batch in buffer.get_batches(batch_size):
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            return_batch = return_batch.to(device)
            adv_batch = adv_batch.to(device)
            old_logprob_batch = old_logprob_batch.to(device)

            log_probs, entropy, values = model.evaluate_action(obs_batch, action_batch)
            entropy = torch.clamp(entropy, min=0.01)

            policy_loss = compute_ppo_loss(log_probs, old_logprob_batch, adv_batch, clip_eps)
            value_loss = compute_value_loss(values, return_batch)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info(f"✅ Epoch {epoch+1}/{total_epochs} 완료")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    logger.info(f"📁 모델 저장 완료: {save_path}")

    RolloutBuffer.delete(buffer_path)

