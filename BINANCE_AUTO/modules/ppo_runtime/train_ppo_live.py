import os
import torch
import torch.optim as optim
import logging
import numpy as np
from typing import Dict
from modules.ppo_runtime.rollout_updater import RolloutBuffer
from modules.training.ppo.core.model import PPOPolicyNetwork
from modules.training.ppo.core.core import (
    compute_ppo_loss, compute_value_loss, compute_explained_variance
)
from modules.config import TIMEFRAMES, PPO_CONFIG  # Import timeframes configuration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_ppo_live(
    direction: str,
    buffer_path: str,
    imitation_model_path: str,
    value_model_path: str,
    save_path: str,
    total_epochs: int = PPO_CONFIG["epochs"],
    batch_size: int = PPO_CONFIG["batch_size"],
    gamma: float = PPO_CONFIG["gamma"],
    lam: float = PPO_CONFIG["lambda"],
    clip_eps: float = PPO_CONFIG["clip_eps"],
    value_coef: float = PPO_CONFIG["value_coef"],
    entropy_coef: float = PPO_CONFIG["entropy_coef"],
    lr: float = PPO_CONFIG["learning_rate"],
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    logger.info(f"🚀 PPO 실전 학습 시작: {direction.upper()}")
    buffer = RolloutBuffer.load(buffer_path)
    
    # Handle both single tensor and MTF dict observations
    sample_obs = buffer.observations[0]
    if isinstance(sample_obs, dict):
        # MTF case: create input_dims mapping for each timeframe
        input_dims = {tf: sample_obs[tf].shape[-1] for tf in TIMEFRAMES}
        logger.info(f"📊 MTF 입력 차원: {input_dims}")
    else:
        # Legacy single tensor case
        input_dim = sample_obs.shape[-1]
        input_dims = None
        logger.info(f"📊 단일 입력 차원: {input_dim}")

    # Initialize model with appropriate input configuration
    if input_dims is not None:
        model = PPOPolicyNetwork(timeframe_dims=input_dims, hidden_dim=PPO_CONFIG["hidden_dim"]).to(device)
    else:
        model = PPOPolicyNetwork(timeframe_dims={"single": input_dim}, hidden_dim=PPO_CONFIG["hidden_dim"]).to(device)
    
    model.load_model(imitation_model_path, allow_partial=True)
    logger.info("📦 모방 학습 모델 로드 완료")

    # 가치망 초기화
    logger.info("🎯 value head 랜덤 초기화")
    model.value_head.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight) if hasattr(m, 'weight') else None)

    buffer.compute_returns_and_advantages(last_value=0.0, gamma=gamma, lam=lam)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    initial_entropy_coef = entropy_coef
    min_entropy_coef = 0.005

    for epoch in range(total_epochs):
        entropy_coef = max(initial_entropy_coef * (1 - epoch / total_epochs), min_entropy_coef)
        model.train()
        epoch_returns = []
        epoch_values = []
        evs = []
        for obs_batch, action_batch, return_batch, adv_batch, old_logprob_batch in buffer.get_batches(batch_size):
            
            # Convert obs_batch to device-aware format
            if isinstance(obs_batch, dict):
                # MTF case: move each timeframe tensor to device
                obs_batch = {tf: obs.to(device) for tf, obs in obs_batch.items()}
            else:
                # Legacy single tensor case
                obs_batch = obs_batch.to(device)
            
            action_batch = action_batch.to(device)
            return_batch = return_batch.to(device)
            adv_batch = adv_batch.to(device)
            old_logprob_batch = old_logprob_batch.to(device)

            # Model now handles both dict and tensor inputs
            log_probs, entropy, values = model.evaluate_action(obs_batch, action_batch)
            entropy = torch.clamp(entropy, min=0.01)

            policy_loss = compute_ppo_loss(log_probs, old_logprob_batch, adv_batch, clip_eps)
            value_loss = compute_value_loss(values, return_batch)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_returns.append(return_batch.mean().item())
            epoch_values.append(values.mean().item())
            evs.append(compute_explained_variance(values.detach(), return_batch.detach()).item())

        avg_return = np.mean(epoch_returns) if epoch_returns else 0.0
        avg_value = np.mean(epoch_values) if epoch_values else 0.0
        ev = np.mean(evs) if evs else 0.0
        logger.info(
            f"📈 Epoch {epoch+1}: Avg Return={avg_return:.3f}, Avg Value={avg_value:.3f}, EV={ev:.3f}"
        )
        logger.info(f"✅ Epoch {epoch+1}/{total_epochs} 완료")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    logger.info(f"📁 모델 저장 완료: {save_path}")

    RolloutBuffer.delete(buffer_path)


def _move_obs_to_device(obs_batch, device):
    """
    Helper function to move observations to device, handling both dict and tensor formats
    """
    if isinstance(obs_batch, dict):
        return {tf: obs.to(device) for tf, obs in obs_batch.items()}
    else:
        return obs_batch.to(device)


def _get_input_dimensions(buffer):
    """
    Extract input dimensions from buffer observations, handling MTF format
    """
    sample_obs = buffer.observations[0]
    if isinstance(sample_obs, dict):
        return {tf: sample_obs[tf].shape[-1] for tf in TIMEFRAMES}
    else:
        return sample_obs.shape[-1]