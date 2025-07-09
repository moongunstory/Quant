import torch
import torch.nn.functional as F
import logging
import numpy as np
logger = logging.getLogger(__name__)

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95, normalize=True):
    """
    Generalized Advantage Estimation (GAE)
    """
    advantages = []
    gae = 0
    values = values + [last_value]

    N = len(rewards)
    # Only log at start, middle and end to reduce log volume
    log_indices = {0, N // 2, N - 1}

    for t in reversed(range(N)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae

        if logger.isEnabledFor(logging.DEBUG) and t in log_indices:
            logger.debug(
                f"[GAE] t={t} | reward={rewards[t]:.3f}, value={values[t]:.3f}, "
                f"delta={delta:.3f}, done={int(dones[t])}, gae={gae:.3f}"
            )

        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)

    if logger.isEnabledFor(logging.DEBUG) and len(advantages) > 0:
        logger.debug(
            f"[GAE] Advantage dist → mean={advantages.mean():.3f}, "
            f"std={advantages.std():.3f}, min={advantages.min():.3f}, max={advantages.max():.3f}"
        )
        logger.debug(
            f"[GAE] Return dist → mean={returns.mean():.3f}, "
            f"std={returns.std():.3f}, min={returns.min():.3f}, max={returns.max():.3f}"
        )

    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def compute_ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2):
    """Compute PPO clipped surrogate loss."""

    ratio = torch.exp(new_log_probs - old_log_probs)

    # Sanity check for extreme ratio values
    if torch.any(~torch.isfinite(ratio)):
        ratio = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))
    ratio = torch.clamp(ratio, 0.0, 10.0)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    return policy_loss


def compute_value_loss(values, returns, normalize=True, class_weights=None):
    """
    가치 함수 손실 계산 - 분류 문제로 전환 (CrossEntropyLoss 사용)
    
    Args:
        values: 모델이 예측한 가치값 (logits)
        returns: 목표 리턴값 (연속값)
        normalize: 정규화 여부 (현재는 사용하지 않음)
    
    Returns:
        CrossEntropy 손실값
    """
    # 리턴값을 분류 라벨로 변환
    # 예: ±1.0 구간 기준으로 라벨링
    # reward >= 0.5 -> label = 2 (강력히 좋은 상태)
    # reward <= -0.5 -> label = 0 (강력히 나쁜 상태)
    # else -> label = 1 (중립 or 애매)
    
    # Ensure returns is on the same device as values
    returns = returns.to(values.device)

    target_labels = torch.ones_like(returns, dtype=torch.long) # Default to neutral (1)
    target_labels[returns >= 0.5] = 2
    target_labels[returns <= -0.5] = 0

    # CrossEntropyLoss는 logits를 입력으로 받음
    return F.cross_entropy(values, target_labels, weight=class_weights)
    

def compute_explained_variance(predicted, actual):
    """
    EV = 1 - Var(actual - predicted) / Var(actual)
    """
    var_actual = torch.var(actual, unbiased=False)
    if var_actual.item() == 0:
        return torch.tensor(0.0)
    return 1.0 - torch.var(actual - predicted, unbiased=False) / var_actual