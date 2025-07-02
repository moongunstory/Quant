import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95, normalize=True):
    """
    Generalized Advantage Estimation (GAE)
    """
    advantages = []
    gae = 0
    values = values + [last_value]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        if logger.isEnabledFor(logging.DEBUG):
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


def compute_value_loss(values, returns, normalize=False):  # True → False로 변경
    """MSE between predicted state value and return."""
    if returns.std() < 1e-3:
        returns = returns + torch.randn_like(returns) * 1e-2

    if normalize:
        # Value와 Return 모두 정규화
        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
        values_norm = (values - values.mean()) / (values.std() + 1e-8)
        return F.mse_loss(values_norm, returns_norm)
    else:
        # Huber Loss 사용하여 outlier에 robust하게
        return F.smooth_l1_loss(values, returns)
    

def compute_explained_variance(predicted, actual):
    """
    EV = 1 - Var(actual - predicted) / Var(actual)
    """
    var_actual = torch.var(actual, unbiased=False)
    if var_actual.item() == 0:
        return torch.tensor(0.0)
    return 1.0 - torch.var(actual - predicted, unbiased=False) / var_actual
