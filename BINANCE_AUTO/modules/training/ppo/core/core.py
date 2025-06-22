import torch
import torch.nn.functional as F

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
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)

    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def compute_ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2):
    """Compute PPO clipped surrogate loss with additional safety checks."""

    # Min-max scale advantages to [-1, 1] regardless of normalization state
    adv_max = advantages.max()
    adv_min = advantages.min()
    if (adv_max - adv_min) > 1e-8:
        advantages = 2 * (advantages - adv_min) / (adv_max - adv_min) - 1
    advantages = torch.clamp(advantages, -1.0, 1.0)

    ratio = torch.exp(new_log_probs - old_log_probs)

    # Sanity check for extreme ratio values
    if torch.any(~torch.isfinite(ratio)):
        ratio = torch.where(torch.isfinite(ratio), ratio, torch.ones_like(ratio))
    ratio = torch.clamp(ratio, 0.0, 10.0)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    return policy_loss


def compute_value_loss(values, returns, normalize=False):
    """MSE between predicted state value and return.

    By default no normalization is applied so that the loss is on the
    same scale as :func:`compute_explained_variance`.
    """
    if returns.std() < 1e-3:
        returns = returns + torch.randn_like(returns) * 1e-2

    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        values = (values - values.mean()) / (values.std() + 1e-8)
    return F.mse_loss(values, returns)


def compute_explained_variance(predicted, actual):
    """
    EV = 1 - Var(actual - predicted) / Var(actual)
    """
    var_actual = torch.var(actual, unbiased=False)
    if var_actual.item() == 0:
        return torch.tensor(0.0)
    return 1.0 - torch.var(actual - predicted, unbiased=False) / var_actual
