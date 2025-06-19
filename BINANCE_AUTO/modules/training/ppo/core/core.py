import torch
import torch.nn.functional as F

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
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

    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)


def compute_ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps=0.2):
    """
    PPO Policy Loss (Clipped surrogate objective)
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    return policy_loss


def compute_value_loss(values, returns):
    """
    MSE between predicted state value and return
    """
    return F.mse_loss(values, returns)


def compute_explained_variance(predicted, actual):
    """
    EV = 1 - Var(actual - predicted) / Var(actual)
    """
    var_actual = torch.var(actual, unbiased=False)
    if var_actual.item() == 0:
        return torch.tensor(0.0)
    return 1.0 - torch.var(actual - predicted, unbiased=False) / var_actual