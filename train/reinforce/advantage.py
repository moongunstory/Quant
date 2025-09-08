# advantage.py

import torch as th

def compute_gae(buffer, gamma=0.99, lam=0.95):
    rewards = buffer.rewards
    values = buffer.values.cpu().tolist()
    dones = buffer.dones

    returns = []
    advantages = []
    gae = 0
    next_value = 0

    for step in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])
        next_value = values[step]

    buffer.returns = th.tensor(returns, dtype=th.float32, device=buffer.device)
    buffer.advantages = th.tensor(advantages, dtype=th.float32, device=buffer.device)
