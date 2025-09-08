# buffer.py

import numpy as np
import torch as th

class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_shape, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.infos = []

    def add(self, obs, action, reward, done, log_prob, value, info=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.infos.append(info or {})

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        self.returns = []
        self.advantages = []
        gae = 0
        values = self.values + [0]
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + gamma * values[i + 1] * (1 - self.dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - self.dones[i]) * gae
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + values[i])

        # Convert to tensors
        self.returns = th.tensor(self.returns, dtype=th.float32, device=self.device)
        self.advantages = th.tensor(self.advantages, dtype=th.float32, device=self.device)
        self.obs = th.tensor(np.array(self.obs), dtype=th.float32, device=self.device)
        self.actions = th.tensor(self.actions, dtype=th.int64, device=self.device)
        self.log_probs = th.tensor(self.log_probs, dtype=th.float32, device=self.device)
        self.values = th.tensor(self.values, dtype=th.float32, device=self.device)

    def get_batches(self, batch_size: int):
        idxs = np.arange(len(self.rewards))
        np.random.shuffle(idxs)
        for start in range(0, len(self.rewards), batch_size):
            batch_idx = idxs[start:start + batch_size]
            yield {
                "obs": self.obs[batch_idx],
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "values": self.values[batch_idx],
                "returns": self.returns[batch_idx],
                "advantages": self.advantages[batch_idx],
            }

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.infos.clear()
