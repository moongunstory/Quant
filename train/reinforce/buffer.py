import numpy as np
import torch as th

class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_space, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.obs = {tf: [] for tf in obs_space.spaces}
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.infos = []
        self._is_tensorized = False

    def add(self, obs: dict, action, reward, done, log_prob, value, info=None):
        for tf in obs:
            self.obs[tf].append(obs[tf])
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
        for tf in self.obs:
            self.obs[tf] = th.tensor(np.array(self.obs[tf]), dtype=th.float32, device=self.device)
        self.actions = th.tensor(self.actions, dtype=th.int64, device=self.device)
        self.log_probs = th.tensor(self.log_probs, dtype=th.float32, device=self.device)
        self.values = th.tensor(self.values, dtype=th.float32, device=self.device)
        self._is_tensorized = True

    def get_batches(self, batch_size: int):
        assert self._is_tensorized, "Call compute_returns_and_advantages() before batching"
        idxs = np.arange(len(self.rewards))
        np.random.shuffle(idxs)
        for start in range(0, len(self.rewards), batch_size):
            batch_idx = idxs[start:start + batch_size]
            yield {
                "obs": {tf: self.obs[tf][batch_idx] for tf in self.obs},
                "actions": self.actions[batch_idx],
                "log_probs": self.log_probs[batch_idx],
                "values": self.values[batch_idx],
                "returns": self.returns[batch_idx],
                "advantages": self.advantages[batch_idx],
            }

    def clear(self):
        tf_keys = list(self.obs.keys())
        self.obs = {tf: [] for tf in tf_keys}
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.infos = []
        self._is_tensorized = False
