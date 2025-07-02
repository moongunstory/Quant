import torch
import numpy as np
from typing import Dict, List, Generator, Tuple
from modules.config import PPO_CONFIG, TIMEFRAMES

class RolloutBuffer:
    def __init__(self, buffer_size: int = PPO_CONFIG["buffer_size"]):
        """
        MTF RolloutBuffer for PPO training
        
        Args:
            buffer_size: Maximum buffer size from config
        """
        self.buffer_size = buffer_size
        self.clear()

    def clear(self):
        """Clear all stored data"""
        self.observations = []  # List[Dict[str, Tensor]]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        # GAE 후 계산될 값들
        self.returns = []
        self.advantages = []

    def add(self, obs: Dict[str, torch.Tensor], action: torch.Tensor, reward: float, 
            done: bool, log_prob: torch.Tensor, value: torch.Tensor):
        """
        Add experience to buffer
        
        Args:
            obs: MTF observation dict {"5min": tensor, "15min": tensor, ...}
            action: Action tensor
            reward: Reward value
            done: Episode done flag
            log_prob: Log probability of action
            value: Value prediction
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, 
                                    lam: float = 0.95, normalize: bool = True):
        # 모든 값을 float로 통일
        values = self.values + [last_value]
        gae = 0.0
        self.returns = []
        self.advantages = []

        for t in reversed(range(len(self.rewards))):
            done_float = 1.0 if self.dones[t] else 0.0  # bool을 명시적으로 float 변환
            delta = self.rewards[t] + gamma * values[t + 1] * (1.0 - done_float) - values[t]
            gae = delta + gamma * lam * (1.0 - done_float) * gae
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + values[t])

        # 안정적인 정규화
        advantages = torch.tensor(self.advantages, dtype=torch.float32)
        returns = torch.tensor(self.returns, dtype=torch.float32)

        if normalize and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size: int) -> Generator[Tuple[Dict[str, torch.Tensor], torch.Tensor, 
                                                             torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """
        Generate batches for training
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            Tuple of (obs_batch_dict, actions, returns, advantages, log_probs)
        """
        n = len(self.observations)
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            # Build MTF observation batch
            obs_batch = self._build_mtf_batch(batch_idx)
            
            # Build other batches
            actions_batch = torch.tensor([self.actions[i] for i in batch_idx], dtype=torch.long)
            returns_batch = self.returns[batch_idx]
            advantages_batch = self.advantages[batch_idx]
            log_probs_batch = torch.tensor([self.log_probs[i] for i in batch_idx], dtype=torch.float32)

            yield (obs_batch, actions_batch, returns_batch, advantages_batch, log_probs_batch)

    def _build_mtf_batch(self, batch_idx: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Build MTF observation batch from indices
        
        Args:
            batch_idx: Indices to batch
            
        Returns:
            Dict with timeframe keys and stacked tensors
        """
        # Get timeframes from first observation
        if not self.observations:
            return {}
        
        available_timeframes = list(self.observations[0].keys())
        obs_batch = {}
        
        for tf in available_timeframes:
            # Stack tensors for this timeframe
            tf_tensors = [self.observations[i][tf] for i in batch_idx]
            obs_batch[tf] = torch.stack(tf_tensors)
        
        return obs_batch

    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self.observations) >= self.buffer_size

    def size(self) -> int:
        """Get current buffer size"""
        return len(self.observations)

    def get_latest_observation(self) -> Dict[str, torch.Tensor]:
        """Get the most recent observation"""
        if self.observations:
            return self.observations[-1]
        return {}

    def to_device(self, device: torch.device):
        """
        Move buffer contents to specified device (optional)
        
        Args:
            device: Target device
        """
        # Move tensor data to device
        if self.returns:
            self.returns = self.returns.to(device)
        if self.advantages:
            self.advantages = self.advantages.to(device)
        
        # Move observations to device
        for i, obs in enumerate(self.observations):
            self.observations[i] = {tf: tensor.to(device) for tf, tensor in obs.items()}
        
        # Move other tensors to device
        for i in range(len(self.actions)):
            if isinstance(self.actions[i], torch.Tensor):
                self.actions[i] = self.actions[i].to(device)
            if isinstance(self.log_probs[i], torch.Tensor):
                self.log_probs[i] = self.log_probs[i].to(device)
            if isinstance(self.values[i], torch.Tensor):
                self.values[i] = self.values[i].to(device)