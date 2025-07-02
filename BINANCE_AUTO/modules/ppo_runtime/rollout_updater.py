import torch
import numpy as np
import pickle
import os
import re
import logging
from typing import Dict, Union, Generator, Tuple
from modules.config import TIMEFRAMES

logger = logging.getLogger(__name__)

class RolloutBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.clear()

    def clear(self):
        self.observations = []  # List[Dict[str, Tensor]]
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

        self.returns = []
        self.advantages = []

    def __len__(self):
        return len(self.rewards)

    def is_ready(self):
        return len(self.rewards) >= self.buffer_size

    def reset(self):
        self.clear()

    def add(self, obs: Dict[str, Union[torch.Tensor, np.ndarray]], action, reward, done, log_prob, value):
        """
        Add experience to buffer with MTF observation support
        
        Args:
            obs: MTF observation dict {"5min": tensor/array, "15min": tensor/array, ...}
            action: Action value
            reward: Reward value
            done: Episode done flag
            log_prob: Log probability of action
            value: Value prediction
        """
        # Convert observations to tensors if they are numpy arrays
        obs_tensors = {}
        for tf, data in obs.items():
            if isinstance(data, np.ndarray):
                obs_tensors[tf] = torch.tensor(data, dtype=torch.float32)
            elif isinstance(data, torch.Tensor):
                obs_tensors[tf] = data.clone().detach()
            else:
                raise ValueError(f"Unsupported data type for {tf}: {type(data)}")
        
        self.observations.append(obs_tensors)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95, normalize=True):
        """Compute returns and advantages using GAE - unchanged logic"""
        values = self.values + [last_value]
        gae = 0
        self.returns = []
        self.advantages = []

        N = len(self.rewards)
        # Only log at start, middle and end to reduce noise
        log_indices = {0, N // 2, N - 1}

        for t in reversed(range(N)):
            # Convert done to float to prevent type errors
            done_float = float(self.dones[t])
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - done_float) - values[t]
            gae = delta + gamma * lam * (1 - done_float) * gae
            if logger.isEnabledFor(logging.DEBUG) and t in log_indices:
                logger.debug(
                    f"[GAE] t={t} | reward={self.rewards[t]:.3f}, value={values[t]:.3f}, "
                    f"delta={delta:.3f}, done={int(done_float)}, gae={gae:.3f}"
                )
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + values[t])

        advantages = torch.tensor(self.advantages, dtype=torch.float32)
        returns = torch.tensor(self.returns, dtype=torch.float32)

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

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size) -> Generator[Tuple[Dict[str, torch.Tensor], torch.Tensor, 
                                                         torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """
        Generate MTF batches for training
        
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
            
            # Build other batches (unchanged)
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
        if not self.observations:
            return {}
        
        # Get available timeframes from first observation
        available_timeframes = list(self.observations[0].keys())
        obs_batch = {}
        
        for tf in available_timeframes:
            # Stack tensors for this timeframe
            tf_tensors = [self.observations[i][tf] for i in batch_idx]
            try:
                obs_batch[tf] = torch.stack(tf_tensors)
                print(f"[BATCH DEBUG] {tf}: stacked shape = {obs_batch[tf].shape}")
            except RuntimeError as e:
                print(f"[ERROR] Failed to stack {tf} tensors: {e}")
                # Print shapes for debugging
                shapes = [t.shape for t in tf_tensors]
                print(f"[ERROR] Tensor shapes for {tf}: {shapes}")
                raise
        
        return obs_batch

    def save(self, path):
        """Save buffer to pickle file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[BUFFER SAVE] Saved to {path}")

    @staticmethod
    def load(path):
        """Load buffer from pickle file"""
        with open(path, 'rb') as f:
            buffer = pickle.load(f)
        print(f"[BUFFER LOAD] Loaded from {path}")
        return buffer

    @staticmethod
    def delete(path):
        """Delete buffer file"""
        try:
            os.remove(path)
            print(f"🗑️ RolloutBuffer 삭제됨: {path}")
        except FileNotFoundError:
            print(f"⚠️ 삭제할 버퍼 없음: {path}")
        except Exception as e:
            print(f"❌ 버퍼 삭제 오류: {e}")

    @staticmethod
    def get_next_rollout_index(dir_path, prefix):
        """
        해당 디렉토리 내에서 주어진 prefix를 가진 pkl 파일들 중
        가장 큰 인덱스를 찾아 다음 인덱스를 반환함.
        예: long_rollout_001.pkl → 다음은 002 반환
        """
        if not os.path.exists(dir_path):
            return 1
            
        files = os.listdir(dir_path)
        pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.pkl")
        indices = [
            int(match.group(1)) for f in files
            if (match := pattern.match(f))
        ]
        return max(indices, default=0) + 1

    def get_observation_info(self) -> Dict:
        """Get information about stored observations"""
        if not self.observations:
            return {"empty": True}
        
        first_obs = self.observations[0]
        info = {
            "empty": False,
            "count": len(self.observations),
            "timeframes": list(first_obs.keys()),
            "shapes": {tf: tensor.shape for tf, tensor in first_obs.items()}
        }
        return info

    def validate_observations(self) -> bool:
        """Validate that all observations have consistent structure"""
        if not self.observations:
            return True
        
        first_timeframes = set(self.observations[0].keys())
        first_shapes = {tf: tensor.shape for tf, tensor in self.observations[0].items()}
        
        for i, obs in enumerate(self.observations[1:], 1):
            current_timeframes = set(obs.keys())
            if current_timeframes != first_timeframes:
                print(f"[ERROR] Timeframe mismatch at index {i}: {current_timeframes} vs {first_timeframes}")
                return False
            
            for tf, tensor in obs.items():
                if tensor.shape != first_shapes[tf]:
                    print(f"[ERROR] Shape mismatch for {tf} at index {i}: {tensor.shape} vs {first_shapes[tf]}")
                    return False
        
        return True