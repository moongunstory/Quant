import torch
import numpy as np
import pickle
import os
import re

class RolloutBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.clear()

    def clear(self):
        self.observations = []
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

    def add(self, obs, action, reward, done, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95, normalize=True):
        values = self.values + [last_value]
        gae = 0
        self.returns = []
        self.advantages = []

        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            self.advantages.insert(0, gae)
            self.returns.insert(0, gae + values[t])

        advantages = torch.tensor(self.advantages, dtype=torch.float32)
        returns = torch.tensor(self.returns, dtype=torch.float32)

        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.advantages = advantages
        self.returns = returns

    def get_batches(self, batch_size):
        n = len(self.observations)
        indices = np.arange(n)
        np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield (
                torch.stack([self.observations[i] for i in batch_idx]),
                torch.tensor([self.actions[i] for i in batch_idx]),
                self.returns[batch_idx],
                self.advantages[batch_idx],
                torch.tensor([self.log_probs[i] for i in batch_idx]),
            )

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"📦 RolloutBuffer 저장됨: {path}")

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            buffer = pickle.load(f)
        print(f"📥 RolloutBuffer 로드됨: {path}")
        return buffer

    @staticmethod
    def delete(path):
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
        files = os.listdir(dir_path)
        pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.pkl")
        indices = [
            int(match.group(1)) for f in files
            if (match := pattern.match(f))
        ]
        return max(indices, default=0) + 1