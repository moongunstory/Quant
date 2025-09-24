# train/reinforce/core/sequence_replay_buffer.py

import numpy as np

class SequenceReplayBuffer:
    def __init__(self, max_size, input_dims, action_dim, seq_lens, batch_size=64, burn_in=16):
        """
        input_dims: {"ohlcv": 66, "funding": 3, ...}
        seq_lens: {"ohlcv": 48, "funding": 7, ...}
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.seq_lens = seq_lens
        self.seq_len_max = max(seq_lens.values())
        self.batch_size = batch_size
        self.burn_in = burn_in
        
        self.obs_buf = {
            k: np.zeros((max_size, d), dtype=np.float32)
            for k, d in input_dims.items()
        }
        self.next_obs_buf = {
            k: np.zeros((max_size, d), dtype=np.float32)
            for k, d in input_dims.items()
        }
        self.action_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)

    def __len__(self):
        return self.size

    def add(self, obs: dict, action, reward, next_obs: dict, done):
        for k in obs:
            self.obs_buf[k][self.ptr] = obs[k][-1]  # 저장은 한 시점
            self.next_obs_buf[k][self.ptr] = next_obs[k][-1]

        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # 시퀀스가 넘치지 않도록 안전 범위 계산
        if self.size < self.seq_len_max:
            raise ValueError(f"Buffer size {self.size} < required seq_len {self.seq_len_max}")
        max_start = self.size - self.seq_len_max
        start_low = 0  # (burn-in은 에이전트 학습에서 활용; 시퀀스 추출은 전체 구간 유지)
        indices = np.random.randint(start_low, max_start + 1, size=batch_size)

        obs_seqs = {k: [] for k in self.obs_buf}
        next_obs_seqs = {k: [] for k in self.next_obs_buf}
        action_list = []
        reward_list = []
        done_list = []

        for idx in indices:
            # 모든 그룹이 동일한 종료 시점(last_idx)을 바라보도록 정렬
            last_idx = idx + self.seq_len_max - 1

            # State는 시퀀스로 반환 (LSTM용)
            for k in self.obs_buf:
                seq_len = self.seq_lens[k]
                start_idx = idx + (self.seq_len_max - seq_len)
                end_idx = start_idx + seq_len
                obs_seqs[k].append(self.obs_buf[k][start_idx:end_idx])
                next_obs_seqs[k].append(self.next_obs_buf[k][start_idx:end_idx])

            # Action/Reward/Done은 모든 그룹 중 가장 긴 시퀀스의 마지막 시점 기준
            action_list.append(self.action_buf[last_idx])
            reward_list.append(self.reward_buf[last_idx])
            done_list.append(self.done_buf[last_idx])

        obs_batch = {k: np.array(v) for k, v in obs_seqs.items()}
        next_obs_batch = {k: np.array(v) for k, v in next_obs_seqs.items()}

        return (
            obs_batch,                    # {k: (batch, seq_len, dim)} - LSTM용 시퀀스
            np.array(action_list),        # (batch, action_dim) - SAC용 단일값
            np.array(reward_list),        # (batch, 1) - SAC용 단일값
            next_obs_batch,               # {k: (batch, seq_len, dim)} - LSTM용 시퀀스  
            np.array(done_list),          # (batch, 1) - SAC용 단일값
        )