import numpy as np

class PPOTradingEnv:
    def __init__(self, data, tp_ratio=0.008, sl_ratio=-0.008, horizon=4):
        """
        data: numpy array of shape (num_steps, seq_len, feature_dim)
        """
        self.data = data
        self.tp_ratio = tp_ratio
        self.sl_ratio = sl_ratio
        self.horizon = horizon

        self.reset()

    def reset(self):
        self.ptr = 0  # 현재 위치
        self.episode_steps = 0
        self.position_open = False
        self.entry_price = None

        self.done = False
        return self._get_state()

    def _get_state(self):
        return self.data[self.ptr]

    def step(self, action):
        """
        action: 0 (Hold), 1 (Long)
        """
        reward = 0.0
        info = {}
        done = False

        current_state = self._get_state()
        entry_price = current_state[-1, 0]  # 마지막 캔들의 종가 (예시: feature 0번이 close라고 가정)

        if action == 0:
            reward = 0.0  # 홀드 → 보상 없음
        else:
            # 롱 진입 후 horizon 동안 TP/SL 조건 검사
            done = True
            horizon_limit = min(self.ptr + self.horizon, len(self.data) - 1)

            prices = self.data[self.ptr + 1:horizon_limit + 1, -1, 0]  # horizon 기간의 close 가격 시퀀스
            returns = (prices - entry_price) / entry_price

            tp_hit = np.any(returns >= self.tp_ratio)
            sl_hit = np.any(returns <= self.sl_ratio)

            if sl_hit:
                reward = -1.0
            elif tp_hit:
                reward = 1.0
            else:
                reward = -0.1  # 진입했는데 아무 것도 못 하고 끝난 경우

        self.ptr += 1
        if self.ptr >= len(self.data) - self.horizon:
            done = True

        next_state = self._get_state()
        return next_state, reward, done, info