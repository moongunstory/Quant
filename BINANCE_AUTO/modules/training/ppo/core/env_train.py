import numpy as np
import pandas as pd

class PPOTradingEnv:
    def __init__(self, csv_path: str, direction: str = "long", seq_len: int = 32, tp_ratio=0.008, sl_ratio=-0.008, horizon=4):
        self.direction = direction.lower()  # ✅ "long" 또는 "short"
        self._logged_direction = False 
        print(f"[ENV INIT] direction = '{self.direction}'")
        """
        csv_path: 라벨링된 CSV 파일 경로
        seq_len: 상태 시퀀스 길이
        """
        self.df = pd.read_csv(csv_path)
        self.seq_len = seq_len
        self.tp_ratio = tp_ratio
        self.sl_ratio = sl_ratio
        self.horizon = horizon

        self._prepare_data()
        self.reset()

    def _prepare_data(self):
        # label이 1인 지점만 학습 대상으로 추림 (long 또는 short)
        self.valid_indices = self.df[self.df["label"] == 1].index.tolist()

        # 전체 feature 컬럼 추출 (label, timestamp 제외)
        self.feature_cols = [col for col in self.df.columns if col not in ["timestamp", "label"]]

        # ✅ NaN 대체 추가
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0.0)

        self.data = self.df[self.feature_cols].values

        self.sequences = []
        self.entry_indices = []

        for idx in self.valid_indices:
            if idx - self.seq_len + 1 < 0:
                continue
            seq = self.data[idx - self.seq_len + 1: idx + 1]
            self.sequences.append(seq)
            self.entry_indices.append(idx)

        self.sequences = np.array(self.sequences)  # shape: (N, seq_len, feature_dim)
        self.entry_indices = np.array(self.entry_indices)

    def reset(self):
        self.ptr = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return self.sequences[self.ptr]

    def step(self, action):
        """
        action: 0 (Hold), 1 (Long)
        """
        done = False
        reward = 0.0
        info = {}

        entry_idx = self.entry_indices[self.ptr]
        # 실제 close 계열 피처 자동 검색
        close_col = next((col for col in self.feature_cols if "close" in col.lower()), None)
        if close_col is None:
            raise ValueError(f"❌ 'close' 관련 컬럼이 feature_cols에 없습니다 → feature_cols={self.feature_cols[:5]}")
        close_idx = self.feature_cols.index(close_col)
        entry_price = self.data[entry_idx][close_idx]

        if action == 0:
            reward = 0.0
        else:
            horizon_limit = min(entry_idx + self.horizon, len(self.data) - 1)
            prices = self.data[entry_idx + 1: horizon_limit + 1, close_idx]
            returns = (prices - entry_price) / entry_price
            if self.direction == "short":
                if not self._logged_direction:
                    print(f"[STEP] direction = {self.direction}, action = {action}")
                    print("🟥 SHORT reward reversal 적용됨")
                    self._logged_direction = True
                returns = -returns
            else:
                if not self._logged_direction:
                    print("🟩 LONG reward 구조 적용됨")
                    self._logged_direction = True
                    
            tp_hit = np.any(returns >= self.tp_ratio)
            sl_hit = np.any(returns <= self.sl_ratio)

            if sl_hit:
                reward = -1.0
            elif tp_hit:
                reward = 1.0
            else:
                reward = -0.1  # 또는 returns[-1] * 스케일링


        self.ptr += 1
        if self.ptr >= len(self.sequences) - 1:
            done = True

        return self._get_state(), reward, done, info
