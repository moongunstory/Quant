import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from typing import Dict
# Import config values
from modules.config import TP_THRESHOLD, SL_THRESHOLD, LABEL_HORIZON, TIMEFRAMES


class PPOTradingEnv:
    def __init__(
        self,
        data_path: str,
        direction: str = "long",
        seq_len: int = 32,
        reference_timeframe: str = "15min",
        hold_reward: float = 0.001,
        include_all_scenarios: bool = True,
        reward_scale: float = 10.0,
    ):
        """
        MTF PPO Trading Environment

        Args:
            data_path: Path to MTF data (Dict[str, pd.DataFrame] pickle or npz file)
            direction: "long" or "short"
            seq_len: Sequence length for each timeframe
            reference_timeframe: Timeframe used for reward calculation and entry signals
            hold_reward: Reward when neither TP nor SL is hit
        """
        self.direction = direction.lower()
        self.seq_len = seq_len
        self.reference_timeframe = reference_timeframe
        self.tp_ratio = TP_THRESHOLD
        self.sl_ratio = SL_THRESHOLD
        self.horizon = LABEL_HORIZON
        self.hold_reward = hold_reward
        self.include_all = include_all_scenarios
        self.reward_scale = reward_scale
        self._logged_direction = False
        self.step_count = 0

        print(
            f"[ENV INIT] direction='{self.direction}', ref_tf='{reference_timeframe}', hold_reward={hold_reward}"
        )
        print(
            f"[ENV INIT] TP={self.tp_ratio}, SL={self.sl_ratio}, horizon={self.horizon}"
        )

        self._load_mtf_data(data_path)
        self._prepare_data()
        self.reset()

    def _load_mtf_data(self, data_path: str):
        """Load MTF data from pickle or npz file"""
        from modules.training.ppo.reinforce.train_ppo import load_cached_pickle  # 필요시 위치 조정

        if data_path.endswith(".pkl"):
            self.mtf_data = load_cached_pickle(data_path)

        elif data_path.endswith(".npz"):
            loaded = np.load(data_path, allow_pickle=True)
            self.mtf_data = loaded["data"].item()  # Convert back to dict
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        if not isinstance(self.mtf_data, dict):
            raise ValueError("Data must be a dictionary of timeframe DataFrames")

        print(f"[DATA LOAD] Available timeframes: {list(self.mtf_data.keys())}")

        # Validate reference timeframe exists
        if self.reference_timeframe not in self.mtf_data:
            raise ValueError(
                f"Reference timeframe '{self.reference_timeframe}' not found in data"
            )

    def _prepare_data(self):
        """Prepare MTF sequences and find valid entry points"""
        # Use reference dataframe
        ref_df = self.mtf_data[self.reference_timeframe]

        if "label" not in ref_df.columns:
            raise ValueError(
                f"Reference timeframe '{self.reference_timeframe}' must have 'label' column"
            )

        if self.include_all:
            self.valid_indices = list(range(len(ref_df)))
        else:
            self.valid_indices = np.where(ref_df["label"] == 1)[0].tolist()

        # Prepare sequences for each timeframe
        self.sequences = {}
        self.entry_indices = []

        # Get feature columns for each timeframe (exclude timestamp, label)
        self.feature_cols = {}
        for tf, df in self.mtf_data.items():
            if tf in ["btc", "dune"]:
                # External features - keep all columns except timestamp
                self.feature_cols[tf] = [
                    col for col in df.columns if col not in ["timestamp"]
                ]
            else:
                # Regular timeframes - exclude timestamp and label
                self.feature_cols[tf] = [
                    col for col in df.columns if col not in ["timestamp", "label"]
                ]

        # Fill NaN values for all timeframes
        for tf, df in self.mtf_data.items():
            feature_cols = self.feature_cols[tf]
            self.mtf_data[tf][feature_cols] = df[feature_cols].fillna(0.0)

        # Create aligned sequences for each timeframe
        valid_sequences = []

        for idx in self.valid_indices:
            # Check if we can create a full sequence for reference timeframe
            if idx < self.seq_len - 1:
                continue

            # Extract sequences for each timeframe
            tf_sequences = {}
            valid_sequence = True

            for tf in TIMEFRAMES + ["btc", "dune"]:
                if tf not in self.mtf_data or self.mtf_data[tf].empty:
                    continue

                df = self.mtf_data[tf]
                feature_cols = self.feature_cols[tf]

                # For external features (btc, dune), use single latest value
                if tf in ["btc", "dune"]:
                    # Find the latest available data point up to the entry index
                    available_data = df.iloc[: idx + 1]
                    if not available_data.empty:
                        tf_sequences[tf] = available_data[feature_cols].iloc[-1].values
                    else:
                        valid_sequence = False
                        break
                else:
                    # For regular timeframes, create sequence
                    if len(df) > idx:
                        seq_data = df.iloc[max(0, idx - self.seq_len + 1) : idx + 1]
                        if len(seq_data) == self.seq_len:
                            tf_sequences[tf] = seq_data[feature_cols].values
                        else:
                            valid_sequence = False
                            break
                    else:
                        valid_sequence = False
                        break

            if valid_sequence and len(tf_sequences) > 0:
                valid_sequences.append(tf_sequences)
                self.entry_indices.append(idx)

        # Convert to arrays and store
        if valid_sequences:
            # Initialize sequences dict
            for tf in valid_sequences[0].keys():
                self.sequences[tf] = []

            # Collect all sequences
            for seq_dict in valid_sequences:
                for tf, seq in seq_dict.items():
                    self.sequences[tf].append(seq)

            # Convert to numpy arrays
            for tf in self.sequences:
                self.sequences[tf] = np.array(self.sequences[tf])
        else:
            self.sequences = {}

        self.entry_indices = np.array(self.entry_indices)

        print(f"[DATA PREP] Found {len(self.entry_indices)} valid sequences")
        for tf, seq in self.sequences.items():
            print(f"[DATA PREP] {tf}: {seq.shape}")

    def reset(self):
        """Reset environment to initial state"""
        # Shuffle order of entry indices and sequences each reset to
        # expose the agent to diverse scenarios across epochs
        if len(self.entry_indices) > 0:
            perm = np.random.permutation(len(self.entry_indices))
            self.entry_indices = self.entry_indices[perm]
            for tf in self.sequences:
                self.sequences[tf] = self.sequences[tf][perm]

        self.ptr = 0
        self.step_count = 0
        self.done = False
        return self._get_state()

    def _get_state(self) -> Dict[str, np.ndarray]:
        """Get current state as dictionary of timeframe sequences"""
        if self.ptr >= len(self.entry_indices):
            # Return empty state if no more data
            return {tf: np.zeros_like(seq[0]) for tf, seq in self.sequences.items()}

        state = {}
        for tf, sequences in self.sequences.items():
            state[tf] = sequences[self.ptr]
        return state

    def step(self, action):
        """
        Take a step in the environment.

        Reward calculation mirrors the labeling logic by checking
        5-minute high/low prices over ``LABEL_HORIZON`` for TP/SL hits.

        Args:
            action: 0 (Hold), 1 (Trade)

        Returns:
            state: Dict[str, np.ndarray] - Next state
            reward: float - Reward for the action
            done: bool - Whether episode is finished
            info: dict - Additional information
        """
        done = False
        reward = 0.0
        info = {}

        if self.ptr >= len(self.entry_indices):
            return self._get_state(), 0.0, True, info

        entry_idx = self.entry_indices[self.ptr]

        # Use reference timeframe for reward calculation
        ref_df = self.mtf_data[self.reference_timeframe]

        # Find close price column in reference timeframe
        close_col = None
        for col in self.feature_cols[self.reference_timeframe]:
            if "close" in col.lower():
                close_col = col
                break

        if close_col is None:
            raise ValueError(
                f"No 'close' column found in reference timeframe '{self.reference_timeframe}'"
            )

        entry_price = ref_df.iloc[entry_idx][close_col]

        horizon_limit = min(entry_idx + self.horizon, len(ref_df) - 1)

        if horizon_limit > entry_idx and "5min" in self.mtf_data:
            df_5m = self.mtf_data["5min"]

            high_col = low_col = None
            for col in self.feature_cols.get("5min", []):
                if "high" in col.lower():
                    high_col = col
                elif "low" in col.lower():
                    low_col = col
            if high_col is None or low_col is None:
                raise ValueError("No 'high' or 'low' column found in 5min timeframe")

            entry_time = ref_df.index[entry_idx]
            end_time = ref_df.index[horizon_limit]

            future_5m = df_5m[(df_5m.index > entry_time) & (df_5m.index <= end_time)]

            if self.direction == "long":
                tp_price = entry_price * (1 + self.tp_ratio)
                sl_price = entry_price * (1 + self.sl_ratio)
                tp_reached = future_5m[high_col] >= tp_price
                sl_reached = future_5m[low_col] <= sl_price
            else:
                tp_price = entry_price * (1 + self.sl_ratio)
                sl_price = entry_price * (1 + self.tp_ratio)
                tp_reached = future_5m[low_col] <= tp_price
                sl_reached = future_5m[high_col] >= sl_price

            tp_first_idx = (
                future_5m[tp_reached].index.min() if tp_reached.any() else pd.NaT
            )
            sl_first_idx = (
                future_5m[sl_reached].index.min() if sl_reached.any() else pd.NaT
            )

            tp_hit = pd.notna(tp_first_idx)
            sl_hit = pd.notna(sl_first_idx)
        else:
            tp_hit = False
            sl_hit = False

        if action == 0:  # Hold
            if sl_hit and not tp_hit:
                reward = 0.1 * self.reward_scale
            else:
                reward = -0.01 * self.reward_scale
        else:  # Trade
            if tp_hit and sl_hit:
                hit_tp = tp_first_idx < sl_first_idx
            else:
                hit_tp = tp_hit and not sl_hit

            if hit_tp:
                reward = 1.0 * self.reward_scale
            elif sl_hit:
                reward = -1.0 * self.reward_scale
            else:
                reward = -0.01 * self.reward_scale

        # Reward Shaping: 가격 움직임 정보 추가
        base_reward = reward
        price_change = 0.0
        shaped_reward = 0.0
        if horizon_limit > entry_idx:
            # 실제 가격 변화율 계산
            final_price = ref_df.iloc[horizon_limit][close_col]
            price_change = (final_price - entry_price) / entry_price

            if self.direction == "long":
                # Long의 경우 가격 상승이 좋음
                shaped_reward = price_change * 0.5 * self.reward_scale
            else:
                # Short의 경우 가격 하락이 좋음
                shaped_reward = -price_change * 0.5 * self.reward_scale

            # Action에 따라 reward 조정
            if action == 1:  # Trade
                reward = base_reward + shaped_reward
            else:  # Hold
                reward = base_reward - abs(shaped_reward) * 0.1  # Hold시 기회비용

        if logger.isEnabledFor(logging.DEBUG) and action == 1 and abs(reward) >= 0.3:
            logger.debug(
                f"[REWARD] idx={self.step_count}, action={action}, "
                f"base={base_reward:.3f}, shaped={shaped_reward:.3f}, total={reward:.3f}"
            )
            logger.debug(
                f"→ TP={tp_hit}, SL={sl_hit}, price_change={price_change:.4f}"
            )

        # Move to next state
        self.ptr += 1
        self.step_count += 1
        if self.ptr >= len(self.entry_indices) - 1:
            done = True
            
        return self._get_state(), reward, done, info

    def get_action_space_size(self):
        """Get size of action space"""
        return 2  # Hold, Trade

    def get_observation_space(self):
        """Get observation space information"""
        obs_space = {}
        for tf, sequences in self.sequences.items():
            if len(sequences) > 0:
                obs_space[tf] = sequences[0].shape
        return obs_space

    def get_input_dims(self) -> Dict[str, int]:
        """각 타임프레임별 입력 feature 수 반환"""
        input_dims = {}
        for tf, data in self.sequences.items():
            if data.ndim == 3:  # (num_seq, seq_len, feature_dim)
                input_dims[tf] = data.shape[2]
            elif (
                data.ndim == 2
            ):  # (num_seq, feature_dim) for external features like btc/dune
                input_dims[tf] = data.shape[1]
        return input_dims
