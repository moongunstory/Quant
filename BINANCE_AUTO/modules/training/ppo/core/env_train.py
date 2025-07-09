import numpy as np
import pandas as pd
import logging
import random

logger = logging.getLogger(__name__)

from typing import Dict
# Import config values
from modules.config import PPO_CONFIG, TIMEFRAMES


class PPOTradingEnv:
    def __init__(
        self,
        data_path: str,
        direction: str = "long",
        seq_len: int = 32,
        reference_timeframe: str = "1min", # 1분봉 기준으로 변경
        liquidation_threshold: float = 0.5, # 청산 확신도 임계치
        max_episode_steps: int = 120, # 한 에피소드의 최대 길이 (분 단위)
    ):
        """
        MTF PPO Trading Environment for Position Management

        Args:
            data_path: Path to MTF data (Dict[str, pd.DataFrame] pickle file)
            direction: "long" or "short" - the type of position this environment manages
            seq_len: Sequence length for each timeframe
            reference_timeframe: Timeframe used for current price and indexing
            liquidation_threshold: Confidence level above which to liquidate
            max_episode_steps: Maximum number of steps (minutes) in an episode
        """
        self.direction = direction.lower()
        if self.direction not in ['long', 'short']:
            raise ValueError("Direction must be 'long' or 'short'.")

        self.seq_len = seq_len
        self.reference_timeframe = reference_timeframe
        self.liquidation_threshold = liquidation_threshold
        self.max_episode_steps = max_episode_steps

        # Position tracking variables
        self.current_position_type = self.direction  # Agent always starts in a position of its type
        self.position_entry_price = None
        self.time_in_position = 0
        self.unrealized_pnl = 0.0
        self.current_idx = 0 # Current index within the reference dataframe
        self.start_idx = 0 # Starting index of the current episode

        logger.info(
            f"""[ENV INIT] Direction={self.direction.upper()}, SeqLen={self.seq_len}, "
            f"RefTF={self.reference_timeframe}, LiqThresh={self.liquidation_threshold}, "
            f"MaxSteps={self.max_episode_steps}"""
        )

        self._load_mtf_data(data_path)
        self._prepare_data()
        self.reset()

    def _load_mtf_data(self, data_path: str):
        """Load MTF data from pickle file"""
        from modules.training.ppo.reinforce.train_ppo import load_cached_pickle  # Assuming this is correct path

        if data_path.endswith(".pkl"):
            self.mtf_data = load_cached_pickle(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # 'dune' 데이터를 로드 단계에서부터 제외 (if it exists)
        if 'dune' in self.mtf_data:
            del self.mtf_data['dune']
            logger.info("🚫 'dune' timeframe data excluded from loading.")

        if not isinstance(self.mtf_data, dict):
            raise ValueError("Data must be a dictionary of timeframe DataFrames")

        logger.info(f"[DATA LOAD] Available timeframes: {list(self.mtf_data.keys())}")

        # Validate reference timeframe exists
        if self.reference_timeframe not in self.mtf_data:
            raise ValueError(
                f"Reference timeframe '{self.reference_timeframe}' not found in data"
            )

    def _prepare_data(self):
        """Prepare data for episode selection and feature extraction"""
        self.ref_df = self.mtf_data[self.reference_timeframe]
        
        # Get feature columns for each timeframe (exclude timestamp, label if present)
        self.feature_cols = {}
        for tf, df in self.mtf_data.items():
            if tf in ["btc"]:
                self.feature_cols[tf] = [col for col in df.columns if col not in ["timestamp"]]
            else:
                self.feature_cols[tf] = [col for col in df.columns if col not in ["timestamp", "label"]]

        # Ensure all dataframes are float32 and fill NaNs (should be handled by collector, but for safety)
        for tf, df in self.mtf_data.items():
            self.mtf_data[tf] = df.astype(np.float32).fillna(0.0)

        # Determine valid starting indices for episodes
        # An episode can start at any point where there's enough historical data (seq_len)
        # and enough future data for at least one step + max_episode_steps
        min_start_idx = self.seq_len - 1
        max_start_idx = len(self.ref_df) - self.max_episode_steps - 1 # Ensure enough future data for max episode length
        
        self.possible_start_indices = list(range(min_start_idx, max_start_idx))
        
        if not self.possible_start_indices:
            raise ValueError("Not enough data to form valid episodes. Check data length and seq_len/max_episode_steps.")

        logger.info(f"[DATA PREP] Found {len(self.possible_start_indices)} possible episode start indices.")

    def reset(self):
        """Reset environment to a new random episode"""
        self.done = False
        self.time_in_position = 0
        self.unrealized_pnl = 0.0

        # Randomly select a starting index for the episode
        self.start_idx = random.choice(self.possible_start_indices)
        self.current_idx = self.start_idx

        # Set initial position entry price
        self.position_entry_price = self.ref_df.iloc[self.current_idx][self._get_close_col_name()]
        
        logger.debug(f"[ENV RESET] New episode starting at index {self.start_idx} (Time: {self.ref_df.index[self.start_idx]}) with entry price {self.position_entry_price:.4f}")

        return self._get_state()

    def _get_close_col_name(self):
        for col in self.feature_cols[self.reference_timeframe]:
            if "close" in col.lower():
                return col
        raise ValueError(f"No 'close' column found in reference timeframe '{self.reference_timeframe}'")

    def _get_high_col_name(self):
        for col in self.feature_cols[self.reference_timeframe]:
            if "high" in col.lower():
                return col
        raise ValueError(f"No 'high' column found in reference timeframe '{self.reference_timeframe}'")

    def _get_low_col_name(self):
        for col in self.feature_cols[self.reference_timeframe]:
            if "low" in col.lower():
                return col
        raise ValueError(f"No 'low' column found in reference timeframe '{self.reference_timeframe}'")

    def _get_state(self) -> Dict[str, np.ndarray]:
        """Get current state as dictionary of timeframe sequences and position info"""
        state = {}
        # Extract features for each timeframe dynamically
        for tf in TIMEFRAMES + ["btc"]:
            if tf not in self.mtf_data or self.mtf_data[tf].empty:
                # Handle missing or empty timeframes by providing zero-filled data
                dim = PPO_CONFIG["input_dims"].get(tf, 0) # Assuming PPO_CONFIG has input_dims now
                if dim == 0: # Fallback if input_dims not in config
                    if tf in ["btc"]:
                        dim = 5 # Example default for btc (OHLCV)
                    else:
                        dim = 26 # Example default for 1min (OHLCV + indicators)

                if tf in ["btc"]:
                    state[tf] = np.zeros(dim, dtype=np.float32)
                else:
                    state[tf] = np.zeros((self.seq_len, dim), dtype=np.float32)
                continue

            df = self.mtf_data[tf]
            feature_cols = self.feature_cols[tf]

            if tf in ["btc"]:
                # For external features (btc), use single latest value
                # Find the latest available data point up to the current_idx
                # Use get_indexer with 'ffill' to find the closest previous index
                pos = df.index.get_indexer([self.ref_df.index[self.current_idx]], method='ffill')[0]
                if pos != -1:
                    state[tf] = df.iloc[pos][feature_cols].values
                else:
                    state[tf] = np.zeros(len(feature_cols), dtype=np.float32)
            else:
                # For regular timeframes, create sequence
                start_loc = max(0, self.current_idx - self.seq_len + 1)
                seq_data = df.iloc[start_loc : self.current_idx + 1][feature_cols].values
                
                # Pad if sequence is too short (shouldn't happen with proper possible_start_indices)
                if len(seq_data) < self.seq_len:
                    padding = np.zeros((self.seq_len - len(seq_data), len(feature_cols)), dtype=np.float32)
                    seq_data = np.vstack([padding, seq_data])
                state[tf] = seq_data

        # Add position-related features to the state
        current_price = self.ref_df.iloc[self.current_idx][self._get_close_col_name()]
        if self.current_position_type == 'long':
            self.unrealized_pnl = (current_price - self.position_entry_price) / self.position_entry_price
        elif self.current_position_type == 'short':
            self.unrealized_pnl = (self.position_entry_price - current_price) / self.position_entry_price
        
        position_info = np.array([
            1.0 if self.current_position_type == 'long' else 0.0, # Is Long
            1.0 if self.current_position_type == 'short' else 0.0, # Is Short
            self.position_entry_price, # Entry Price
            self.time_in_position, # Time in Position
            self.unrealized_pnl # Unrealized PnL
        ], dtype=np.float32)
        state['position_info'] = position_info

        return state

    def step(self, action_confidence: float):
        """
        Take a step in the environment based on liquidation confidence.

        Args:
            action_confidence: float (0.0 to 1.0) - Model's confidence to liquidate

        Returns:
            state: Dict[str, np.ndarray] - Next state
            reward: float - Reward for the action
            done: bool - Whether episode is finished
            info: dict - Additional information
        """
        reward = 0.0
        info = {}
        self.done = False

        # Get current price for reward calculation
        prev_price = self.ref_df.iloc[self.current_idx][self._get_close_col_name()]
        
        # Increment index for next state calculation
        self.current_idx += 1
        self.time_in_position += 1

        # Check if episode ends due to max steps or end of data
        if self.time_in_position >= self.max_episode_steps or self.current_idx >= len(self.ref_df):
            # Force liquidation at end of episode
            current_price = self.ref_df.iloc[self.current_idx - 1][self._get_close_col_name()] # Use last valid price
            if self.current_position_type == 'long':
                reward = (current_price - self.position_entry_price) / self.position_entry_price
            else: # short
                reward = (self.position_entry_price - current_price) / self.position_entry_price
            self.done = True
            logger.debug(f"[ENV STEP] Episode ended (Max steps or End of data). Final PnL: {reward:.4f}")

        # Agent's decision based on confidence
        elif action_confidence >= self.liquidation_threshold:
            # Agent decides to liquidate
            current_price = self.ref_df.iloc[self.current_idx - 1][self._get_close_col_name()] # Use last valid price
            if self.current_position_type == 'long':
                reward = (current_price - self.position_entry_price) / self.position_entry_price
            else: # short
                reward = (self.position_entry_price - current_price) / self.position_entry_price
            self.done = True
            logger.debug(f"[ENV STEP] Agent liquidated. Confidence: {action_confidence:.2f}, Final PnL: {reward:.4f}")

        else: # Agent decides to HOLD
            # Reward for holding: PnL for this single step
            current_price = self.ref_df.iloc[self.current_idx - 1][self._get_close_col_name()] # Use last valid price
            if self.current_position_type == 'long':
                reward = (current_price - prev_price) / prev_price # Per-step PnL
            else: # short
                reward = (prev_price - current_price) / prev_price # Per-step PnL
            
            # Small time penalty to encourage timely exits, not just holding forever
            reward -= 0.00001 # Very small penalty
            logger.debug(f"[ENV STEP] Agent held. Confidence: {action_confidence:.2f}, Step PnL: {reward:.4f}")

        return self._get_state(), float(reward), self.done, info

    def get_action_space_size(self):
        """Get size of action space (continuous output for confidence)"""
        return 1  # Single continuous output for liquidation confidence

    def get_observation_space(self):
        """Get observation space information"""
        obs_space = {}
        # This will be dynamically determined by the first state returned by reset()
        # For now, return a placeholder or rely on PPOPolicyNetwork to infer
        return obs_space # PPOPolicyNetwork will infer from first observation

    def get_input_dims(self) -> Dict[str, int]:
        """각 타임프레임별 입력 feature 수 반환"""
        input_dims = {}
        for tf, df in self.mtf_data.items():
            feature_cols = self.feature_cols[tf]
            if tf in ["btc"]:
                input_dims[tf] = len(feature_cols)
            else:
                input_dims[tf] = len(feature_cols)
        # Add position_info_dim manually as it's not from mtf_data
        # Assuming position_info has 5 features: is_long, is_short, entry_price, time_in_position, unrealized_pnl
        input_dims['position_info'] = 5
        return input_dims
