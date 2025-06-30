import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, Union

# Import config values
from modules.config import (
    TP_THRESHOLD,
    SL_THRESHOLD, 
    LABEL_HORIZON,
    TIMEFRAMES
)

class PPOTradingEnv:
    def __init__(self, data_path: str, direction: str = "long", seq_len: int = 32,
                 reference_timeframe: str = "15min", hold_reward: float = 0.01):
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
        self._logged_direction = False
        
        print(f"[ENV INIT] direction='{self.direction}', ref_tf='{reference_timeframe}', hold_reward={hold_reward}")
        print(f"[ENV INIT] TP={self.tp_ratio}, SL={self.sl_ratio}, horizon={self.horizon}")
        
        self._load_mtf_data(data_path)
        self._prepare_data()
        self.reset()

    def _load_mtf_data(self, data_path: str):
        """Load MTF data from pickle or npz file"""
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.mtf_data = pickle.load(f)
        elif data_path.endswith('.npz'):
            loaded = np.load(data_path, allow_pickle=True)
            self.mtf_data = loaded['data'].item()  # Convert back to dict
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        if not isinstance(self.mtf_data, dict):
            raise ValueError("Data must be a dictionary of timeframe DataFrames")
        
        print(f"[DATA LOAD] Available timeframes: {list(self.mtf_data.keys())}")
        
        # Validate reference timeframe exists
        if self.reference_timeframe not in self.mtf_data:
            raise ValueError(f"Reference timeframe '{self.reference_timeframe}' not found in data")

    def _prepare_data(self):
        """Prepare MTF sequences and find valid entry points"""
        # Use reference timeframe to find entry points (where label == 1)
        ref_df = self.mtf_data[self.reference_timeframe]
        
        if 'label' not in ref_df.columns:
            raise ValueError(f"Reference timeframe '{self.reference_timeframe}' must have 'label' column")
        
        # Find valid entry indices from reference timeframe
        self.valid_indices = ref_df[ref_df["label"] == 1].index.to_series().reset_index(drop=True).index.tolist()
        
        # Prepare sequences for each timeframe
        self.sequences = {}
        self.entry_indices = []
        
        # Get feature columns for each timeframe (exclude timestamp, label)
        self.feature_cols = {}
        for tf, df in self.mtf_data.items():
            if tf in ['btc', 'dune']:
                # External features - keep all columns except timestamp
                self.feature_cols[tf] = [col for col in df.columns if col not in ["timestamp"]]
            else:
                # Regular timeframes - exclude timestamp and label
                self.feature_cols[tf] = [col for col in df.columns if col not in ["timestamp", "label"]]
        
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
            
            for tf in TIMEFRAMES + ['btc', 'dune']:
                if tf not in self.mtf_data or self.mtf_data[tf].empty:
                    continue
                    
                df = self.mtf_data[tf]
                feature_cols = self.feature_cols[tf]
                
                # For external features (btc, dune), use single latest value
                if tf in ['btc', 'dune']:
                    # Find the latest available data point up to the entry index
                    available_data = df.iloc[:idx+1]
                    if not available_data.empty:
                        tf_sequences[tf] = available_data[feature_cols].iloc[-1].values
                    else:
                        valid_sequence = False
                        break
                else:
                    # For regular timeframes, create sequence
                    if len(df) > idx:
                        seq_data = df.iloc[max(0, idx - self.seq_len + 1):idx + 1]
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
        self.ptr = 0
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
        Take a step in the environment
        
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
            raise ValueError(f"No 'close' column found in reference timeframe '{self.reference_timeframe}'")
        
        entry_price = ref_df.iloc[entry_idx][close_col]

        if action == 0:  # Hold
            reward = 0.0
        else:  # Trade
            # Calculate returns over horizon
            horizon_limit = min(entry_idx + self.horizon, len(ref_df) - 1)
            
            if horizon_limit > entry_idx:
                future_prices = ref_df.iloc[entry_idx + 1:horizon_limit + 1][close_col].values
                returns = (future_prices - entry_price) / entry_price
                
                # Reverse returns for short direction
                if self.direction == "short":
                    if not self._logged_direction:
                        print(f"[STEP] direction = {self.direction}, action = {action}")
                        print("🟥 SHORT reward reversal applied")
                        self._logged_direction = True
                    returns = -returns
                else:
                    if not self._logged_direction:
                        print("🟩 LONG reward structure applied")
                        self._logged_direction = True
                
                # Check TP/SL conditions
                tp_hit = np.any(returns >= self.tp_ratio)
                sl_hit = np.any(returns <= self.sl_ratio)

                if sl_hit:
                    reward = -0.5
                elif tp_hit:
                    reward = 1.0
                else:
                    reward = self.hold_reward
            else:
                reward = self.hold_reward

        # Move to next state
        self.ptr += 1
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
            elif data.ndim == 2:  # (num_seq, feature_dim) for external features like btc/dune
                input_dims[tf] = data.shape[1]
        return input_dims
