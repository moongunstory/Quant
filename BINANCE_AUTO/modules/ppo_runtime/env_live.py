import pandas as pd
import numpy as np
from typing import Dict
from modules.config import (
    TP_THRESHOLD, 
    SL_THRESHOLD, 
    LABEL_HORIZON,
    TRADE_SYMBOL
)

class LivePPOEnv:
    def __init__(self, mtf_data: Dict[str, pd.DataFrame], seq_len: int = 32, 
                 reference_timeframe: str = "15min"):
        """
        MTF Live PPO Environment
        
        Args:
            mtf_data: Dictionary of timeframe DataFrames
                     Example: {"5min": df_5m, "15min": df_15m, "30min": df_30m, "1H": df_1h}
            seq_len: Sequence length for each timeframe
            reference_timeframe: Timeframe used for reward calculation
        """
        self.mtf_data = mtf_data.copy()
        self.seq_len = seq_len
        self.reference_timeframe = reference_timeframe
        self.current_step = 0

        # Position tracking variables
        self.current_position_type = 'none'  # 'long', 'short', 'none'
        self.position_entry_price = None
        self.time_in_position = 0
        self.unrealized_pnl = 0.0
        
        # Validate reference timeframe exists
        if reference_timeframe not in mtf_data:
            raise ValueError(f"Reference timeframe '{reference_timeframe}' not found in mtf_data")
        
        # Validate reference timeframe has required columns
        ref_df = mtf_data[reference_timeframe]
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in ref_df.columns]
        if missing_cols:
            # Try with prefix
            prefixed_cols = [f"{reference_timeframe}_{col}" for col in required_cols]
            missing_prefixed = [col for col in prefixed_cols if col not in ref_df.columns]
            if missing_prefixed:
                raise ValueError(f"Reference timeframe missing required columns: {missing_cols}")
            else:
                self.price_cols = {
                    'high': f"{reference_timeframe}_high",
                    'low': f"{reference_timeframe}_low", 
                    'close': f"{reference_timeframe}_close"
                }
        else:
            self.price_cols = {'high': 'high', 'low': 'low', 'close': 'close'}
        
        print(f"[LIVE ENV] Reference timeframe: {reference_timeframe}")
        print(f"[LIVE ENV] Price columns: {self.price_cols}")
        print(f"[LIVE ENV] Available timeframes: {list(mtf_data.keys())}")

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        # Reset position tracking variables
        self.current_position_type = 'none'
        self.position_entry_price = None
        self.time_in_position = 0
        self.unrealized_pnl = 0.0
        return self._get_observation()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current observation as dictionary of timeframe sequences
        
        Returns:
            Dict[str, np.ndarray]: Observations for each timeframe
        """
        observations = {}
        
        for tf_name, df in self.mtf_data.items():
            end = self.current_step + 1
            start = max(0, end - self.seq_len)
            
            # Extract sequence for this timeframe
            obs_df = df.iloc[start:end]
            
            # Pad with zeros if sequence is too short
            if len(obs_df) < self.seq_len:
                padding_rows = self.seq_len - len(obs_df)
                padding = np.zeros((padding_rows, len(df.columns)))
                obs_values = np.vstack([padding, obs_df.values])
            else:
                obs_values = obs_df.values
            
            observations[tf_name] = obs_values.astype(np.float32)

        # Add position-related features to the observations
        position_info = np.array([
            1.0 if self.current_position_type == 'long' else 0.0,
            1.0 if self.current_position_type == 'short' else 0.0,
            self.position_entry_price if self.position_entry_price is not None else 0.0,
            self.time_in_position,
            self.unrealized_pnl
        ], dtype=np.float32)
        observations['position_info'] = position_info
        
        return observations

    def step(self, action: int):
        """
        Take a step in the environment.
        
        Args:
            action: 0: ATTEMPT_LONG, 1: ATTEMPT_SHORT, 2: CLOSE_POSITION, 3: NO_ACTION
            
        Returns:
            observation: Dict[str, np.ndarray] - Next observation
            reward: float - Reward for the action
            done: bool - Whether episode is finished
            info: dict - Additional information
        """
        reward = 0.0
        done = False
        info = {}

        # Get current price from reference timeframe
        ref_df = self.mtf_data[self.reference_timeframe]
        current_price = ref_df[self.price_cols['close']].iloc[self.current_step]

        # Update time in position and unrealized PnL if holding a position
        if self.current_position_type != 'none':
            self.time_in_position += 1
            if self.current_position_type == 'long':
                self.unrealized_pnl = (current_price - self.position_entry_price) / self.position_entry_price
            elif self.current_position_type == 'short':
                self.unrealized_pnl = (self.position_entry_price - current_price) / self.position_entry_price
            
            # Small penalty for holding position to encourage timely exits
            reward -= 0.001 # Small holding penalty

        # Action interpretation and position management
        if action == 0: # ATTEMPT_LONG
            if self.current_position_type == 'none':
                self.current_position_type = 'long'
                self.position_entry_price = current_price
                self.time_in_position = 0
                self.unrealized_pnl = 0.0
            elif self.current_position_type == 'short':
                # Close short position first, then attempt long
                # Reward for closing short position
                reward += (self.position_entry_price - current_price) / self.position_entry_price # Realized PnL from short
                self.current_position_type = 'long'
                self.position_entry_price = current_price
                self.time_in_position = 0
                self.unrealized_pnl = 0.0
            # If already long, just continue holding (no change in position state, time_in_position updated above)
            
        elif action == 1: # ATTEMPT_SHORT
            if self.current_position_type == 'none':
                self.current_position_type = 'short'
                self.position_entry_price = current_price
                self.time_in_position = 0
                self.unrealized_pnl = 0.0
            elif self.current_position_type == 'long':
                # Close long position first, then attempt short
                # Reward for closing long position
                reward += (current_price - self.position_entry_price) / self.position_entry_price # Realized PnL from long
                self.current_position_type = 'short'
                self.position_entry_price = current_price
                self.time_in_position = 0
                self.unrealized_pnl = 0.0
            # If already short, just continue holding
            
        elif action == 2: # CLOSE_POSITION
            if self.current_position_type == 'long':
                reward += (current_price - self.position_entry_price) / self.position_entry_price # Realized PnL from long
            elif self.current_position_type == 'short':
                reward += (self.position_entry_price - current_price) / self.position_entry_price # Realized PnL from short
            
            self.current_position_type = 'none'
            self.position_entry_price = None
            self.time_in_position = 0
            self.unrealized_pnl = 0.0
            
        elif action == 3: # NO_ACTION
            # If no position, remain none. If holding, continue holding (time_in_position updated above)
            pass # No explicit position change, just pass to next step
        
        # Ensure reward is not None
        reward = float(reward)

        # Move to next step
        self.current_step += 1
        
        # Check if we've reached the end of data
        if self.current_step >= min(len(df) for df in self.mtf_data.values()) - self.seq_len:
            done = True

        obs = self._get_observation()
        info = {
            'current_position_type': self.current_position_type,
            'position_entry_price': self.position_entry_price,
            'time_in_position': self.time_in_position,
            'unrealized_pnl': self.unrealized_pnl,
            'current_price': current_price
        }
        
        return obs, reward, done, info

    def get_observation_space(self):
        """Get observation space information"""
        obs_space = {}
        for tf_name, df in self.mtf_data.items():
            obs_space[tf_name] = (self.seq_len, len(df.columns))
        return obs_space
    
    def get_action_space_size(self):
        """Get size of action space"""
        return 4  # 0: ATTEMPT_LONG, 1: ATTEMPT_SHORT, 2: CLOSE_POSITION, 3: NO_ACTION