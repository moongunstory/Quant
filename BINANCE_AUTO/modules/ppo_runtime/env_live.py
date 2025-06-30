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
        
        return observations

    def step(self, action: str):
        """
        Take a step in the environment
        
        Args:
            action: "long", "short", or "hold"
            
        Returns:
            observation: Dict[str, np.ndarray] - Next observation
            reward: float - Reward for the action
            done: bool - Whether episode is finished
            info: dict - Additional information
        """
        start_idx = self.current_step + 1
        end_idx = start_idx + LABEL_HORIZON
        
        # Use reference timeframe for reward calculation
        ref_df = self.mtf_data[self.reference_timeframe]
        
        if end_idx >= len(ref_df):
            return self._get_observation(), 0, True, {}

        # Get entry price from reference timeframe
        entry_price = ref_df[self.price_cols['close']].iloc[self.current_step]
        
        # Get future data from reference timeframe
        future = ref_df.iloc[start_idx:end_idx]
        highs = future[self.price_cols['high']].values
        lows = future[self.price_cols['low']].values

        tp_hit = False
        sl_hit = False
        reward = 0
        done = False

        if action == 'long':
            for i, (h, l) in enumerate(zip(highs, lows)):
                if (h - entry_price) / entry_price >= TP_THRESHOLD:
                    tp_hit = True
                    break
                if (l - entry_price) / entry_price <= SL_THRESHOLD:
                    sl_hit = True
                    break
            
            reward = 1 if tp_hit else -1 if sl_hit else 0
            done = tp_hit or sl_hit or (i == LABEL_HORIZON - 1)

        elif action == 'short':
            for i, (l, h) in enumerate(zip(lows, highs)):
                if (entry_price - l) / entry_price >= TP_THRESHOLD:
                    tp_hit = True
                    break
                if (entry_price - h) / entry_price <= SL_THRESHOLD:
                    sl_hit = True
                    break
            
            reward = 1 if tp_hit else -1 if sl_hit else 0
            done = tp_hit or sl_hit or (i == LABEL_HORIZON - 1)

        elif action == 'hold':
            reward = 0  # HOLD는 중립적 행동으로 보상 없음
            done = True  # HOLD는 즉시 종료

        else:
            raise ValueError(f"Unknown action: {action}")

        # Move to next step
        self.current_step += 1
        
        # Check if we've reached the end of data
        if self.current_step >= min(len(df) for df in self.mtf_data.values()) - self.seq_len:
            done = True

        obs = self._get_observation()
        info = {
            'tp_hit': tp_hit,
            'sl_hit': sl_hit,
            'entry_price': entry_price,
            'reference_timeframe': self.reference_timeframe
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
        return 3  # long, short, hold