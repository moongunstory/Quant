import torch
import numpy as np
from typing import Dict, Tuple, Union, Optional
from modules.config import (
    PPO_FINAL_MODEL_PATHS,
    LONG_THRESHOLD,
    SHORT_THRESHOLD,
    TIMEFRAMES
)
from modules.training.ppo.core.model import PPOPolicyNetwork


class Predictor:
    def __init__(self, timeframe_dims: Dict[str, int], position_info_dim: int = 5):
        """
        MTF Predictor
        
        Args:
            timeframe_dims: Dictionary mapping timeframe names to their input dimensions
                          Example: {"5min": 16, "15min": 32, "30min": 8, "1H": 6, "btc": 15, "dune": 20}
            position_info_dim: Dimension of position-related features (default to 5)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timeframe_dims = timeframe_dims
        self.position_info_dim = position_info_dim

        self.models = {
            'long': PPOPolicyNetwork(timeframe_dims=timeframe_dims, position_info_dim=self.position_info_dim, hidden_dim=256, action_dim=4).to(self.device),
            'short': PPOPolicyNetwork(timeframe_dims=timeframe_dims, position_info_dim=self.position_info_dim, hidden_dim=256, action_dim=4).to(self.device)
        }

        self.models['long'].load_model(PPO_FINAL_MODEL_PATHS['long'])
        self.models['short'].load_model(PPO_FINAL_MODEL_PATHS['short'])

        for model in self.models.values():
            model.eval()

        print(f"[PREDICTOR INIT] Device: {self.device}")
        print(f"[PREDICTOR INIT] Timeframe dims: {timeframe_dims}")

    def _preprocess(self, mtf_state: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Preprocess MTF state to tensor dictionary
        
        Args:
            mtf_state: Dictionary of timeframe arrays
                      Example: {"5min": array(seq_len, feat_dim), "15min": array(seq_len, feat_dim)}
        
        Returns:
            Dictionary of tensors moved to device
        """
        if not isinstance(mtf_state, dict):
            raise ValueError(f"Expected dict input, got {type(mtf_state)}")
        
        # Validate timeframes
        unknown_timeframes = set(mtf_state.keys()) - set(self.timeframe_dims.keys())
        if unknown_timeframes:
            print(f"[WARNING] Unknown timeframes: {unknown_timeframes}")
        
        processed_state = {}
        
        for tf_name, array in mtf_state.items():
            if tf_name == 'position_info':
                # Position info is a 1D array, convert directly to tensor
                processed_state[tf_name] = torch.tensor(array, dtype=torch.float32).unsqueeze(0).to(self.device)
                continue

            if tf_name not in self.timeframe_dims:
                continue
                
            # Convert to numpy array with proper dtype
            array = np.asarray(array, dtype=np.float32)
            
            # Handle different input shapes
            if tf_name in ['btc', 'dune']:
                # External features - single vector
                if array.ndim == 1:
                    tensor = torch.tensor(array).unsqueeze(0).to(self.device)  # (1, feat_dim)
                elif array.ndim == 2 and array.shape[0] == 1:
                    tensor = torch.tensor(array).to(self.device)  # (1, feat_dim)
                else:
                    raise ValueError(f"Invalid shape for {tf_name}: {array.shape}, expected (feat_dim,) or (1, feat_dim)")
            else:
                # Sequential timeframes
                if array.ndim == 2:
                    tensor = torch.tensor(array).unsqueeze(0).to(self.device)  # (1, seq_len, feat_dim)
                elif array.ndim == 3 and array.shape[0] == 1:
                    tensor = torch.tensor(array).to(self.device)  # (1, seq_len, feat_dim)
                else:
                    raise ValueError(f"Invalid shape for {tf_name}: {array.shape}, expected (seq_len, feat_dim) or (1, seq_len, feat_dim)")
            
            processed_state[tf_name] = tensor
        
        return processed_state

    def predict_policy(self, mtf_state: Dict[str, np.ndarray]) -> Tuple[int, float, float, torch.Tensor]:
        """
        학습용: 모델이 뽑은 행동과 log_prob (PPO 학습용), value, probs 모두 반환
        
        Args:
            mtf_state: MTF state dictionary
            
        Returns:
            Tuple of (action_idx, log_prob, value, probs)
        """
        state_tensors = self._preprocess(mtf_state)
        
        # Debug logging per timeframe
        print(f"[LOG] ▶️ [predict_policy] MTF input:")
        for tf_name, tensor in state_tensors.items():
            print(f"[LOG]   {tf_name}: shape = {tensor.shape}")
            if tensor.numel() > 0:
                print(f"[LOG]   {tf_name}: sample = {tensor.flatten()[:5]}")

        with torch.no_grad():
            action, log_prob, value, probs = self.models['long'].get_action(state_tensors) # Use 'long' model for action prediction

        return action.item(), float(log_prob.item()), float(value.item()), probs.squeeze(0)

    def predict_filtered(self, mtf_state: Dict[str, np.ndarray]) -> Tuple[str, float, float]:
        """
        실전용: threshold 적용. 확신도 없으면 NO_ACTION 반환
        
        Args:
            mtf_state: MTF state dictionary
            
        Returns:
            Tuple of (action_str, prob, value)
        """
        action_idx, log_prob, value, probs = self.predict_policy(mtf_state)

        # Extract probabilities for specific actions
        long_prob = probs[0].item() # ATTEMPT_LONG
        short_prob = probs[1].item() # ATTEMPT_SHORT
        close_prob = probs[2].item() # CLOSE_POSITION
        no_action_prob = probs[3].item() # NO_ACTION

        # Determine the action based on probabilities and thresholds
        final_action_str = 'no_action'
        final_prob = no_action_prob

        # Check for ATTEMPT_LONG
        long_condition = long_prob >= LONG_THRESHOLD
        # Check for ATTEMPT_SHORT
        short_condition = short_prob >= SHORT_THRESHOLD

        if long_condition and short_condition:
            # Both conditions met, choose the one with higher probability
            if long_prob > short_prob:
                final_action_str = 'attempt_long'
                final_prob = long_prob
            else:
                final_action_str = 'attempt_short'
                final_prob = short_prob
        elif long_condition:
            final_action_str = 'attempt_long'
            final_prob = long_prob
        elif short_condition:
            final_action_str = 'attempt_short'
            final_prob = short_prob
        else:
            # If neither long nor short conditions are met, consider CLOSE_POSITION or NO_ACTION
            # Prioritize CLOSE_POSITION if its probability is higher than NO_ACTION
            if close_prob > no_action_prob:
                final_action_str = 'close_position'
                final_prob = close_prob
            else:
                final_action_str = 'no_action'
                final_prob = no_action_prob

        print(f"[DEBUG] [predict_filtered()] long_prob={long_prob:.3f} (thresh={LONG_THRESHOLD}) | short_prob={short_prob:.3f} (thresh={SHORT_THRESHOLD}) | close_prob={close_prob:.3f} | no_action_prob={no_action_prob:.3f} -> Final Action: {final_action_str} (Prob: {final_prob:.3f})")

        return final_action_str, final_prob, value

    def validate_input(self, mtf_state: Dict[str, np.ndarray]) -> bool:
        """
        Validate MTF input format
        
        Args:
            mtf_state: MTF state dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(mtf_state, dict):
            print(f"[ERROR] Expected dict, got {type(mtf_state)}")
            return False
        
        for tf_name, array in mtf_state.items():
            if tf_name not in self.timeframe_dims:
                print(f"[WARNING] Unknown timeframe: {tf_name}")
                continue
                
            expected_dim = self.timeframe_dims[tf_name]
            
            if tf_name in ['btc', 'dune']:
                # External features
                if len(array) != expected_dim:
                    print(f"[ERROR] {tf_name} dimension mismatch: got {len(array)}, expected {expected_dim}")
                    return False
            else:
                # Sequential timeframes
                if array.ndim != 2 or array.shape[1] != expected_dim:
                    print(f"[ERROR] {tf_name} shape mismatch: got {array.shape}, expected (seq_len, {expected_dim})")
                    return False
        
        return True

    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "device": str(self.device),
            "timeframe_dims": self.timeframe_dims,
            "model_paths": PPO_FINAL_MODEL_PATHS,
            "thresholds": {
                "long": LONG_THRESHOLD,
                "short": SHORT_THRESHOLD
            }
        }