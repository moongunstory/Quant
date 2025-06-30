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
    def __init__(self, timeframe_dims: Dict[str, int]):
        """
        MTF Predictor
        
        Args:
            timeframe_dims: Dictionary mapping timeframe names to their input dimensions
                          Example: {"5min": 16, "15min": 32, "30min": 8, "1H": 6, "btc": 15, "dune": 20}
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timeframe_dims = timeframe_dims

        self.models = {
            'long': PPOPolicyNetwork(timeframe_dims=timeframe_dims, hidden_dim=256).to(self.device),
            'short': PPOPolicyNetwork(timeframe_dims=timeframe_dims, hidden_dim=256).to(self.device)
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

    def predict_policy(self, mtf_state: Dict[str, np.ndarray], direction: str = None) -> Tuple[str, Optional[float], float, float]:
        """
        학습용: 모델이 뽑은 행동과 log_prob (PPO 학습용), ENTER 확률 (로그 출력용) 모두 반환
        
        Args:
            mtf_state: MTF state dictionary
            direction: "long" or "short", if None then compare both
            
        Returns:
            Tuple of (action_str, log_prob, value, prob)
        """
        state_tensors = self._preprocess(mtf_state)
        
        # Debug logging per timeframe
        print(f"[LOG] ▶️ [predict_policy] MTF input:")
        for tf_name, tensor in state_tensors.items():
            print(f"[LOG]   {tf_name}: shape = {tensor.shape}")
            if tensor.numel() > 0:
                print(f"[LOG]   {tf_name}: sample = {tensor.flatten()[:5]}")

        if direction:
            with torch.no_grad():
                action, log_prob, value, probs = self.models[direction].get_action(state_tensors)

            action_str = direction if action.item() == 0 else 'hold'
            prob = float(probs[0, 0].item())
            return action_str, float(log_prob.item()), float(value.item()), prob

        else:
            # 양방향 비교 (확신도 높은 방향 선택, 샘플링은 없음)
            result = {}
            for dir_ in ['long', 'short']:
                with torch.no_grad():
                    _, _, value, probs = self.models[dir_].get_action(state_tensors)
                    prob = float(probs[0, 0].item())
                    result[dir_] = {'prob': prob, 'value': float(value.item())}

            if result['long']['prob'] > result['short']['prob']:
                return 'long', None, result['long']['value'], result['long']['prob']
            else:
                return 'short', None, result['short']['value'], result['short']['prob']

    def predict_filtered(self, mtf_state: Dict[str, np.ndarray], direction: str = None) -> Tuple[str, float, float]:
        """
        실전용: threshold 적용. 확신도 없으면 HOLD 반환
        
        Args:
            mtf_state: MTF state dictionary
            direction: "long" or "short"
            
        Returns:
            Tuple of (action_str, prob, value)
        """
        action, log_prob, value, prob = self.predict_policy(mtf_state, direction)

        threshold = LONG_THRESHOLD if direction == 'long' else SHORT_THRESHOLD
        print(f"[DEBUG] [predict_filtered()] dir = {direction} | prob = {prob:.3f} | threshold = {threshold}")

        if prob >= threshold and action == direction:
            return direction, prob, value
        else:
            return 'hold', prob, value

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