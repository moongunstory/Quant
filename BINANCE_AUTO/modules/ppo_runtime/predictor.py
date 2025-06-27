import torch
import numpy as np
from modules.config import (
    PPO_FINAL_MODEL_PATHS,
    LONG_THRESHOLD,
    SHORT_THRESHOLD
)
from modules.training.ppo.core.model import PPOPolicyNetwork


class Predictor:
    def __init__(self, input_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.models = {
            'long': PPOPolicyNetwork(input_dim=input_dim, hidden_dim=256).to(self.device),
            'short': PPOPolicyNetwork(input_dim=input_dim, hidden_dim=256).to(self.device)
        }

        self.models['long'].load_model(PPO_FINAL_MODEL_PATHS['long'])
        self.models['short'].load_model(PPO_FINAL_MODEL_PATHS['short'])

        for model in self.models.values():
            model.eval()

    def _preprocess(self, state_series: np.ndarray) -> torch.Tensor:
        state_series = np.asarray(state_series, dtype=np.float32)
        if state_series.ndim == 1:
            return torch.tensor(state_series).unsqueeze(0).unsqueeze(0).to(self.device)
        elif state_series.ndim == 2:
            return torch.tensor(state_series).unsqueeze(0).to(self.device)
        else:
            raise ValueError(f"Invalid input shape: {state_series.shape}")

    def predict_policy(self, state_series: np.ndarray, direction: str = None):
        """
        학습용: 확신도와 관계없이 policy가 뽑은 action 정보 반환
        """
        state_tensor = self._preprocess(state_series)
        directions = [direction] if direction else ['long', 'short']
        result = {}

        for dir_ in directions:
            with torch.no_grad():
                _, _, value, probs = self.models[dir_].get_action(state_tensor)
                prob = float(probs[0, 0].item())
                result[dir_] = {
                    'prob': prob,
                    'value': float(value.item())
                }

        if direction:
            return direction, result[direction]['prob'], result[direction]['value']
        else:
            # 양방향 비교: 확신도 높은 쪽 선택
            long_prob = result['long']['prob']
            short_prob = result['short']['prob']
            if long_prob > short_prob:
                return 'long', long_prob, result['long']['value']
            else:
                return 'short', short_prob, result['short']['value']

    def predict_filtered(self, state_series: np.ndarray, direction: str = None):
        """
        실전용: threshold 적용. 확신도 없으면 HOLD 반환
        """
        direction, prob, value = self.predict_policy(state_series, direction)

        # 단일 방향
        if direction in ['long', 'short']:
            threshold = LONG_THRESHOLD if direction == 'long' else SHORT_THRESHOLD
            print(f"[DEBUG] [predict_filtered()] dir = {direction} | prob = {prob:.3f} | threshold = {threshold}")
            if prob >= threshold:
                return direction, prob, value
            else:
                return 'hold', prob, value

        # 혹시 None이 들어온 경우 양방향 판단 (양방향 정책 비교 시)
        else:
            # 이 코드는 predict_policy에서 이미 'long' 또는 'short'만 반환함
            raise ValueError("Unexpected fallback in predict_filtered().")
