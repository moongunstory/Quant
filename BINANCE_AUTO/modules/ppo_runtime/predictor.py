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
        학습용: 모델이 뽑은 행동과 log_prob (PPO 학습용), ENTER 확률 (로그 출력용) 모두 반환
        """
        state_tensor = self._preprocess(state_series)

        if direction:
            with torch.no_grad():
                action, log_prob, value, probs = self.models[direction].get_action(state_tensor)

            action_str = direction if action.item() == 0 else 'hold'
            prob = float(probs[0, 0].item())
            return action_str, float(log_prob.item()), float(value.item()), prob

        else:
            # 양방향 비교 (확신도 높은 방향 선택, 샘플링은 없음)
            result = {}
            for dir_ in ['long', 'short']:
                with torch.no_grad():
                    _, _, value, probs = self.models[dir_].get_action(state_tensor)
                    prob = float(probs[0, 0].item())
                    result[dir_] = {'prob': prob, 'value': float(value.item())}

            if result['long']['prob'] > result['short']['prob']:
                return 'long', None, result['long']['value'], result['long']['prob']
            else:
                return 'short', None, result['short']['value'], result['short']['prob']

    def predict_filtered(self, state_series: np.ndarray, direction: str = None):
        """
        실전용: threshold 적용. 확신도 없으면 HOLD 반환
        """
        action, log_prob, value, prob = self.predict_policy(state_series, direction)

        threshold = LONG_THRESHOLD if direction == 'long' else SHORT_THRESHOLD
        print(f"[DEBUG] [predict_filtered()] dir = {direction} | prob = {prob:.3f} | threshold = {threshold}")

        if prob >= threshold and action == direction:
            return direction, prob, value
        else:
            return 'hold', prob, value
