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

    def predict(self, state_series: np.ndarray, direction: str = None):
        state_series = np.asarray(state_series, dtype=np.float32)

        if state_series.ndim == 1:
            state_tensor = torch.tensor(state_series, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif state_series.ndim == 2:
            state_tensor = torch.tensor(state_series, dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError(f"Invalid input shape for state_series: {state_series.shape}")

        state_tensor = state_tensor.to(self.device)

        result = {}
        directions = [direction] if direction else ['long', 'short']

        for dir_ in directions:
            with torch.no_grad():
                _, _, value, probs = self.models[dir_].get_action(state_tensor)
                # Debugging: log probability vector and selected index
                print(f"probs = {probs}")
                print(f"direction = {dir_}, selected prob = {probs[0, 1].item()}")
                prob = float(probs[0, 1].item())  # 확신도 = action=1 (진입) 확률
                result[dir_] = {
                    'prob': prob,
                    'value': float(value.item())
                }

        if direction:
            prob = result[direction]['prob']
            value = result[direction]['value']
            threshold = LONG_THRESHOLD if direction == 'long' else SHORT_THRESHOLD
            if prob >= threshold:
                return direction, prob, value
            else:
                return 'hold', prob, value

        long_prob = result['long']['prob']
        short_prob = result['short']['prob']
        long_value = result['long']['value']
        short_value = result['short']['value']

        long_sig = long_prob >= LONG_THRESHOLD
        short_sig = short_prob >= SHORT_THRESHOLD

        if long_sig and not short_sig:
            return 'long', long_prob, long_value
        elif short_sig and not long_sig:
            return 'short', short_prob, short_value
        elif long_sig and short_sig:
            if long_prob > short_prob:
                return 'long', long_prob, long_value
            else:
                return 'short', short_prob, short_value
        else:
            if long_prob > short_prob:
                return 'hold', long_prob, long_value
            else:
                return 'hold', short_prob, short_value
