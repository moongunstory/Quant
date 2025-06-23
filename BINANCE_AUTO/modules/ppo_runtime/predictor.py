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

        # 모델 로드
        self.models = {
            'long': PPOPolicyNetwork(input_dim=input_dim, hidden_dim=256).to(self.device),
            'short': PPOPolicyNetwork(input_dim=input_dim, hidden_dim=256).to(self.device)
        }

        self.models['long'].load_model(PPO_FINAL_MODEL_PATHS['long'])
        self.models['short'].load_model(PPO_FINAL_MODEL_PATHS['short'])

        for model in self.models.values():
            model.eval()

    def predict(self, state_series: np.ndarray, direction: str = None):
        """
        direction: 'long' 또는 'short' 지정 시 해당 방향만 반환 (main.py와 연동 위해)
        state_series: np.ndarray, shape = (feature_dim,) or (seq_len, feature_dim)
        """
        # ▶ 타입 강제 변환 및 shape 표준화
        state_series = np.asarray(state_series, dtype=np.float32)

        if state_series.ndim == 1:
            # [feature_dim] → [1, 1, feature_dim]
            state_tensor = torch.tensor(state_series, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        elif state_series.ndim == 2:
            # [seq_len, feature_dim] → [1, seq_len, feature_dim]
            state_tensor = torch.tensor(state_series, dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError(f"Invalid input shape for state_series: {state_series.shape}")

        state_tensor = state_tensor.to(self.device)

        result = {}
        directions = [direction] if direction else ['long', 'short']

        for dir_ in directions:
            with torch.no_grad():
                action, log_prob, value = self.models[dir_].get_action(state_tensor)
                result[dir_] = {
                    'action': int(action.item()),
                    'log_prob': float(log_prob.exp().item()),  # ✅ 여기 수정
                    'value': float(value.item())
                }

        if direction:
            action = result[direction]['action']
            log_prob = result[direction]['log_prob']
            value = result[direction]['value']
            threshold = LONG_THRESHOLD if direction == 'long' else SHORT_THRESHOLD

            if action == 1 and log_prob >= threshold:
                return direction, log_prob, value
            else:
                return 'hold', log_prob, value

        # dual mode 판단
        long_sig = result['long']['action'] == 1 and result['long']['log_prob'] >= LONG_THRESHOLD
        short_sig = result['short']['action'] == 1 and result['short']['log_prob'] >= SHORT_THRESHOLD

        if long_sig and not short_sig:
            return 'long', result['long']['log_prob'], result['long']['value']
        elif short_sig and not long_sig:
            return 'short', result['short']['log_prob'], result['short']['value']
        elif long_sig and short_sig:
            if result['long']['log_prob'] > result['short']['log_prob']:
                return 'long', result['long']['log_prob'], result['long']['value']
            else:
                return 'short', result['short']['log_prob'], result['short']['value']
        else:
            if result['long']['log_prob'] > result['short']['log_prob']:
                return 'hold', result['long']['log_prob'], result['long']['value']
            else:
                return 'hold', result['short']['log_prob'], result['short']['value']
