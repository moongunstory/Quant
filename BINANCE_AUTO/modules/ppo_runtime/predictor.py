import torch
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler

from modules.config import (
    PPO_FINAL_MODEL_PATHS,
    SCALER_PATH, # 스케일러 경로 추가
    LONG_THRESHOLD,
    SHORT_THRESHOLD,
    TIMEFRAMES
)
from modules.training.ppo.core.model import PPOPolicyNetwork


class Predictor:
    def __init__(self, timeframe_dims: Dict[str, int]):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timeframe_dims = timeframe_dims

        # ✅ 스케일러 로드
        try:
            with open(SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"[PREDICTOR INIT] 스케일러 로드 완료: {SCALER_PATH}")
        except FileNotFoundError:
            print(f"[ERROR] 스케일러 파일을 찾을 수 없습니다: {SCALER_PATH}")
            self.scaler = None

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
        if not isinstance(mtf_state, dict):
            raise ValueError(f"Expected dict input, got {type(mtf_state)}")

        # ✅ 스케일링 적용
        if self.scaler:
            # DataFrame으로 변환하여 스케일링
            df_dict = {tf: pd.DataFrame(data) for tf, data in mtf_state.items()}
            
            # 모든 피처 컬럼을 가져와서 정렬
            all_feature_names = list(self.scaler.feature_names_in_)
            
            # 모든 DF에 대해 컬럼 맞추기
            aligned_dfs = {}
            for tf, df in df_dict.items():
                for col in all_feature_names:
                    if col not in df.columns:
                        df[col] = 0 # 훈련 시 없던 컬럼은 0으로 채움
                aligned_dfs[tf] = df[all_feature_names]

            # 스케일링
            scaled_dfs = {}
            for tf, df in aligned_dfs.items():
                scaled_data = self.scaler.transform(df)
                scaled_dfs[tf] = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
            
            # 원래 mtf_state 형식(numpy array dict)으로 복원
            mtf_state = {tf: df.values for tf, df in scaled_dfs.items()}
            print("[PREDICTOR] 피처 스케일링 완료.")

        processed_state = {}
        for tf_name, array in mtf_state.items():
            if tf_name not in self.timeframe_dims:
                continue
            
            array = np.asarray(array, dtype=np.float32)
            
            if tf_name in ['btc', 'dune']:
                tensor = torch.tensor(array).unsqueeze(0).to(self.device)
            else:
                tensor = torch.tensor(array).unsqueeze(0).to(self.device)
            
            processed_state[tf_name] = tensor
        
        return processed_state

    def predict_policy(self, mtf_state: Dict[str, np.ndarray], direction: str = None) -> Tuple[str, Optional[float], float, float]:
        state_tensors = self._preprocess(mtf_state)
        
        print(f"[LOG] ▶️ [predict_policy] MTF input:")
        for tf_name, tensor in state_tensors.items():
            print(f"[LOG]   {tf_name}: shape = {tensor.shape}")

        if direction:
            with torch.no_grad():
                action, log_prob, value, probs = self.models[direction].get_action(state_tensors)

            action_str = direction if action.item() == 0 else 'hold'
            prob = float(probs[0, 0].item())
            return action_str, float(log_prob.item()), float(value.item()), prob

        else:
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
        action, log_prob, value, prob = self.predict_policy(mtf_state, direction)

        threshold = LONG_THRESHOLD if direction == 'long' else SHORT_THRESHOLD
        print(f"[DEBUG] [predict_filtered()] dir = {direction} | prob = {prob:.3f} | threshold = {threshold}")

        if prob >= threshold and action == direction:
            return direction, prob, value
        else:
            return 'hold', prob, value

    def validate_input(self, mtf_state: Dict[str, np.ndarray]) -> bool:
        if not isinstance(mtf_state, dict):
            print(f"[ERROR] Expected dict, got {type(mtf_state)}")
            return False
        
        for tf_name, array in mtf_state.items():
            if tf_name not in self.timeframe_dims:
                print(f"[WARNING] Unknown timeframe: {tf_name}")
                continue
                
            expected_dim = self.timeframe_dims[tf_name]
            
            if tf_name in ['btc', 'dune']:
                if len(array) != expected_dim:
                    print(f"[ERROR] {tf_name} dimension mismatch: got {len(array)}, expected {expected_dim}")
                    return False
            else:
                if array.ndim != 2 or array.shape[1] != expected_dim:
                    print(f"[ERROR] {tf_name} shape mismatch: got {array.shape}, expected (seq_len, {expected_dim})")
                    return False
        
        return True

    def get_model_info(self) -> Dict:
        return {
            "device": str(self.device),
            "timeframe_dims": self.timeframe_dims,
            "model_paths": PPO_FINAL_MODEL_PATHS,
            "thresholds": {
                "long": LONG_THRESHOLD,
                "short": SHORT_THRESHOLD
            }
        }