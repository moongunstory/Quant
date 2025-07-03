import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Optional, Tuple

class PPOPolicyNetwork(nn.Module):
    def __init__(self, timeframe_dims: Dict[str, int], hidden_dim: int = 128, action_dim: int = 2, create_value_head: bool = True):
        """
        MTF PPO Policy/Value Network

        Args:
            timeframe_dims: 각 타임프레임의 입력 차원 딕셔너리
            hidden_dim: LSTM 및 FC 레이어의 은닉 차원
            action_dim: 행동의 차원 (기본값: 2, 진입/보류)
            create_value_head: 가치망(value head) 생성 여부 (기본값: True)
        """
        super().__init__()
        
        self.timeframe_dims = timeframe_dims
        self.hidden_dim = hidden_dim
        self.timeframes = list(timeframe_dims.keys())
        self.has_value_head = create_value_head

        # 각 타임프레임별 LSTM 또는 피처 프로젝터 생성
        self.lstm_layers = nn.ModuleDict()
        self.feature_projectors = nn.ModuleDict()
        total_feature_dim = 0
        
        for tf_name, input_dim in timeframe_dims.items():
            # 외부 피처 (btc, dune 등)는 간단한 프로젝션 레이어 사용
            if tf_name in ['btc', 'dune']:
                self.feature_projectors[tf_name] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                total_feature_dim += hidden_dim // 4
            # 시계열 데이터는 LSTM 사용
            else:
                self.lstm_layers[tf_name] = nn.LSTM(
                    input_size=input_dim, 
                    hidden_size=hidden_dim, 
                    num_layers=2,
                    batch_first=True,
                    dropout=0.1
                )
                total_feature_dim += hidden_dim
        
        # 모든 피처를 결합하는 레이어
        self.feature_combiner = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # 정책망 (Action 확률 분포)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # 가치망 (선택적으로 생성)
        if self.has_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1)
            )
            self._init_value_head()
        else:
            self.value_head = None

    def _init_value_head(self):
        """가치망 가중치 초기화"""
        if self.value_head is None: return
        for m in self.value_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
                if m.out_features == 1:
                    nn.init.uniform_(m.weight, -0.01, 0.01)

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Returns:
            logits: 정책 로짓 (batch_size, action_dim)
            value: 가치 예측값 (batch_size,) 또는 None
        """
        batch_size = next(iter(x.values())).size(0)
        device = next(iter(x.values())).device
        features = []
        
        # 각 타임프레임 피처 추출
        for tf_name in self.timeframes:
            if tf_name not in x:
                # 누락된 타임프레임은 0으로 채움
                dim = self.hidden_dim // 4 if tf_name in ['btc', 'dune'] else self.hidden_dim
                features.append(torch.zeros(batch_size, dim, device=device))
                continue
            
            tf_input = x[tf_name]
            if tf_name in self.feature_projectors:
                features.append(self.feature_projectors[tf_name](tf_input))
            elif tf_name in self.lstm_layers:
                lstm_out, _ = self.lstm_layers[tf_name](tf_input)
                features.append(lstm_out[:, -1, :])
        
        # 피처 결합 및 처리
        combined_features = torch.cat(features, dim=-1)
        processed_features = self.feature_combiner(combined_features)
        
        # 정책 로짓 생성
        logits = self.policy_head(processed_features)
        
        # 가치 예측 (가치망이 존재할 경우)
        value = self.value_head(processed_features).squeeze(-1) if self.has_value_head else None
        
        return logits, value

    def get_action(self, x: Dict[str, torch.Tensor]):
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, probs

    def evaluate_action(self, x: Dict[str, torch.Tensor], action: torch.Tensor):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"[MODEL SAVE] 모델 저장 완료: {path}")

    def load_model(self, path: str, allow_partial: bool = False):
        state_dict = torch.load(path, map_location='cpu')
        if allow_partial:
            self.load_state_dict(state_dict, strict=False)
            print(f"[MODEL LOAD] 부분 로드 완료: {path}")
        else:
            self.load_state_dict(state_dict)
            print(f"[MODEL LOAD] 전체 로드 완료: {path}")
        self.eval()
    
    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "timeframes": self.timeframes,
            "hidden_dim": self.hidden_dim,
            "has_value_head": self.has_value_head,
            "total_params": total_params,
            "trainable_params": trainable_params
        }
