import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Dict, Optional, Tuple

class PPOPolicyNetwork(nn.Module):
    def __init__(self, timeframe_dims: Dict[str, int], position_info_dim: int, hidden_dim: int = 128, action_dim: int = 1, create_value_head: bool = True, num_value_classes: int = 1):
        """
        MTF PPO Policy/Value Network

        Args:
            timeframe_dims: 각 타임프레임의 입력 차원 딕셔너리
            position_info_dim: 포지션 관련 피처의 입력 차원
            hidden_dim: LSTM 및 FC 레이어의 은닉 차원
            action_dim: 행동의 차원 (기본값: 1, 청산 확신도)
            create_value_head: 가치망(value head) 생성 여부 (기본값: True)
        """
        super().__init__()
        
        self.timeframe_dims = timeframe_dims
        self.hidden_dim = hidden_dim
        self.timeframes = list(timeframe_dims.keys())
        self.has_value_head = create_value_head
        self.action_dim = action_dim # Store action_dim
        self.num_value_classes = num_value_classes # Store num_value_classes

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
        
        # 포지션 정보 처리 레이어 추가
        self.position_info_projector = nn.Sequential(
            nn.Linear(position_info_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        total_feature_dim += hidden_dim // 4

        # 모든 피처를 결합하는 레이어
        self.feature_combiner = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # 정책망 (Action 확률 분포 - 연속형)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2) # Output mean and log_std
        )

        # 가치망 (선택적으로 생성)
        if self.has_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, self.num_value_classes),
                nn.Identity()
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

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: # Returns mean, log_std, value
        """
        Forward pass
        
        Returns:
            mean: 정책 분포의 평균 (batch_size, action_dim)
            log_std: 정책 분포의 로그 표준편차 (batch_size, action_dim)
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
        
        # 포지션 정보 피처 추출 및 추가
        if 'position_info' in x:
            position_info_features = self.position_info_projector(x['position_info'])
            features.append(position_info_features)
        else:
            # position_info가 없는 경우 0으로 채움
            features.append(torch.zeros(batch_size, self.hidden_dim // 4, device=device))

        # 피처 결합 및 처리
        combined_features = torch.cat(features, dim=-1)
        processed_features = self.feature_combiner(combined_features)
        
        # 정책 로짓 생성 (mean and log_std)
        policy_output = self.policy_head(processed_features)
        mean = policy_output[:, :self.action_dim]
        log_std = policy_output[:, self.action_dim:]
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, min=-20, max=2) # Common range for log_std

        # 가치 예측 (가치망이 존재할 경우)
        value = self.value_head(processed_features) if self.has_value_head else None
        
        return mean, log_std, value

    def get_action(self, x: Dict[str, torch.Tensor]):
        mean, log_std, value = self.forward(x)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        action = dist.sample()
        # Clip action to be within [0, 1] for liquidation confidence
        action = torch.clamp(action, 0, 1) 
        
        log_prob = dist.log_prob(action).sum(axis=-1) # Sum log_probs for multi-dim actions
        entropy = dist.entropy().sum(axis=-1) # Sum entropy for multi-dim actions
        
        return action, log_prob, value, mean # Return mean for debugging/analysis

    def evaluate_action(self, x: Dict[str, torch.Tensor], action: torch.Tensor):
        mean, log_std, value = self.forward(x)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        
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