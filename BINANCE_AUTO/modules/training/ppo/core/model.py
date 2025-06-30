import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict
from modules.config import TIMEFRAMES

class PPOPolicyNetwork(nn.Module):
    def __init__(self, timeframe_dims: Dict[str, int], hidden_dim: int = 128, action_dim: int = 2):
        """
        MTF PPO Policy Network
        
        Args:
            timeframe_dims: Dictionary mapping timeframe names to their input dimensions
                          Example: {"5min": 16, "15min": 32, "30min": 8, "1H": 6, "btc": 15, "dune": 20}
            hidden_dim: Hidden dimension for LSTM and fully connected layers
            action_dim: Number of actions (default: 2 for enter/hold)
        """
        super().__init__()
        
        self.timeframe_dims = timeframe_dims
        self.hidden_dim = hidden_dim
        self.timeframes = list(timeframe_dims.keys())
        
        # Create separate LSTM for each timeframe that has sequential data
        self.lstm_layers = nn.ModuleDict()
        self.feature_projectors = nn.ModuleDict()
        
        total_feature_dim = 0
        
        for tf_name, input_dim in timeframe_dims.items():
            if tf_name in ['btc', 'dune']:
                # External features - use simple projection layer
                self.feature_projectors[tf_name] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                total_feature_dim += hidden_dim // 4
            else:
                # Sequential timeframes - use LSTM
                self.lstm_layers[tf_name] = nn.LSTM(
                    input_size=input_dim, 
                    hidden_size=hidden_dim, 
                    batch_first=True,
                    dropout=0.1 if hidden_dim > 64 else 0
                )
                total_feature_dim += hidden_dim
        
        print(f"[MTF NETWORK] Timeframes: {self.timeframes}")
        print(f"[MTF NETWORK] Total feature dim: {total_feature_dim}")
        
        # Combined feature processing
        self.feature_combiner = nn.Sequential(
            nn.Linear(total_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim)
        )
        
        # Policy head (action 확률 분포)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # action_dim = 2 (Enter vs Hold)
        )

        # Value head (LayerNorm 포함)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: Dict[str, torch.Tensor]):
        """
        MTF forward pass
        
        Args:
            x: Dictionary of timeframe tensors
               Sequential TFs: (batch_size, seq_len, input_dim)
               External TFs: (batch_size, input_dim)
        
        Returns:
            logits: (batch_size, action_dim)
            value: (batch_size,)
        """
        batch_size = next(iter(x.values())).size(0)
        features = []
        
        # Process each timeframe
        for tf_name in self.timeframes:
            if tf_name not in x:
                # Handle missing timeframes with zero padding
                if tf_name in ['btc', 'dune']:
                    feature_dim = self.hidden_dim // 4
                else:
                    feature_dim = self.hidden_dim
                zero_feature = torch.zeros(batch_size, feature_dim, device=next(iter(x.values())).device)
                features.append(zero_feature)
                continue
            
            tf_input = x[tf_name]
            
            if tf_name in ['btc', 'dune']:
                # External features - project to smaller dimension
                tf_feature = self.feature_projectors[tf_name](tf_input)
            else:
                # Sequential timeframes - use LSTM
                lstm_out, _ = self.lstm_layers[tf_name](tf_input)
                tf_feature = lstm_out[:, -1, :]  # Take last hidden state
            
            features.append(tf_feature)
        
        # Combine all timeframe features
        combined_features = torch.cat(features, dim=-1)  # (batch_size, total_feature_dim)
        
        # Process combined features
        processed_features = self.feature_combiner(combined_features)  # (batch_size, hidden_dim)
        
        # Generate policy logits and value
        logits = self.policy_head(processed_features)  # (batch_size, action_dim)
        value = self.value_head(processed_features).squeeze(-1)  # (batch_size,)
        
        return logits, value

    def get_action(self, x: Dict[str, torch.Tensor]):
        """
        행동 샘플링: 정책 분포에서 하나 선택 + log_prob + 가치 예측 + 확신도 벡터 반환
        
        Args:
            x: Dictionary of timeframe tensors
            
        Returns:
            action: Sampled action (0: enter, 1: hold)
            log_prob: Log probability of the action
            value: State value prediction
            probs: Action probabilities [enter, hold]
        """
        logits, value = self.forward(x)
        # Reorder probabilities so index 0 corresponds to "enter" and 1 to "hold"
        probs = torch.softmax(logits, dim=-1)[:, [1, 0]]
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, probs

    def evaluate_action(self, x: Dict[str, torch.Tensor], action: torch.Tensor):
        """
        PPO 학습 시: 행동의 log_prob, entropy, value 예측
        
        Args:
            x: Dictionary of timeframe tensors
            action: Actions to evaluate
            
        Returns:
            log_prob: Log probabilities of actions
            entropy: Policy entropy
            value: State value predictions
        """
        logits, value = self.forward(x)
        # Use same [enter, hold] ordering during evaluation
        logits = logits[:, [1, 0]]
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value

    def save_model(self, path: str):
        """Save model state dict"""
        torch.save(self.state_dict(), path)
        print(f"[MODEL SAVE] Saved to {path}")

    def load_model(self, path: str, allow_partial: bool = False):
        """Load model state dict with optional partial loading"""
        state_dict = torch.load(path, map_location='cpu')

        if allow_partial:
            current_state = self.state_dict()
            loaded, skipped = [], []

            for k, v in state_dict.items():
                if k in current_state and current_state[k].shape == v.shape:
                    current_state[k].copy_(v)
                    loaded.append(k)
                else:
                    skipped.append(k)

            self.load_state_dict(current_state)
            print(f"[MODEL LOAD] Partially loaded from {path} ({len(loaded)} tensors)")
            if skipped:
                print(f"[MODEL LOAD WARNING] Skipped {len(skipped)} tensors: {skipped}")
        else:
            self.load_state_dict(state_dict)
            print(f"[MODEL LOAD] Loaded from {path}")

        self.eval()
    
    def get_model_info(self):
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            "timeframes": self.timeframes,
            "timeframe_dims": self.timeframe_dims,
            "hidden_dim": self.hidden_dim,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "lstm_layers": len(self.lstm_layers),
            "feature_projectors": len(self.feature_projectors)
        }
        return info