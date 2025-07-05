import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple

# 프로젝트 루트 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../"))
sys.path.append(PROJECT_ROOT)

# 내부 모듈 임포트
from modules.config import (
    PPO_CONFIG,
)
from modules.training.ppo.core.model import PPOPolicyNetwork

# 로깅 설정 (디버깅용으로 INFO 레벨로 설정)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# --- 더미 데이터 생성 함수 ---
class DummyValuePretrainDataset(Dataset):
    def __init__(self, num_samples: int, input_dims: Dict[str, int], output_dim: int = 1):
        self.num_samples = num_samples
        self.input_dims = input_dims
        self.output_dim = output_dim

        # 더미 데이터 생성 (선형 관계 + 노이즈)
        self.mtf_features = {}
        for tf, dim in input_dims.items():
            self.mtf_features[tf] = np.random.rand(num_samples, dim).astype(np.float32)
        
        # returns를 features의 선형 조합으로 생성
        # 예시: 5min 타임프레임의 첫 번째 피처를 기반으로 returns 생성
        # 실제 데이터의 복잡성을 흉내내기 위해 약간의 노이즈 추가
        base_returns = self.mtf_features["5min"][:, 0] * 0.5 + np.random.randn(num_samples) * 0.1
        self.returns = base_returns.reshape(-1, output_dim).astype(np.float32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        features = {tf: torch.FloatTensor(data[idx]) for tf, data in self.mtf_features.items()}
        features = {}
        for tf, data in self.mtf_features.items():
            if tf in ['btc', 'dune']:
                # 외부 피처는 2D 유지
                features[tf] = torch.FloatTensor(data[idx])
            else:
                # 시계열 피처는 LSTM 입력에 맞게 시퀀스 차원 추가
                features[tf] = torch.FloatTensor(data[idx]).unsqueeze(0)
        return features, torch.tensor(self.returns[idx].item())

def create_dummy_pretrain_data(num_samples: int = 1000) -> Tuple[Dataset, Dataset, Dict]:
    logger.info(f"🛠️  더미 가치 사전 학습 데이터 준비 시작 ({num_samples}개 샘플)")
    
    # 실제 모델의 input_dims와 유사하게 설정 (PPOPolicyNetwork의 timeframe_dims에 맞춰야 함)
    # 실제 config.py나 model.py를 보고 정확한 차원 정보를 가져와야 합니다.
    # 여기서는 예시로 임의의 차원을 사용합니다.
    input_dims = {
        "5min": 64,  # 예시 차원
        "15min": 64,
        "30min": 64,
        "1H": 64,
        "btc": 64,
    }
    
    train_ds = DummyValuePretrainDataset(int(num_samples * 0.8), input_dims)
    val_ds = DummyValuePretrainDataset(int(num_samples * 0.2), input_dims)

    logger.info(f"   - 더미 데이터 준비 완료: Train {len(train_ds)}개, Val {len(val_ds)}개")
    return train_ds, val_ds, input_dims

# --- 가치망 사전 학습 실행 함수 (pretrain.py에서 복사 및 수정) ---
def run_debug_value_pretraining():
    logger.info(f"\n{'='*60}\n🎯 [DEBUG] 가치망 사전 학습 시작\n{'='*60}")
    
    # 1. 더미 데이터 준비
    train_ds, val_ds, input_dims = create_dummy_pretrain_data(num_samples=1000) # 샘플 수 조절
    train_loader = DataLoader(train_ds, batch_size=PPO_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=PPO_CONFIG["batch_size"], shuffle=False)

    # 2. 모델 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPOPolicyNetwork(
        timeframe_dims=input_dims, 
        hidden_dim=PPO_CONFIG["hidden_dim"], 
        action_dim=PPO_CONFIG["action_dim"],
        create_value_head=True
    ).to(device)

    # 3. 모방학습된 정책망 가중치 로드 (디버깅에서는 생략하거나 더미 로드)
    # 실제 로드 로직은 주석 처리하거나, 더미 모델을 로드하도록 변경할 수 있습니다.
    # 여기서는 PPOPolicyNetwork가 초기화될 때 가중치가 랜덤으로 설정되므로,
    # 가치망 학습 테스트를 위해 굳이 모방 정책을 로드할 필요는 없습니다.
    # 만약 로드해야 한다면, 빈 state_dict를 로드하는 등의 처리가 필요합니다.
    # logger.info(f"   - 모방 정책 로드 (디버깅에서는 생략)")

    # 4. 가치망만 학습하도록 설정
    for name, param in model.named_parameters():
        param.requires_grad = False  # 모든 파라미터를 기본적으로 동결
    
    for name, param in model.value_head.named_parameters():
        param.requires_grad = True
        logger.info(f"   - 학습 대상 파라미터 (가치망): {name}")

    # 5. 가치망 학습
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.value_head.parameters(), lr=PPO_CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    epochs = 5 # 디버깅을 위해 에포크 수 감소

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, returns in train_loader:
            features = {tf: data.to(device) for tf, data in features.items()}
            returns = returns.to(device)

            optimizer.zero_grad()
            _, value = model(features)
            logger.debug(f"Value shape: {value.shape}, Returns shape: {returns.shape}")
            loss = criterion(value, returns.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, returns in val_loader:
                features = {tf: data.to(device) for tf, data in features.items()}
                returns = returns.to(device)
                _, value = model(features)
                logger.debug(f"Value shape: {value.shape}, Returns shape: {returns.shape}")
                val_loss += criterion(value, returns).item()
        val_loss /= len(val_loader)

        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        scheduler.step(val_loss)

        # 디버깅에서는 모델 저장 생략
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     logger.info(f"   ✅ 새로운 최저 검증 손실 달성 (디버깅에서는 저장 생략)")

    logger.info(f"\n{'='*60}\n✅ [DEBUG] 가치망 사전 학습 완료\n{'='*60}")

if __name__ == "__main__":
    run_debug_value_pretraining()
