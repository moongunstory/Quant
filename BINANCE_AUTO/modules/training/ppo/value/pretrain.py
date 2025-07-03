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
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
sys.path.append(PROJECT_ROOT)

# 내부 모듈 임포트
from modules.config import (
    TRAIN_PICKLE_PATHS,
    PPO_IMITATION_MODEL_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH,
    PPO_CONFIG,
)
from modules.training.ppo.core.model import PPOPolicyNetwork
from modules.training.ppo.imitation.train_imitation import (
    prepare_features,
    validate_data,
    align_timeframes,
    calculate_tp_sl_hits_optimized,
)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValuePretrainDataset(Dataset):
    """가치 사전 학습을 위한 데이터셋 (상태, 실제 누적 보상)"""
    def __init__(self, mtf_features: Dict[str, np.ndarray], returns: np.ndarray):
        self.mtf_features = {tf: torch.FloatTensor(features) for tf, features in mtf_features.items()}
        self.returns = torch.FloatTensor(returns)

    def __len__(self) -> int:
        return len(self.returns)

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        features = {tf: data[idx] for tf, data in self.mtf_features.items()}
        return features, self.returns[idx]

def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """주어진 보상 시퀀스로부터 할인된 누적 보상(return)을 계산"""
    returns = np.zeros_like(rewards, dtype=float)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
    returns[t] = running_add
    return returns

def generate_rewards(tp_hit_array: np.ndarray, sl_hit_array: np.ndarray) -> np.ndarray:
    """리워드 생성"""
    rewards = np.where(tp_hit_array, 1.0, np.where(sl_hit_array, -1.0, 0.0))
    
    reward_dist = {
        'positive': (rewards > 0).sum(),
        'negative': (rewards < 0).sum(),
        'neutral': (rewards == 0).sum()
    }
    
    logger.info(f"📊 Reward 분포: +1: {reward_dist['positive']}, -1: {reward_dist['negative']}, 0: {reward_dist['neutral']}")
    logger.info(f"   평균 reward: {rewards.mean():.4f}")
    
    return rewards

def create_pretrain_data(direction: str) -> Tuple[ValuePretrainDataset, ValuePretrainDataset, Dict]:
    """가치 사전 학습을 위한 데이터 준비"""
    logger.info(f"🛠️  [{direction.upper()}] 가치 사전 학습 데이터 준비 시작")
    raw_data = pd.read_pickle(TRAIN_PICKLE_PATHS[direction])

    # 리팩토링된 피처 준비 함수 사용
    features, input_dims, data_len = prepare_features(raw_data)
    
    # 보상 및 누적 보상 계산
    entry_tf, eval_tf = "15min", "5min"
    df_entry = validate_data(raw_data[entry_tf].copy(), f"{direction}-entry")
    df_eval = validate_data(raw_data[eval_tf].copy(), f"{direction}-eval")
    
    df_entry, df_eval = align_timeframes(df_entry, df_eval)
    
    # 데이터 정렬
    df_entry_aligned = df_entry.iloc[:data_len].copy()

    tp_hit, sl_hit = calculate_tp_sl_hits_optimized(df_entry_aligned, df_eval, direction)
    rewards = generate_rewards(tp_hit, sl_hit)
    returns = compute_discounted_returns(rewards, PPO_CONFIG["gamma"])

    # 데이터 분할
    indices = np.arange(data_len)
    train_size = int(0.8 * data_len)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    # 훈련 데이터 기준으로 정규화
    returns_mean, returns_std = returns[train_indices].mean(), returns[train_indices].std() + 1e-8
    normalized_returns = (returns - returns_mean) / returns_std

    train_features = {tf: data[train_indices] for tf, data in features.items()}
    val_features = {tf: data[val_indices] for tf, data in features.items()}

    train_dataset = ValuePretrainDataset(train_features, normalized_returns[train_indices])
    val_dataset = ValuePretrainDataset(val_features, normalized_returns[val_indices])
    
    logger.info(f"   - 데이터 준비 완료: Train {len(train_dataset)}개, Val {len(val_dataset)}개")
    return train_dataset, val_dataset, input_dims

def run_value_pretraining_for(direction: str):
    """지정된 방향에 대해 가치망 사전 학습 실행"""
    logger.info(f"""
{'='*60}
🎯 [{direction.upper()}] 가치망 사전 학습 시작
{'='*60}""")
    
    # 1. 데이터 준비
    train_ds, val_ds, input_dims = create_pretrain_data(direction)
    train_loader = DataLoader(train_ds, batch_size=PPO_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=PPO_CONFIG["batch_size"], shuffle=False)

    # 2. 모델 준비
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 정책망과 가치망을 모두 가진 모델 생성
    model = PPOPolicyNetwork(
        timeframe_dims=input_dims, 
        hidden_dim=PPO_CONFIG["hidden_dim"], 
        action_dim=PPO_CONFIG["action_dim"],
        create_value_head=True
    ).to(device)

    # 3. 모방학습된 정책망 가중치 로드
    imitation_policy_path = PPO_IMITATION_MODEL_PATHS[direction]
    logger.info(f"   - 모방 정책 로드: {imitation_policy_path}")
    # strict=False로 설정하여 가치망 부분은 로드하지 않음
    model.load_state_dict(torch.load(imitation_policy_path, map_location=device), strict=False)

    # 4. 정책망 파라미터 고정
    for name, param in model.named_parameters():
        if 'value_head' not in name:
            param.requires_grad = False
        else:
            logger.info(f"   - 학습 대상 파라미터: {name}")

    # 5. 가치망 학습
    criterion = nn.MSELoss()
    # value_head의 파라미터만 학습하도록 필터링
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=PPO_CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    epochs = PPO_CONFIG.get("pretrain_epochs", 10)  # 사전학습 epoch 설정

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, returns in train_loader:
            features = {tf: data.to(device) for tf, data in features.items()}
            returns = returns.to(device)

            optimizer.zero_grad()
            _, value = model(features)
            loss = criterion(value.squeeze(), returns)
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
                val_loss += criterion(value.squeeze(), returns).item()
        val_loss /= len(val_loader)

        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = VALUE_PRETRAIN_OUTPUT_PATH[direction]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # 가치망의 state_dict만 저장
            torch.save(model.value_head.state_dict(), output_path)
            logger.info(f"   ✅ 새로운 최저 검증 손실 달성. 가치망 저장: {output_path}")

    logger.info(f"""
{'='*60}
✅ [{direction.upper()}] 가치망 사전 학습 완료
{'='*60}""")

def main():
    for direction in ['long', 'short']:
        try:
            run_value_pretraining_for(direction)
        except Exception as e:
            logger.error(f"❌ [{direction.upper()}] 처리 중 심각한 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()
