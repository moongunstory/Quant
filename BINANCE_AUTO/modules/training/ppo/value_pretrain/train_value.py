import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    PPO_IMITATION_MODEL_PATHS,
    TRAIN_LABEL_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH
)

from modules.training.ppo.core.model import PPOPolicyNetwork
from modules.training.ppo.imitation.train_imitation import generate_flatten_features


class ValuePretrainDataset(Dataset):
    def __init__(self, sequences, rewards):
        self.sequences = torch.FloatTensor(sequences)
        self.rewards = torch.FloatTensor(rewards)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.rewards[idx]


def train_value_network(direction='long', epochs=10, batch_size=1024, lr=3e-4):
    print(f"\n🚀 [VALUE 사전학습 시작] direction={direction.upper()}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print("🔧 시드 고정 완료: 42")

    # 1. 데이터 로드 및 reshape
    df = pd.read_csv(TRAIN_LABEL_PATHS[direction])
    X_df = generate_flatten_features(df, window=32)
    X_raw = X_df.values
    X_raw = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-8)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=1e6, neginf=-1e6)

    num_windows = X_raw.shape[0]
    feature_dim = X_raw.shape[1] // 32
    X_seq = X_raw.reshape(num_windows, 32, feature_dim)

    print(f"📊 데이터 정보: {num_windows}개 윈도우, {feature_dim}개 피처")

    # 2. 모방학습된 PPOPolicyNetwork 로드 (CNN+LSTM)
    model_path = PPO_IMITATION_MODEL_PATHS[direction]
    print(f"📂 PPO 모방학습 모델 로딩: {model_path}")

    model = PPOPolicyNetwork(input_dim=feature_dim, hidden_dim=256, action_dim=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"🔧 사용 디바이스: {device}")

    # 3. 확신도 추출
    dataset_temp = torch.utils.data.TensorDataset(torch.FloatTensor(X_seq))
    loader = DataLoader(dataset_temp, batch_size=1024, shuffle=False)

    softmax = nn.Softmax(dim=1)
    probs_list = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            probs = softmax(logits)
            probs_list.append(probs[:, 1].cpu())

    rewards = torch.cat(probs_list, dim=0).numpy()

    avg_confidence = rewards.mean()
    high_confidence_rate = (rewards > 0.6).sum() / len(rewards)
    print(f"📈 확신도 통계: 평균 확신도={avg_confidence:.4f}, 고확신도(>0.6) 비율={high_confidence_rate:.4f}")

    # 4. PPO 구조 그대로 value_head만 학습
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for param in model.value_head.parameters():
        param.requires_grad = True

    dataset = ValuePretrainDataset(X_seq, rewards)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.value_head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"🎯 value_head 학습 시작 (epochs={epochs})")
    for epoch in range(epochs):
        total_loss = 0
        for sequences, target in dataloader:
            sequences, target = sequences.to(device), target.to(device)
            _, predicted = model(sequences)
            loss = criterion(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # 5. 저장 (value_head만)
    os.makedirs(os.path.dirname(VALUE_PRETRAIN_OUTPUT_PATH[direction]), exist_ok=True)
    torch.save(model.value_head.state_dict(), VALUE_PRETRAIN_OUTPUT_PATH[direction])
    print(f"\n📎 [저장 완료] {VALUE_PRETRAIN_OUTPUT_PATH[direction]}")
    print(f"✅ {direction.upper()} Value Head 사전학습 완료!")



if __name__ == "__main__":
    print("=" * 60)
    print("🎯 PPO 구조 기반 Value Head 사전학습 시작")
    print("=" * 60)

    train_value_network('long')
    train_value_network('short')

    print("\n" + "=" * 60)
    print("🎉 모든 Value 사전학습 완료!")
    print("💡 저장된 value_head는 PPO와 100% 호환됩니다.")
    print("=" * 60)
