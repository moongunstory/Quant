import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
from typing import Dict, Any, Tuple
import random
import logging
import warnings
import pickle
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(PROJECT_ROOT)

# 내부 모듈 임포트
from modules.config import (
    TRAIN_PICKLE_PATHS,
    PPO_IMITATION_MODEL_PATHS,
    SCALER_PATH, # 스케일러 경로 추가
    TIMEFRAMES,
    AUX_TIMEFRAMES,
    PPO_CONFIG,
    IMITATION_CONFIG,
)
from modules.training.ppo.core.model import PPOPolicyNetwork

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field

@dataclass
class ProcessedData:
    features: Dict[str, np.ndarray]
    labels: pd.Series
    input_dims: Dict[str, int]
    final_index: pd.Index

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ✅ 수정: pretrain.py와 동일한 스케일러 적용 함수
def prepare_features(raw_data: Dict[str, pd.DataFrame], seq_len: int) -> ProcessedData:
    logger.info("데이터 전처리 및 스케일링 시작...")
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"스케일러 로드 완료: {SCALER_PATH}")

    all_feature_names = list(scaler.feature_names_in_)
    
    processed_dfs = {}
    for tf, df in raw_data.items():
        if tf == '15min': # 15분봉 데이터는 라벨을 포함하고 있을 수 있음
            label_series = df.get('label')
            feature_df = df.drop(columns=['label'], errors='ignore')
        else:
            feature_df = df
            label_series = None

        for col in all_feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0
        processed_dfs[tf] = feature_df[all_feature_names]
        if label_series is not None:
            processed_dfs['label'] = label_series

    scaled_dfs = {}
    for tf, df in processed_dfs.items():
        if tf == 'label': continue
        scaled_data = scaler.transform(df)
        scaled_dfs[tf] = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        logger.info(f"[{tf}] 스케일링 완료.")

    main_df = scaled_dfs["15min"]
    aligned_dfs = {tf: df.reindex(main_df.index, method='ffill').fillna(0) for tf, df in scaled_dfs.items()}

    sequences = {tf: [] for tf in scaled_dfs.keys()}
    for i in range(len(main_df) - seq_len + 1):
        for tf in scaled_dfs.keys():
            window = aligned_dfs[tf].iloc[i : i + seq_len].values
            sequences[tf].append(window)
    
    final_features = {tf: np.array(arr) for tf, arr in sequences.items()}
    final_index = main_df.index[seq_len - 1:]
    input_dims = {tf: df.shape[1] for tf, df in scaled_dfs.items()}
    
    # 라벨도 최종 인덱스에 맞게 슬라이싱
    final_labels = processed_dfs['label'].reindex(final_index, method='ffill')

    return ProcessedData(final_features, final_labels, input_dims, final_index)

class ImitationDataset(Dataset):
    def __init__(self, mtf_features: Dict[str, np.ndarray], labels: np.ndarray):
        self.mtf_features = {tf: torch.FloatTensor(features_np) for tf, features_np in mtf_features.items()}
        self.labels = torch.LongTensor(labels)
        logger.info(f"📊 Dataset created: Total {len(self.labels)} samples")
        logger.info(f"   Label distribution: {dict(Counter(labels.tolist()))}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        features = {tf: data_tensor[idx] for tf, data_tensor in self.mtf_features.items()}
        return features, self.labels[idx]

def evaluate_policy(model: PPOPolicyNetwork, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, targets in loader:
            features = {tf: data.to(device) for tf, data in features.items()}
            policy_logits, _ = model(features)
            preds = policy_logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    return accuracy, f1

def train_policy_network(model: PPOPolicyNetwork, train_loader: DataLoader, val_loader: DataLoader, 
                         output_path: str) -> Tuple[PPOPolicyNetwork, float]:
    logger.info("🚀 정책망 모방 학습 시작")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"   사용 디바이스: {device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=IMITATION_CONFIG["learning_rate"], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3)

    best_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(IMITATION_CONFIG["epochs"]):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for features, targets in train_loader:
            features = {tf: data.to(device) for tf, data in features.items()}
            targets = targets.to(device)

            optimizer.zero_grad()
            policy_logits, _ = model(features)
            loss = criterion(policy_logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()
            preds = policy_logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
        
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_samples

        val_acc, val_f1 = evaluate_policy(model, val_loader, device)
        scheduler.step(val_f1)

        logger.info(f"Epoch {epoch+1}/{IMITATION_CONFIG['epochs']} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} | "
                    f"Val Acc: {val_acc:.3f}, Val F1: {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), output_path)
            logger.info(f"   🎯 새로운 최고 F1 달성: {val_f1:.3f}. 모델 저장: {output_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= IMITATION_CONFIG['early_stopping_patience']:
                logger.info(f"   조기 종료: {patience_counter} epoch 동안 성능 개선 없음.")
                break
    
    model.load_state_dict(torch.load(output_path))
    return model, best_f1

def run_imitation_learning_for(direction: str):
    logger.info(f'"""
{'='*60}
🎯 [{direction.upper()}] 모방 학습 파이프라인 시작
{'='*60}"""')
    set_seed(42)

    with open(TRAIN_PICKLE_PATHS[direction], 'rb') as f:
        raw_data = pickle.load(f)
    logger.info(f"   - 원본 데이터 로드 완료: {TRAIN_PICKLE_PATHS[direction]}")
    
    processed_data = prepare_features(raw_data, PPO_CONFIG["seq_len"])

    if processed_data.labels is None or processed_data.labels.empty:
        raise ValueError("라벨 데이터를 생성할 수 없습니다.")

    labels_np = processed_data.labels.values.astype(int)
    dataset = ImitationDataset(processed_data.features, labels_np)
    
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=IMITATION_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=IMITATION_CONFIG["batch_size"], shuffle=False)
    logger.info(f"   - 데이터셋 분할 완료: Train {train_size}개, Validation {val_size}개")

    model = PPOPolicyNetwork(
        timeframe_dims=processed_data.input_dims, 
        hidden_dim=PPO_CONFIG["hidden_dim"], 
        action_dim=PPO_CONFIG["action_dim"],
        create_value_head=False 
    )
    logger.info(f"   - 정책망 모델 생성 완료. {model.get_model_info()}")

    output_path = PPO_IMITATION_MODEL_PATHS[direction]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    final_model, best_f1 = train_policy_network(model, train_loader, val_loader, output_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_accuracy, final_f1 = evaluate_policy(final_model, val_loader, device)
    
    logger.info(f'"""
{'='*60}
✅ [{direction.upper()}] 모방 학습 완료
{'='*60}"""')
    logger.info(f"   - 최종 검증 정확도: {final_accuracy:.3f}")
    logger.info(f"   - 최종 검증 F1-Score: {final_f1:.3f} (Best F1: {best_f1:.3f})")
    logger.info(f"   - 저장된 모델 경로: {output_path}")
    logger.info(f'"""{'='*60}
"""')

def main():
    for direction in ['long', 'short']:
        try:
            run_imitation_learning_for(direction)
        except Exception as e:
            logger.error(f"❌ [{direction.upper()}] 처리 중 심각한 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()