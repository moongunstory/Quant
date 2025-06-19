import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging
import joblib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PPOPolicyNetwork(nn.Module):
    """PPO 정책 네트워크 정의 (예시)"""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # action_dim = 2 (hold, long)
        )
    
    def forward(self, x):
        return self.network(x)

class ImitationDataset(Dataset):
    """모방 학습용 데이터셋"""
    def __init__(self, features: np.ndarray, soft_labels: np.ndarray, hard_labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.soft_labels = torch.FloatTensor(soft_labels)
        self.hard_labels = torch.LongTensor(hard_labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.soft_labels[idx], self.hard_labels[idx]

class EnhancedImitationPretrainerLong:
    """향상된 모방 학습 사전 훈련기 - 롱 전용"""
    
    def __init__(
        self,
        bundle_path: str = "data/lgbm_long_bundle.pkl",
        model_save_path: str = "models/ppo_staging/long_imitation.pt",
        sequence_length: int = 32,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        max_steps: int = 5000,
        patience: int = 50,
        kl_weight: float = 0.3,
        distribution_weight: float = 0.2,
        curriculum_ratio: float = 0.7,
        device: Optional[str] = None
    ):
        self.bundle_path = bundle_path
        self.model_save_path = model_save_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.patience = patience
        self.kl_weight = kl_weight
        self.distribution_weight = distribution_weight
        self.curriculum_ratio = curriculum_ratio
        
        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # 모델 저장 디렉토리 생성
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
    
    def load_data(self) -> Tuple[Dict, np.ndarray, List[str]]:
        """단일 모델 및 csv 데이터 로드"""
        # 모델만 저장된 경우 joblib으로 로드
        lgbm_model = joblib.load(self.bundle_path)

        # 대응하는 데이터 경로 추론 또는 고정 지정
        csv_path = "new/data/label/train_long.csv"  # 실제 라벨링된 csv 경로
        df = pd.read_csv(csv_path)

        # 피처 선택 기준
        feature_names = [col for col in df.columns if col not in ["timestamp", "label"]]  # 필요한 경우 수정
        feature_data = df[feature_names].values

        bundle = {
            "model": lgbm_model,
            "features": feature_names,
            "data": df,
            "threshold": 0.6  # 기본값
        }

        logger.info(f"Loaded data shape: {df.shape}")
        logger.info(f"Features: {len(feature_names)}")

        return bundle, feature_data, feature_names
    
    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """시계열 시퀀스 생성"""
        sequences = []
        for i in range(self.sequence_length, len(data)):
            seq = data[i-self.sequence_length:i].flatten()
            sequences.append(seq)
        
        return np.array(sequences)
    
    def generate_labels(self, bundle: Dict, feature_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Expert 모델로부터 soft/hard label 생성 (롱 전용)"""
        lgbm_model = bundle['model']
        threshold = bundle.get('threshold', 0.6)
        
        # 시퀀스 데이터 생성
        sequences = self.create_sequences(feature_data)
        
        # LGBM은 마지막 시점의 feature만 사용
        last_features = sequences 
        
        # Soft labels (확률)
        soft_labels = lgbm_model.predict_proba(last_features)
        
        # Long 전용: 확률 구조를 [hold, long]로 변경
        if soft_labels.shape[1] == 2:
            # 이미 [hold, long] 구조라고 가정
            long_soft_labels = soft_labels.copy()
        
        # Hard labels (threshold 기반) - Long 신호
        long_probs = long_soft_labels[:, 1]  # Long 확률
        hard_labels = (long_probs >= threshold).astype(int)
        
        # Expert signal rate 계산 (Long 신호 비율)
        expert_signal_rate = np.mean(hard_labels)
        
        logger.info(f"Expert long signal rate: {expert_signal_rate:.4f}")
        logger.info(f"Label distribution - Hold: {np.sum(hard_labels == 0)}, Long: {np.sum(hard_labels == 1)}")
        
        return long_soft_labels, hard_labels, expert_signal_rate
    
    def create_curriculum_indices(self, soft_labels: np.ndarray, hard_labels: np.ndarray, step: int) -> np.ndarray:
        """Curriculum learning을 위한 인덱스 생성"""
        # Confidence 계산 (max probability)
        confidences = np.max(soft_labels, axis=1)
        
        # 진행도에 따라 confidence threshold 조정
        progress = min(step / (self.max_steps * 0.5), 1.0)  # 절반 지점까지 진행
        confidence_threshold = 0.9 - 0.3 * progress  # 0.9에서 0.6으로 감소
        
        # 높은 confidence 샘플 선택
        high_conf_indices = np.where(confidences >= confidence_threshold)[0]
        
        # Curriculum ratio에 따라 샘플 수 조정
        n_curriculum = int(len(soft_labels) * self.curriculum_ratio)
        n_random = len(soft_labels) - n_curriculum
        
        # 높은 confidence에서 샘플링
        if len(high_conf_indices) >= n_curriculum:
            curriculum_indices = np.random.choice(high_conf_indices, n_curriculum, replace=False)
        else:
            curriculum_indices = high_conf_indices
            n_random += n_curriculum - len(high_conf_indices)
        
        # 나머지는 무작위 샘플링
        remaining_indices = np.setdiff1d(np.arange(len(soft_labels)), curriculum_indices)
        if len(remaining_indices) >= n_random:
            random_indices = np.random.choice(remaining_indices, n_random, replace=False)
        else:
            random_indices = remaining_indices
        
        selected_indices = np.concatenate([curriculum_indices, random_indices])
        np.random.shuffle(selected_indices)
        
        return selected_indices
    
    def create_stratified_dataloader(
        self, 
        sequences: np.ndarray, 
        soft_labels: np.ndarray, 
        hard_labels: np.ndarray, 
        indices: np.ndarray
    ) -> DataLoader:
        """Stratified sampling을 통한 DataLoader 생성"""
        # 선택된 인덱스의 데이터만 사용
        selected_sequences = sequences[indices]
        selected_soft_labels = soft_labels[indices]
        selected_hard_labels = hard_labels[indices]
        
        # 클래스별 인덱스 분리
        class_0_indices = np.where(selected_hard_labels == 0)[0]  # Hold
        class_1_indices = np.where(selected_hard_labels == 1)[0]  # Long
        
        # 균형 잡힌 배치 생성을 위한 샘플링
        min_class_size = min(len(class_0_indices), len(class_1_indices))
        
        if min_class_size > 0:
            # 각 클래스에서 동일한 수만큼 샘플링
            balanced_indices = []
            n_batches = len(selected_sequences) // self.batch_size + 1
            
            for _ in range(n_batches):
                batch_size_per_class = self.batch_size // 2
                
                if len(class_0_indices) >= batch_size_per_class:
                    batch_0 = np.random.choice(class_0_indices, batch_size_per_class, replace=False)
                else:
                    batch_0 = np.random.choice(class_0_indices, batch_size_per_class, replace=True)
                
                if len(class_1_indices) >= batch_size_per_class:
                    batch_1 = np.random.choice(class_1_indices, batch_size_per_class, replace=False)
                else:
                    batch_1 = np.random.choice(class_1_indices, batch_size_per_class, replace=True)
                
                balanced_indices.extend(batch_0)
                balanced_indices.extend(batch_1)
            
            balanced_indices = np.array(balanced_indices[:len(selected_sequences)])
            np.random.shuffle(balanced_indices)
            
            # 균형 잡힌 데이터셋 생성
            final_sequences = selected_sequences[balanced_indices]
            final_soft_labels = selected_soft_labels[balanced_indices]
            final_hard_labels = selected_hard_labels[balanced_indices]
        else:
            final_sequences = selected_sequences
            final_soft_labels = selected_soft_labels
            final_hard_labels = selected_hard_labels
        
        dataset = ImitationDataset(final_sequences, final_soft_labels, final_hard_labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
    def compute_losses(
        self, 
        logits: torch.Tensor, 
        soft_labels: torch.Tensor, 
        hard_labels: torch.Tensor,
        expert_signal_rate: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """복합 손실 함수 계산 (롱 전용)"""
        # CrossEntropy Loss (hard labels)
        ce_loss = nn.CrossEntropyLoss()(logits, hard_labels)
        
        # KL Divergence Loss (soft labels)
        log_softmax = nn.LogSoftmax(dim=1)(logits)
        kl_loss = nn.KLDivLoss(reduction='batchmean')(log_softmax, soft_labels)
        
        # Distribution Matching Loss (Long 신호 비율 맞추기)
        student_probs = torch.softmax(logits, dim=1)
        student_signal_rate = torch.mean(student_probs[:, 1])  # Long 확률
        target_signal_rate = torch.tensor(expert_signal_rate, device=self.device)
        dist_loss = torch.abs(student_signal_rate - target_signal_rate)
        
        # 총 손실
        total_loss = ce_loss + self.kl_weight * kl_loss + self.distribution_weight * dist_loss
        
        loss_dict = {
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
            'dist_loss': dist_loss.item(),
            'total_loss': total_loss.item(),
            'student_long_rate': student_signal_rate.item()
        }
        
        return total_loss, loss_dict
    
    def validate(
        self, 
        model: nn.Module, 
        val_sequences: np.ndarray, 
        val_soft_labels: np.ndarray, 
        val_hard_labels: np.ndarray
    ) -> Tuple[float, float]:
        """검증"""
        model.eval()
        correct = 0
        total = 0
        signal_predictions = []
        
        with torch.no_grad():
            val_dataset = ImitationDataset(val_sequences, val_soft_labels, val_hard_labels)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            
            for features, _, labels in val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                logits = model(features)
                predictions = torch.argmax(logits, dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                signal_predictions.extend(predictions.cpu().numpy())
        
        accuracy = correct / total
        long_signal_rate = np.mean(signal_predictions)  # Long 신호 비율
        
        return accuracy, long_signal_rate
    
    def run(self) -> Dict[str, float]:
        """모방 학습 실행 (롱 전용)"""
        # 데이터 로드
        bundle, feature_data, feature_names = self.load_data()
        
        # 라벨 생성
        soft_labels, hard_labels, expert_signal_rate = self.generate_labels(bundle, feature_data)
        
        # 시퀀스 생성
        sequences = self.create_sequences(feature_data)
        
        logger.info(f"Sequence shape: {sequences.shape}")
        
        # Train/Validation 분할
        train_indices, val_indices = train_test_split(
            np.arange(len(sequences)), 
            test_size=0.2, 
            stratify=hard_labels,
            random_state=42
        )
        
        train_sequences = sequences[train_indices]
        train_soft_labels = soft_labels[train_indices]
        train_hard_labels = hard_labels[train_indices]
        
        val_sequences = sequences[val_indices]
        val_soft_labels = soft_labels[val_indices]
        val_hard_labels = hard_labels[val_indices]
        
        # 모델 초기화
        input_dim = sequences.shape[1]
        model = PPOPolicyNetwork(input_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # 훈련 변수
        best_val_accuracy = 0.0
        best_model_state = None
        patience_counter = 0
        
        logger.info("Starting long imitation training...")
        
        for step in range(self.max_steps):
            # Curriculum learning 인덱스 생성
            curriculum_indices = self.create_curriculum_indices(
                train_soft_labels, train_hard_labels, step
            )
            
            # Stratified DataLoader 생성
            train_loader = self.create_stratified_dataloader(
                train_sequences, train_soft_labels, train_hard_labels, curriculum_indices
            )
            
            # 훈련
            model.train()
            epoch_losses = []
            
            for features, soft_labels_batch, hard_labels_batch in train_loader:
                features = features.to(self.device)
                soft_labels_batch = soft_labels_batch.to(self.device)
                hard_labels_batch = hard_labels_batch.to(self.device)
                
                optimizer.zero_grad()
                
                logits = model(features)
                loss, loss_dict = self.compute_losses(
                    logits, soft_labels_batch, hard_labels_batch, expert_signal_rate
                )
                
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss_dict)
            
            # 검증 (매 10 step마다)
            if (step + 1) % 10 == 0:
                val_accuracy, val_long_rate = self.validate(
                    model, val_sequences, val_soft_labels, val_hard_labels
                )
                
                signal_rate_diff = abs(val_long_rate - expert_signal_rate)
                
                # 평균 손실 계산
                avg_losses = {}
                for key in epoch_losses[0].keys():
                    avg_losses[key] = np.mean([loss[key] for loss in epoch_losses])
                
                logger.info(
                    f"Step {step+1}/{self.max_steps} | "
                    f"Val Acc: {val_accuracy:.4f} | "
                    f"Long Rate: {val_long_rate:.4f} | "
                    f"Expert Rate: {expert_signal_rate:.4f} | "
                    f"Rate Diff: {signal_rate_diff:.4f} | "
                    f"Total Loss: {avg_losses['total_loss']:.4f}"
                )
                
                # Best 모델 저장
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at step {step+1}")
                    break
        
        # Best 모델 저장
        if best_model_state is not None:
            torch.save(best_model_state, self.model_save_path)
            logger.info(f"Best long model saved to {self.model_save_path}")
        
        # 최종 검증
        model.load_state_dict(best_model_state)
        final_val_accuracy, final_long_rate = self.validate(
            model, val_sequences, val_soft_labels, val_hard_labels
        )
        
        results = {
            'final_val_accuracy': final_val_accuracy,
            'signal_rate_diff': abs(final_long_rate - expert_signal_rate),
            'expert_long_rate': expert_signal_rate,
            'student_long_rate': final_long_rate,
            'total_steps': step + 1,
            'best_val_accuracy': best_val_accuracy
        }
        
        logger.info("Long imitation training completed!")
        logger.info(f"Final results: {results}")
        
        return results

# 사용 예시
if __name__ == "__main__":
    trainer = EnhancedImitationPretrainerLong()
    results = trainer.run()
    print("Long Training Results:", results)