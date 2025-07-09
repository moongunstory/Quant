import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from sklearn.metrics import mean_absolute_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 프로젝트 루트 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "BINANCE_AUTO"))

# 내부 모듈 임포트
from modules.config import (
    TRAIN_PICKLE_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH,
    PPO_CONFIG,
)
from modules.training.ppo.core.model import PPOPolicyNetwork
from torch.utils.data import DataLoader, IterableDataset

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 보상 클래스 변환을 위한 임계값
POSITIVE_THRESHOLD = 0.8
NEGATIVE_THRESHOLD = -0.8

def map_reward_to_class(reward: float) -> int:
    if reward >= POSITIVE_THRESHOLD:
        return 2  # Good
    elif reward <= NEGATIVE_THRESHOLD:
        return 0  # Bad
    else:
        return 1  # Neutral


class ValuePretrainIterableDataset(IterableDataset):
    """
    가치 사전 학습을 위한 Iterable 데이터셋.
    메모리 문제를 피하기 위해 데이터를 실시간으로 생성합니다.
    """
    def __init__(self, raw_data: Dict[str, pd.DataFrame], indices: np.ndarray, 
                 class_labels: np.ndarray, input_dims: Dict[str, int], 
                 seq_len: int, direction: str):
        super().__init__()
        self.raw_data = {tf: df.astype(np.float32) for tf, df in raw_data.items()}
        self.indices = indices
        self.class_labels = class_labels
        self.input_dims = input_dims
        self.seq_len = seq_len
        self.direction = direction
        
        self.scalers = {}
        for tf, df in self.raw_data.items():
            if not df.empty:
                scaler = StandardScaler()
                self.raw_data[tf] = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
                self.scalers[tf] = scaler
            else:
                self.scalers[tf] = None # Handle empty dataframe case

        self.ref_df = self.raw_data["1min"]
        self.close_col = next((col for col in self.ref_df.columns if "close" in col.lower()), None)
        self.high_col = next((col for col in self.ref_df.columns if "high" in col.lower()), "high")
        self.low_col = next((col for col in self.ref_df.columns if "low" in col.lower()), "low")
        self.value_horizon_steps = 60

        self.clean_data = {tf: df.dropna() for tf, df in self.raw_data.items()}
        self.df_indexers = {tf: pd.Index(df.index) for tf, df in self.clean_data.items()}

    def __len__(self) -> int:
        return len(self.indices)

    def _get_features_at_idx(self, actual_idx: pd.Timestamp) -> Dict[str, torch.Tensor]:
        features = {}
        for tf, df_clean in self.clean_data.items():
            if df_clean.empty:
                dim = self.input_dims[tf]
                features[tf] = torch.zeros(self.seq_len if tf not in ['btc', 'dune'] else dim, dim, dtype=torch.float32)
                continue

            if tf in ['btc', 'dune']:
                pos = self.df_indexers[tf].get_indexer([actual_idx], method='ffill')[0]
                data_values = df_clean.iloc[pos].values if pos != -1 else np.zeros(len(df_clean.columns), dtype=np.float32)
                features[tf] = torch.from_numpy(data_values)
            else:
                try:
                    end_loc = df_clean.index.get_loc(actual_idx)
                    start_loc = max(0, end_loc - self.seq_len + 1)
                    seq_data = df_clean.iloc[start_loc:end_loc+1].values
                    if len(seq_data) < self.seq_len:
                        padding = np.zeros((self.seq_len - len(seq_data), len(df_clean.columns)), dtype=np.float32)
                        seq_data = np.vstack([padding, seq_data])
                    features[tf] = torch.from_numpy(seq_data)
                except (KeyError, IndexError):
                    features[tf] = torch.zeros(self.seq_len, self.input_dims[tf], dtype=torch.float32)
        return features

    def __iter__(self):
        # 매 에포크마다 훈련 데이터의 순서를 섞음
        shuffled_indices = np.random.permutation(len(self.indices))
        for i in shuffled_indices:
            actual_idx = self.indices[i]
            features = self._get_features_at_idx(actual_idx)
            target_class = torch.tensor(self.class_labels[i], dtype=torch.long)
            yield features, target_class


def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """주어진 보상 시퀀스로부터 할인된 누적 보상(return)을 계산"""
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        returns[t] = running_add
    return returns


def create_pretrain_data(direction: str) -> Tuple[IterableDataset, IterableDataset, Dict]:
    """가치 사전 학습용 데이터 생성 (메모리 효율적인 생성기 방식)"""
    logger.info(f"🛠️  [{direction.upper()}] 가치 사전 학습 데이터 준비 시작 (생성기 방식)")
    
    raw_data = pd.read_pickle(TRAIN_PICKLE_PATHS[direction])
    
    # 임시로 데이터 10%만 샘플링
    SAMPLE_RATIO = 0.1 
    if SAMPLE_RATIO < 1.0:
        logger.info(f"📊 전체 데이터의 {SAMPLE_RATIO*100:.0f}%만 샘플링하여 사용합니다.")
        sampled_raw_data = {}
        for tf, df in raw_data.items():
            # 날짜 인덱스를 유지하며 랜덤 샘플링
            sampled_indices = sorted(np.random.choice(df.index, size=int(len(df) * SAMPLE_RATIO), replace=False))
            sampled_raw_data[tf] = df.loc[sampled_indices]
        raw_data = sampled_raw_data

    ref_df = raw_data["1min"]
    seq_len = PPO_CONFIG["seq_len"]
    valid_indices = ref_df.index[seq_len-1:]
    input_dims = {tf: len(df.columns) for tf, df in raw_data.items()}

    logger.info("📊 전체 데이터에 대한 보상을 미리 계산합니다...")
    close_col = next((col for col in ref_df.columns if "close" in col.lower()), None)
    if not close_col:
        raise ValueError("Could not find 'close' column for return calculation.")

    VALUE_HORIZON_STEPS = 60
    MIN_PROFIT_TARGET = PPO_CONFIG["min_profit_target"]
    MAX_LOSS_TOLERANCE = PPO_CONFIG["max_loss_tolerance"]
    
    rewards_list, final_valid_indices = [], []
    for current_time_idx in valid_indices:
        try:
            entry_price = ref_df.loc[current_time_idx, close_col]
            current_loc = ref_df.index.get_loc(current_time_idx)
            
            next_loc = current_loc + 1 # 1분 뒤 스텝
            if next_loc >= len(ref_df): # 데이터 끝에 도달
                continue # 다음 스텝이 없으면 건너뛰기

            # 미래 데이터 슬라이싱 (VALUE_HORIZON_STEPS + 1분 더 필요)
            future_slice = ref_df.iloc[current_loc + 1 : current_loc + 1 + VALUE_HORIZON_STEPS + 1]
            
            if future_slice.empty or len(future_slice) <= VALUE_HORIZON_STEPS: # +1 for the actual 60th minute
                continue

            tp_level = entry_price * (1 + MIN_PROFIT_TARGET)
            sl_level = entry_price * (1 - MAX_LOSS_TOLERANCE)

            tp_hit = (future_slice['high'] >= tp_level).any()
            sl_hit = (future_slice['low'] <= sl_level).any()

            reward = 0.0 # Default to 0.0

            if tp_hit and not sl_hit:
                reward = 1.0
            elif sl_hit and not tp_hit:
                reward = -1.0
            else:
                # Use price_t_plus_1 as final_price for the continuous reward
                final_price = ref_df.iloc[next_loc][close_col]
                if entry_price == 0: # Prevent division by zero
                    log_return = 0.0
                else:
                    log_return = np.log(final_price / entry_price)
                
                clipped_ret = np.clip(log_return, -0.01, 0.01)
                reward = clipped_ret * 10
            
            # Apply reward clipping as per new strategy
            reward = np.clip(reward, -1.0, 1.0)

            # Filter out meaningless samples
            if abs(reward) > 0.05: # Only include samples where reward is significant
                rewards_list.append(reward)
                final_valid_indices.append(current_time_idx)

        except (KeyError, IndexError): 
            continue
    
    rewards_array = np.array(rewards_list, dtype=np.float32)
    final_valid_indices = np.array(final_valid_indices)
    
    logger.info(f"📊 Reward 분포: +: {(rewards_array > 0).sum()}, -: {(rewards_array < 0).sum()}, 0: {(rewards_array == 0).sum()}")
    logger.info(f"   평균 reward: {rewards_array.mean():.6f}")
    logger.info(f"[DEBUG] rewards mean/std = {rewards_array.mean():.6f} / {rewards_array.std():.6f}")

    train_size = int(0.8 * len(rewards_array))
    train_indices, val_indices = final_valid_indices[:train_size], final_valid_indices[train_size:]
    train_rewards, val_rewards = rewards_array[:train_size], rewards_array[train_size:]

    # 보상을 클래스 레이블로 변환
    train_class_labels = np.array([map_reward_to_class(r) for r in train_rewards], dtype=np.int64)
    val_class_labels = np.array([map_reward_to_class(r) for r in val_rewards], dtype=np.int64)

    train_dataset = ValuePretrainIterableDataset(raw_data, train_indices, train_class_labels, input_dims, seq_len, direction)
    val_dataset = ValuePretrainIterableDataset(raw_data, val_indices, val_class_labels, input_dims, seq_len, direction)

    logger.info(f"✅ 데이터 준비 완료: Train {len(train_dataset)}개, Val {len(val_dataset)}개 (전체 데이터 사용)")
    return train_dataset, val_dataset, input_dims


def run_value_pretraining_for(direction: str):
    """지정된 방향에 대해 가치망 사전 학습 실행"""
    logger.info(f"\n{'='*60}\n🎯 [{direction.upper()}] 가치망 사전 학습 시작\n{'='*60}")
    
    # 1. 데이터 준비
    train_ds, val_ds, input_dims = create_pretrain_data(direction)
    # IterableDataset은 shuffle=False 또는 shuffle 미지정
    train_loader = DataLoader(train_ds, batch_size=PPO_CONFIG["batch_size"], shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=PPO_CONFIG["batch_size"], shuffle=False)

    # 2. 모델 준비 (가치망만 학습)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPOPolicyNetwork(
        timeframe_dims=input_dims, 
        position_info_dim=5,
        hidden_dim=PPO_CONFIG["hidden_dim"], 
        action_dim=PPO_CONFIG["action_dim"],
        create_value_head=True,
        num_value_classes=3 # 3개의 클래스 (bad, neutral, good)
    ).to(device)

    # 가치망 초기화
    for m in model.value_head.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01) # 작은 표준편차로 초기화
            nn.init.constant_(m.bias, 0.0) # 바이어스 0으로 초기화

    # 3. 가치망만 학습하도록 설정
    for name, param in model.named_parameters():
        param.requires_grad = False  # 모든 파라미터 동결
    
    # 가치망과 feature_combiner만 학습 가능하도록 설정
    for name, param in model.value_head.named_parameters():
        param.requires_grad = True
    
    for name, param in model.feature_combiner.named_parameters():
        param.requires_grad = True

    # 4. 학습 설정
    criterion = nn.CrossEntropyLoss() # MSELoss 대신 CrossEntropyLoss 사용
    optimizer = optim.Adam([
        {'params': model.value_head.parameters()}, 
        {'params': model.feature_combiner.parameters()}
    ], lr=PPO_CONFIG["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # 5. 학습 루프
    best_val_loss = float('inf')
    epochs = PPO_CONFIG.get("pretrain_epochs", 10)

    for epoch in range(epochs):
        # 훈련
        model.train()
        train_loss = 0
        for features, returns in train_loader:
            features = {tf: data.to(device) for tf, data in features.items()}
            returns = returns.to(device)

            optimizer.zero_grad()
            _, _, value = model(features)
            loss = criterion(value, returns)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # 검증
        model.eval()
        val_loss = 0
        all_returns, all_preds = [], []
        
        with torch.no_grad():
            for batch_idx, (features, returns) in enumerate(val_loader):
                features = {tf: data.to(device) for tf, data in features.items()}
                returns = returns.to(device)
                _, _, value = model(features)
                
                # Softmax 확률 및 클래스 예측 계산
                probabilities = torch.softmax(value, dim=-1)
                predicted_classes = torch.argmax(probabilities, dim=-1)

                # 🔍 디버깅 로그 추가 - 첫 배치만
                if batch_idx == 0:
                    logger.info(f"[DEBUG] returns[:5] = {returns[:5].cpu().numpy()}")
                    logger.info(f"[DEBUG] value_logits[:5] = {value[:5].cpu().numpy()}")
                    logger.info(f"[DEBUG] probabilities[:5] = {probabilities[:5].cpu().numpy()}")
                    logger.info(f"[DEBUG] predicted_classes[:5] = {predicted_classes[:5].cpu().numpy()}")

                val_loss += criterion(value, returns).item()
                all_returns.extend(returns.cpu().numpy())
                all_preds.extend(predicted_classes.cpu().numpy())

        val_loss /= len(val_loader)
        # 분류 정확도 계산
        val_accuracy = np.mean(np.array(all_returns) == np.array(all_preds))
        val_precision = precision_score(all_returns, all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_returns, all_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(all_returns, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_returns, all_preds)

        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch+1}/{epochs} | "
                   f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                   f"Val Accuracy: {val_accuracy:.6f} | Val Precision: {val_precision:.6f} | "
                   f"Val Recall: {val_recall:.6f} | Val F1: {val_f1:.6f} | LR: {current_lr:.1e}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        scheduler.step(val_loss)

        # 6. 최고 성능 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = VALUE_PRETRAIN_OUTPUT_PATH[direction]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 가치망의 state_dict만 저장
            torch.save(model.value_head.state_dict(), output_path)
            logger.info(f"💾 Best model saved: {output_path}")

    logger.info(f"\n{'='*60}\n✅ [{direction.upper()}] 가치망 사전 학습 완료\n{'='*60}")


def main():
    """메인 실행 함수"""
    for direction in ['long', 'short']:
        try:
            run_value_pretraining_for(direction)
        except Exception as e:
            logger.error(f"❌ [{direction.upper()}] 처리 중 오류 발생: {e}", exc_info=True)


if __name__ == "__main__":
    main()