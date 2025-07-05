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
sys.path.append(os.path.join(PROJECT_ROOT, "BINANCE_AUTO"))

# 내부 모듈 임포트
from modules.config import (
    TRAIN_PICKLE_PATHS,
    PPO_IMITATION_MODEL_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH,
    PPO_CONFIG,
    TP_THRESHOLD, 
    SL_THRESHOLD, 
    LABEL_HORIZON
)
from modules.training.ppo.core.model import PPOPolicyNetwork
from modules.training.ppo.imitation.train_imitation import (
    prepare_features,
    validate_data,
    ProcessedData,
)
DEBUG_LOG_LIMIT = 100 
REWARD_SCALE_FACTOR = 100
# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s') # INFO -> DEBUG
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_counts = {
    'tp_first': 0, 'sl_first': 0, 'only_tp': 0, 'only_sl': 0, 'no_hit': 0, 'no_future_data': 0
}
max_logs = {
    'tp_first': 10, 'sl_first': 10, 'only_tp': 10, 'only_sl': 10, 'no_hit': 10, 'no_future_data': 10
}

class ValuePretrainDataset(Dataset):
    """가치 사전 학습을 위한 데이터셋 (상태, 실제 누적 보상)"""
    def __init__(self, mtf_features: Dict[str, np.ndarray], returns: np.ndarray, indices: np.ndarray, input_dims: Dict[str, int]):
        self.mtf_features = mtf_features  # 원본 데이터 유지
        self.returns = returns
        self.indices = indices  # 사용할 인덱스 목록
        self.input_dims = input_dims

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        actual_idx = self.indices[idx]  # 실제 데이터 인덱스
        
        features = {}
        for tf, data in self.mtf_features.items():
            if actual_idx < len(data):
                if tf in ['btc', 'dune']:
                    # 외부 피처는 2D 유지
                    features[tf] = torch.FloatTensor(data[actual_idx])
                else:
                    # 시계열 피처는 LSTM 입력에 맞게 시퀀스 차원 추가
                    features[tf] = torch.FloatTensor(data[actual_idx])
            else:
                # 데이터가 없는 경우 제로 패딩
                # 이 부분은 PPOPolicyNetwork의 forward에서 처리되므로, 여기서는 단순히 0으로 채움
                # PPOPolicyNetwork의 forward 로직과 일치하도록 수정 필요
                # 현재는 input_dims를 사용하여 0으로 채우지만, 실제 데이터에서는 이런 경우가 없어야 함
                dim = self.input_dims[tf]
                if tf in ['btc', 'dune']:
                    features[tf] = torch.zeros(dim, dtype=torch.float32)
                else:
                    features[tf] = torch.zeros(PPO_CONFIG["seq_len"], dim, dtype=torch.float32) # 시퀀스 차원 유지
        
        return features, torch.tensor(self.returns[actual_idx].item())

def compute_discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """주어진 보상 시퀀스로부터 할인된 누적 보상(return)을 계산"""
    returns = np.zeros_like(rewards, dtype=float)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        returns[t] = running_add
    return returns

def generate_rewards(returns_array: np.ndarray) -> np.ndarray:
    """
    실제 수익률 배열을 받아 보상으로 사용합니다.
    """
    # 여기서는 받은 수익률 배열을 그대로 보상으로 사용합니다.
    # 필요하다면 추가적인 변환 로직을 여기에 추가할 수 있습니다.
    epsilon = 1e-6
    reward_dist = {
        'positive': (returns_array > 0).sum(),
        'negative': (returns_array < 0).sum(),
        'neutral': ((returns_array >= -epsilon) & (returns_array <= epsilon)).sum()
    }

    logger.info(f"📊 Reward 분포: +: {reward_dist['positive']}, -: {reward_dist['negative']}, 0: {reward_dist['neutral']}")
    logger.info(f"   평균 reward: {returns_array.mean():.4f}")

    return returns_array

def calculate_horizon_returns(df_entry: pd.DataFrame, df_eval: pd.DataFrame, direction: str) -> Tuple[np.ndarray, np.ndarray]:
    returns_array = np.zeros(len(df_entry), dtype=float)
    valid_samples_mask = np.ones(len(df_entry), dtype=bool)

    close_col = next((col for col in df_entry.columns if "close" in col.lower()), None)
    high_col = next((col for col in df_eval.columns if "high" in col.lower()), None)
    low_col = next((col for col in df_eval.columns if "low" in col.lower()), None)

    if not all([close_col, high_col, low_col]):
        raise ValueError("Could not find required 'close', 'high', 'low' columns for return calculation.")

    amplification = 30  # tanh 스케일링 계수

    for i in range(len(df_entry) - LABEL_HORIZON):
        entry_time = df_entry.index[i]
        entry_price = df_entry.iloc[i][close_col]
        log_this_sample = (i < DEBUG_LOG_LIMIT)

        if direction == "long":
            tp_price = entry_price * (1 + TP_THRESHOLD)
            sl_price = entry_price * (1 + SL_THRESHOLD)
        elif direction == "short":
            tp_price = entry_price * (1 + SL_THRESHOLD)
            sl_price = entry_price * (1 + TP_THRESHOLD)
        else:
            raise ValueError("Direction must be 'long' or 'short'.")

        end_time = df_entry.index[i + LABEL_HORIZON]
        future_eval_data = df_eval[(df_eval.index > entry_time) & (df_eval.index <= end_time)]

        if future_eval_data.empty:
            returns_array[i] = 0.0
            valid_samples_mask[i] = False
            if log_this_sample:
                logger.debug(f"[{i}] No future_eval_data. Marked as invalid. Entry Time: {entry_time}")
            continue

        # TP/SL 도달 시점 확인
        if direction == "long":
            tp_hit_times = future_eval_data.index[future_eval_data[high_col] >= tp_price]
            sl_hit_times = future_eval_data.index[future_eval_data[low_col] <= sl_price]
        else:
            tp_hit_times = future_eval_data.index[future_eval_data[low_col] <= tp_price]
            sl_hit_times = future_eval_data.index[future_eval_data[high_col] >= sl_price]

        tp_first_time = tp_hit_times.min() if not tp_hit_times.empty else pd.NaT
        sl_first_time = sl_hit_times.min() if not sl_hit_times.empty else pd.NaT

        is_tp_hit = pd.notna(tp_first_time)
        is_sl_hit = pd.notna(sl_first_time)

        # 도달한 시점 기준 최종 평가 가격 결정
        if is_tp_hit and (not is_sl_hit or tp_first_time <= sl_first_time):
            final_price = df_eval.loc[tp_first_time][close_col]
        elif is_sl_hit and (not is_tp_hit or sl_first_time < tp_first_time):
            final_price = df_eval.loc[sl_first_time][close_col]
        else:
            final_price = future_eval_data.iloc[-1][close_col]

        # 수익률 계산
        calculated_return = (final_price - entry_price) / entry_price
        if direction == "short":
            calculated_return *= -1

        # tanh 기반 shaping reward
        reward = float(np.tanh(calculated_return * amplification))
        returns_array[i] = reward

        if log_this_sample:
            logger.debug(
                f"[{i}] Entry: {entry_time}, EntryPrice: {entry_price:.2f}, FinalPrice: {final_price:.2f}, "
                f"Return: {calculated_return:.4f}, Reward: {reward:.4f}, TP: {is_tp_hit}, SL: {is_sl_hit}"
            )

    return returns_array, valid_samples_mask

def create_pretrain_data(direction: str) -> Tuple[ValuePretrainDataset, ValuePretrainDataset, Dict]:
    logger.info(f"🛠️  [{direction.upper()}] 가치 사전 학습 데이터 준비 시작")
    raw_data = pd.read_pickle(TRAIN_PICKLE_PATHS[direction])

    # 1. 리팩토링된 prepare_features 호출 (ProcessedData 객체 반환)
    processed_data = prepare_features(raw_data)
    
    features_dict = processed_data.features
    input_dims = processed_data.input_dims
    final_seq_index = processed_data.final_index # 시퀀싱 후의 최종 인덱스
    raw_features_df = processed_data.raw_features_for_pretrain

    # 2. 보상 계산을 위한 데이터프레임 준비
    entry_tf, eval_tf = "15min", "5min"
    df_entry = raw_features_df[entry_tf]
    df_eval = raw_features_df[eval_tf]

    # 3. 보상 및 유효 샘플 마스크 계산 (시퀀싱 전 인덱스 기준)
    rewards_array, valid_samples_mask = calculate_horizon_returns(df_entry, df_eval, direction)

    # 4. 보상과 마스크를 pandas Series로 변환하여, 최종 인덱스로 정렬 (핵심 수정)
    rewards_series = pd.Series(rewards_array, index=df_entry.index)
    mask_series = pd.Series(valid_samples_mask, index=df_entry.index)

    # 시퀀싱 후의 최종 인덱스(final_seq_index)를 사용하여 완벽하게 정렬
    aligned_rewards = rewards_series.loc[final_seq_index]
    aligned_mask = mask_series.loc[final_seq_index]

    # 5. 정렬된 마스크를 사용하여 모든 피처와 보상 필터링
    filtered_rewards = aligned_rewards[aligned_mask].values
    filtered_data_len = len(filtered_rewards)

    filtered_features = {}
    for tf, data_array in features_dict.items():
        # 이제 features_dict와 aligned_mask는 길이가 동일하므로 직접 필터링 가능
        filtered_features[tf] = data_array[aligned_mask.values]

    # 6. 보상 생성 및 데이터셋 구성
    rewards = generate_rewards(filtered_rewards)
    returns = compute_discounted_returns(rewards, PPO_CONFIG["gamma"])

    indices = np.arange(filtered_data_len)
    train_size = int(0.8 * filtered_data_len)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    returns_mean, returns_std = returns[train_indices].mean(), returns[train_indices].std() + 1e-8
    normalized_returns = (returns - returns_mean) / returns_std

    train_dataset = ValuePretrainDataset(filtered_features, normalized_returns, train_indices, input_dims)
    val_dataset = ValuePretrainDataset(filtered_features, normalized_returns, val_indices, input_dims)

    original_data_len = len(next(iter(raw_data.values())))
    logger.info(f"   - 데이터 준비 완료: Train {len(train_dataset)}개, Val {len(val_dataset)}개 (원본 {original_data_len}개 중 유효 {filtered_data_len}개)")
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

    # 4. 가치망만 학습하도록 설정
    for name, param in model.named_parameters():
        param.requires_grad = False  # 모든 파라미터를 기본적으로 동결
    
    # 가치망(value_head)의 파라미터만 학습 가능하도록 설정
    for name, param in model.value_head.named_parameters():
        param.requires_grad = True
        logger.info(f"   - 학습 대상 파라미터 (가치망): {name}")

    # 5. 가치망 학습
    criterion = nn.MSELoss()
    # value_head의 파라미터만 학습하도록 필터링
    optimizer = optim.Adam(model.value_head.parameters(), lr=PPO_CONFIG["learning_rate"])
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
