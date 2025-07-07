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
from sklearn.metrics import mean_absolute_error, r2_score

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

# 로깅 설정 (INFO 레벨로 간소화)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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
    epsilon = 1e-6
    reward_dist = {
        'positive': (returns_array > 0).sum(),
        'negative': (returns_array < 0).sum(),
        'neutral': ((returns_array >= -epsilon) & (returns_array <= epsilon)).sum()
    }

    logger.info(f"📊 Reward 분포: +: {reward_dist['positive']}, -: {reward_dist['negative']}, 0: {reward_dist['neutral']}")
    logger.info(f"   평균 reward: {returns_array.mean():.4f}")

    return returns_array

def calculate_horizon_returns(
    df_entry: pd.DataFrame,
    df_eval: pd.DataFrame,
    direction: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    TP/SL 기반 수익률 보상 계산
    """
    returns_array = np.zeros(len(df_entry), dtype=float)
    valid_samples_mask = np.ones(len(df_entry), dtype=bool)

    close_col = next((col for col in df_entry.columns if "close" in col.lower()), None)
    high_col = next((col for col in df_eval.columns if "high" in col.lower()), None)
    low_col = next((col for col in df_eval.columns if "low" in col.lower()), None)

    if not all([close_col, high_col, low_col]):
        raise ValueError("Could not find required 'close', 'high', 'low' columns for return calculation.")

    REWARD_SCALE = 10.0

    def get_no_hit_reward(ret: float, direction: str) -> float:
        neutral_band = PPO_CONFIG.get("neutral_band_ratio", 0.1) # 기본값 0.1
        
        tp_thresh = TP_THRESHOLD
        sl_thresh = abs(SL_THRESHOLD) # SL_THRESHOLD는 음수이므로 절대값 사용

        # 중립 구간 경계 설정
        positive_neutral_thresh = tp_thresh * neutral_band
        negative_neutral_thresh = sl_thresh * neutral_band

        if direction == "long":
            if 0 <= ret < positive_neutral_thresh:
                return 0.0
            if ret >= positive_neutral_thresh:
                # 중립 구간을 제외한 범위에서 0~1로 정규화
                return (ret - positive_neutral_thresh) / (tp_thresh - positive_neutral_thresh)
            if ret < 0:
                # 손실은 SL 임계값으로 정규화 (0 ~ -1 범위)
                return ret / sl_thresh

        elif direction == "short":
            # Short 포지션의 수익률(ret)은 (수익시 양수, 손실시 음수)로 변환되었음
            # 따라서 ret > 0 이면 이익, ret < 0 이면 손실
            if 0 <= ret < positive_neutral_thresh:
                return 0.0
            if ret >= positive_neutral_thresh:
                # 이익 구간 (ret은 양수)
                return (ret - positive_neutral_thresh) / (tp_thresh - positive_neutral_thresh)
            if ret < 0:
                # 손실 구간 (ret은 음수)
                return ret / sl_thresh
        
        return 0.0 # 그 외의 경우는 0

    hit_counts = {'TP': 0, 'SL': 0, 'No Hit': 0}

    for i in range(len(df_entry) - LABEL_HORIZON):
        entry_time = df_entry.index[i]
        entry_price = df_entry.iloc[i][close_col]

        if direction == "long":
            tp_price = entry_price * (1 + TP_THRESHOLD)
            sl_price = entry_price * (1 - abs(SL_THRESHOLD))
        else:
            tp_price = entry_price * (1 - TP_THRESHOLD)
            sl_price = entry_price * (1 + abs(SL_THRESHOLD))

        end_time = df_entry.index[i + LABEL_HORIZON]
        future_eval_data = df_eval[(df_eval.index > entry_time) & (df_eval.index <= end_time)]

        if future_eval_data.empty:
            returns_array[i] = 0.0
            valid_samples_mask[i] = False
            hit_counts['No Hit'] += 1 # No Hit 카운트 증가
            logger.debug(f"[{i}] No Hit (No Future Data).")
            continue

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

        logger.debug(f"[{i}] is_tp_hit: {is_tp_hit}, is_sl_hit: {is_sl_hit}")
        logger.debug(f"[{i}] tp_first_time: {tp_first_time}, sl_first_time: {sl_first_time}")
        # logger.debug(f"[{i}] future_eval_data:\n{future_eval_data.to_string()}") # 너무 많은 로그가 발생할 수 있으므로 필요시 주석 해제

        if is_tp_hit and (not is_sl_hit or tp_first_time <= sl_first_time):
            final_price = df_eval.loc[tp_first_time][close_col]
            hit_counts['TP'] += 1
            logger.debug(f"[{i}] TP Hit. final_price: {final_price:.4f}")
        elif is_sl_hit and (not is_tp_hit or sl_first_time < tp_first_time):
            final_price = df_eval.loc[sl_first_time][close_col]
            hit_counts['SL'] += 1
            logger.debug(f"[{i}] SL Hit. final_price: {final_price:.4f}")
        else:
            final_price = future_eval_data.iloc[-1][close_col]
            hit_counts['No Hit'] += 1
            logger.debug(f"[{i}] No Hit. final_price: {final_price:.4f}")

        # 순수 수익률 계산 (부호 변경 없음)
        ret = (final_price - entry_price) / entry_price
        logger.debug(f"[{i}] Calculated ret: {ret:.4f}")
        
        reward = (
            10.0 if is_tp_hit else
            -1.0 if is_sl_hit else
            get_no_hit_reward(ret, direction) # 순수 ret을 전달
        )
        logger.debug(f"[{i}] Final reward: {reward:.4f}")
        returns_array[i] = reward * REWARD_SCALE

    logger.info(f"   - 보상 집계: TP {hit_counts['TP']}건, SL {hit_counts['SL']}건, No Hit {hit_counts['No Hit']}건 (총 {sum(hit_counts.values())}건)")

    return returns_array, valid_samples_mask, hit_counts

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
    rewards_array, valid_samples_mask, _ = calculate_horizon_returns(df_entry, df_eval, direction)

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
    logger.info(f"\n{'='*60}\n🎯 [{direction.upper()}] 가치망 사전 학습 시작\n{'='*60}")
    
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
        all_returns, all_preds = [], []
        with torch.no_grad():
            for features, returns in val_loader:
                features = {tf: data.to(device) for tf, data in features.items()}
                returns = returns.to(device)
                _, value = model(features)
                val_loss += criterion(value.squeeze(), returns).item()
                all_returns.extend(returns.cpu().numpy())
                all_preds.extend(value.squeeze().cpu().numpy())
        
        val_loss /= len(val_loader)
        val_mae = mean_absolute_error(all_returns, all_preds)
        val_r2 = r2_score(all_returns, all_preds)
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f} | Val R2: {val_r2:.6f} | LR: {current_lr:.1e}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = VALUE_PRETRAIN_OUTPUT_PATH[direction]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # 가치망의 state_dict만 저장
            torch.save(model.value_head.state_dict(), output_path)

    logger.info(f"\n{'='*60}\n✅ [{direction.upper()}] 가치망 사전 학습 완료\n{'='*60}")

def main():
    for direction in ['long', 'short']:
        try:
            run_value_pretraining_for(direction)
        except Exception as e:
            logger.error(f"❌ [{direction.upper()}] 처리 중 심각한 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()