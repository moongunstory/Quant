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
import pickle
from sklearn.preprocessing import StandardScaler

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
    SCALER_PATH, # 스케일러 경로 추가
    PPO_CONFIG,
    TP_THRESHOLD, 
    SL_THRESHOLD, 
    LABEL_HORIZON
)
from modules.training.ppo.core.model import PPOPolicyNetwork
# train_imitation의 prepare_features를 더 이상 사용하지 않음

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ✅ 추가: 스케일러를 적용하는 새로운 prepare_features 함수
class ProcessedData:
    def __init__(self, features, input_dims, final_index, raw_features_for_pretrain):
        self.features = features
        self.input_dims = input_dims
        self.final_index = final_index
        self.raw_features_for_pretrain = raw_features_for_pretrain

def prepare_features(raw_data: Dict[str, pd.DataFrame], seq_len: int) -> ProcessedData:
    logger.info("데이터 전처리 및 스케일링 시작...")
    
    # 스케일러 로드
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"스케일러 로드 완료: {SCALER_PATH}")

    # 모든 타임프레임의 컬럼을 스케일러의 순서에 맞게 정렬
    all_feature_names = list(scaler.feature_names_in_)
    
    processed_dfs = {}
    for tf, df in raw_data.items():
        # 현재 데이터프레임에 없는 스케일러 피처는 0으로 채움
        for col in all_feature_names:
            if col not in df.columns:
                df[col] = 0
        # 스케일러 순서에 맞게 컬럼 정렬
        processed_dfs[tf] = df[all_feature_names]

    # 스케일링 적용
    scaled_dfs = {}
    for tf, df in processed_dfs.items():
        scaled_data = scaler.transform(df)
        scaled_dfs[tf] = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        logger.info(f"[{tf}] 스케일링 완료.")

    # 시퀀스 데이터 생성
    # ... (기존 train_imitation.py의 prepare_features 로직과 유사하게 진행)
    # 여기서는 15min 데이터를 기준으로 시퀀스를 생성합니다.
    main_df = scaled_dfs["15min"]
    
    # 모든 타임프레임의 데이터를 15분봉 인덱스에 맞게 재정렬
    aligned_dfs = {tf: df.reindex(main_df.index, method='ffill').fillna(0) for tf, df in scaled_dfs.items()}

    # Numpy 배열로 변환 및 시퀀스 생성
    sequences = {tf: [] for tf in scaled_dfs.keys()}
    for i in range(len(main_df) - seq_len + 1):
        for tf in scaled_dfs.keys():
            window = aligned_dfs[tf].iloc[i : i + seq_len].values
            sequences[tf].append(window)
    
    final_features = {tf: np.array(arr) for tf, arr in sequences.items()}
    final_index = main_df.index[seq_len - 1:]
    input_dims = {tf: df.shape[1] for tf, df in scaled_dfs.items()}

    return ProcessedData(final_features, input_dims, final_index, raw_data)


class ValuePretrainDataset(Dataset):
    """가치 사전 학습을 위한 데이터셋 (상태, 실제 누적 보상)"""
    def __init__(self, mtf_features: Dict[str, np.ndarray], returns: np.ndarray, indices: np.ndarray, input_dims: Dict[str, int]):
        self.mtf_features = {tf: data for tf, data in mtf_features.items()} # 데이터 복사
        self.returns = returns.copy()
        self.indices = indices.copy()
        self.input_dims = input_dims

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        actual_idx = self.indices[idx]
        
        features = {}
        for tf, data in self.mtf_features.items():
            feature_data = data[actual_idx]
            features[tf] = torch.FloatTensor(feature_data)

        return features, torch.tensor(self.returns[actual_idx].item())

def soft_clip(x: np.ndarray, max_abs: float = 5.0) -> np.ndarray:
    """tanh를 이용해 보상 값을 부드럽게 클리핑"""
    return np.tanh(x / max_abs) * max_abs

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    TP/SL 기반 수익률 보상 계산 및 NoHit 마스크 반환
    """
    seq_len = PPO_CONFIG["seq_len"]
    max_index = len(df_entry) - LABEL_HORIZON - seq_len + 1

    returns_array = np.zeros(len(df_entry), dtype=float)
    valid_samples_mask = np.zeros(len(df_entry), dtype=bool)
    nohit_mask = np.zeros(len(df_entry), dtype=bool) # NoHit 샘플 식별용 마스크

    close_col = next((col for col in df_entry.columns if "close" in col.lower()), None)
    high_col = next((col for col in df_eval.columns if "high" in col.lower()), None)
    low_col = next((col for col in df_eval.columns if "low" in col.lower()), None)

    if not all([close_col, high_col, low_col]):
        raise ValueError("Could not find required 'close', 'high', 'low' columns for return calculation.")

    REWARD_SCALE = 10.0
    hit_counts = {'TP': 0, 'SL': 0, 'No Hit': 0}

    for i in range(max_index):
        reward_index = i + seq_len - 1
        final_index = df_entry.index[reward_index]
        entry_price = df_entry.loc[final_index][close_col]

        if direction == "long":
            tp_price = entry_price * (1 + TP_THRESHOLD)
            sl_price = entry_price * (1 - abs(SL_THRESHOLD))
        else:
            tp_price = entry_price * (1 - TP_THRESHOLD)
            sl_price = entry_price * (1 + abs(SL_THRESHOLD))

        future_start = final_index
        future_end = df_entry.index[reward_index + LABEL_HORIZON]
        future_eval_data = df_eval[(df_eval.index > future_start) & (df_eval.index <= future_end)]

        if future_eval_data.empty:
            returns_array[reward_index] = 0.0
            valid_samples_mask[reward_index] = False
            hit_counts['No Hit'] += 1
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

        is_no_hit = False
        if is_tp_hit and (not is_sl_hit or tp_first_time <= sl_first_time):
            hit_counts['TP'] += 1
            reward = 10.0
        elif is_sl_hit and (not is_tp_hit or sl_first_time < tp_first_time):
            hit_counts['SL'] += 1
            reward = -1.0
        else:
            hit_counts['No Hit'] += 1
            is_no_hit = True
            reward = 0

        reward_scaled = reward * REWARD_SCALE
        returns_array[reward_index] = reward_scaled
        valid_samples_mask[reward_index] = True
        nohit_mask[reward_index] = is_no_hit

    logger.info(f"[DEBUG]  Reward (raw) 분포: mean={np.mean(returns_array[valid_samples_mask]):.4f}, std={np.std(returns_array[valid_samples_mask]):.4f}")
    logger.info(f"   - 보상 집계: TP {hit_counts['TP']}건, SL {hit_counts['SL']}건, No Hit {hit_counts['No Hit']}건 (총 {sum(hit_counts.values())}건)")

    return returns_array, valid_samples_mask, nohit_mask, hit_counts

def create_pretrain_data(direction: str) -> Tuple[ValuePretrainDataset, ValuePretrainDataset, Dict]:
    logger.info(f"🛠️  [{direction.upper()}] 가치 사전 학습 데이터 준비 시작")
    with open(TRAIN_PICKLE_PATHS[direction], 'rb') as f:
        raw_data = pickle.load(f)

    # ✅ 수정: 스케일러를 적용하는 새로운 prepare_features 함수 사용
    processed_data = prepare_features(raw_data, PPO_CONFIG["seq_len"])
    
    features_dict = processed_data.features
    input_dims = processed_data.input_dims
    final_seq_index = processed_data.final_index
    raw_features_df = processed_data.raw_features_for_pretrain

    entry_tf, eval_tf = "15min", "5min"
    df_entry = raw_features_df[entry_tf]
    df_eval = raw_features_df[eval_tf]

    rewards_array, valid_samples_mask, nohit_mask, _ = calculate_horizon_returns(df_entry, df_eval, direction)

    rewards_series = pd.Series(rewards_array, index=df_entry.index)
    mask_series = pd.Series(valid_samples_mask, index=df_entry.index)
    nohit_series = pd.Series(nohit_mask, index=df_entry.index)

    aligned_rewards = rewards_series.reindex(final_seq_index, fill_value=0)
    aligned_mask = mask_series.reindex(final_seq_index, fill_value=False)
    aligned_nohit = nohit_series.reindex(final_seq_index, fill_value=False)

    valid_indices = final_seq_index[aligned_mask]
    tp_sl_indices = valid_indices[~aligned_nohit.loc[valid_indices]]
    nohit_indices = valid_indices[aligned_nohit.loc[valid_indices]]

    if len(tp_sl_indices) > 0:
        target_nohit_len = min(len(nohit_indices), 2 * len(tp_sl_indices))
        sampled_nohit_idx = np.random.choice(nohit_indices, size=target_nohit_len, replace=False)
        final_indices_list = np.concatenate([tp_sl_indices, sampled_nohit_idx])
    else:
        final_indices_list = nohit_indices
        
    final_indices = pd.Index(final_indices_list).sort_values()

    final_mask = final_seq_index.isin(final_indices)

    filtered_features = {tf: data[final_mask] for tf, data in features_dict.items()}
    filtered_rewards = aligned_rewards[final_mask].values
    filtered_data_len = len(filtered_rewards)

    logger.info(f"[DEBUG]  Downsampled 학습 시퀀스 수: {filtered_data_len} / 유효 시퀀스 수: {aligned_mask.sum()}")

    rewards = generate_rewards(filtered_rewards)
    
    logger.info(f"[DEBUG]  Raw Reward 통계: mean={rewards.mean():.4f}, std={rewards.std():.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}")
    rewards = soft_clip(rewards, max_abs=5.0)
    logger.info(f"[DEBUG]  Clipped Reward 통계: mean={rewards.mean():.4f}, std={rewards.std():.4f}, min={rewards.min():.4f}, max={rewards.max():.4f}")
    
    returns = rewards

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
    
    train_ds, val_ds, input_dims = create_pretrain_data(direction)
    train_loader = DataLoader(train_ds, batch_size=PPO_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=PPO_CONFIG["batch_size"], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PPOPolicyNetwork(
        timeframe_dims=input_dims, 
        hidden_dim=PPO_CONFIG["hidden_dim"], 
        action_dim=PPO_CONFIG["action_dim"],
        create_value_head=True
    ).to(device)

    hidden_dim = PPO_CONFIG["hidden_dim"]
    model.value_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.LayerNorm(hidden_dim * 2),
        nn.ReLU(),
        nn.Linear(hidden_dim * 2, 1)
    ).to(device)

    imitation_policy_path = PPO_IMITATION_MODEL_PATHS[direction]
    logger.info(f"   - 모방 정책 로드: {imitation_policy_path}")
    model.load_state_dict(torch.load(imitation_policy_path, map_location=device), strict=False)

    for name, param in model.named_parameters():
        param.requires_grad = False
    
    for name, param in model.value_head.named_parameters():
        param.requires_grad = True
        if "weight" in name and "norm" not in name:
            nn.init.normal_(param, mean=0.0, std=0.1)
        elif "bias" in name:
            nn.init.constant_(param, 0)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.value_head.parameters(), lr=1e-4)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)

    best_val_loss = float('inf')
    epochs = PPO_CONFIG.get("pretrain_epochs", 10)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_debug_limit = 5
        batch_idx = 0

        for features, returns in train_loader:
            features = {tf: data.to(device) for tf, data in features.items()}
            returns = returns.to(device)

            optimizer.zero_grad()
            _, value = model(features)
            loss = criterion(value.squeeze(), returns)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx < batch_debug_limit and (epoch == 0 or (epoch+1) % 2 == 0):
                with torch.no_grad():
                    value_np = value.squeeze().detach().cpu().numpy()
                    logger.info(f"[DEBUG]  Value 예측 분포 (배치): mean={value_np.mean():.4f}, std={value_np.std():.4f}, min={value_np.min():.4f}, max={value_np.max():.4f}")
                
                total_norm = 0.0
                for p in model.value_head.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                logger.info(f"[DEBUG]  Gradient L2 Norm (value_head): {total_norm:.6f}")

            batch_idx += 1
        
        train_loss /= len(train_loader)

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

        if len(all_returns) > 5:
            sample_idx = np.random.randint(0, len(all_returns), size=5)
            for idx in sample_idx:
                logger.info(f"[DEBUG]  예측 비교: GT={all_returns[idx]:.4f}, Pred={all_preds[idx]:.4f}")

        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f} | Val R2: {val_r2:.6f} | LR: {current_lr:.1e}")
        
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            output_path = VALUE_PRETRAIN_OUTPUT_PATH[direction]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
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