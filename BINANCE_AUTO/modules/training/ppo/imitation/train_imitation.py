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
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(PROJECT_ROOT)

# 내부 모듈 임포트
from modules.config import (
    TRAIN_PICKLE_PATHS,
    PPO_IMITATION_MODEL_PATHS,
    TIMEFRAMES,
    AUX_TIMEFRAMES,  # 새로 추가
    PPO_CONFIG,
    IMITATION_CONFIG,
    TP_THRESHOLD,
    SL_THRESHOLD,
    LABEL_HORIZON,
)
from modules.training.ppo.core.model import PPOPolicyNetwork

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from dataclasses import dataclass, field

@dataclass
class ProcessedData:
    """가공된 피처/라벨 데이터를 담는 전용 구조체"""
    features: Dict[str, np.ndarray]               # 모델 입력 피처 (3D/2D)
    labels: pd.Series                             # 정답 라벨
    input_dims: Dict[str, int]                    # 타임프레임별 input 차원
    final_index: pd.Index                         # 모든 데이터의 최종 기준 인덱스
    raw_features_for_pretrain: Dict[str, pd.DataFrame] = field(default_factory=dict) # 가치망 학습용 원본 피처
    final_indices: Dict[str, pd.Index] = field(default_factory=dict)

def set_seed(seed=42):
    """재현성을 위한 랜덤 시드 고정""" 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def validate_data(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """데이터 유효성 검사 및 정리"""
    logger.info(f"🔍 {name} 데이터 검증 시작")
    
    # NaN 체크
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
        logger.warning(f"⚠️ {name}에서 NaN 발견: {nan_cols}")
        # OHLC 데이터의 NaN은 forward fill
        ohlc_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlc_cols:
            if col in df.columns and col in nan_cols:
                df[col] = df[col].fillna(method='ffill')
                logger.info(f"   {col} 컬럼 forward fill 완료")
    
    # 시간 인덱스 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # 정렬
    df = df.sort_index()
    
    # 데이터 정보 출력 (비활성화)
    # logger.info(f"   시간 범위: {df.index[0]} ~ {df.index[-1]}")
    # logger.info(f"   샘플 수: {len(df)}")
    # if 'close' in df.columns:
    #     logger.info(f"   가격 범위: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    
    return df

def create_sequential_data(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, pd.Index]:
    """2D DataFrame에서 3D 순차 데이터를 생성합니다."""
    data = df.values
    num_samples = len(data) - seq_len + 1

    # NumPy의 stride_tricks를 사용하면 메모리 복사 없이 효율적으로 3D 배열을 생성할 수 있습니다.
    shape = (num_samples, seq_len, data.shape[1])
    strides = (data.strides[0], data.strides[0], data.strides[1])
    sequential_data = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    # 3D 데이터에 해당하는 시간 인덱스 (각 시퀀스의 마지막 시점)
    sequential_index = df.index[seq_len - 1:]

    return sequential_data, sequential_index

class ImitationDataset(Dataset):
    """모방 학습을 위한 데이터셋 (상태, 정책 라벨)"""
    def __init__(self, mtf_features: Dict[str, np.ndarray], labels: np.ndarray):
            self.mtf_features = {}
            # Convert numpy arrays to torch tensors, handling 2D vs 3D shapes
            for tf, features_np in mtf_features.items():
                self.mtf_features[tf] = torch.FloatTensor(features_np)

            self.labels = torch.LongTensor(labels)

            # Store lengths for on-the-fly padding
            self.timeframe_lengths = {tf: len(data) for tf, data in self.mtf_features.items()}

            # Initialize zero_templates based on the first sample's shape for each timeframe
            self.zero_templates = {}
            for tf, data_tensor in self.mtf_features.items():
                if data_tensor.ndim == 3:  # For OHLCV (seq_len, num_features)
                    self.zero_templates[tf] = torch.zeros_like(data_tensor[0])
                elif data_tensor.ndim == 2:  # For auxiliary (num_features)
                    self.zero_templates[tf] = torch.zeros_like(data_tensor[0])
                else:
                    raise ValueError(f"Unexpected tensor dimension for {tf}: {data_tensor.ndim}")

            logger.info(f"📊 Dataset created: Total {len(self.labels)} samples")
            logger.info(f"   Label distribution: {dict(Counter(labels.tolist()))}")
            logger.info(f"   Timeframe lengths in dataset: {self.timeframe_lengths}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
            features = {}
            for tf, data_tensor in self.mtf_features.items():
                # Design Guideline 4: Handle potential length mismatch with on-the-fly zero padding
                if idx < self.timeframe_lengths[tf]:
                    features[tf] = data_tensor[idx]
                else:
                    features[tf] = self.zero_templates[tf]
            return features, self.labels[idx]

def evaluate_policy(model: PPOPolicyNetwork, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """정책망의 성능(정확도, F1 스코어)을 평가"""
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
    """모방 학습을 통해 정책망을 훈련"""
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
    
    # 가장 성능이 좋았던 모델을 다시 로드
    model.load_state_dict(torch.load(output_path))
    return model, best_f1

def prepare_features(raw_data: Dict[str, pd.DataFrame], master_tf: str = "5min") -> ProcessedData:
    """
    (전문가 모드 리팩토링) 이벤트 기반 비동기 처리 방식으로 피처를 생성합니다.
    - bfill을 완전히 제거하여 미래 정보 유출을 원천 차단합니다.
    - 가장 짧은 master_tf를 기준으로 모든 데이터를 ffill하여 정렬합니다.
    - 데이터 손실을 최소화하고, 실제 매매 환경과 동일한 데이터 시점을 보장합니다.
    """
    logger.info(f"🛠️  Feature preparation started (v4 - Async Event-based). Master Timeframe: {master_tf}")

    # 1. 데이터 역할 정의
    LABEL_COLUMNS = ['label']
    FUTURE_COLUMNS = ['target', 'tp_hit', 'sl_hit', 'next_', 'forward_']

    # 2. 타임프레임별 데이터프레임 준비 및 역할 분리
    mtf_dfs = {"feature": {}, "label": {}}
    for timeframe, df_raw in raw_data.items():
        if timeframe not in TIMEFRAMES:
            continue
        df = validate_data(df_raw.copy(), f"MTF-{timeframe}")
        
        feature_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in LABEL_COLUMNS and not any(k in col.lower() for k in FUTURE_COLUMNS)]
        label_col = next((col for col in df.columns if col in LABEL_COLUMNS), None)

        if feature_cols:
            mtf_dfs["feature"][timeframe] = df[feature_cols]
        if label_col:
            mtf_dfs["label"][timeframe] = df[label_col]

    # 3. 마스터 인덱스 설정 및 데이터 정렬 (ffill only)
    if master_tf not in mtf_dfs["feature"]:
        raise ValueError(f"마스터 타임프레임 '{master_tf}'에 해당하는 데이터가 없습니다.")
    master_index = mtf_dfs["feature"][master_tf].index

    aligned_features = {tf: df.reindex(master_index, method='ffill') for tf, df in mtf_dfs["feature"].items()}
    aligned_labels = {tf: s.reindex(master_index, method='ffill') for tf, s in mtf_dfs["label"].items()}

    # 4. ffill 후에도 남은 초기 NaN 제거를 위한 공통 유효 인덱스 찾기
    valid_index = master_index
    for tf, df in aligned_features.items():
        valid_index = valid_index.intersection(df.dropna().index)
    logger.info(f"초기 NaN 제거 후 유효 데이터 길이: {len(valid_index)}")

    # 5. 유효 인덱스로 모든 데이터 최종 슬라이싱
    final_features = {tf: df.loc[valid_index] for tf, df in aligned_features.items()}
    final_labels_series = {tf: s.loc[valid_index] for tf, s in aligned_labels.items()}

    # 6. 피처 정규화 및 시퀀싱
    mtf_features_array = {}
    input_dims = {}
    processed_data_for_alignment = {}

    for tf, df in final_features.items():
        X_normalized = (df - df.mean()) / (df.std() + 1e-8)
        
        if tf in AUX_TIMEFRAMES:
            processed_data_for_alignment[tf] = pd.DataFrame(X_normalized, index=valid_index)
        else:
            sequential_features, sequential_index = create_sequential_data(X_normalized, PPO_CONFIG["seq_len"])
            seq_df = pd.DataFrame(sequential_features.reshape(len(sequential_features), -1), index=sequential_index)
            processed_data_for_alignment[tf] = seq_df

    # 7. 시퀀싱 후 최종 공통 인덱스 생성
    final_seq_index = valid_index
    for tf, df in processed_data_for_alignment.items():
        final_seq_index = final_seq_index.intersection(df.index)
    final_seq_index = final_seq_index.dropna()
    logger.info(f"시퀀싱 후 최종 데이터 길이: {len(final_seq_index)}")

    # 8. 최종 결과물 생성
    for tf, df in processed_data_for_alignment.items():
        data_final_df = df.loc[final_seq_index]
        feature_values = data_final_df.values
        
        if tf in AUX_TIMEFRAMES:
            mtf_features_array[tf] = feature_values
            input_dims[tf] = feature_values.shape[1]
        else:
            num_features = len(final_features[tf].columns)
            mtf_features_array[tf] = feature_values.reshape(len(data_final_df), PPO_CONFIG["seq_len"], num_features)
            input_dims[tf] = num_features

    # 9. 최종 라벨 및 pretrain용 데이터 준비
    # 모방학습의 라벨은 보통 진입 타임프레임(e.g., 15min)을 따르지만, 여기서는 마스터 타임프레임 기준으로 정렬된 것을 사용
    final_labels = final_labels_series.get("15min").loc[final_seq_index]
    raw_features_for_pretrain = {tf: df.loc[final_seq_index] for tf, df in final_features.items()}

    # 10. 시퀀싱된 시점 인덱스를 포함하는 final_indices 추가
    final_indices = {"15min": final_seq_index}

    return ProcessedData(
        features=mtf_features_array,
        labels=final_labels,
        input_dims=input_dims,
        final_index=final_seq_index,
        raw_features_for_pretrain=raw_features_for_pretrain,
        final_indices=final_indices  # ✅ 이 줄 추가
    )

def run_imitation_learning_for(direction: str):
    """지정된 방향(long/short)에 대한 모방 학습 전체 파이프라인 실행"""
    logger.info(f"""
{'='*60}
🎯 [{direction.upper()}] 모방 학습 파이프라인 시작
{'='*60}""")
    set_seed(42)

    # 1. 데이터 로드 및 새로운 prepare_features 호출
    raw_data = pd.read_pickle(TRAIN_PICKLE_PATHS[direction])
    logger.info(f"   - 원본 데이터 로드 완료: {TRAIN_PICKLE_PATHS[direction]}")
    
    processed_data = prepare_features(raw_data)

    if processed_data.labels is None or processed_data.labels.empty:
        raise ValueError("라벨 데이터를 생성할 수 없습니다. 15min 데이터에 'label' 컬럼이 있는지 확인하세요.")

    # 2. 데이터셋 및 데이터로더 생성
    labels_np = processed_data.labels.values.astype(int)
    dataset = ImitationDataset(processed_data.features, labels_np)
    
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=IMITATION_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=IMITATION_CONFIG["batch_size"], shuffle=False)
    logger.info(f"   - 데이터셋 분할 완료: Train {train_size}개, Validation {val_size}개")

    # 3. 모델 생성
    model = PPOPolicyNetwork(
        timeframe_dims=processed_data.input_dims, 
        hidden_dim=PPO_CONFIG["hidden_dim"], 
        action_dim=PPO_CONFIG["action_dim"],
        create_value_head=False 
    )
    logger.info(f"   - 정책망 모델 생성 완료. {model.get_model_info()}")

    # 4. 모델 훈련
    output_path = PPO_IMITATION_MODEL_PATHS[direction]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    final_model, best_f1 = train_policy_network(model, train_loader, val_loader, output_path)

    # 5. 최종 평가 및 결과 요약
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_accuracy, final_f1 = evaluate_policy(final_model, val_loader, device)
    
    logger.info(f"""
{'='*60}
✅ [{direction.upper()}] 모방 학습 완료
{'='*60}""")
    logger.info(f"   - 최종 검증 정확도: {final_accuracy:.3f}")
    logger.info(f"   - 최종 검증 F1-Score: {final_f1:.3f} (Best F1: {best_f1:.3f})")
    logger.info(f"   - 저장된 모델 경로: {output_path}")
    logger.info(f"""{'='*60}
""")

def main():
    """메인 실행 함수"""
    for direction in ['long', 'short']:
        try:
            run_imitation_learning_for(direction)
        except Exception as e:
            logger.error(f"❌ [{direction.upper()}] 처리 중 심각한 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()