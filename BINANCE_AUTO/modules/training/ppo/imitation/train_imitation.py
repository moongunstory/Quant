import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from typing import Dict, Any, Tuple
import random
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    LGBM_MODEL_PATHS,
    TRAIN_PICKLE_PATHS,
    PPO_IMITATION_MODEL_PATHS,
    TIMEFRAMES,
    WINDOW_SIZE,
    HIDDEN_DIM,
    ACTION_DIM,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    VALUE_LOSS_COEF,  # 새로 추가된 config
    TP_THRESHOLD,     # TP 임계값
    SL_THRESHOLD      # SL 임계값
)
from modules.training.ppo.core.model import PPOPolicyNetwork

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """재현 가능한 결과를 위한 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"시드 고정 완료: {seed}")

def generate_mtf_features(mtf_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """MTF 방식으로 각 타임프레임별 피처 생성 (MTF dict 기반)"""
    logger.info(f"🔄 MTF 피처 생성 시작 - timeframes: {TIMEFRAMES}, window: {WINDOW_SIZE}")

    mtf_features = {}

    for timeframe in TIMEFRAMES:
        if timeframe not in mtf_dict:
            logger.warning(f"⛔ 타임프레임 {timeframe} 데이터 없음 → 건너뜀")
            continue

        df = mtf_dict[timeframe].copy().ffill()

        # 제외 키워드 (라벨/미래 변수 등)
        exclude_keywords = ['label', 'target', 'tp_hit', 'sl_hit', 'next_', 'forward_']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if not any(k in col.lower() for k in exclude_keywords)]

        logger.info(f"[{timeframe}] ▶️ 선택 피처 수: {len(feature_cols)}")

        # 윈도우 기반 flatten 시퀀스 생성
        flatten_dfs = []
        for col in feature_cols:
            for i in range(WINDOW_SIZE):
                shifted = df[col].shift(i + 1)
                flatten_dfs.append(shifted.rename(f"{col}_t-{i+1}"))

        tf_feature_df = pd.concat(flatten_dfs, axis=1).dropna()
        mtf_features[timeframe] = tf_feature_df

        logger.info(f"[{timeframe}] ✅ shape: {tf_feature_df.shape}")

    return mtf_features

def calculate_tp_sl_hits(entry_df: pd.DataFrame, eval_df: pd.DataFrame, direction: str, 
                        reward_horizon: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    TP/SL 히트 상태 계산
    
    Args:
        entry_df: 15min 진입 신호 데이터프레임
        eval_df: 5min 평가용 데이터프레임  
        direction: 'long' 또는 'short'
        reward_horizon: 리워드 평가 기간 (5min 캔들 개수)
    
    Returns:
        tp_hit_array, sl_hit_array: TP/SL 히트 여부 배열
    """
    logger.info(f"🎯 TP/SL 히트 계산 시작 - direction: {direction}, horizon: {reward_horizon}")
    
    # 필수 컬럼 확인
    required_cols = ['close', 'high', 'low']
    for col in required_cols:
        if col not in entry_df.columns:
            raise ValueError(f"Entry 데이터에 '{col}' 컬럼이 없습니다.")
        if col not in eval_df.columns:
            raise ValueError(f"Eval 데이터에 '{col}' 컬럼이 없습니다.")
    
    # datetime 인덱스 확인 및 정렬
    if not isinstance(entry_df.index, pd.DatetimeIndex):
        raise ValueError("Entry 데이터의 인덱스가 DatetimeIndex가 아닙니다.")
    if not isinstance(eval_df.index, pd.DatetimeIndex):
        raise ValueError("Eval 데이터의 인덱스가 DatetimeIndex가 아닙니다.")
    
    entry_df = entry_df.sort_index()
    eval_df = eval_df.sort_index()
    
    tp_hits = []
    sl_hits = []
    tp_count = 0
    sl_count = 0
    neutral_count = 0
    
    logger.info(f"Entry 데이터 범위: {entry_df.index[0]} ~ {entry_df.index[-1]}")
    logger.info(f"Eval 데이터 범위: {eval_df.index[0]} ~ {eval_df.index[-1]}")
    
    for entry_time, entry_row in entry_df.iterrows():
        entry_price = entry_row['close']
        
        # TP/SL 가격 레벨 계산
        if direction == 'long':
            tp_price = entry_price * (1 + TP_THRESHOLD)
            sl_price = entry_price * (1 - SL_THRESHOLD)
        else:  # short
            tp_price = entry_price * (1 - TP_THRESHOLD)
            sl_price = entry_price * (1 + SL_THRESHOLD)
        
        # 진입 시점 이후의 5min 캔들들 찾기
        future_candles = eval_df[eval_df.index > entry_time].head(reward_horizon)
        
        tp_hit = False
        sl_hit = False
        
        # 미래 캔들들을 순차적으로 검사하여 TP/SL 히트 확인
        for eval_time, eval_row in future_candles.iterrows():
            high_price = eval_row['high']
            low_price = eval_row['low']
            
            if direction == 'long':
                # Long 포지션: high가 TP 도달하면 TP 히트
                if high_price >= tp_price:
                    tp_hit = True
                    break
                # low가 SL 도달하면 SL 히트
                elif low_price <= sl_price:
                    sl_hit = True
                    break
            else:  # short
                # Short 포지션: low가 TP 도달하면 TP 히트
                if low_price <= tp_price:
                    tp_hit = True
                    break
                # high가 SL 도달하면 SL 히트
                elif high_price >= sl_price:
                    sl_hit = True
                    break
        
        tp_hits.append(tp_hit)
        sl_hits.append(sl_hit)
        
        # 카운팅
        if tp_hit:
            tp_count += 1
        elif sl_hit:
            sl_count += 1
        else:
            neutral_count += 1
    
    tp_hit_array = np.array(tp_hits)
    sl_hit_array = np.array(sl_hits)
    
    # 통계 로깅
    total_samples = len(tp_hit_array)
    logger.info(f"📊 TP/SL 히트 분포:")
    logger.info(f"  - TP Hit: {tp_count}개 ({tp_count/total_samples*100:.2f}%)")
    logger.info(f"  - SL Hit: {sl_count}개 ({sl_count/total_samples*100:.2f}%)")
    logger.info(f"  - Neutral: {neutral_count}개 ({neutral_count/total_samples*100:.2f}%)")
    logger.info(f"  - TP Threshold: {TP_THRESHOLD:.4f}, SL Threshold: {SL_THRESHOLD:.4f}")
    
    return tp_hit_array, sl_hit_array

def generate_rewards(tp_hit_array: np.ndarray, sl_hit_array: np.ndarray) -> np.ndarray:
    """TP/SL 히트 배열 기반 reward 생성"""
    logger.info("🎯 TP/SL 기반 reward 생성 시작")
    
    rewards = []
    tp_count = 0
    sl_count = 0
    neutral_count = 0
    
    for tp_hit, sl_hit in zip(tp_hit_array, sl_hit_array):
        if tp_hit:
            reward = 1.0
            tp_count += 1
        elif sl_hit:
            reward = -1.0
            sl_count += 1
        else:
            reward = 0.0
            neutral_count += 1
        
        rewards.append(reward)
    
    rewards = np.array(rewards)
    
    # Reward 분포 로깅
    logger.info(f"📊 Reward 분포:")
    logger.info(f"  - TP Hit (reward=1.0): {tp_count}개 ({tp_count/len(rewards)*100:.2f}%)")
    logger.info(f"  - SL Hit (reward=-1.0): {sl_count}개 ({sl_count/len(rewards)*100:.2f}%)")
    logger.info(f"  - Neutral (reward=0.0): {neutral_count}개 ({neutral_count/len(rewards)*100:.2f}%)")
    logger.info(f"  - 평균 Reward: {rewards.mean():.4f}")
    
    return rewards

class MTFTimeSeriesDataset(Dataset):
    """Multi-timeframe 시계열 데이터셋 클래스 (Policy + Value 지원)"""
    
    def __init__(self, mtf_features: Dict[str, np.ndarray], labels: np.ndarray, rewards: np.ndarray):
        self.mtf_features = {tf: torch.FloatTensor(features) for tf, features in mtf_features.items()}
        self.labels = torch.LongTensor(labels)
        self.rewards = torch.FloatTensor(rewards)
        
        logger.info(f"📊 데이터셋 생성 완료:")
        logger.info(f"  - 샘플 수: {len(self.labels)}")
        logger.info(f"  - 타임프레임: {list(self.mtf_features.keys())}")
        logger.info(f"  - 라벨 분포: {Counter(labels.tolist())}")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        features_dict = {tf: features[idx] for tf, features in self.mtf_features.items()}
        return features_dict, self.labels[idx], self.rewards[idx]

def load_lgbm_model(model_path: str):
    """LGBM 모델 로딩"""
    logger.info(f"LGBM 모델 로딩 경로: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LGBM 모델 파일을 찾을 수 없습니다: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"LGBM 모델 로딩 완료")
    return model

def train_imitation_model(model: PPOPolicyNetwork, train_loader: DataLoader, val_loader: DataLoader, 
                         output_model_path: str) -> Tuple[PPOPolicyNetwork, float]:
    """PPO 모방학습 모델 훈련 (Policy + Value Head)"""
    logger.info("PPO 구조 기반 모방학습 훈련 시작 (Policy + Value Head)")
    
    # 손실 함수 정의
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    # 최고 성능 추적을 위한 변수
    best_f1 = 0.0
    best_model_path = output_model_path
    
    device = next(model.parameters()).device
    
    for epoch in range(EPOCHS):
        # 훈련 지표 추적
        total_policy_loss = 0
        total_value_loss = 0
        total_loss = 0
        correct = 0
        total = 0
        
        # 예측값 및 실제 보상 추적 (평균 계산용)
        epoch_predicted_values = []
        epoch_actual_rewards = []
        
        # 훈련 단계
        model.train()
        for batch_idx, (mtf_data, target, reward) in enumerate(train_loader):
            # MTF 데이터를 디바이스로 이동
            mtf_data = {tf: data.to(device) for tf, data in mtf_data.items()}
            target = target.to(device)
            reward = reward.to(device)
            
            optimizer.zero_grad()
            
            # PPOPolicyNetwork는 (policy_logits, value)를 반환
            policy_logits, value = model(mtf_data)
            
            # 손실 계산
            policy_loss = policy_criterion(policy_logits, target)
            value_loss = value_criterion(value.squeeze(), reward)
            total_batch_loss = policy_loss + VALUE_LOSS_COEF * value_loss
            
            total_batch_loss.backward()
            optimizer.step()
            
            # 지표 누적
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += total_batch_loss.item()
            
            # 정확도 계산
            pred = policy_logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # 예측값과 실제 보상 저장
            epoch_predicted_values.extend(value.squeeze().detach().cpu().numpy())
            epoch_actual_rewards.extend(reward.detach().cpu().numpy())
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}, '
                           f'Policy Loss: {policy_loss.item():.4f}, '
                           f'Value Loss: {value_loss.item():.4f}, '
                           f'Total Loss: {total_batch_loss.item():.4f}')
        
        scheduler.step()
        
        # 훈련 지표 계산
        avg_policy_loss = total_policy_loss / len(train_loader)
        avg_value_loss = total_value_loss / len(train_loader)
        avg_total_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        
        # 예측값 vs 실제 보상 평균
        avg_predicted_value = np.mean(epoch_predicted_values)
        avg_actual_reward = np.mean(epoch_actual_rewards)
        
        # 검증 단계
        val_accuracy, val_f1, val_precision, val_recall, val_signal_rate = evaluate_model(model, val_loader)
        
        # 훈련 로그 출력
        logger.info(f'Epoch {epoch+1}/{EPOCHS} 완료 - Train Acc: {train_accuracy:.4f}')
        logger.info(f'🔧 Avg Policy Loss: {avg_policy_loss:.4f} | Avg Value Loss: {avg_value_loss:.4f} | Total Loss: {avg_total_loss:.4f}')
        logger.info(f'📉 Predicted Value (mean): {avg_predicted_value:.4f} | Ground Truth Reward (mean): {avg_actual_reward:.4f}')
        logger.info(f'📊 검증 지표 - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Signal Rate: {val_signal_rate:.4f}')
        
        # 최고 F1 스코어 기준으로 모델 저장
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            logger.info(f'🎯 새로운 최고 F1 스코어! 모델 저장: {val_f1:.4f}')
        
        # 조기 종료 조건
        if val_accuracy < 0.5 and epoch > 2:
            logger.warning("⚠️ 성능 기준 미달로 조기 종료 (Accuracy < 0.5)")
            break
    
    logger.info(f"PPO 구조 기반 모방학습 훈련 완료 (Policy + Value Head) - 최고 F1: {best_f1:.4f}")
    return model, best_f1

def evaluate_model(model: PPOPolicyNetwork, test_loader: DataLoader) -> Tuple[float, float, float, float, float]:
    """PPO 모델 평가 (MTF 지원)"""
    model.eval()
    predictions = []
    targets = []
    
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for mtf_data, target, reward in test_loader:
            # MTF 데이터를 디바이스로 이동
            mtf_data = {tf: data.to(device) for tf, data in mtf_data.items()}
            
            # PPOPolicyNetwork에서 policy logits만 사용 (평가는 정책만)
            policy_logits, _ = model(mtf_data)
            pred = policy_logits.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    # 다양한 평가 지표 계산
    accuracy = accuracy_score(targets, predictions)
    
    # 클래스가 모두 존재하는지 확인 후 지표 계산
    unique_targets = set(targets)
    unique_preds = set(predictions)
    
    if len(unique_targets) > 1 and len(unique_preds) > 1:
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    else:
        logger.warning("⚠️ 단일 클래스 예측 감지 - F1, Precision, Recall을 0으로 설정")
        f1 = precision = recall = 0.0
    
    signal_rate = np.mean(np.array(predictions) == 1)
    
    return accuracy, f1, precision, recall, signal_rate

def train_model_for_direction(direction: str) -> Dict[str, Any]:
    """특정 방향(long/short)에 대한 PPO 모델 훈련 (Policy + Value Head)"""
    logger.info(f"=== {direction.upper()} PPO 모방학습 모델 훈련 시작 (Policy + Value Head) ===")
    
    # 시드 고정
    set_seed(42)
    
    # 파일 경로 설정
    lgbm_model_path = LGBM_MODEL_PATHS[direction]
    train_data_path = TRAIN_PICKLE_PATHS[direction]
    output_model_path = PPO_IMITATION_MODEL_PATHS[direction]

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    # 2. 훈련 데이터 로딩
    logger.info(f"훈련 데이터 로딩: {train_data_path}")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"훈련 데이터 파일을 찾을 수 없습니다: {train_data_path}")
        
    # MTF dict 로딩
    raw = pd.read_pickle(train_data_path)  # Dict[str, pd.DataFrame] 구조
    entry_tf = "15min"  # 진입 신호용 타임프레임
    eval_tf = "5min"    # TP/SL 평가용 타임프레임

    if entry_tf not in raw:
        raise ValueError(f"'{entry_tf}' timeframe이 데이터에 없습니다.")
    if eval_tf not in raw:
        raise ValueError(f"'{eval_tf}' timeframe이 데이터에 없습니다.")

    df_entry = raw[entry_tf]  # 라벨 추출용 및 진입 신호용
    df_eval = raw[eval_tf]    # TP/SL 평가용

    # 3. MTF 방식으로 피처 생성
    mtf_features_df = generate_mtf_features(raw)  # full MTF dict 입력

    # 4. 라벨 및 리워드 생성
    if "label" not in df_entry.columns:
        raise ValueError(f"{entry_tf} 데이터에 'label' 컬럼이 없습니다.")

    min_len = min([features.shape[0] for features in mtf_features_df.values()])
    labels = df_entry["label"].values[-min_len:].astype(int)

    # TP/SL 히트 계산 (내부 계산)
    logger.info(f"🔍 TP/SL 히트 계산 - Entry: {entry_tf}, Eval: {eval_tf}")
    
    # 길이 맞춤을 위해 entry 데이터도 잘라냄
    df_entry_aligned = df_entry.iloc[-min_len:].copy()
    
    tp_hit_array, sl_hit_array = calculate_tp_sl_hits(
        entry_df=df_entry_aligned,
        eval_df=df_eval,
        direction=direction,
        reward_horizon=100  # 5min 캔들 100개 = 약 8.3시간
    )

    rewards = generate_rewards(tp_hit_array, sl_hit_array)

    # 5. 각 타임프레임별로 정규화 및 시계열 복원
    mtf_features_array = {}
    input_dims = {}

    for timeframe, features_df in mtf_features_df.items():
        
        # 정규화 적용
        X_values = features_df.values
        X_values = (X_values - X_values.mean(axis=0)) / (X_values.std(axis=0) + 1e-8)
        X_values = np.nan_to_num(X_values, nan=0.0, posinf=1e6, neginf=-1e6)

        # 시계열 복원
        num_windows = features_df.shape[0]
        num_features = int(features_df.shape[1] / WINDOW_SIZE)
        X_reshaped = X_values.reshape(num_windows, WINDOW_SIZE, num_features)
        
        mtf_features_array[timeframe] = X_reshaped
        input_dims[timeframe] = num_features
        
        logger.info(f"[🔍 {timeframe} 시계열 shape] X: {X_reshaped.shape}")

    # 6. 라벨 및 리워드 분포 확인
    label_distribution = Counter(labels)
    logger.info(f"📊 라벨 분포: {dict(label_distribution)}")
    
    # 라벨 불균형 경고
    if len(label_distribution) == 1:
        logger.warning("⚠️ 심각한 라벨 불균형 감지: 단일 클래스만 존재!")
    elif label_distribution.get(1, 0) / len(labels) < 0.05:
        logger.warning("⚠️ 라벨 불균형 경고: Signal(label=1) 비율이 5% 미만입니다.")

    # 7. MTF 데이터셋 생성 (Policy + Value)
    dataset = MTFTimeSeriesDataset(mtf_features_array, labels, rewards)
    
    # 훈련/검증 분할 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    logger.info(f"데이터셋 준비 완료 - 훈련: {train_size}, 검증: {val_size}")
    
    # 8. PPOPolicyNetwork 모델 초기화 (MTF 지원)
    model = PPOPolicyNetwork(
        timeframe_dims=input_dims,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM
    )
    logger.info(f"PPOPolicyNetwork 초기화 완료 - input_dims: {input_dims}, hidden_dim: {HIDDEN_DIM}, action_dim: {ACTION_DIM}")
    logger.info(f"Value Loss Coefficient: {VALUE_LOSS_COEF}")

    # GPU 사용 가능 시 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"사용 디바이스: {device}")
    
    # 9. 모델 훈련
    model, best_f1 = train_imitation_model(model, train_loader, val_loader, output_model_path)
    
    # 10. 최종 모델 평가
    logger.info("최종 PPO 모델 평가 시작")
    accuracy, f1, precision, recall, signal_rate = evaluate_model(model, val_loader)
    
    # 11. PPO 구조 모델 저장 확인
    logger.info(f"PPO 구조 기반 모방학습 모델 저장 완료: {output_model_path}")
    logger.info(f"저장된 모델은 PPO 본 학습과 100% 동일한 구조입니다 (Policy + Value Head 모두 훈련)")
    
    # 12. 최종 성능 로그 출력
    logger.info(f"\n=== {direction.upper()} PPO 모방학습 모델 훈련 완료 (Policy + Value Head) ===")
    logger.info(f"📊 최종 평가 지표:")
    logger.info(f"  - Accuracy: {accuracy:.4f}")
    logger.info(f"  - F1 Score: {f1:.4f} (Best: {best_f1:.4f})")
    logger.info(f"  - Precision: {precision:.4f}")
    logger.info(f"  - Recall: {recall:.4f}")
    logger.info(f"  - Signal Rate (label=1 예측 비율): {signal_rate:.4f}")
    logger.info(f"저장된 파일 경로: {output_model_path}")
    logger.info(f"모델 구조: PPOPolicyNetwork (Policy + Value Head 모두 훈련)")
    logger.info("=" * 50)
    
    return {
        'direction': direction,
        'accuracy': accuracy,
        'f1_score': f1,
        'best_f1_score': best_f1,
        'precision': precision,
        'recall': recall,
        'signal_rate': signal_rate,
        'model_path': output_model_path,
        'input_dims': input_dims
    }

def main():
    """메인 실행 함수"""
    logger.info("PPO 구조 기반 모방학습 훈련 시작 (Policy + Value Head)")
    logger.info("=" * 60)
    
    # 전역 시드 고정
    set_seed(42)
    
    results = []
    
    # Long 모델 훈련
    try:
        long_result = train_model_for_direction('long')
        results.append(long_result)
    except Exception as e:
        logger.error(f"Long 모델 훈련 중 오류 발생: {str(e)}")
        raise
    
    # Short 모델 훈련  
    try:
        short_result = train_model_for_direction('short')
        results.append(short_result)
    except Exception as e:
        logger.error(f"Short 모델 훈련 중 오류 발생: {str(e)}")
        raise
    
    # 전체 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("전체 PPO 모방학습 훈련 결과 요약 (Policy + Value Head)")
    logger.info("=" * 60)
    
    for result in results:
        logger.info(f"\n{result['direction'].upper()} PPO 모방학습 모델:")
        logger.info(f"  📊 평가 지표:")
        logger.info(f"    - Accuracy: {result['accuracy']:.4f}")
        logger.info(f"    - F1 Score: {result['f1_score']:.4f} (Best: {result['best_f1_score']:.4f})")
        logger.info(f"    - Precision: {result['precision']:.4f}")
        logger.info(f"    - Recall: {result['recall']:.4f}")
        logger.info(f"    - Signal Rate: {result['signal_rate']:.4f}")
        logger.info(f"  📁 저장 경로: {result['model_path']}")
        logger.info(f"  🏗️ 모델 구조: PPOPolicyNetwork (Policy + Value Head)")
        logger.info(f"  📐 입력 차원: {result['input_dims']}")
    
    logger.info(f"\n🎉 모든 PPO 구조 기반 모방학습 훈련이 완료되었습니다! (Policy + Value Head)")
    logger.info(f"💡 저장된 모델들은 PPO 본 학습과 100% 동일한 구조를 가집니다.")
    logger.info(f"🔧 Policy Head: 진입 타이밍 식별 (label=1 예측)")
    logger.info(f"🎯 Value Head: 예상 보상 추정 (내부 계산된 TP/SL 기반 reward)")
    logger.info(f"⚖️ Value Loss Coefficient: {VALUE_LOSS_COEF}")
    logger.info(f"📊 TP Threshold: {TP_THRESHOLD:.4f}, SL Threshold: {SL_THRESHOLD:.4f}")
    logger.info(f"🕒 Multi-timeframe 입력을 완벽 지원합니다.")

if __name__ == "__main__":
    main()