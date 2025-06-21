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
import random
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    LGBM_MODEL_PATHS,
    TRAIN_LABEL_PATHS,
    PPO_IMITATION_MODEL_PATHS
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

def generate_flatten_features(df, window=32):
    """LGBM 학습과 동일한 방식으로 flatten된 피처 생성"""
    print(f"[🔄 Flatten 피처 생성 시작] window={window}")
    
    df = df.ffill()
    
    # 제외할 키워드들
    future_keywords = ['label', 'target', 'tp_hit', 'sl_hit', 'next_', 'forward_']
    
    # 수치형 컬럼만 선택하고 future_keywords 제외
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if not any(keyword in col.lower() for keyword in future_keywords)]
    
    print(f" - 원본 컬럼 수: {len(df.columns)}개")
    print(f" - 수치형 컬럼 수: {len(numeric_cols)}개") 
    print(f" - 피처 컬럼 수: {len(feature_cols)}개")
    print(f" - 제외된 컬럼 예시: {[col for col in df.columns if col not in feature_cols][:5]}")
    
    flatten_dfs = []
    for col in feature_cols:
        for i in range(window):
            shifted_col = df[col].shift(i + 1)
            new_col_name = f"{col}_t-{i+1}"
            flatten_dfs.append(shifted_col.rename(new_col_name))
    
    X_flatten = pd.concat(flatten_dfs, axis=1).dropna()
    
    print(f" - Flatten 후 피처 수: {len(X_flatten.columns)}개")
    print(f" - Flatten 후 데이터 행: {len(X_flatten)}개")
    print(f" - 피처명 예시: {X_flatten.columns[:5].tolist()}")
    
    return X_flatten

class TimeSeriesDataset(Dataset):
    """시계열 데이터셋 클래스"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_lgbm_model(model_path):
    """LGBM 모델 로딩"""
    logger.info(f"LGBM 모델 로딩 경로: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LGBM 모델 파일을 찾을 수 없습니다: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"LGBM 모델 로딩 완료")
    return model

def train_imitation_model(model, train_loader, val_loader, output_model_path, epochs=5, lr=0.001):
    """PPO 모방학습 모델 훈련 (개선된 버전)"""
    logger.info("PPO 구조 기반 모방학습 훈련 시작")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    # 최고 성능 추적을 위한 변수
    best_f1 = 0.0
    best_model_path = output_model_path
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # 훈련 단계
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # PPOPolicyNetwork는 (policy_logits, value)를 반환하므로 policy만 사용
            policy_logits, _ = model(data)
            loss = criterion(policy_logits, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = policy_logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        # 검증 단계
        val_accuracy, val_f1, val_precision, val_recall, val_signal_rate = evaluate_model(model, val_loader)
        
        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        
        logger.info(f'Epoch {epoch+1}/{epochs} 완료 - Train Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}')
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
    
    logger.info(f"PPO 구조 기반 모방학습 훈련 완료 - 최고 F1: {best_f1:.4f}")
    return model, best_f1

def evaluate_model(model, test_loader):
    """PPO 모델 평가 (개선된 지표 포함)"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # PPOPolicyNetwork에서 policy logits만 사용
            policy_logits, _ = model(data)
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

def train_model_for_direction(direction):
    """특정 방향(long/short)에 대한 PPO 모델 훈련"""
    logger.info(f"=== {direction.upper()} PPO 모방학습 모델 훈련 시작 ===")
    
    # 시드 고정
    set_seed(42)
    
    # 파일 경로 설정
    lgbm_model_path = LGBM_MODEL_PATHS[direction]
    train_data_path = TRAIN_LABEL_PATHS[direction]
    output_model_path = PPO_IMITATION_MODEL_PATHS[direction]

    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    # 1. LGBM 모델 로딩
    lgbm_model = load_lgbm_model(lgbm_model_path)
    
    # 2. 훈련 데이터 로딩
    logger.info(f"훈련 데이터 로딩: {train_data_path}")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"훈련 데이터 파일을 찾을 수 없습니다: {train_data_path}")
        
    df = pd.read_csv(train_data_path)

    # 3. LGBM 학습과 동일한 flatten 피처 생성
    X_df = generate_flatten_features(df, window=32)

    # 4. LGBM 예측값을 imitation label로 사용
    labels = lgbm_model.predict(X_df)
    
    # 5. 라벨 분포 확인 및 로깅
    label_distribution = Counter(labels)
    logger.info(f"📊 라벨 분포: {dict(label_distribution)}")
    
    # 라벨 불균형 경고
    if len(label_distribution) == 1:
        logger.warning("⚠️ 심각한 라벨 불균형 감지: 단일 클래스만 존재!")
    elif label_distribution.get(1, 0) / len(labels) < 0.05:
        logger.warning("⚠️ 라벨 불균형 경고: Signal(label=1) 비율이 5% 미만입니다.")

    # 6. 정규화 적용
    X_values = X_df.values
    X_values = (X_values - X_values.mean(axis=0)) / (X_values.std(axis=0) + 1e-8)
    X_values = np.nan_to_num(X_values, nan=0.0, posinf=1e6, neginf=-1e6)

    # 7. 시계열 복원
    num_windows = X_df.shape[0]
    num_features = int(X_df.shape[1] / 32)
    X = X_values.reshape(num_windows, 32, num_features)

    logger.info(f"[🔍 복원된 시계열 shape] X: {X.shape}")

    # 8. 데이터셋 생성
    dataset = TimeSeriesDataset(X, labels)
    
    # 훈련/검증 분할 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)
    
    logger.info(f"데이터셋 준비 완료 - 훈련: {train_size}, 검증: {val_size}")
    
    # 9. PPOPolicyNetwork 모델 초기화
    input_dim = X.shape[2]  # 피처 차원
    hidden_dim = 256
    action_dim = 2
    
    model = PPOPolicyNetwork(input_dim=input_dim, hidden_dim=hidden_dim, action_dim=action_dim)
    logger.info(f"PPOPolicyNetwork 초기화 완료 - input_dim: {input_dim}, hidden_dim: {hidden_dim}, action_dim: {action_dim}")

    # GPU 사용 가능 시 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"사용 디바이스: {device}")
    
    # 10. 모델 훈련 (개선된 버전)
    model, best_f1 = train_imitation_model(model, train_loader, val_loader, output_model_path, epochs=5)
    
    # 11. 최종 모델 평가
    logger.info("최종 PPO 모델 평가 시작")
    accuracy, f1, precision, recall, signal_rate = evaluate_model(model, val_loader)
    
    # 12. PPO 구조 모델 저장 확인
    logger.info(f"PPO 구조 기반 모방학습 모델 저장 완료: {output_model_path}")
    logger.info(f"저장된 모델은 PPO 본 학습과 100% 동일한 구조입니다")
    
    # 13. 최종 성능 로그 출력
    logger.info(f"\n=== {direction.upper()} PPO 모방학습 모델 훈련 완료 ===")
    logger.info(f"📊 최종 평가 지표:")
    logger.info(f"  - Accuracy: {accuracy:.4f}")
    logger.info(f"  - F1 Score: {f1:.4f} (Best: {best_f1:.4f})")
    logger.info(f"  - Precision: {precision:.4f}")
    logger.info(f"  - Recall: {recall:.4f}")
    logger.info(f"  - Signal Rate (label=1 예측 비율): {signal_rate:.4f}")
    logger.info(f"저장된 파일 경로: {output_model_path}")
    logger.info(f"모델 구조: PPOPolicyNetwork (PPO 본 학습과 동일)")
    logger.info("=" * 50)
    
    return {
        'direction': direction,
        'accuracy': accuracy,
        'f1_score': f1,
        'best_f1_score': best_f1,
        'precision': precision,
        'recall': recall,
        'signal_rate': signal_rate,
        'model_path': output_model_path
    }

def main():
    """메인 실행 함수"""
    logger.info("PPO 구조 기반 모방학습 훈련 시작")
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
    logger.info("전체 PPO 모방학습 훈련 결과 요약")
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
        logger.info(f"  🏗️ 모델 구조: PPOPolicyNetwork")
    
    logger.info(f"\n🎉 모든 PPO 구조 기반 모방학습 훈련이 완료되었습니다!")
    logger.info(f"💡 저장된 모델들은 PPO 본 학습과 100% 동일한 구조를 가집니다.")
    logger.info(f"🔧 시드 고정으로 재현 가능한 결과를 보장합니다.")

if __name__ == "__main__":
    main()