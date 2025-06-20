import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(PROJECT_ROOT)

from modules.config import (
    LGBM_MODEL_PATHS,
    TRAIN_LABEL_PATHS,
    PPO_IMITATION_MODEL_PATHS,
)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedImitationPretrainer(nn.Module):
    """PPO 모방학습을 위한 Enhanced Imitation Pretrainer"""

    def __init__(self, input_shape, action_dim, hidden_dim=256):
        super().__init__()
        self.input_shape = input_shape  # (seq_len, feature_dim)
        self.action_dim = action_dim

        feature_dim = input_shape[1]

        # CNN layers for time series feature extraction
        self.conv1 = nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # LSTM for sequential modeling
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 + hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, action_dim)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size(0)

        # x: (batch, seq_len=32, feature_dim)
        x_cnn = x.transpose(1, 2)  # → (batch, feature_dim, seq_len) for Conv1d
        x_cnn = self.relu(self.conv1(x_cnn))
        x_cnn = self.relu(self.conv2(x_cnn))
        x_cnn = self.pool(x_cnn).squeeze(-1)  # → (batch, 64)

        lstm_out, (h_n, c_n) = self.lstm(x)  # → LSTM input: (batch, seq_len, feature_dim)
        x_lstm = h_n[-1]  # → (batch, hidden_dim)

        x_combined = torch.cat([x_cnn, x_lstm], dim=1)

        x = self.relu(self.fc1(x_combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

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

    model = joblib.load(model_path)  # ✅ 올바른 방식
    logger.info(f"LGBM 모델 로딩 완료")
    return model

def create_time_series_data(df, window_size=32):
    logger.info(f"시계열 데이터 생성 시작 - Window size: {window_size}")

    # ✅ LGBM 학습 시 사용한 feature 수에 맞춤
    feature_cols = df.select_dtypes(include=[np.number]).columns[:61].tolist()
    features = df[feature_cols].values

    X, y = [], []
    for i in range(len(features) - window_size + 1):
        window_data = features[i:i + window_size]
        X.append(window_data)

        if 'target' in df.columns:
            y.append(df.iloc[i + window_size - 1]['target'])
        else:
            y.append(0)

    X = np.array(X)  # (N, 32, 61)
    y = np.array(y)
    
    logger.info(f"시계열 데이터 생성 완료 - Shape: {X.shape}")
    return X, y

def flatten_for_lgbm(X):
    """LGBM을 위한 flatten 처리"""
    # Shape: (N, 32, 56) -> (N, 32*56)
    return X.reshape(X.shape[0], -1)

def generate_imitation_labels(lgbm_model, X_flat, confidence_threshold=0.85):
    """LGBM 확신도 기반 모방학습 라벨 생성"""
    logger.info("LGBM 확신도 기반 라벨 생성 시작")
    
    # LGBM predict_proba
    probabilities = lgbm_model.predict_proba(X_flat)
    
    # 최대 확률값을 확신도로 사용
    confidences = np.max(probabilities, axis=1)
    
    # 확신도 통계
    avg_confidence = np.mean(confidences)
    high_confidence_rate = np.mean(confidences >= confidence_threshold)
    
    logger.info(f"확신도 평균: {avg_confidence:.4f}")
    logger.info(f"확신도 {confidence_threshold} 이상 비율: {high_confidence_rate:.4f}")
    
    # 라벨 생성: 확신도 >= threshold일 때 1, 이외 0
    labels = (confidences >= confidence_threshold).astype(int)
    
    logger.info(f"생성된 라벨 분포 - 0: {np.sum(labels == 0)}, 1: {np.sum(labels == 1)}")
    
    return labels, confidences

def train_imitation_model(model, train_loader, epochs=5, lr=0.001):
    """모방학습 모델 훈련"""
    logger.info("모방학습 훈련 시작")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        logger.info(f'Epoch {epoch+1}/{epochs} 완료 - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    logger.info("모방학습 훈련 완료")
    return model

def evaluate_model(model, test_loader):
    """모델 평가"""
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    accuracy = accuracy_score(targets, predictions)
    signal_rate = np.mean(np.array(predictions) == 1)
    
    return accuracy, signal_rate

def train_model_for_direction(direction):
    """특정 방향(long/short)에 대한 모델 훈련"""
    logger.info(f"=== {direction.upper()} 모델 훈련 시작 ===")
    
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
    logger.info(f"훈련 데이터 로딩 완료 - Shape: {df.shape}")
    
    # 3. 시계열 데이터 생성
    X, _ = create_time_series_data(df, window_size=32)
    
    # 4. LGBM 확신도 기반 라벨 생성
    X_flat = flatten_for_lgbm(X)
    labels, confidences = generate_imitation_labels(lgbm_model, X_flat, confidence_threshold=0.85)
    
    # 5. 데이터셋 생성
    dataset = TimeSeriesDataset(X, labels)
    
    # 훈련/검증 분할 (8:2)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)
    
    logger.info(f"데이터셋 준비 완료 - 훈련: {train_size}, 검증: {val_size}")
    
    # 6. 모델 초기화
    model = EnhancedImitationPretrainer(input_shape=X.shape[1:], action_dim=2)

    # GPU 사용 가능 시 GPU로 이동
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"사용 디바이스: {device}")
    
    # 7. 모델 훈련
    model = train_imitation_model(model, train_loader, epochs=5)
    
    # 8. 모델 평가
    logger.info("모델 평가 시작")
    accuracy, signal_rate = evaluate_model(model, val_loader)
    
    # 9. 모델 저장
    torch.save(model.state_dict(), output_model_path)
    logger.info(f"모델 저장 완료: {output_model_path}")
    
    # 10. 성능 로그 출력
    logger.info(f"\n=== {direction.upper()} 모델 훈련 완료 ===")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Signal Rate (label=1 예측 비율): {signal_rate:.4f}")
    logger.info(f"저장된 파일 경로: {output_model_path}")
    logger.info("=" * 50)
    
    return {
        'direction': direction,
        'accuracy': accuracy,
        'signal_rate': signal_rate,
        'model_path': output_model_path,
        'avg_confidence': np.mean(confidences),
        'high_confidence_rate': np.mean(confidences >= 0.85)
    }

def main():
    """메인 실행 함수"""
    logger.info("PPO 모방학습 훈련 시작")
    logger.info("=" * 60)
    
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
    logger.info("전체 훈련 결과 요약")
    logger.info("=" * 60)
    
    for result in results:
        logger.info(f"\n{result['direction'].upper()} 모델:")
        logger.info(f"  - Accuracy: {result['accuracy']:.4f}")
        logger.info(f"  - Signal Rate: {result['signal_rate']:.4f}")
        logger.info(f"  - 평균 확신도: {result['avg_confidence']:.4f}")
        logger.info(f"  - 고확신도 비율: {result['high_confidence_rate']:.4f}")
        logger.info(f"  - 저장 경로: {result['model_path']}")
    
    logger.info("\n모든 PPO 모방학습 훈련이 완료되었습니다!")

if __name__ == "__main__":
    main()