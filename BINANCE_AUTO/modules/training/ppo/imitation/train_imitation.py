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
    
    # 데이터 정보 출력
    logger.info(f"   시간 범위: {df.index[0]} ~ {df.index[-1]}")
    logger.info(f"   샘플 수: {len(df)}")
    
    if 'close' in df.columns:
        logger.info(f"   가격 범위: {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    
    return df

def align_timeframes(entry_df: pd.DataFrame, eval_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """타임프레임 정렬 - 15분 entry와 5분 eval을 정확히 매칭"""
    logger.info("🔄 타임프레임 정렬 시작")
    
    # 시간 인덱스를 datetime으로 확실히 변환
    entry_df.index = pd.to_datetime(entry_df.index)
    eval_df.index = pd.to_datetime(eval_df.index)
    
    # 공통 시간 범위 찾기
    common_start = max(entry_df.index[0], eval_df.index[0])
    common_end = min(entry_df.index[-1], eval_df.index[-1])
    
    logger.info(f"공통 시간 범위: {common_start} ~ {common_end}")
    
    # 공통 범위로 필터링
    entry_df = entry_df[common_start:common_end]
    eval_df = eval_df[common_start:common_end]
    
    # 마지막 entry 시간 이후 충분한 eval 데이터 확보
    last_entry_time = entry_df.index[-LABEL_HORIZON]  # 마지막 LABEL_HORIZON개는 미래 데이터 필요
    entry_df = entry_df[:last_entry_time]
    
    logger.info(f"정렬 후 - Entry: {len(entry_df)} 샘플, Eval: {len(eval_df)} 샘플")
    
    return entry_df, eval_df

def calculate_tp_sl_hits_optimized(entry_df: pd.DataFrame, eval_df: pd.DataFrame, 
                                  direction: str) -> Tuple[np.ndarray, np.ndarray]:
    """최적화된 TP/SL 히트 계산"""
    logger.info(f"🎯 TP/SL 히트 계산 시작 - direction: {direction}")
    
    # 타임존 통일 - 모두 UTC로 변환하거나 타임존 제거
    if entry_df.index.tz is not None:
        entry_df = entry_df.copy()
        entry_df.index = entry_df.index.tz_convert('UTC')
    else:
        entry_df = entry_df.copy()
        entry_df.index = pd.to_datetime(entry_df.index).tz_localize('UTC')
    
    if eval_df.index.tz is not None:
        eval_df = eval_df.copy()
        eval_df.index = eval_df.index.tz_convert('UTC')
    else:
        eval_df = eval_df.copy()
        eval_df.index = pd.to_datetime(eval_df.index).tz_localize('UTC')
    
    tp_thresh = abs(TP_THRESHOLD)
    sl_thresh = abs(SL_THRESHOLD)
    
    # 결과 배열 초기화
    tp_hits = np.zeros(len(entry_df), dtype=bool)
    sl_hits = np.zeros(len(entry_df), dtype=bool)
    
    # 디버깅용 통계
    debug_stats = {
        'no_future_data': 0,
        'tp_hit': 0,
        'sl_hit': 0,
        'both_hit': 0,
        'neutral': 0
    }
    
    # 배치 처리를 위한 준비
    entry_times = entry_df.index
    entry_prices = entry_df['close'].values
    
    # 디버깅: 처음 몇 개 샘플 확인
    logger.info(f"Entry 타임존: {entry_df.index.tz}, Eval 타임존: {eval_df.index.tz}")
    logger.info(f"처음 3개 entry 시간: {entry_times[:3].tolist()}")
    
    # 각 entry에 대해 계산
    for i in range(len(entry_df)):
        entry_time = entry_times[i]
        entry_price = entry_prices[i]
        
        # TP/SL 가격 계산
        if direction == 'long':
            tp_price = entry_price * (1 + tp_thresh)
            sl_price = entry_price * (1 - sl_thresh)
        else:
            tp_price = entry_price * (1 - tp_thresh)
            sl_price = entry_price * (1 + sl_thresh)
        
        # 미래 데이터 선택 (라벨링 로직과 동일하게 첫 5분봉부터)
        future_start = entry_time  # 직후 5분 봉부터 포함
        future_end = entry_time + pd.Timedelta(minutes=15 * LABEL_HORIZON)
        
        # loc 사용하여 안전하게 슬라이싱
        future_mask = (eval_df.index >= future_start) & (eval_df.index <= future_end)
        future_data = eval_df.loc[future_mask]
        
        if len(future_data) == 0:
            debug_stats['no_future_data'] += 1
            continue
        
        # TP/SL 히트 확인
        if direction == 'long':
            tp_hit_mask = future_data['high'] >= tp_price
            sl_hit_mask = future_data['low'] <= sl_price
        else:
            tp_hit_mask = future_data['low'] <= tp_price
            sl_hit_mask = future_data['high'] >= sl_price
        
        # 첫 번째 히트 시점 찾기
        tp_hit_idx = np.where(tp_hit_mask)[0]
        sl_hit_idx = np.where(sl_hit_mask)[0]
        
        if len(tp_hit_idx) > 0 and len(sl_hit_idx) > 0:
            # 둘 다 히트한 경우
            debug_stats['both_hit'] += 1
            if tp_hit_idx[0] <= sl_hit_idx[0]:
                tp_hits[i] = True
                debug_stats['tp_hit'] += 1
            else:
                sl_hits[i] = True
                debug_stats['sl_hit'] += 1
        elif len(tp_hit_idx) > 0:
            tp_hits[i] = True
            debug_stats['tp_hit'] += 1
        elif len(sl_hit_idx) > 0:
            sl_hits[i] = True
            debug_stats['sl_hit'] += 1
        else:
            debug_stats['neutral'] += 1
        
        # 처음 몇 개 샘플 상세 디버깅
        if i < 3:
            logger.debug(f"샘플 {i}: entry_time={entry_time}, future_data 수={len(future_data)}")
            if len(future_data) > 0:
                logger.debug(f"  미래 데이터 시간 범위: {future_data.index[0]} ~ {future_data.index[-1]}")
    
    # 결과 통계 출력
    total_samples = len(entry_df)
    logger.info(f"📊 TP/SL 히트 분포:")
    logger.info(f"  TP 히트: {debug_stats['tp_hit']}개 ({debug_stats['tp_hit']/total_samples*100:.1f}%)")
    logger.info(f"  SL 히트: {debug_stats['sl_hit']}개 ({debug_stats['sl_hit']/total_samples*100:.1f}%)")
    logger.info(f"  Neutral: {debug_stats['neutral']}개 ({debug_stats['neutral']/total_samples*100:.1f}%)")
    logger.info(f"  미래 데이터 없음: {debug_stats['no_future_data']}개")
    logger.info(f"  동시 히트: {debug_stats['both_hit']}개")
    
    # 샘플 디버깅
    if debug_stats['tp_hit'] + debug_stats['sl_hit'] == 0:
        logger.warning("⚠️ TP/SL 히트가 전혀 없습니다. 샘플 분석:")
        for i in range(min(3, len(entry_df))):
            entry_time = entry_times[i]
            entry_price = entry_prices[i]
            tp_price = entry_price * (1 + tp_thresh) if direction == 'long' else entry_price * (1 - tp_thresh)
            sl_price = entry_price * (1 - sl_thresh) if direction == 'long' else entry_price * (1 + sl_thresh)
            
            # 타임존 안전한 슬라이싱
            future_end = entry_time + pd.Timedelta(minutes=60)
            future_mask = (eval_df.index > entry_time) & (eval_df.index <= future_end)
            future_data = eval_df.loc[future_mask]
            
            if len(future_data) > 0:
                max_move = (future_data['high'].max() - entry_price) / entry_price
                min_move = (future_data['low'].min() - entry_price) / entry_price
                logger.warning(f"  샘플 {i}: Entry={entry_price:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}")
                logger.warning(f"    최대 상승: {max_move*100:.2f}%, 최대 하락: {min_move*100:.2f}%")
    
    return tp_hits, sl_hits

def generate_mtf_features(mtf_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """MTF 피처 생성 - 개선된 버전"""
    logger.info(f"🔄 MTF 피처 생성 시작")
    mtf_features = {}
    
    for timeframe in TIMEFRAMES:
        if timeframe not in mtf_dict:
            continue
        
        df = mtf_dict[timeframe].copy()
        
        # 데이터 검증
        df = validate_data(df, f"MTF-{timeframe}")
        
        # 피처 선택
        exclude_keywords = ['label', 'target', 'tp_hit', 'sl_hit', 'next_', 'forward_']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if not any(k in col.lower() for k in exclude_keywords)]
        
        if len(feature_cols) == 0:
            logger.warning(f"⚠️ {timeframe}에 사용 가능한 피처가 없습니다")
            continue
        
        # 시계열 윈도우 생성
        flatten_dfs = []
        for col in feature_cols:
            for i in range(PPO_CONFIG["seq_len"]):
                shifted = df[col].shift(i + 1)
                flatten_dfs.append(shifted.rename(f"{col}_t-{i+1}"))
        
        tf_feature_df = pd.concat(flatten_dfs, axis=1).dropna()
        mtf_features[timeframe] = tf_feature_df
        logger.info(f"[{timeframe}] shape: {tf_feature_df.shape}, 피처 수: {len(feature_cols)}")
    
    return mtf_features

class ImitationDataset(Dataset):
    """모방 학습을 위한 데이터셋 (상태, 정책 라벨)"""
    def __init__(self, mtf_features: Dict[str, np.ndarray], labels: np.ndarray):
        self.mtf_features = {tf: torch.FloatTensor(features) for tf, features in mtf_features.items()}
        self.labels = torch.LongTensor(labels)
        
        logger.info(f"📊 데이터셋 생성: 총 {len(self.labels)} 샘플")
        logger.info(f"   라벨 분포: {dict(Counter(labels.tolist()))}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        features = {tf: data[idx] for tf, data in self.mtf_features.items()}
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

def prepare_features(raw_data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, np.ndarray], Dict[str, int], int]:
    """원시 데이터로부터 MTF 피처를 생성, 정규화 및 재구성"""
    logger.info("🛠️  피처 준비 시작")
    
    # 1. MTF 피처 생성
    mtf_features_df = {}
    for tf in TIMEFRAMES:
        if tf not in raw_data: 
            continue
        df = raw_data[tf].copy()
        
        # 피처로 사용할 컬럼 선택 (라벨 등 제외)
        exclude_keywords = ['label', 'target', 'tp_hit', 'sl_hit', 'next_', 'forward_']
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if not any(k in col.lower() for k in exclude_keywords)]
        
        if len(feature_cols) == 0:
            logger.warning(f"⚠️ {tf}에 사용 가능한 피처가 없습니다")
            continue
        
        # 시계열 윈도우 생성
        flatten_dfs = [df[col].shift(i + 1).rename(f"{col}_t-{i+1}") for col in feature_cols for i in range(PPO_CONFIG["seq_len"])]
        tf_feature_df = pd.concat(flatten_dfs, axis=1).dropna()
        mtf_features_df[tf] = tf_feature_df
        logger.info(f"   - [{tf}] 피처 생성 완료. Shape: {tf_feature_df.shape}")

    # 2. 데이터 길이 정렬
    entry_df = raw_data['15min'].copy()
    min_len = min(len(df) for df in mtf_features_df.values())
    min_len = min(min_len, len(entry_df))
    
    for tf in mtf_features_df:
        mtf_features_df[tf] = mtf_features_df[tf].iloc[-min_len:]
    
    # 3. 피처 정규화 및 Reshape
    mtf_features_array = {}
    input_dims = {}
    for tf, features_df in mtf_features_df.items():
        X = features_df.values
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
        X_normalized = (X - mean) / std
        
        num_features = len(features_df.columns) // PPO_CONFIG["seq_len"]
        X_reshaped = X_normalized.reshape(len(X), PPO_CONFIG["seq_len"], num_features)
        
        mtf_features_array[tf] = X_reshaped
        input_dims[tf] = num_features
        logger.info(f"   - [{tf}] 정규화 및 Reshape 완료. 최종 Shape: {X_reshaped.shape}")

    return mtf_features_array, input_dims, min_len

def run_imitation_learning_for(direction: str):
    """지정된 방향(long/short)에 대한 모방 학습 전체 파이프라인 실행"""
    logger.info(f"""
{'='*60}
🎯 [{direction.upper()}] 모방 학습 파이프라인 시작
{'='*60}""")
    set_seed(42)

    # 1. 데이터 로드 및 준비
    raw_data = pd.read_pickle(TRAIN_PICKLE_PATHS[direction])
    logger.info(f"   - 원본 데이터 로드 완료: {TRAIN_PICKLE_PATHS[direction]}")
    
    # 타임프레임 설정
    entry_tf, eval_tf = "15min", "5min"
    
    # 데이터 검증
    df_entry = validate_data(raw_data[entry_tf].copy(), f"{direction}-entry")
    df_eval = validate_data(raw_data[eval_tf].copy(), f"{direction}-eval")
    
    # 타임프레임 정렬
    df_entry, df_eval = align_timeframes(df_entry, df_eval)
    
    # MTF 피처 생성
    mtf_features_df = generate_mtf_features(raw_data)
    
    # 모든 타임프레임의 최소 길이에 맞춰 정렬
    min_len = min([features.shape[0] for features in mtf_features_df.values()])
    min_len = min(min_len, len(df_entry))  # entry 데이터 길이도 고려
    
    # 데이터 정렬
    labels = df_entry["label"].values[:min_len].astype(int)

    mtf_features_array = {}
    input_dims = {}
    
    for timeframe, features_df in mtf_features_df.items():
        # 길이 맞추기
        features_df = features_df.iloc[:min_len]
        
        X_values = features_df.values
        
        # 정규화
        mean = X_values.mean(axis=0)
        std = X_values.std(axis=0) + 1e-8
        X_values = (X_values - mean) / std
        X_values = np.nan_to_num(X_values, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Reshape
        num_windows = features_df.shape[0]
        num_features = int(features_df.shape[1] / PPO_CONFIG["seq_len"])
        
        try:
            X_reshaped = X_values.reshape(num_windows, PPO_CONFIG["seq_len"], num_features)
        except ValueError as e:
            logger.error(f"Reshape 오류 - {timeframe}: {e}")
            logger.error(f"Shape: {X_values.shape}, Target: ({num_windows}, {PPO_CONFIG['seq_len']}, {num_features})")
            raise
        
        mtf_features_array[timeframe] = X_reshaped
        input_dims[timeframe] = num_features
    
    # 2. 데이터셋 및 데이터로더 생성
    dataset = ImitationDataset(mtf_features_array, labels)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=IMITATION_CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=IMITATION_CONFIG["batch_size"], shuffle=False)
    logger.info(f"   - 데이터셋 분할 완료: Train {train_size}개, Validation {val_size}개")

    # 3. 모델 생성
    model = PPOPolicyNetwork(
        timeframe_dims=input_dims, 
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