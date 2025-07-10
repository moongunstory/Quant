import pandas as pd
import numpy as np
import os
import sys
import joblib
import lightgbm as lgb
import optuna
import bisect
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from typing import Dict, Tuple, List
import pickle
from sklearn.preprocessing import StandardScaler

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

sys.path.append(PROJECT_ROOT)

from modules.config import (
   TRAIN_PICKLE_PATHS,
   LGBM_THRESHOLD,
   LGBM_MODEL_PATHS,
   SCALER_PATH, # 스케일러 경로 추가
   TIMEFRAMES,
   AUX_TIMEFRAMES,
   PPO_CONFIG,
   RAW_DATA_PATH,
)

def load_mtf_data() -> Dict[str, pd.DataFrame]:
   """MTF 개별 pickle 파일들을 로드"""
   save_dir = os.path.dirname(os.path.join(PROJECT_ROOT, RAW_DATA_PATH))
   mtf_data = {}
   
   data_keys = TIMEFRAMES + AUX_TIMEFRAMES
   
   for key in data_keys:
       file_path = os.path.join(save_dir, f"market_data_{key}.pkl")
       if os.path.exists(file_path):
           df = pd.read_pickle(file_path)
           mtf_data[key] = df
           print(f"[로드] {key}: {len(df)} rows, {len(df.columns)} columns")
       else:
           print(f"[경고] {key} 파일 없음: {file_path}")
   
   return mtf_data

def load_labeled_data(data_type: str = "long") -> pd.DataFrame:
   """Long/Short 이진분류 데이터 로딩"""
   with open(TRAIN_PICKLE_PATHS[data_type], 'rb') as f:
        mtf_dict = pickle.load(f)
   df = mtf_dict["15min"]
   print(f"[{data_type.upper()} 라벨 데이터 로딩 완료] 행: {len(df)}, 컬럼: {len(df.columns)}")
   return df

# ✅ 수정: 스케일러를 적용하고 시퀀스를 생성하는 함수
def create_mtf_sequences(mtf_data: Dict[str, pd.DataFrame], 
                         target_index: pd.DatetimeIndex) -> Tuple[Dict[str, np.ndarray], pd.DatetimeIndex]:
    """스케일링을 적용하고 기준 인덱스에 맞춰 정렬된 MTF 시퀀스 데이터 생성"""
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"[INFO] 스케일러 로드 완료: {SCALER_PATH}")
    all_feature_names = list(scaler.feature_names_in_)

    processed_data = {}
    for timeframe, df_orig in mtf_data.items():
        df = df_orig.copy()
        # 빠진 컬럼을 한 번에 추가하여 PerformanceWarning 방지
        missing_cols = set(all_feature_names) - set(df.columns)
        if missing_cols:
            padding_df = pd.DataFrame(0, index=df.index, columns=list(missing_cols))
            df = pd.concat([df, padding_df], axis=1)
        
        df = df[all_feature_names]
        
        # 스케일링 적용
        scaled_values = scaler.transform(df)
        df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
        
        # 원본에 있던 컬럼만 다시 선택하여 데이터 보존
        original_cols = [col for col in df_orig.columns if col in df_scaled.columns]
        processed_data[timeframe] = df_scaled[original_cols]

    # target_index를 기준으로 모든 데이터를 재정렬 (forward-fill)
    aligned_data = {}
    for timeframe, df in processed_data.items():
        # fillna(0)을 추가하여 ffill 후에도 남는 NaN 값을 처리
        aligned_data[timeframe] = df.reindex(target_index, method='ffill').fillna(0)
        print(f"[🔄 {timeframe}] 데이터를 target_index에 정렬 완료. Shape: {aligned_data[timeframe].shape}")

    mtf_sequences = {}
    seq_len = PPO_CONFIG['seq_len']
    
    # 모든 타임프레임에 대해 시퀀스 생성
    for timeframe, df in aligned_data.items():
        values = df.values
        sequences = []
        # 슬라이딩 윈도우를 사용하여 시퀀스 생성
        if len(df) >= seq_len:
            for i in range(len(df) - seq_len + 1):
                sequences.append(values[i:i+seq_len])
        
        if sequences:
            mtf_sequences[timeframe] = {
                'sequences': np.array(sequences),
                'columns': df.columns.tolist()
            }
            print(f"[✅ {timeframe}] 시퀀스 생성 완료: {np.array(sequences).shape}")

    # 시퀀스 생성으로 인해 잘려나간 앞부분을 고려하여 최종 인덱스 조정
    final_index = target_index[seq_len - 1:]
    
    # 모든 시퀀스의 길이를 final_index 길이에 맞춤 (가장 짧은 시퀀스 기준)
    min_len = min(len(data['sequences']) for data in mtf_sequences.values()) if mtf_sequences else 0
    
    for timeframe in mtf_sequences.keys():
        mtf_sequences[timeframe]['sequences'] = mtf_sequences[timeframe]['sequences'][:min_len]

    final_index = final_index[:min_len]

    return mtf_sequences, final_index

# ✅ 수정: 피처 이름 생성 로직 변경
def flatten_mtf_sequences(mtf_sequences: Dict[str, dict]) -> Tuple[np.ndarray, List[str]]:
    """MTF 시퀀스를 LGBM용 평면 피처로 변환"""
    flattened_features = []
    new_feature_names = []
    
    for timeframe, data in mtf_sequences.items():
        sequences = data['sequences']
        columns = data['columns']
        n_samples, seq_len, n_features = sequences.shape
        
        flat_data = sequences.reshape(n_samples, -1)
        flattened_features.append(flat_data)
        
        for t in range(seq_len):
            for f_idx, f_name in enumerate(columns):
                new_feature_names.append(f"{timeframe}_t{t}_{f_name}")
    
    if flattened_features:
        X_flat = np.concatenate(flattened_features, axis=1)
        print(f"[🔄 MTF 평면화 완료] Shape: {X_flat.shape}")
        return X_flat, new_feature_names
    else:
        raise ValueError("평면화할 MTF 시퀀스가 없습니다.")

def create_time_based_split(X: np.ndarray, y: np.ndarray, 
                          timestamps: pd.DatetimeIndex, 
                          train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
   """시간 기반 train/validation 분할"""
   n_samples = len(X)
   train_size = int(n_samples * train_ratio)
   
   X_train, X_val = X[:train_size], X[train_size:]
   y_train, y_val = y[:train_size], y[train_size:]

   timestamps = timestamps[:len(X)]
   train_period = f"{timestamps[0].strftime('%Y-%m-%d')} ~ {timestamps[train_size-1].strftime('%Y-%m-%d')}"
   val_period = f"{timestamps[train_size].strftime('%Y-%m-%d')} ~ {timestamps[-1].strftime('%Y-%m-%d')}"
   
   print(f"[시간 기반 분할] Train: {len(X_train)} ({train_period}), Val: {len(X_val)} ({val_period})")
   return X_train, X_val, y_train, y_val

def optimize_model(X_train: np.ndarray, y_train: np.ndarray, 
                 X_val: np.ndarray, y_val: np.ndarray, 
                 data_type: str = "long") -> Tuple:
   """Optuna 하이퍼파라미터 최적화"""
   print(f"[{data_type.upper()} 하이퍼파라미터 최적화 시작]")
   
   def objective(trial):
       params = {
           "objective": "binary", "boosting_type": "gbdt",
           "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
           "num_leaves": trial.suggest_int("num_leaves", 16, 64),
           "max_depth": trial.suggest_int("max_depth", 4, 10),
           "n_estimators": 2000, "class_weight": "balanced", "random_state": 42, "verbosity": -1,
       }
       model = lgb.LGBMClassifier(**params)
       model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
       y_prob = model.predict_proba(X_val)[:, 1]
       y_pred = (y_prob >= LGBM_THRESHOLD).astype(int)
       return f1_score(y_val, y_pred, zero_division=0)
   
   study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
   study.optimize(objective, n_trials=15, show_progress_bar=False)
   print(f"[{data_type.upper()} 최적화 완료] 최고 F1: {study.best_value:.4f}")
   return study, []

def train_model(X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray, 
              data_type: str = "long", use_optuna: bool = True) -> lgb.LGBMClassifier:
   if use_optuna:
       study, _ = optimize_model(X_train, y_train, X_val, y_val, data_type)
       best_params = study.best_params
       best_params.update({"objective": "binary", "boosting_type": "gbdt", "n_estimators": 2000, "class_weight": "balanced", "random_state": 42, "verbosity": -1})
       model = lgb.LGBMClassifier(**best_params)
   else:
       model = lgb.LGBMClassifier(n_estimators=1000, class_weight="balanced", random_state=42, verbosity=-1)
   
   model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
   print(f"[{data_type.upper()} 모델 학습 완료]")
   return model

def evaluate_model(model: lgb.LGBMClassifier, X_val: np.ndarray, y_val: np.ndarray, 
                   data_type: str = "long", feature_names: List[str] = None) -> Dict:
   print(f"[{data_type.upper()} 모델 평가 중...")
   y_prob = model.predict_proba(X_val)[:, 1]
   y_pred = (y_prob >= LGBM_THRESHOLD).astype(int)
   
   metrics = {
       'f1': f1_score(y_val, y_pred, zero_division=0),
       'precision': precision_score(y_val, y_pred, zero_division=0),
       'recall': recall_score(y_val, y_pred, zero_division=0),
       'signal_rate': (y_prob >= LGBM_THRESHOLD).mean()
   }
   print(f"[{data_type.upper()} 평가 결과] F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
   return metrics

def save_model(model: lgb.LGBMClassifier, metrics: Dict, data_type: str = "long"):
   model_path = LGBM_MODEL_PATHS[data_type]
   os.makedirs(os.path.dirname(model_path), exist_ok=True)
   joblib.dump(model, model_path)
   print(f"[{data_type.upper()} 모델 저장 완료] {model_path}")

def show_feature_importance(model: lgb.LGBMClassifier, feature_names: List[str], 
                          data_type: str = "long", top_k: int = 20):
   importance_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
   print(f"[{data_type.upper()} 상위 {top_k}개 중요 피처]")
   print(importance_df.head(top_k).to_string(index=False))

def train_pipeline(data_type: str = "long", use_optuna: bool = True) -> Tuple:
    print(f"\n{'='*60}\n{data_type.upper()} MTF LGBM 모델 학습 시작\n{'='*60}")
    
    mtf_data = load_mtf_data()
    labeled_df = load_labeled_data(data_type)

    mtf_sequences, common_indices = create_mtf_sequences(mtf_data, labeled_df.index)

    if not common_indices:
        print(f"[{data_type.upper()}] 공통 인덱스가 없어 학습을 중단합니다.")
        return None, None

    valid_timestamps = pd.DatetimeIndex(common_indices)
    
    # label_df.index가 timezone-aware이고 valid_timestamps가 naive일 수 있으므로 통일
    if labeled_df.index.tz is not None and valid_timestamps.tz is None:
        valid_timestamps = valid_timestamps.tz_localize(labeled_df.index.tz)
    elif labeled_df.index.tz is None and valid_timestamps.tz is not None:
        valid_timestamps = valid_timestamps.tz_convert(None)

    aligned_labels = labeled_df.loc[labeled_df.index.intersection(valid_timestamps), 'label'].values

    X_flat, flat_feature_names = flatten_mtf_sequences(mtf_sequences)

    # 데이터 정렬 확인
    if X_flat.shape[0] != len(aligned_labels):
        print(f"[경고] X_flat과 라벨의 길이가 다릅니다. X_flat: {X_flat.shape[0]}, Labels: {len(aligned_labels)}")
        # 길이를 맞추기 위한 추가 로직이 필요할 수 있음 (예: 재정렬)
        # 임시 해결: 라벨 길이에 맞춰 X_flat을 자름 (데이터 손실 가능성 있음)
        min_len = min(X_flat.shape[0], len(aligned_labels))
        X_flat = X_flat[:min_len]
        aligned_labels = aligned_labels[:min_len]
        valid_timestamps = valid_timestamps[:min_len]

    assert X_flat.shape[0] == len(aligned_labels), "Features and labels size mismatch after alignment"

    print(f"[{data_type.upper()} 데이터 준비 완료] 최종 샘플 수: {len(X_flat)}")

    X_train, X_val, y_train, y_val = create_time_based_split(X_flat, aligned_labels, valid_timestamps)

    model = train_model(X_train, y_train, X_val, y_val, data_type, use_optuna)
    metrics = evaluate_model(model, X_val, y_val, data_type, flat_feature_names)
    show_feature_importance(model, flat_feature_names, data_type)
    save_model(model, metrics, data_type)

    print(f"\n[{data_type.upper()} MTF LGBM 학습 완료]")
    return model, metrics

def main():
   print("=" * 80)
   print("MTF LGBM Long/Short 모델 학습 시작")
   print("=" * 80)
   
   results = {}
   long_model, long_metrics = train_pipeline("long", use_optuna=True)
   results["long"] = {"model": long_model, "metrics": long_metrics}
   
   short_model, short_metrics = train_pipeline("short", use_optuna=True)
   results["short"] = {"model": short_model, "metrics": short_metrics}
   
   print(f"\n{'=' * 80}\nMTF LGBM Long/Short 모델 학습 완료 요약\n{'=' * 80}")
   print(f"Long 모델  - F1: {long_metrics['f1']:.4f}")
   print(f"Short 모델 - F1: {short_metrics['f1']:.4f}")
   
   return results

if __name__ == "__main__":
   results = main()