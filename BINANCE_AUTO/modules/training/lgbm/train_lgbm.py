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
                         target_index: pd.DatetimeIndex) -> Tuple[Dict[str, np.ndarray], List[str], List[pd.Timestamp]]:
    """스케일링을 적용한 MTF 시퀀스 데이터 생성"""
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"[INFO] 스케일러 로드 완료: {SCALER_PATH}")
    all_feature_names = list(scaler.feature_names_in_)

    mtf_sequences = {}
    final_valid_indices = None

    for timeframe in TIMEFRAMES + AUX_TIMEFRAMES:
        if timeframe not in mtf_data:
            continue

        df = mtf_data[timeframe].copy()
        
        original_cols = df.columns.tolist()
        for col in all_feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[all_feature_names]
        
        scaled_values = scaler.transform(df)
        df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
        
        # 원본에 있던 컬럼만 다시 선택
        df_processed = df_scaled[original_cols].ffill().fillna(0)
        print(f"[🔧 {timeframe}] 피처 수: {len(df_processed.columns)}, 시퀀스 길이: {PPO_CONFIG['seq_len']}")

        index_list = df_processed.index.to_list()
        values = df_processed.values
        sequences = []
        valid_indices = []

        for t in target_index:
            if df_processed.index.tz is not None:
                t = t.tz_convert(df_processed.index.tz)
            else:
                t = t.tz_localize(None)

            pos = bisect.bisect_right(index_list, t)

            if pos >= PPO_CONFIG["seq_len"]:
                seq = values[pos - PPO_CONFIG["seq_len"]:pos]
                sequences.append(seq)
                valid_indices.append(index_list[pos - 1])

        if sequences:
            mtf_sequences[timeframe] = {
                'sequences': np.array(sequences),
                'columns': df_processed.columns.tolist()
            }
            print(f"[✅ {timeframe}] 시퀀스 생성 완료: {mtf_sequences[timeframe]['sequences'].shape}")

            if final_valid_indices is None:
                final_valid_indices = valid_indices
            else:
                final_valid_indices = list(set(final_valid_indices) & set(valid_indices))

    return mtf_sequences, sorted(final_valid_indices)

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

    mtf_sequences, valid_indices = create_mtf_sequences(mtf_data, labeled_df.index)

    valid_timestamps = pd.DatetimeIndex(valid_indices)
    aligned_labels = labeled_df.loc[valid_timestamps, 'label'].values

    X_flat, flat_feature_names = flatten_mtf_sequences(mtf_sequences)

    valid_indices_pos = labeled_df.index.get_indexer(valid_timestamps)
    X_flat = X_flat[valid_indices_pos]
    y = aligned_labels

    assert X_flat.shape[0] == len(y), "Features and labels size mismatch"

    print(f"[{data_type.upper()} 데이터 준비 완료] 최종 샘플 수: {len(X_flat)}")

    X_train, X_val, y_train, y_val = create_time_based_split(X_flat, y, valid_timestamps)

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