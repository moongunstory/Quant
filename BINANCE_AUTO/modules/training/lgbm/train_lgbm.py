import pandas as pd
import numpy as np
import os
import sys
import joblib
import lightgbm as lgb
import optuna
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
    TIMEFRAMES,
    AUX_TIMEFRAMES,
    PPO_CONFIG,
    RAW_DATA_PATH,
)

def load_mtf_data() -> Dict[str, pd.DataFrame]:
    """MTF 개별 pickle 파일들을 로드 (선택적 타임프레임)"""
    save_dir = os.path.dirname(os.path.join(PROJECT_ROOT, RAW_DATA_PATH))
    mtf_data = {}
    
    data_keys = ['5min', '15min', '30min', '1H']
    print(f"[INFO] 선택된 타임프레임: {data_keys}")

    for key in data_keys:
        file_path = os.path.join(save_dir, f"market_data_{key}.pkl")
        if os.path.exists(file_path):
            df = pd.read_pickle(file_path)
            float_cols = df.select_dtypes(include=['float64']).columns
            df[float_cols] = df[float_cols].astype(np.float32)
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

def create_mtf_sequences(mtf_data: Dict[str, pd.DataFrame], 
                         target_index: pd.DatetimeIndex) -> Tuple[Dict[str, np.ndarray], pd.DatetimeIndex]:
    """기준 인덱스에 맞춰 정렬된 MTF 시퀀스 데이터 생성 (스케일링 없음)"""
    processed_data = {}
    
    # 모든 데이터프레임의 컬럼을 통합하여 전체 피처 목록 생성
    all_columns = set()
    for df in mtf_data.values():
        all_columns.update(df.columns)
    all_columns = sorted(list(all_columns))

    for timeframe, df_orig in mtf_data.items():
        df = df_orig.copy()
        missing_cols = set(all_columns) - set(df.columns)
        if missing_cols:
            padding_df = pd.DataFrame(0, index=df.index, columns=list(missing_cols), dtype=np.float32)
            df = pd.concat([df, padding_df], axis=1)
        
        processed_data[timeframe] = df[all_columns]

    aligned_data = {}
    for timeframe, df in processed_data.items():
        aligned_data[timeframe] = df.reindex(target_index, method='ffill').fillna(0)
        print(f"[🔄 {timeframe}] 데이터를 target_index에 정렬 완료. Shape: {aligned_data[timeframe].shape}")

    mtf_sequences = {}
    seq_len = PPO_CONFIG['seq_len']
    
    for timeframe, df in aligned_data.items():
        values = df.values.astype(np.float32)
        sequences = []
        if len(df) >= seq_len:
            for i in range(len(df) - seq_len + 1):
                sequences.append(values[i:i+seq_len])
        
        if sequences:
            mtf_sequences[timeframe] = {
                'sequences': np.array(sequences, dtype=np.float32),
                'columns': df.columns.tolist()
            }
            print(f"[✅ {timeframe}] 시퀀스 생성 완료: {np.array(sequences).shape}")

    final_index = target_index[seq_len - 1:]
    
    min_len = min(len(data['sequences']) for data in mtf_sequences.values()) if mtf_sequences else 0
    
    for timeframe in mtf_sequences.keys():
        mtf_sequences[timeframe]['sequences'] = mtf_sequences[timeframe]['sequences'][:min_len]

    final_index = final_index[:min_len]

    return mtf_sequences, final_index

def engineer_features(sequences: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    시퀀스 데이터로부터 통계적/모멘텀 피처를 엔지니어링.
    (n_samples, seq_len, n_features) -> (n_samples, n_engineered_features)
    """
    n_samples, seq_len, n_features = sequences.shape
    
    # 주요 피처 선택 (예: close, volume, rsi 등)
    key_features_indices = [i for i, name in enumerate(feature_names) if 'close' in name or 'volume' in name or 'rsi' in name or 'macd' in name]
    
    if not key_features_indices:
        # 키 피처가 없으면 마지막 스텝의 값만 사용
        return pd.DataFrame(sequences[:, -1, :], columns=feature_names)

    key_sequences = sequences[:, :, key_features_indices]
    key_feature_names = [feature_names[i] for i in key_features_indices]
    
    engineered_features = {}
    
    # 1. 롤링 통계 (평균, 표준편차)
    windows = [5, 15, 30]
    for win in windows:
        if seq_len < win: continue
        rolling_mean = np.mean(key_sequences[:, -win:, :], axis=1)
        rolling_std = np.std(key_sequences[:, -win:, :], axis=1)
        for i, name in enumerate(key_feature_names):
            engineered_features[f'{name}_mean_{win}'] = rolling_mean[:, i]
            engineered_features[f'{name}_std_{win}'] = rolling_std[:, i]

    # 2. 모멘텀 (가격 변화율)
    if seq_len > 1:
        momentum = key_sequences[:, -1, :] - key_sequences[:, -16, :]
        for i, name in enumerate(key_feature_names):
            engineered_features[f'{name}_mom_16'] = momentum[:, i]

    # 3. 마지막 스텝의 값
    for i, name in enumerate(feature_names):
        engineered_features[f'{name}_last'] = sequences[:, -1, i]
        
    df = pd.DataFrame(engineered_features)
    print(f"[⚙️ 피처 엔지니어링 완료] Shape: {df.shape}")
    return df

def train_model(X_train: pd.DataFrame, y_train: np.ndarray, 
              X_val: pd.DataFrame, y_val: np.ndarray, 
              data_type: str = "long", use_optuna: bool = True) -> lgb.LGBMClassifier:
    """Optuna를 사용하거나 기본 파라미터로 모델 학습"""
    if use_optuna:
        # Optuna 최적화 (n_trials는 예시로 줄임)
        def objective(trial):
            params = {
                "objective": "binary", "boosting_type": "gbdt",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 20, 60),
                "max_depth": trial.suggest_int("max_depth", 5, 10),
                "n_estimators": 1000, "class_weight": "balanced", "random_state": 42, "verbosity": -1,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(50, verbose=False)])
            y_prob = model.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= LGBM_THRESHOLD).astype(int)
            return f1_score(y_val, y_pred, zero_division=0)
        
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=15, show_progress_bar=False) # 실제 사용 시 n_trials 증가 권장
        best_params = study.best_params
        best_params.update({"objective": "binary", "boosting_type": "gbdt", "n_estimators": 2000, "class_weight": "balanced", "random_state": 42, "verbosity": -1})
        model = lgb.LGBMClassifier(**best_params)
    else:
        model = lgb.LGBMClassifier(n_estimators=1000, class_weight="balanced", random_state=42, verbosity=-1)
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"[{data_type.upper()} 모델 학습 완료]")
    return model

def evaluate_model(model: lgb.LGBMClassifier, X_val: pd.DataFrame, y_val: np.ndarray) -> Tuple[Dict, np.ndarray]:
    """모델 평가 및 예측 확률 반환"""
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= LGBM_THRESHOLD).astype(int)
    
    metrics = {
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'signal_rate': (y_prob >= LGBM_THRESHOLD).mean()
    }
    return metrics, y_prob

def save_model(model: lgb.LGBMClassifier, scaler: StandardScaler, data_type: str = "long"):
    """모델과 스케일러 저장"""
    # 모델 저장
    model_path = LGBM_MODEL_PATHS[data_type]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"[{data_type.upper()} 모델 저장 완료] {model_path}")
    
    # 스케일러 저장
    scaler_path = os.path.join(os.path.dirname(model_path), f"lgbm_scaler_{data_type}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[{data_type.upper()} 스케일러 저장 완료] {scaler_path}")

def show_feature_importance(model: lgb.LGBMClassifier, feature_names: List[str], data_type: str = "long", top_k: int = 20):
    """피처 중요도 출력"""
    importance_df = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    print(f"\n[{data_type.upper()} 상위 {top_k}개 중요 피처]")
    print(importance_df.head(top_k).to_string(index=False))

def train_pipeline(data_type: str = "long", use_optuna: bool = True):
    """데이터 준비부터 모델 학습, 평가, 저장까지의 전체 파이프라인 (Walk-Forward Validation)"""
    print(f"\n{'='*60}\n{data_type.upper()} MTF LGBM 모델 학습 시작 (Walk-Forward Validation)\n{'='*60}")
    
    # 1. 데이터 로드 및 시퀀스 생성
    mtf_data = load_mtf_data()
    labeled_df = load_labeled_data(data_type)
    mtf_sequences, final_index = create_mtf_sequences(mtf_data, labeled_df.index)

    if final_index.empty:
        print(f"[{data_type.upper()}] 처리할 데이터가 없어 학습을 중단합니다.")
        return None, None

    # 모든 타임프레임의 시퀀스를 하나로 결합 (n_samples, seq_len, n_total_features)
    combined_sequences = np.concatenate([data['sequences'] for data in mtf_sequences.values()], axis=2)
    combined_feature_names = [col for data in mtf_sequences.values() for col in data['columns']]
    
    # 라벨 정렬
    y = labeled_df.loc[labeled_df.index.intersection(final_index), 'label'].values
    min_len = min(len(combined_sequences), len(y))
    X_seq, y = combined_sequences[:min_len], y[:min_len]

    # 2. Walk-Forward Validation 설정
    n_samples = len(X_seq)
    initial_train_size = int(n_samples * 0.7)
    validation_size = int(n_samples * 0.1)
    n_splits = (n_samples - initial_train_size) // validation_size
    
    all_val_preds, all_val_labels = [], []

    print(f"\n[INFO] Walk-Forward Validation 시작: 총 {n_splits} 스플릿")
    for i in range(n_splits):
        train_end = initial_train_size + i * validation_size
        val_end = train_end + validation_size
        
        train_indices = range(train_end)
        val_indices = range(train_end, val_end)
        
        print(f"\n--- 스플릿 {i+1}/{n_splits}: Train={len(train_indices)}, Val={len(val_indices)} ---")

        # 3. 피처 엔지니어링
        X_train_seq, X_val_seq = X_seq[train_indices], X_seq[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        X_train_eng = engineer_features(X_train_seq, combined_feature_names)
        X_val_eng = engineer_features(X_val_seq, combined_feature_names)
        
        # 4. 데이터 스케일링 (Data Leakage 방지)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_eng)
        X_val_scaled = scaler.transform(X_val_eng)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_eng.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val_eng.columns)

        # 5. 모델 학습 및 평가
        model = train_model(X_train_scaled, y_train, X_val_scaled, y_val, data_type, use_optuna=(i==n_splits-1)) # 마지막 스플릿에서만 Optuna
        
        metrics, y_prob = evaluate_model(model, X_val_scaled, y_val)
        print(f"[평가 결과] F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        
        all_val_preds.extend(y_prob)
        all_val_labels.extend(y_val)

    # 6. 전체 검증 결과 집계
    print(f"\n{'='*60}\nWalk-Forward 최종 평가 결과\n{'='*60}")
    final_preds = (np.array(all_val_preds) >= LGBM_THRESHOLD).astype(int)
    report = classification_report(all_val_labels, final_preds, target_names=['Hold', data_type.capitalize()])
    print(report)

    # 7. 최종 모델 학습 및 저장
    print("\n[INFO] 전체 데이터로 최종 모델 학습 중...")
    final_X_eng = engineer_features(X_seq, combined_feature_names)
    final_scaler = StandardScaler()
    final_X_scaled = final_scaler.fit_transform(final_X_eng)
    final_X_scaled = pd.DataFrame(final_X_scaled, columns=final_X_eng.columns)
    
    final_model = lgb.LGBMClassifier(**model.get_params()) # 마지막 학습된 모델의 파라미터 사용
    final_model.fit(final_X_scaled, y)
    
    print("[INFO] 최종 모델 학습 완료.")
    show_feature_importance(final_model, final_X_eng.columns, data_type)
    save_model(final_model, final_scaler, data_type)
    
    final_metrics = {
        'f1': f1_score(all_val_labels, final_preds, zero_division=0),
        'precision': precision_score(all_val_labels, final_preds, zero_division=0),
        'recall': recall_score(all_val_labels, final_preds, zero_division=0)
    }
    
    return final_model, final_metrics

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
    if long_metrics:
        print(f"Long 모델  - F1: {long_metrics['f1']:.4f}, Precision: {long_metrics['precision']:.4f}, Recall: {long_metrics['recall']:.4f}")
    if short_metrics:
        print(f"Short 모델 - F1: {short_metrics['f1']:.4f}, Precision: {short_metrics['precision']:.4f}, Recall: {short_metrics['recall']:.4f}")
    
    return results

if __name__ == "__main__":
    results = main()
