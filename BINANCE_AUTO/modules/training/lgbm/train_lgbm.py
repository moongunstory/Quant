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

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

sys.path.append(PROJECT_ROOT)

from modules.config import (
    TRAIN_LABEL_PATHS, 
    LGBM_THRESHOLD, 
    LGBM_MODEL_PATHS,
    TIMEFRAMES,
    FEATURE_CATEGORIES_BY_TF,
    SEQ_LEN,
    RAW_DATA_PATH
)

def load_mtf_data() -> Dict[str, pd.DataFrame]:
    """MTF 개별 pickle 파일들을 로드"""
    save_dir = os.path.dirname(os.path.join(PROJECT_ROOT, RAW_DATA_PATH))
    mtf_data = {}
    
    # 각 타임프레임 + BTC/DUNE 데이터 로드
    data_keys = TIMEFRAMES + ["btc", "dune"]
    
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
    data_path = TRAIN_LABEL_PATHS[data_type]
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"[📊 {data_type.upper()} 라벨 데이터 로딩 완료] 행: {len(df)}, 컬럼: {len(df.columns)}")
    return df

def filter_features_by_timeframe(df: pd.DataFrame, timeframe: str) -> List[str]:
    """타임프레임별 피처 필터링"""
    if timeframe in FEATURE_CATEGORIES_BY_TF:
        allowed_features = FEATURE_CATEGORIES_BY_TF[timeframe]
        available_features = [col for col in df.columns if any(feat in col for feat in allowed_features)]
    else:
        # 모든 피처 사용 (타임프레임별 카테고리가 정의되지 않은 경우)
        available_features = df.columns.tolist()
    
    # 미래 정보 제거
    future_keywords = ['label', 'target', 'tp_hit', 'sl_hit', 'next_', 'forward_']
    filtered_features = [col for col in available_features 
                        if not any(keyword in col.lower() for keyword in future_keywords)]
    
    return filtered_features

def create_mtf_sequences(mtf_data: Dict[str, pd.DataFrame], 
                         target_index: pd.DatetimeIndex) -> Tuple[Dict[str, np.ndarray], List[pd.Timestamp]]:
    """MTF 시퀀스 데이터 생성 (고속/저메모리 최적화)"""
    mtf_sequences = {}
    final_valid_indices = None  # 공통으로 존재하는 인덱스만 유지

    for timeframe in TIMEFRAMES:
        if timeframe not in mtf_data:
            print(f"[경고] {timeframe} 데이터가 없습니다.")
            continue

        df = mtf_data[timeframe].copy()

        # 피처 선택 및 전처리
        feature_cols = filter_features_by_timeframe(df, timeframe)
        df = df[feature_cols].ffill().fillna(0)

        print(f"[🔧 {timeframe}] 피처 수: {len(feature_cols)}, 시퀀스 길이: {SEQ_LEN}")

        # 미리 인덱스 및 값 추출
        index_list = df.index.to_list()
        values = df.values
        sequences = []
        valid_indices = []

        for t in target_index:
            if df.index.tz is not None:
                t = t.tz_convert(df.index.tz)
            else:
                t = t.tz_localize(None)

            pos = bisect.bisect_right(index_list, t)

            if pos >= SEQ_LEN:
                seq = values[pos - SEQ_LEN:pos]
                sequences.append(seq)
                valid_indices.append(index_list[pos - 1])
            elif pos > 0:
                pad_len = SEQ_LEN - pos
                padded = np.vstack([
                    np.zeros((pad_len, values.shape[1])),
                    values[:pos]
                ])
                sequences.append(padded)
                valid_indices.append(index_list[pos - 1])
            # else: 데이터가 하나도 없는 경우 무시

        if sequences:
            mtf_sequences[timeframe] = np.array(sequences)
            print(f"[✅ {timeframe}] 시퀀스 생성 완료: {mtf_sequences[timeframe].shape}")

            if final_valid_indices is None:
                final_valid_indices = valid_indices
            else:
                final_valid_indices = list(set(final_valid_indices) & set(valid_indices))
        else:
            print(f"[❌ {timeframe}] 유효한 시퀀스가 없습니다.")

    return mtf_sequences, sorted(final_valid_indices)

def flatten_mtf_sequences(mtf_sequences: Dict[str, np.ndarray]) -> np.ndarray:
    """MTF 시퀀스를 LGBM용 평면 피처로 변환"""
    flattened_features = []
    feature_names = []
    
    for timeframe in TIMEFRAMES:
        if timeframe not in mtf_sequences:
            continue
        
        sequences = mtf_sequences[timeframe]  # Shape: (N, SEQ_LEN, features)
        n_samples, seq_len, n_features = sequences.shape
        
        # 시퀀스를 평면화 (각 타임스텝과 피처를 별도 컬럼으로)
        flat_data = sequences.reshape(n_samples, -1)  # Shape: (N, SEQ_LEN * features)
        flattened_features.append(flat_data)
        
        # 피처명 생성
        for t in range(seq_len):
            for f in range(n_features):
                feature_names.append(f"{timeframe}_t{t}_f{f}")
    
    if flattened_features:
        X_flat = np.concatenate(flattened_features, axis=1)
        print(f"[🔄 MTF 평면화 완료] Shape: {X_flat.shape}, 피처 수: {len(feature_names)}")
        return X_flat, feature_names
    else:
        raise ValueError("평면화할 MTF 시퀀스가 없습니다.")

def create_time_based_split(X: np.ndarray, y: np.ndarray, 
                           timestamps: pd.DatetimeIndex, 
                           train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """시간 기반 train/validation 분할 (셔플 금지)"""
    n_samples = len(X)
    train_size = int(n_samples * train_ratio)
    
    # 시간 순서를 유지하며 분할
    X_train = X[:train_size]
    X_val = X[train_size:]
    y_train = y[:train_size]
    y_val = y[train_size:]

    timestamps = timestamps[:len(X)]
    train_period = f"{timestamps[0].strftime('%Y-%m-%d')} ~ {timestamps[train_size-1].strftime('%Y-%m-%d')}"
    val_period = f"{timestamps[train_size].strftime('%Y-%m-%d')} ~ {timestamps[-1].strftime('%Y-%m-%d')}"
    
    print(f"[📊 시간 기반 분할]")
    print(f"  - 훈련: {len(X_train)}개 ({train_period})")
    print(f"  - 검증: {len(X_val)}개 ({val_period})")
    
    return X_train, X_val, y_train, y_val

def optimize_model(X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray, 
                  data_type: str = "long") -> Tuple:
    """Optuna 하이퍼파라미터 최적화"""
    print(f"[🔧 {data_type.upper()} 하이퍼파라미터 최적화 시작] (threshold={LGBM_THRESHOLD} 고정)")
    
    results = []
    
    def objective(trial):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "n_estimators": 2000,
            "random_state": 42,
            "class_weight": "balanced",
            "verbosity": -1,
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        # 평가
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= LGBM_THRESHOLD).astype(int)
        
        recall = recall_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results.append({
            "trial": trial.number,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            **params
        })
        
        if trial.number % 20 == 0:
            print(f"  Trial {trial.number}: F1={f1:.4f}, Recall={recall:.4f}, Precision={precision:.4f}")
        
        return f1
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    
    print(f"[✅ {data_type.upper()} 최적화 완료] 최고 F1: {study.best_value:.4f}")
    print(f"  최적 파라미터: {study.best_params}")
    
    return study, results

def train_model(X_train: np.ndarray, y_train: np.ndarray, 
               X_val: np.ndarray, y_val: np.ndarray, 
               data_type: str = "long", use_optuna: bool = True) -> lgb.LGBMClassifier:
    """LGBM 모델 학습"""
    if use_optuna:
        print(f"[🚀 {data_type.upper()} LGBM 모델 학습 시작] (Optuna 최적화 포함)")
        
        # Optuna 최적화
        study, optuna_results = optimize_model(X_train, y_train, X_val, y_val, data_type)
        
        # 최적 파라미터로 최종 모델 학습
        best_params = study.best_params
        best_params.update({
            "objective": "binary",
            "boosting_type": "gbdt", 
            "n_estimators": 2000,
            "class_weight": "balanced",
            "random_state": 42,
            "verbosity": -1
        })
        
        model = lgb.LGBMClassifier(**best_params)
        
    else:
        print(f"[🚀 {data_type.upper()} LGBM 모델 학습 시작] (기본 파라미터)")
        
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            class_weight="balanced",
            random_state=42,
            verbosity=-1
        )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    
    print(f"[✅ {data_type.upper()} 모델 학습 완료]")
    return model

def evaluate_model(model: lgb.LGBMClassifier, X_val: np.ndarray, y_val: np.ndarray, 
                  data_type: str = "long") -> Dict:
    """모델 평가"""
    print(f"[🧪 {data_type.upper()} 모델 평가 중...]")
    
    # 확률 예측
    y_prob = model.predict_proba(X_val)[:, 1]
    
    # Threshold 기반 예측
    y_pred = (y_prob >= LGBM_THRESHOLD).astype(int)
    
    # 메트릭 계산
    f1 = f1_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    signal_rate = (y_prob >= LGBM_THRESHOLD).mean()
    
    signal_label = "Long" if data_type == "long" else "Short"
    
    print(f"[📊 {data_type.upper()} 평가 결과] (threshold={LGBM_THRESHOLD})")
    print(f"  - F1-Score: {f1:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - 신호율: {signal_rate:.2%}")
    
    # 분류 리포트
    report = classification_report(y_val, y_pred, target_names=['Hold', signal_label])
    print(f"[📋 {data_type.upper()} 분류 리포트]\n{report}")
    
    # 확률 분포 확인
    print(f"[🔍 {data_type.upper()} 확률 분포]")
    print(f"  - 평균: {y_prob.mean():.3f}")
    print(f"  - 표준편차: {y_prob.std():.3f}")
    print(f"  - 최소값: {y_prob.min():.3f}")
    print(f"  - 최대값: {y_prob.max():.3f}")
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'signal_rate': signal_rate
    }

def save_model(model: lgb.LGBMClassifier, metrics: Dict, data_type: str = "long"):
    """모델 저장"""
    model_path = LGBM_MODEL_PATHS[data_type]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    print(f"[💾 {data_type.upper()} 모델 저장 완료] {model_path}")
    
    # 메타데이터 로깅
    print(f"[📋 {data_type.upper()} 모델 메타데이터]")
    print(f"  - Threshold: {LGBM_THRESHOLD}")
    print(f"  - F1-Score: {metrics['f1']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - Signal Rate: {metrics['signal_rate']:.2%}")
    print(f"  - Timeframes: {TIMEFRAMES}")
    print(f"  - Sequence Length: {SEQ_LEN}")

def show_feature_importance(model: lgb.LGBMClassifier, feature_names: List[str], 
                           data_type: str = "long", top_k: int = 20):
    """피처 중요도 분석"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"[🎯 {data_type.upper()} 상위 {top_k}개 중요 피처]")
    for i, (_, row) in enumerate(importance_df.head(top_k).iterrows()):
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")
    
    # 타임프레임별 중요도 집계
    tf_importance = {}
    for tf in TIMEFRAMES:
        tf_features = importance_df[importance_df['feature'].str.startswith(tf)]
        tf_importance[tf] = tf_features['importance'].sum()
    
    print(f"\n[📊 {data_type.upper()} 타임프레임별 피처 중요도]")
    for tf, importance in sorted(tf_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tf}: {importance:.4f}")

def train_pipeline(data_type: str = "long", use_optuna: bool = True) -> Tuple:
    """MTF LGBM 학습 파이프라인"""
    print(f"\n{'='*60}")
    print(f"🚀 {data_type.upper()} MTF LGBM 모델 학습 시작")
    print(f"🕒 지원 Timeframes: {TIMEFRAMES}")
    print(f"📏 Sequence Length: {SEQ_LEN}")
    print(f"{'='*60}")
    
    # 1. MTF 데이터 로딩
    mtf_data = load_mtf_data()
    
    # 2. 라벨 데이터 로딩
    labeled_df = load_labeled_data(data_type)
    
    # 3. MTF 시퀀스 생성 (PPO와 동일한 방식)
    mtf_sequences, valid_indices = create_mtf_sequences(mtf_data, labeled_df.index)
    
    # 4. 유효한 인덱스에 맞춰 라벨 정렬
    valid_timestamps = pd.DatetimeIndex(valid_indices)
    aligned_labels = labeled_df.loc[valid_timestamps, 'label'].values
    
    print(f"[🔧 {data_type.upper()} 데이터 정렬 완료]")
    print(f"  - 유효 샘플 수: {len(aligned_labels)}")
    print(f"  - 라벨 분포: {data_type}={sum(aligned_labels)}, hold={len(aligned_labels)-sum(aligned_labels)}")
    
    # 5. MTF 시퀀스를 LGBM용 평면 피처로 변환
    X_flat, feature_names = flatten_mtf_sequences(mtf_sequences)

    # 라벨 데이터프레임의 인덱스에서 valid_timestamps가 위치한 행을 찾아
    # 해당 위치의 피처만 선택한다. get_indexer는 각 타임스탬프의 위치를
    # 반환하며, 존재하지 않는 경우 -1을 반환하므로 필터링 전에 검증한다.
    valid_indices_pos = labeled_df.index.get_indexer(valid_timestamps)
    if (valid_indices_pos < 0).any():
        raise ValueError("일부 유효 타임스탬프가 라벨 데이터프레임에 존재하지 않습니다.")

    X_flat = X_flat[valid_indices_pos]
    y = aligned_labels

    # 피처와 라벨의 행 수가 일치하는지 확인
    assert X_flat.shape[0] == len(aligned_labels), "Features and labels size mismatch"
    
    # 6. 시간 기반 train/validation 분할 (셔플 금지)
    X_train, X_val, y_train, y_val = create_time_based_split(
        X_flat, y, valid_timestamps, train_ratio=0.8
    )
    
    # 7. 모델 학습
    model = train_model(X_train, y_train, X_val, y_val, data_type, use_optuna)
    
    # 8. 모델 평가
    metrics = evaluate_model(model, X_val, y_val, data_type)
    
    # 9. 피처 중요도 분석
    show_feature_importance(model, feature_names, data_type)
    
    # 10. 모델 저장
    save_model(model, metrics, data_type)
    
    print(f"\n[🎉 {data_type.upper()} MTF LGBM 학습 완료]")
    print(f"   F1-Score: {metrics['f1']:.4f}")
    print(f"   {data_type.title()} 신호율: {metrics['signal_rate']:.2%}")
    
    return model, metrics

def main():
    """메인 실행 함수 - Long/Short MTF LGBM 모델 모두 학습"""
    print("=" * 80)
    print("🚀 MTF LGBM Long/Short 이진분류 모델 학습 시작")
    print(f"🕒 지원 Timeframes: {TIMEFRAMES}")
    print(f"📏 Sequence Length: {SEQ_LEN}")
    print("=" * 80)
    
    results = {}
    
    # Long 모델 학습
    print("\n" + "🟢 " * 30 + " LONG 모델 " + "🟢 " * 30)
    long_model, long_metrics = train_pipeline("long", use_optuna=True)
    results["long"] = {"model": long_model, "metrics": long_metrics}
    
    # Short 모델 학습  
    print("\n" + "🔴 " * 30 + " SHORT 모델 " + "🔴 " * 30)
    short_model, short_metrics = train_pipeline("short", use_optuna=True)
    results["short"] = {"model": short_model, "metrics": short_metrics}
    
    # 전체 요약
    print(f"\n{'=' * 80}")
    print(f"🎉 MTF LGBM Long/Short 모델 학습 완료 요약")
    print(f"{'=' * 80}")
    print(f"📊 Long 모델  - F1: {long_metrics['f1']:.4f}, 신호율: {long_metrics['signal_rate']:.2%}")
    print(f"📊 Short 모델 - F1: {short_metrics['f1']:.4f}, 신호율: {short_metrics['signal_rate']:.2%}")
    print(f"\n💾 저장된 모델:")
    print(f"   ✅ {LGBM_MODEL_PATHS['long']}")
    print(f"   ✅ {LGBM_MODEL_PATHS['short']}")
    print(f"\n🔧 MTF 설정:")
    print(f"   📊 Timeframes: {TIMEFRAMES}")
    print(f"   📏 Sequence Length: {SEQ_LEN}")
    print(f"   🎯 Threshold: {LGBM_THRESHOLD}")
    print(f"{'=' * 80}")
    
    return results

if __name__ == "__main__":
    # MTF LGBM Long/Short 이진분류 모델 학습
    results = main()