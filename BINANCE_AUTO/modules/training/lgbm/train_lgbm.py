import pandas as pd
import numpy as np
import os
import sys
import joblib
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

sys.path.append(PROJECT_ROOT)

from modules.config import TRAIN_LABEL_PATHS, LGBM_THRESHOLD

def load_data(data_type="long"):
    """Long/Short 이진분류 데이터 로딩"""
    if data_type == "long":
        data_path = TRAIN_LABEL_PATHS["long"]
    else:  # short
        data_path = TRAIN_LABEL_PATHS["short"]
    
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"[📊 {data_type.upper()} 데이터 로딩 완료] 행: {len(df)}, 컬럼: {len(df.columns)}")
    return df

def generate_flatten_features(df, window=32):
    print(f"[🔄 Flatten 피처 생성 시작] window={window}")

    # ✅ 결측값 먼저 채우기 (ffill)
    df = df.ffill()

    # ✅ 미래 정보 포함 가능성 있는 열 제거
    future_keywords = ['label', 'target', 'tp_hit', 'sl_hit', 'next_', 'forward_']
    feature_cols = [col for col in df.columns if not any(keyword in col.lower() for keyword in future_keywords)]

    flatten_dfs = []
    for col in feature_cols:
        for i in range(window):  # 0부터 시작
            shifted_col = df[col].shift(i + 1)  # ✅ t-1 ~ t-32 로 변경
            new_col_name = f"{col}_t-{i+1}"     # ✅ 이름도 시점 반영
            flatten_dfs.append(shifted_col.rename(new_col_name))

    X_flatten = pd.concat(flatten_dfs, axis=1)
    X_flatten = X_flatten.dropna()

    print(f"  - Flatten 후 피처 수: {len(X_flatten.columns)}개")
    print(f"  - Flatten 후 데이터 행: {len(X_flatten)}개")
    print(f"  - 피처명 예시: {X_flatten.columns[:5].tolist()}")

    return X_flatten

def optimize_model(X_train, y_train, X_val, y_val, data_type="long"):
    """Optuna 하이퍼파라미터 최적화"""
    print(f"[🔧 {data_type.upper()} 하이퍼파라미터 최적화 시작] (threshold={LGBM_THRESHOLD} 고정)")
    
    results = []
    
    def objective(trial):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05),
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 40, 100),
            "n_estimators": 1500,
            "random_state": 42,
            "class_weight": "balanced",
            "verbosity": -1,
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False)]
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
    study.optimize(objective, n_trials=100, show_progress_bar=False)
    
    print(f"[✅ {data_type.upper()} 최적화 완료] 최고 F1: {study.best_value:.4f}")
    print(f"  최적 파라미터: {study.best_params}")
    
    return study, results

def train_model(X_train, y_train, X_valid, y_valid, data_type="long", use_optuna=False):
    """LGBM 모델 학습 (Optuna 옵션)"""
    if use_optuna:
        print(f"[🚀 {data_type.upper()} LGBM 모델 학습 시작] (Optuna 최적화 포함)")
        
        # Optuna 최적화
        study, optuna_results = optimize_model(X_train, y_train, X_valid, y_valid, data_type)
        
        # 최적 파라미터로 최종 모델 학습
        best_params = study.best_params
        best_params.update({
            "objective": "binary",
            "boosting_type": "gbdt", 
            "n_estimators": 1500,
            "class_weight": "balanced",
            "random_state": 42,
            "verbosity": -1
        })
        
        model = lgb.LGBMClassifier(**best_params)
        
    else:
        print(f"[🚀 {data_type.upper()} LGBM 모델 학습 시작] (기본 파라미터)")
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbosity=-1
        )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(10, verbose=False)]
    )
    
    print(f"[✅ {data_type.upper()} 모델 학습 완료]")
    return model

def evaluate_model(model, X_valid, y_valid, data_type="long"):
    """모델 평가"""
    print(f"[🧪 {data_type.upper()} 모델 평가 중...]")
    
    # 확률 예측
    y_prob = model.predict_proba(X_valid)[:, 1]
    
    # Threshold 기반 예측
    y_pred = (y_prob >= LGBM_THRESHOLD).astype(int)
    
    # 메트릭 계산
    f1 = f1_score(y_valid, y_pred)
    precision = precision_score(y_valid, y_pred, zero_division=0)
    recall = recall_score(y_valid, y_pred, zero_division=0)
    signal_rate = (y_prob >= LGBM_THRESHOLD).mean()
    
    signal_label = "Long" if data_type == "long" else "Short"
    
    print(f"[📊 {data_type.upper()} 평가 결과] (threshold={LGBM_THRESHOLD})")
    print(f"  - F1-Score: {f1:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - 신호율: {signal_rate:.2%}")
    
    # 분류 리포트
    report = classification_report(y_valid, y_pred, target_names=['Hold', signal_label])
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

def save_model(model, metrics, data_type="long"):
    """모델만 저장 (메타데이터는 로그 출력)"""
    # 저장 디렉토리 생성
    model_dir = os.path.join(PROJECT_ROOT, "models", "lgbm")
    os.makedirs(model_dir, exist_ok=True)
    
    # 모델만 저장
    model_path = os.path.join(model_dir, f"lgbm_{data_type}.pkl")
    joblib.dump(model, model_path)
    
    print(f"[💾 {data_type.upper()} 모델 저장 완료] {model_path}")
    
    # 메타데이터는 로그로만 출력
    print(f"[📋 {data_type.upper()} 모델 메타데이터]")
    print(f"  - Threshold: {LGBM_THRESHOLD}")
    print(f"  - F1-Score: {metrics['f1']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"  - Signal Rate: {metrics['signal_rate']:.2%}")

def show_feature_importance(model, feature_names, data_type="long"):
    """피처 중요도 로그 출력"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"[🎯 {data_type.upper()} 상위 10개 중요 피처]")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

def train_pipeline(data_type="long", use_optuna=True):
    """LGBM 학습 파이프라인 (Long 또는 Short)"""
    print(f"\n{'='*60}")
    print(f"🚀 {data_type.upper()} 모델 학습 시작")
    print(f"🔧 Optuna 하이퍼파라미터 최적화 활성화")
    print(f"{'='*60}")
    
    # 1. 데이터 로딩
    df = load_data(data_type)
    
    # 2. Flatten 피처 생성 (NaN 처리 포함)
    X = generate_flatten_features(df, window=32)
    y = df.loc[X.index, 'label']
    
    # 3. 피처 준비 완료 로그
    print(f"[🔧 {data_type.upper()} 피처 준비 완료]")
    print(f"  - 전체 피처 수: {len(X.columns)}개")
    print(f"  - 피처 예시: {X.columns[:10].tolist()}")
    print(f"  - 라벨 분포: {data_type}={sum(y)}, hold={len(y)-sum(y)}")
    
    # 4. 데이터 분할
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42,
        shuffle=True
    )
    
    print(f"[📊 {data_type.upper()} 데이터 분할] 학습: {len(X_train)}, 검증: {len(X_valid)}")
    
    # 5. 모델 학습 (Optuna 최적화)
    model = train_model(X_train, y_train, X_valid, y_valid, data_type, use_optuna)
    
    # 6. 모델 평가
    metrics = evaluate_model(model, X_valid, y_valid, data_type)
    
    # 7. 피처 중요도 분석
    show_feature_importance(model, X.columns.tolist(), data_type)
    
    # 8. 모델 저장
    save_model(model, metrics, data_type)
    
    print(f"\n[🎉 {data_type.upper()} 학습 완료]")
    print(f"   F1-Score: {metrics['f1']:.4f}")
    print(f"   {data_type.title()} 신호율: {metrics['signal_rate']:.2%}")
    
    return model, metrics

def main():
    """메인 실행 함수 - Long/Short 모델 모두 학습"""
    print("=" * 80)
    print("🚀 LGBM Long/Short 이진분류 모델 학습 시작")
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
    print(f"🎉 LGBM Long/Short 모델 학습 완료 요약")
    print(f"{'=' * 80}")
    print(f"📊 Long 모델  - F1: {long_metrics['f1']:.4f}, 신호율: {long_metrics['signal_rate']:.2%}")
    print(f"📊 Short 모델 - F1: {short_metrics['f1']:.4f}, 신호율: {short_metrics['signal_rate']:.2%}")
    print(f"\n💾 저장된 모델:")
    print(f"   ✅ models/lgbm/lgbm_long.pkl")
    print(f"   ✅ models/lgbm/lgbm_short.pkl")
    print(f"{'=' * 80}")
    
    return results

if __name__ == "__main__":
    # LGBM Long/Short 이진분류 모델 학습 (Optuna 최적화 포함)
    results = main()