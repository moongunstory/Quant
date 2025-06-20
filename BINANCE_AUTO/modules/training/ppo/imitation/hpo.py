import os
import sys
import json
import optuna
import torch
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
sys.path.append(PROJECT_ROOT)

from modules.training.ppo.imitation import train_imitation

HPO_DIR = os.path.join(PROJECT_ROOT, 'data', 'models', 'hpo')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'data', 'models', 'ppo_staging')
os.makedirs(HPO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 각 방향별 최적 성능 추적용 전역 변수
best_performances = {}


def objective(trial, direction: str):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_int('batch_size', 256, 2048, step=256)
    epochs = trial.suggest_int('epochs', 3, 15)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256, step=32)

    # 디버깅을 위해 return_metrics=True 설정
    metrics = train_imitation.train(
        direction=direction,
        lr=lr,
        batch_size=batch_size,
        epochs=epochs,
        hidden_dim=hidden_dim,
        save_model=False,
        return_metrics=True
    )

    performance_score = metrics.get('val_loss', 999.0)  # fallback

    # 디버깅 로그 추가
    print(f"[TRIAL RESULT] Direction: {direction.upper()}, "
        f"Loss: {metrics['val_loss']:.6f}, Acc: {metrics['accuracy']:.4f}, "
        f"F1: {metrics['f1_score']:.4f}, Total Return: {metrics.get('total_return', 0):.4f}")

    # 클래스 분포 출력 (확률 예측이 편향됐는지 확인)
    if 'pred_distribution' in metrics:
        pred_ratio = metrics['pred_distribution']
        print(f"🔍 Predicted class distribution: [0]: {pred_ratio[0]*100:.2f}%, [1]: {pred_ratio[1]*100:.2f}%")
    
    # 베스트 성능 체크 및 모델 저장
    if direction not in best_performances or performance_score < best_performances[direction]:
        best_performances[direction] = performance_score
        print(f"[🎯 NEW BEST] Direction: {direction.upper()}, Score: {performance_score:.6f} (Improved!)")
        
        # 베스트 모델 저장 (기존 방식)
        try:
            train_imitation.train(
                direction=direction,
                lr=lr,
                batch_size=batch_size,
                epochs=epochs,
                hidden_dim=hidden_dim,
                save_model=True
            )
            print(f"[💾 MODEL SAVED] Best model saved for {direction.upper()}")
        except Exception as e:
            print(f"[WARNING] Failed to save best model: {e}")
    else:
        print(f"[NO IMPROVEMENT] Score: {performance_score:.6f}")
    
    # Optuna는 minimize 방향이므로 최소화할 값 반환
    return performance_score


def calculate_composite_score(results):
    """
    복합 성능 지표 계산
    낮을수록 좋은 점수 (minimize 방향)
    """
    # 가중치 설정
    weights = {
        'validation_loss': 0.3,      # 검증 손실 (낮을수록 좋음)
        'accuracy': -0.2,            # 정확도 (높을수록 좋음, 음수 가중치)
        'f1_score': -0.2,           # F1 점수 (높을수록 좋음, 음수 가중치)
        'sharpe_ratio': -0.15,      # 샤프 비율 (높을수록 좋음, 음수 가중치)
        'max_drawdown': 0.1,        # 최대 손실 (낮을수록 좋음, 절댓값)
        'total_return': -0.05       # 총 수익률 (높을수록 좋음, 음수 가중치)
    }
    
    score = 0
    for metric, weight in weights.items():
        if metric in results:
            value = results[metric]
            if metric == 'max_drawdown':
                value = abs(value)  # 최대 손실은 절댓값 사용
            score += weight * value
    
    return score


def run_hpo(direction: str, n_trials: int = 20):
    print(f"\n{'='*60}")
    print(f"[HPO START] Direction: {direction.upper()}, Total Trials: {n_trials}")
    print(f"{'='*60}")
    
    # TPESampler with seed=42 사용
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # 방향별 best 성능 초기화
    best_performances[direction] = float('inf')
    
    def callback(study, trial):
        # 각 trial 완료 시마다 현재 best trial과 성능 출력
        current_trial = len(study.trials)
        current_best_score = study.best_value if study.best_value is not None else float('inf')
        
        print(f"\n[HPO PROGRESS] Direction: {direction.upper()}, Trial {current_trial}/{n_trials}")
        print(f"  ➜ Current Best Score: {current_best_score:.6f}")
        print(f"  ➜ Progress: {current_trial/n_trials*100:.1f}%")
        
        if current_trial > 1:
            # 최근 몇 개 trial의 개선 추세 출력
            recent_values = [t.value for t in study.trials[-3:] if t.value is not None]
            if len(recent_values) >= 2:
                trend = "📈" if recent_values[-1] < recent_values[0] else "📉"
                print(f"  ➜ Recent Trend: {trend} (Last 3 trials: {[f'{v:.4f}' for v in recent_values]})")
        print("-" * 60)
    
    # HPO 실행
    study.optimize(lambda trial: objective(trial, direction), n_trials=n_trials, callbacks=[callback])
    
    # HPO 완료 후 결과 출력
    print(f"\n{'='*60}")
    print(f"[🎉 HPO COMPLETED] Direction: {direction.upper()}")
    print(f"{'='*60}")
    print(f"Best Composite Score: {study.best_value:.6f}")
    print(f"Best Params: {study.best_params}")
    print(f"Total Trials: {len(study.trials)}")
    
    # 성능 개선 통계
    all_values = [t.value for t in study.trials if t.value is not None]
    improvement = 0
    if len(all_values) > 1:
        improvement = ((all_values[0] - study.best_value) / all_values[0]) * 100
        print(f"Improvement: {improvement:.2f}% (from {all_values[0]:.6f} to {study.best_value:.6f})")
    
    # 결과 저장 (JSON만 저장, 모델은 저장 안 함)
    result_path = os.path.join(HPO_DIR, f'ppo_{direction}_imitation_hpo_enhanced.json')
    result_data = {
        'direction': direction,
        'best_composite_score': study.best_value,
        'best_params': study.best_params,
        'n_trials': n_trials,
        'improvement_pct': improvement,
        'all_trial_values': all_values[:10],
        'evaluation_metrics': [
            'val_loss', 'accuracy', 'f1_score', 
            'sharpe_ratio', 'max_drawdown', 'total_return'
        ],
        'note': 'HPO completed - use these params for final model training'
    }
    
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"[💾 JSON SAVED] HPO results saved to: {result_path}")
    print(f"[📝 NEXT STEP] Run train_imitation.py to train final model with best params")
    print(f"{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced HPO for PPO Imitation Learning')
    parser.add_argument('--direction', choices=['long', 'short'], default=None,
                       help='Trading direction (if not specified, runs both long and short)')
    parser.add_argument('--n_trials', type=int, default=20,
                       help='Number of trials for HPO (default: 20)')
    args = parser.parse_args()
    
    print("=== Enhanced HPO Framework (Currently using Loss only) ===")
    print("Ready for: Loss, Accuracy, F1-Score, Sharpe Ratio, Max Drawdown, Total Return")
    print("Current: Training Loss optimization (expandable framework)")
    print("=" * 70)
    
    if args.direction:
        run_hpo(args.direction, args.n_trials)
    else:
        print("=== Starting HPO for both LONG and SHORT directions ===")
        run_hpo('long', args.n_trials)
        run_hpo('short', args.n_trials)
        print("\n=== All Enhanced HPO completed ===")


if __name__ == '__main__':
    main()