import os
import json
import logging
from typing import Dict, Optional, Tuple
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# 이전에 정의한 클래스들을 import한다고 가정
from train_long import EnhancedImitationPretrainerLong
from train_short import EnhancedImitationPretrainerShort
# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directories():
    """필요한 디렉토리 구조 생성"""
    directories = [
        "new/data/raw",
        "new/data/label", 
        "new/data/buffer",
        "new/data/models/lgbm",
        "new/data/models/ppo_staging",
        "new/data/models/ppo",
        "new/data/models/hpo"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")

def create_objective_function(direction: str):
    """Optuna objective 함수 생성"""
    
    def objective(trial):
        """Optuna trial 실행 함수"""
        try:
            # 하이퍼파라미터 샘플링
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            kl_weight = trial.suggest_float('kl_weight', 0.1, 0.7)
            distribution_weight = trial.suggest_float('distribution_weight', 0.05, 0.3)
            curriculum_ratio = trial.suggest_float('curriculum_ratio', 0.3, 0.9)
            
            # 방향에 따른 경로 설정
            if direction.lower() == "long":
                bundle_path = "new/data/models/lgbm/lgbm_long.pkl"  # 수정된 경로
                model_save_path = f"new/data/models/ppo_staging/long_imitation_trial_{trial.number}.pt"
                trainer_class = EnhancedImitationPretrainerLong
            elif direction.lower() == "short":
                bundle_path = "new/data/models/lgbm/lgbm_short.pkl"  # 수정된 경로
                model_save_path = f"new/data/models/ppo_staging/short_imitation_trial_{trial.number}.pt"
                trainer_class = EnhancedImitationPretrainerShort
            else:
                raise ValueError(f"Invalid direction: {direction}. Must be 'long' or 'short'")
            
            logger.info(f"Trial {trial.number} - Testing hyperparameters:")
            logger.info(f"  learning_rate: {learning_rate:.6f}")
            logger.info(f"  kl_weight: {kl_weight:.3f}")
            logger.info(f"  distribution_weight: {distribution_weight:.3f}")
            logger.info(f"  curriculum_ratio: {curriculum_ratio:.3f}")
            
            # 트레이너 초기화
            trainer = trainer_class(
                bundle_path=bundle_path,
                model_save_path=model_save_path,
                learning_rate=learning_rate,
                kl_weight=kl_weight,
                distribution_weight=distribution_weight,
                curriculum_ratio=curriculum_ratio,
                max_steps=3000,  # HPO를 위해 단축된 학습
                patience=30      # 조기 종료를 위한 작은 patience
            )
            
            # 학습 실행
            results = trainer.run()
            
            # 최적화 목표: best validation accuracy
            score = results.get('best_val_accuracy', 0.0)
            
            logger.info(f"Trial {trial.number} - Score: {score:.4f}")
            logger.info(f"Trial {trial.number} - Results: {results}")
            
            # 임시 모델 파일 정리 (선택사항)
            if os.path.exists(model_save_path):
                try:
                    os.remove(model_save_path)
                except:
                    pass
            
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            return 0.0  # 실패 시 최저 점수
    
    return objective

def run_hpo(direction: str, n_trials: int = 30) -> Dict:
    """
    모방 학습 하이퍼파라미터 최적화 실행
    
    Args:
        direction: "long" 또는 "short"
        n_trials: 최적화 trial 횟수
    
    Returns:
        Dict: 최적 하이퍼파라미터 및 결과
    """
    
    # 방향 검증
    if direction.lower() not in ["long", "short"]:
        raise ValueError(f"Invalid direction: {direction}. Must be 'long' or 'short'")
    
    # 디렉토리 구조 확인
    ensure_directories()
    
    # 결과 저장 경로 설정 (새로운 구조에 맞게)
    result_path = f"new/data/models/hpo/ppo_{direction.lower()}_imitation_hpo.json"
    
    logger.info(f"Starting hyperparameter optimization for {direction.upper()} direction")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Result will be saved to: {result_path}")
    
    # Optuna study 생성
    study_name = f"imitation_hpo_{direction.lower()}"
    sampler = TPESampler(seed=42)  # 재현성을 위한 시드 설정
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        direction="maximize",  # best_val_accuracy 최대화
        sampler=sampler,
        pruner=pruner,
        study_name=study_name
    )
    
    # Objective 함수 생성
    objective = create_objective_function(direction)
    
    # 최적화 실행
    try:
        logger.info("=" * 60)
        logger.info(f"HYPERPARAMETER OPTIMIZATION STARTED - {direction.upper()}")
        logger.info("=" * 60)
        
        study.optimize(objective, n_trials=n_trials)
        
        logger.info("=" * 60)
        logger.info(f"HYPERPARAMETER OPTIMIZATION COMPLETED - {direction.upper()}")
        logger.info("=" * 60)
        
        # 최적 결과 정리
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        logger.info(f"Best trial number: {best_trial.number}")
        logger.info(f"Best validation accuracy: {best_score:.6f}")
        logger.info("Best hyperparameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # 결과 딕셔너리 생성
        hpo_results = {
            "model_type": "ppo_imitation",
            "direction": direction.lower(),
            "n_trials": n_trials,
            "best_trial_number": best_trial.number,
            "best_score": best_score,
            "best_params": best_params,
            "optimization_history": [
                {
                    "trial_number": trial.number,
                    "score": trial.value,
                    "params": trial.params
                }
                for trial in study.trials if trial.value is not None
            ],
            "study_statistics": {
                "total_trials": len(study.trials),
                "completed_trials": len([t for t in study.trials if t.value is not None]),
                "failed_trials": len([t for t in study.trials if t.value is None])
            }
        }
        
        # JSON으로 저장
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(hpo_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"HPO results saved to: {result_path}")
        
        # 최적화 히스토리 요약
        completed_trials = [t for t in study.trials if t.value is not None]
        failed_trials = [t for t in study.trials if t.value is None]
        
        logger.info("=" * 60)
        logger.info(f"OPTIMIZATION SUMMARY - {direction.upper()}")
        logger.info("=" * 60)
        logger.info(f"Total trials: {len(study.trials)}")
        logger.info(f"Completed trials: {len(completed_trials)}")
        logger.info(f"Failed trials: {len(failed_trials)}")
        logger.info(f"Best score: {best_score:.6f}")
        
        if len(completed_trials) > 1:
            scores = [t.value for t in completed_trials]
            logger.info(f"Score statistics:")
            logger.info(f"  Mean: {sum(scores)/len(scores):.6f}")
            logger.info(f"  Min: {min(scores):.6f}")
            logger.info(f"  Max: {max(scores):.6f}")
        
        return best_params
        
    except Exception as e:
        logger.error(f"HPO failed with error: {str(e)}")
        # 에러 발생 시에도 빈 결과 저장
        error_results = {
            "model_type": "ppo_imitation",
            "direction": direction.lower(),
            "n_trials": n_trials,
            "error": str(e),
            "best_params": None
        }
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(error_results, f, indent=2, ensure_ascii=False)
        
        raise

def load_best_params(direction: str) -> Optional[Dict]:
    """저장된 최적 하이퍼파라미터 로드"""
    result_path = f"new/data/models/hpo/ppo_{direction.lower()}_imitation_hpo.json"
    
    if not os.path.exists(result_path): 
        logger.warning(f"HPO result file not found: {result_path}")
        return None
    
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results.get('best_params')
    
    except Exception as e:
        logger.error(f"Failed to load HPO results: {str(e)}")
        return None

def run_with_best_params(direction: str) -> Dict:
    """
    최적 하이퍼파라미터로 최종 학습 실행
    
    Args:
        direction: "long" 또는 "short"
    
    Returns:
        Dict: 최종 학습 결과
    """
    
    # 최적 파라미터 로드
    best_params = load_best_params(direction)
    
    if best_params is None:
        logger.warning(f"No best parameters found for {direction}. Using default parameters.")
        best_params = {
            'learning_rate': 1e-3,
            'kl_weight': 0.3,
            'distribution_weight': 0.2,
            'curriculum_ratio': 0.7
        }
    
    logger.info(f"Running final training with best parameters for {direction.upper()}:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")
    
    # 방향에 따른 설정 (새로운 경로 구조)
    if direction.lower() == "long":
        bundle_path = "new/data/models/lgbm/lgbm_long.pkl"
        model_save_path = "new/data/models/ppo_staging/long_imitation.pt"
        trainer_class = EnhancedImitationPretrainerLong
    else:
        bundle_path = "new/data/models/lgbm/lgbm_short.pkl"
        model_save_path = "new/data/models/ppo_staging/short_imitation.pt"
        trainer_class = EnhancedImitationPretrainerShort
    
    # 최종 트레이너 초기화 및 실행
    trainer = trainer_class(
        bundle_path=bundle_path,
        model_save_path=model_save_path,
        **best_params,
        max_steps=5000,  # 최종 학습은 충분한 스텝
        patience=50      # 최종 학습은 충분한 patience
    )
    
    results = trainer.run()
    
    logger.info(f"Final training completed for {direction.upper()}")
    logger.info(f"Final results: {results}")
    
    return results

def run_sequential_hpo(n_trials: int = 30) -> Tuple[Dict, Dict]:
    """
    Long과 Short 방향 HPO를 순차적으로 실행
    
    Args:
        n_trials: 각 방향별 trial 횟수
    
    Returns:
        Tuple[Dict, Dict]: (long_best_params, short_best_params)
    """
    
    logger.info("=" * 80)
    logger.info("SEQUENTIAL HPO EXECUTION STARTED")
    logger.info("=" * 80)
    
    # 1. Long 방향 HPO 실행
    logger.info("🚀 Starting HPO for LONG direction...")
    try:
        best_params_long = run_hpo("long", n_trials=n_trials)
        logger.info(f"✅ LONG HPO completed successfully")
        logger.info(f"Best params for LONG: {best_params_long}")
    except Exception as e:
        logger.error(f"❌ LONG HPO failed: {str(e)}")
        best_params_long = None
    
    # 2. Short 방향 HPO 실행
    logger.info("\n🚀 Starting HPO for SHORT direction...")
    try:
        best_params_short = run_hpo("short", n_trials=n_trials)
        logger.info(f"✅ SHORT HPO completed successfully")
        logger.info(f"Best params for SHORT: {best_params_short}")
    except Exception as e:
        logger.error(f"❌ SHORT HPO failed: {str(e)}")
        best_params_short = None
    
    # 3. 전체 결과 요약
    logger.info("=" * 80)
    logger.info("SEQUENTIAL HPO EXECUTION COMPLETED")
    logger.info("=" * 80)
    
    if best_params_long:
        logger.info("✅ LONG direction HPO: SUCCESS")
        logger.info(f"   Best params: {best_params_long}")
    else:
        logger.info("❌ LONG direction HPO: FAILED")
    
    if best_params_short:
        logger.info("✅ SHORT direction HPO: SUCCESS")
        logger.info(f"   Best params: {best_params_short}")
    else:
        logger.info("❌ SHORT direction HPO: FAILED")
    
    # 결과 파일 경로 정보
    logger.info("\n📁 HPO Results saved to:")
    logger.info(f"   LONG: new/data/models/hpo/ppo_long_imitation_hpo.json")
    logger.info(f"   SHORT: new/data/models/hpo/ppo_short_imitation_hpo.json")
    
    return best_params_long, best_params_short

def run_final_training_both_directions():
    """
    Long과 Short 양방향 최종 학습 실행
    """
    
    logger.info("=" * 80)
    logger.info("FINAL TRAINING WITH BEST PARAMETERS")
    logger.info("=" * 80)
    
    # Long 방향 최종 학습
    logger.info("🏁 Running final training for LONG direction...")
    try:
        final_results_long = run_with_best_params("long")
        logger.info("✅ LONG final training completed")
    except Exception as e:
        logger.error(f"❌ LONG final training failed: {str(e)}")
        final_results_long = None
    
    # Short 방향 최종 학습
    logger.info("\n🏁 Running final training for SHORT direction...")
    try:
        final_results_short = run_with_best_params("short")
        logger.info("✅ SHORT final training completed")
    except Exception as e:
        logger.error(f"❌ SHORT final training failed: {str(e)}")
        final_results_short = None
    
    logger.info("=" * 80)
    logger.info("FINAL TRAINING COMPLETED")
    logger.info("=" * 80)
    
    return final_results_long, final_results_short

# 사용 예시 및 메인 실행부
if __name__ == "__main__":
    # 전체 파이프라인 실행
    print("🚀 Starting complete HPO pipeline...")
    
    # 1. 순차적 HPO 실행 (Long -> Short)
    best_params_long, best_params_short = run_sequential_hpo(n_trials=20)
    
    # 2. 선택사항: 최적 파라미터로 최종 학습 실행
    user_input = input("\n🤔 Run final training with best parameters? (y/n): ")
    if user_input.lower() in ['y', 'yes']:
        print("🏁 Running final training...")
        final_results_long, final_results_short = run_final_training_both_directions()
        print("✅ Complete pipeline finished!")
    else:
        print("✅ HPO pipeline completed. Final training skipped.")
    
    print("\n📊 Summary:")
    print(f"   HPO Results: new/data/models/hpo/")
    print(f"   Models: new/data/models/ppo_staging/")
    print("   Ready for next pipeline stage! 🎉")