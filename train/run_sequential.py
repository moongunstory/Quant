
import sys
import os

# Add the parent directory of train to the Python path to allow imports like reinforce.manager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train import fetch
from train import fe
from train.reinforce import manager
from train.reinforce import worker

def run_all_training_steps():
    print("모든 학습 단계를 순차적으로 실행합니다...")
    
    print("\n1. 데이터 가져오기 (fetch)...")
    fetch.main_fetch()
    
    print("\n2. 특징 공학 (feature engineering) 실행 (fe)...")
    fe.main_fe()
    
    print("\n3. 매니저 모델 학습 (manager)...")
    manager.main_manager()
    
    print("\n4. 워커 모델 학습 (worker)...")
    worker.main_worker()
    
    print("\n모든 학습 단계가 완료되었습니다.")

if __name__ == "__main__":
    run_all_training_steps()
