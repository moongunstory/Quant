import time
import torch
import traceback
import os
import shutil
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any

from modules.trading_executor import FuturesTradeExecutor
from modules.ppo_runtime.predictor import Predictor
from modules.data_collector import RealTimeDataCollector
from modules.ppo_runtime.rollout_updater import RolloutBuffer
from modules.ppo_runtime.env_live import LivePPOEnv
from modules.ppo_runtime.train_ppo_live import train_ppo_live
from modules.config import (
    PPO_FINAL_MODEL_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH,
    PPO_BUFFER_PATHS,
    PPO_CONFIG,
    TIMEFRAMES,
    LONG_THRESHOLD,
    SHORT_THRESHOLD,
    FEATURE_CATEGORIES_BY_TF,
)

def convert_state_to_tensor_dict(state: Dict[str, np.ndarray], device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """MTF 상태를 텐서 딕셔너리로 변환"""
    return {
        tf: torch.tensor(state[tf], dtype=torch.float32).unsqueeze(0).to(device)
        for tf in state.keys()
    }

def squeeze_tensor_dict(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """텐서 딕셔너리에서 배치 차원 제거"""
    return {tf: tensor.squeeze(0) for tf, tensor in tensor_dict.items()}

def log_mtf_shapes(state: Dict[str, np.ndarray], prefix: str = ""):
    """MTF 상태의 shape 정보 로깅"""
    shape_info = {tf: arr.shape for tf, arr in state.items()}
    print(f"📐 [DEBUG] {prefix} MTF shapes: {shape_info}")

def validate_mtf_state(state: Dict[str, np.ndarray]) -> bool:
    """MTF 상태 유효성 검증"""
    if not isinstance(state, dict):
        print(f"⚠️ State는 dict여야 합니다. 현재 타입: {type(state)}")
        return False
    
    for tf in TIMEFRAMES:
        if tf not in state:
            print(f"⚠️ Timeframe {tf}가 state에 없습니다.")
            return False
        
        if not isinstance(state[tf], np.ndarray):
            print(f"⚠️ {tf} 데이터가 numpy array가 아닙니다: {type(state[tf])}")
            return False
        
        if len(state[tf].shape) != 2:
            print(f"⚠️ {tf} 데이터 shape가 잘못되었습니다: {state[tf].shape} (기대: (seq_len, features))")
            return False
    
    return True

def get_current_price_from_state(state: Dict[str, np.ndarray]) -> float:
    """MTF 상태에서 현재 가격 추출"""
    # 가장 작은 타임프레임의 마지막 close 가격 사용
    primary_tf = TIMEFRAMES[0]  # 첫 번째 타임프레임 (보통 5m)
    
    if primary_tf in state:
        # 마지막 타임스텝의 close 가격 (컬럼 순서에 따라 조정 필요)
        # 일반적으로 close는 첫 번째 또는 특정 인덱스에 위치
        close_price = state[primary_tf][-1, 0]  # 마지막 행, 첫 번째 컬럼 (close 가정)
        return float(close_price)
    
    # 대안: 다른 타임프레임에서 가격 추출
    for tf in TIMEFRAMES:
        if tf in state and state[tf].size > 0:
            return float(state[tf][-1, 0])
    
    raise ValueError("현재 가격을 추출할 수 없습니다.")

def main():
    # MTF 지원 컴포넌트 초기화
    collector = RealTimeDataCollector()
    
    # Predictor 초기화 (position_info_dim 포함)
    timeframe_dims = {tf: len(FEATURE_CATEGORIES_BY_TF.get(tf, [])) for tf in TIMEFRAMES}
    # Ensure 'btc' is handled if it's not in TIMEFRAMES but has features
    if 'btc' not in timeframe_dims and 'btc' in FEATURE_CATEGORIES_BY_TF:
        timeframe_dims['btc'] = len(FEATURE_CATEGORIES_BY_TF['btc'])
    # 'dune' is currently excluded from TIMEFRAMES, so no need to add it here unless re-enabled

    predictor = Predictor(timeframe_dims=timeframe_dims, position_info_dim=5)
    trade_executor = FuturesTradeExecutor()

    force_immediate_run = True  # 테스트 시 True, 실전 시 False
    
    print(f"🚀 MTF Trading System 시작")
    print(f"📊 지원 Timeframes: {TIMEFRAMES}")
    print(f"📏 Sequence Length: {PPO_CONFIG["seq_len"]}")

    while True:
        try:
            now = datetime.now(timezone.utc)

            # 매 1분마다 실행 (또는 특정 시간 간격)
            # if not force_immediate_run and now.minute % 1 != 0: # Changed from 30 to 1 for 1-min data
            #     time.sleep(5)
            #     continue

            force_immediate_run = False  # 한 번 실행하고 False로 변경

            print(f"✅ {now.strftime('%H:%M')} → 매매 판단 시작 (MTF)")
            # time.sleep(5) # Removed this sleep as data collection might take time

            # MTF 상태 수집
            mtf_data_raw = collector.run()  # Returns Dict[str, pd.DataFrame]
            if mtf_data_raw is None:
                print("🚨 피처 결측 → 스킵")
                time.sleep(5) # Short sleep before next attempt
                continue

            # LivePPOEnv를 사용하여 현재 상태 (observation) 가져오기
            # LivePPOEnv는 내부적으로 position_info를 관리하고 observation에 포함
            live_env = LivePPOEnv(mtf_data=mtf_data_raw, seq_len=PPO_CONFIG["seq_len"])
            current_observation = live_env._get_observation() # Get current observation including position_info

            # MTF 상태 유효성 검증 (position_info 포함)
            # validate_mtf_state 함수를 업데이트하여 position_info도 검증하도록 해야 함
            # 현재는 mtf_data_raw만 검증하는 것으로 보임. _get_observation이 반환하는 형태에 맞춰야 함.
            # For now, let's assume current_observation is valid if mtf_data_raw is valid.
            # if not validate_mtf_state(current_observation):
            #     print("🚨 MTF 상태 유효성 검증 실패 → 스킵")
            #     time.sleep(5)
            #     continue

            log_mtf_shapes(current_observation, "Collected State for Predictor")

            # 현재 가격 추출 (LivePPOEnv의 내부 상태에서 가져오는 것이 더 정확)
            # current_price = get_current_price_from_state(current_observation) # This function needs to be updated
            # For now, let's get it from the raw data
            current_price = mtf_data_raw[TIMEFRAMES[0]]['close'].iloc[-1] # Get close price from the smallest timeframe

            # MTF 상태를 텐서로 변환
            obs_tensor_dict = convert_state_to_tensor_dict(current_observation)
            
            # 1. MTF 기반 정책 예측 (새로운 인터페이스 사용)
            # predict_filtered는 이제 direction 인자를 받지 않고, 4가지 액션 문자열을 반환
            predicted_action_str, confidence_prob, value_pred = predictor.predict_filtered(obs_tensor_dict)

            # 2. RL 에이전트의 액션 실행
            trade_executor.execute_rl_action(predicted_action_str, current_price)

            # 3. 버퍼 및 PPO Live 학습 (기존 로직 유지, imitation model path 제거)
            # PPO_BUFFER_PATHS는 long/short 키를 가지고 있으므로, 이를 어떻게 사용할지 결정해야 함
            # 현재는 단일 PPO 모델이 4가지 액션을 모두 처리하므로, 버퍼도 단일화하거나,
            # 학습 시 direction을 제거해야 함.
            # For now, let's use 'long' as a placeholder for buffer path.
            buffer_path = PPO_BUFFER_PATHS['long'] # Assuming a single buffer for the unified model
            dir_path = os.path.dirname(buffer_path)
            prefix = os.path.splitext(os.path.basename(buffer_path))[0]

            # 5. 버퍼 로드 또는 생성
            if os.path.exists(buffer_path):
                buffer = RolloutBuffer.load(buffer_path)
            else:
                buffer = RolloutBuffer(buffer_size=PPO_CONFIG["buffer_size"])

            # 6. MTF 관찰값을 버퍼에 저장
            # reward와 done은 live_env.step()에서 가져와야 함. 현재는 execute_rl_action에서 직접 보상을 주지 않음.
            # 이 부분은 LivePPOEnv의 step 함수를 호출하여 reward와 done을 받아와야 함.
            # For now, let's use dummy reward and done, and focus on the structure.
            # The actual reward calculation for live training needs to be carefully designed.
            
            # Simulate a step in the live environment to get reward and done status
            # This requires mapping predicted_action_str back to an integer action for live_env.step()
            action_map = {
                'attempt_long': 0,
                'attempt_short': 1,
                'close_position': 2,
                'no_action': 3
            }
            action_int = action_map.get(predicted_action_str, 3) # Default to NO_ACTION

            # This step will update the internal state of live_env and return reward/done
            # However, the reward from live_env.step() is based on the environment's internal logic,
            # which might not directly correspond to the realized PnL from trade_executor.
            # This is a critical point for online learning reward design.
            # For now, let's use the reward from live_env.step() as a placeholder.
            _, reward_from_env, done_from_env, _ = live_env.step(action_int)

            # Add to buffer
            buffer.add(
                current_observation, # Use the observation from live_env._get_observation()
                action_int,
                reward_from_env,
                done_from_env,
                log_prob=0.0, # Placeholder, actual log_prob comes from model.get_action
                value=value_pred
            )

            buffer.save(buffer_path)
            record_count = len(buffer)
            indexed_path = os.path.join(dir_path, f"{prefix}_{record_count:04d}.pkl")
            shutil.copy2(buffer_path, indexed_path)

            # 7. 로그 출력
            emoji_map = {'no_action': '⏸️', 'attempt_long': '⚡', 'attempt_short': '⛓️', 'close_position': '🛑'}
            emoji = emoji_map.get(predicted_action_str, '❓')
            print(
                f"🧐 Action: {predicted_action_str.upper()} {emoji} | "
                f"Confidence: {confidence_prob:.3f} | Reward (Env): {reward_from_env:.3f} | "
                f"Buffer: {len(buffer)}/{PPO_CONFIG["buffer_size"]}"
            )

            # 8. MTF 기반 PPO 학습 실행
            if buffer.is_ready():
                print(f"🚀 PPO 학습 시작 (MTF)")
                try:
                    train_ppo_live(
                        direction='unified', # Use a unified direction for the single model
                        buffer_path=indexed_path,
                        value_model_path=VALUE_PRETRAIN_OUTPUT_PATH['long'], # Assuming long path for unified model
                        save_path=PPO_FINAL_MODEL_PATHS['long'], # Assuming long path for unified model
                        total_epochs=PPO_CONFIG["epochs"]
                    )
                    print(f"✅ PPO 학습 완료")
                    buffer.reset()
                    RolloutBuffer.delete(indexed_path)
                except Exception as train_error:
                    print(f"❌ PPO 학습 오류: {train_error}")
                    traceback.print_exc()

        except Exception as e:
            print(f"❌ 메인 루프 오류: {e}")
            traceback.print_exc()
            time.sleep(5) # Short sleep on error

def test_mtf_components():
    """MTF 컴포넌트 테스트 함수"""
    print("🧪 MTF 컴포넌트 테스트 시작")
    
    # 테스트용 MTF 상태 생성
    test_state = {}
    for tf in TIMEFRAMES:
        # 임의의 시퀀스 데이터 생성 (seq_len, features)
        # Note: The actual feature dimension for each timeframe is defined in FEATURE_CATEGORIES_BY_TF
        # For testing, we can use a dummy value or try to infer from FEATURE_CATEGORIES_BY_TF
        # For now, let's use a dummy 10 features
        test_state[tf] = np.random.randn(PPO_CONFIG["seq_len"], 10)  # 10개 피처 가정
    
    # Add dummy position_info
    test_state['position_info'] = np.random.randn(5) # 5 features for position_info

    print("✅ 테스트 MTF 상태 생성 완료")
    log_mtf_shapes(test_state, "Test State")
    
    # 상태 유효성 검증
    if validate_mtf_state(test_state):
        print("✅ MTF 상태 유효성 검증 통과")
    else:
        print("❌ MTF 상태 유효성 검증 실패")
        return
    
    # 텐서 변환 테스트
    try:
        tensor_dict = convert_state_to_tensor_dict(test_state)
        print("✅ MTF 텐서 변환 성공")
        log_mtf_shapes({tf: tensor.numpy() for tf, tensor in tensor_dict.items()}, "Tensor Dict")
    except Exception as e:
        print(f"❌ MTF 텐서 변환 실패: {e}")
        return
    
    # 가격 추출 테스트
    try:
        price = get_current_price_from_state(test_state)
        print(f"✅ 현재 가격 추출 성공: {price:.2f}")
    except Exception as e:
        print(f"❌ 현재 가격 추출 실패: {e}")
    
    print("🎉 MTF 컴포넌트 테스트 완료")

if __name__ == "__main__":
    # 테스트 모드 체크
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_mtf_components()
    else:
        main()

def test_mtf_components():
    """MTF 컴포넌트 테스트 함수"""
    print("🧪 MTF 컴포넌트 테스트 시작")
    
    # 테스트용 MTF 상태 생성
    test_state = {}
    for tf in TIMEFRAMES:
        # 임의의 시퀀스 데이터 생성 (seq_len, features)
        test_state[tf] = np.random.randn(PPO_CONFIG["seq_len"], 10)  # 10개 피처 가정
    
    print("✅ 테스트 MTF 상태 생성 완료")
    log_mtf_shapes(test_state, "Test State")
    
    # 상태 유효성 검증
    if validate_mtf_state(test_state):
        print("✅ MTF 상태 유효성 검증 통과")
    else:
        print("❌ MTF 상태 유효성 검증 실패")
        return
    
    # 텐서 변환 테스트
    try:
        tensor_dict = convert_state_to_tensor_dict(test_state)
        print("✅ MTF 텐서 변환 성공")
        log_mtf_shapes({tf: tensor.numpy() for tf, tensor in tensor_dict.items()}, "Tensor Dict")
    except Exception as e:
        print(f"❌ MTF 텐서 변환 실패: {e}")
        return
    
    # 가격 추출 테스트
    try:
        price = get_current_price_from_state(test_state)
        print(f"✅ 현재 가격 추출 성공: {price:.2f}")
    except Exception as e:
        print(f"❌ 현재 가격 추출 실패: {e}")
    
    print("🎉 MTF 컴포넌트 테스트 완료")

if __name__ == "__main__":
    # 테스트 모드 체크
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_mtf_components()
    else:
        main()