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
    PPO_IMITATION_MODEL_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH,
    PPO_BUFFER_PATHS,
    PPO_BUFFER_SIZE,
    PPO_EPOCHS,
    TIMEFRAMES,
    SEQ_LEN,
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
    timeframe_dims = {tf: len(FEATURE_CATEGORIES_BY_TF.get(tf, [])) for tf in TIMEFRAMES}
    timeframe_dims.update({"btc": 12, "dune": 6})
    predictor = Predictor(timeframe_dims=timeframe_dims)  # MTF 지원 버전
    executors = {
        "long": FuturesTradeExecutor(),
        "short": FuturesTradeExecutor()
    }

    force_immediate_run = True  # 테스트 시 True, 실전 시 False
    
    print(f"🚀 MTF Trading System 시작")
    print(f"📊 지원 Timeframes: {TIMEFRAMES}")
    print(f"📏 Sequence Length: {SEQ_LEN}")

    while True:
        try:
            now = datetime.now(timezone.utc)

            if not force_immediate_run and now.minute % 30 != 0:
                time.sleep(60)
                continue

            force_immediate_run = False  # 한 번 실행하고 False로 변경

            print(f"✅ {now.strftime('%H:%M')} → 매매 판단 시작 (MTF)")
            time.sleep(60)

            # MTF 상태 수집
            state = collector.run()  # Returns Dict[str, np.ndarray]
            if state is None:
                print("🚨 피처 결측 → 스킵")
                time.sleep(60)
                continue

            # MTF 상태 유효성 검증
            if not validate_mtf_state(state):
                print("🚨 MTF 상태 유효성 검증 실패 → 스킵")
                time.sleep(60)
                continue

            log_mtf_shapes(state, "Collected State")

            for direction in ["long", "short"]:
                if not executors[direction].should_enter():
                    executors[direction].monitor_position()
                    continue

                # MTF 상태를 텐서로 변환
                obs_tensor_dict = convert_state_to_tensor_dict(state)
                
                print(f"🔍 임계값 확인: LONG_THRESHOLD = {LONG_THRESHOLD}, SHORT_THRESHOLD = {SHORT_THRESHOLD}")
                
                # 1. MTF 기반 정책 예측
                policy_action, log_prob, value, prob = predictor.predict_policy(
                    obs_tensor_dict, direction=direction
                )

                # 2. 확신도 임계값 필터 적용
                threshold = LONG_THRESHOLD if direction == 'long' else SHORT_THRESHOLD
                if policy_action == direction and prob >= threshold:
                    executors[direction].cancel_existing_orders()
                    current_price = get_current_price_from_state(state)
                    executors[direction].enter_position(direction=direction, current_price=current_price)
                    print(f"✅ [{direction.upper()}] 포지션 진입: price={current_price:.2f}")
                else:
                    print(
                        f"⛔ [{direction.upper()}] 진입 조건 불일치 → 생략 "
                        f"(action={policy_action.upper()}, prob={prob:.3f}, threshold={threshold:.3f})"
                    )

                # 3. 환경에서 정책 행동 평가 (진입 여부와 무관)
                market_df = collector.get_recent_market_df(tf='5min')
                env = LivePPOEnv({"5min": market_df}, reference_timeframe="5min")

                # 정책 행동을 환경에 전달
                _, reward, done, _ = env.step(policy_action)
                
                log_mtf_shapes(state, f"[{direction.upper()}] Environment Input")

                # 4. 버퍼 경로 및 파일명 설정
                temp_path = PPO_BUFFER_PATHS[direction]
                dir_path = os.path.dirname(temp_path)
                prefix = os.path.splitext(os.path.basename(temp_path))[0]

                # 5. 버퍼 로드 또는 생성
                if os.path.exists(temp_path):
                    buffer = RolloutBuffer.load(temp_path)
                else:
                    buffer = RolloutBuffer(buffer_size=PPO_BUFFER_SIZE)

                # 6. MTF 관찰값을 버퍼에 저장
                action_idx = 0 if policy_action == direction else 1
                obs_tensor_cpu = squeeze_tensor_dict(obs_tensor_dict)  # CPU로 이동하여 배치 차원 제거
                
                buffer.add(
                    obs_tensor_cpu,  # Dict[str, torch.Tensor] 형태로 저장
                    action_idx,
                    reward,
                    done,
                    log_prob=log_prob,
                    value=value
                )

                buffer.save(temp_path)
                record_count = len(buffer)
                indexed_path = os.path.join(dir_path, f"{prefix}_{record_count:04d}.pkl")
                shutil.copy2(temp_path, indexed_path)

                # 7. 로그 출력: policy_action 기준
                emoji_map = {'hold': '⏸️', 'long': '⚡', 'short': '⛓️'}
                emoji = emoji_map.get(policy_action, '❓')
                print(
                    f"🧐 [{direction.upper()}] Action: {policy_action.upper()} {emoji} | "
                    f"Confidence: {prob:.3f} | Reward: {reward:.3f} | "
                    f"Buffer: {len(buffer)}/{PPO_BUFFER_SIZE}"
                )

                # 8. MTF 기반 PPO 학습 실행
                if buffer.is_ready():
                    print(f"🚀 PPO 학습 시작: {direction.upper()} (MTF)")
                    try:
                        train_ppo_live(
                            direction=direction,
                            buffer_path=indexed_path,
                            imitation_model_path=PPO_IMITATION_MODEL_PATHS[direction],
                            value_model_path=VALUE_PRETRAIN_OUTPUT_PATH[direction],
                            save_path=PPO_FINAL_MODEL_PATHS[direction],
                            total_epochs=PPO_EPOCHS
                        )
                        print(f"✅ PPO 학습 완료: {direction.upper()}")
                        buffer.reset()
                        RolloutBuffer.delete(indexed_path)
                    except Exception as train_error:
                        print(f"❌ PPO 학습 오류 ({direction.upper()}): {train_error}")
                        traceback.print_exc()

        except Exception as e:
            print(f"❌ 메인 루프 오류: {e}")
            traceback.print_exc()
            time.sleep(60)

def test_mtf_components():
    """MTF 컴포넌트 테스트 함수"""
    print("🧪 MTF 컴포넌트 테스트 시작")
    
    # 테스트용 MTF 상태 생성
    test_state = {}
    for tf in TIMEFRAMES:
        # 임의의 시퀀스 데이터 생성 (seq_len, features)
        test_state[tf] = np.random.randn(SEQ_LEN, 10)  # 10개 피처 가정
    
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