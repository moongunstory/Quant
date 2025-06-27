import time
import torch
import traceback
import os
import shutil
from datetime import datetime
from modules.trading_executor import FuturesTradeExecutor
from modules.ppo_runtime.predictor import Predictor
from modules.data_collector import RealTimeDataCollector
from modules.ppo_runtime.rollout_updater import RolloutBuffer
from modules.ppo_runtime.env_live import LivePPOEnv
from datetime import datetime, timezone
from modules.ppo_runtime.train_ppo_live import train_ppo_live
from modules.config import (
    PPO_FINAL_MODEL_PATHS,
    PPO_IMITATION_MODEL_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH,
    PPO_BUFFER_PATHS,
    PPO_BUFFER_SIZE,
    PPO_EPOCHS,
    PPO_INPUT_DIM,
    LONG_THRESHOLD,
    SHORT_THRESHOLD,
)

def main():
    collector = RealTimeDataCollector()
    predictor = Predictor(input_dim=PPO_INPUT_DIM)
    executors = {
        "long": FuturesTradeExecutor(),
        "short": FuturesTradeExecutor()
    }

    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.minute % 30 != 0:
                time.sleep(60)
                continue

            print(f"✅ {now.strftime('%H:%M')} → 메매 판단 시작")
            time.sleep(60)

            state = collector.run()
            if state is None:
                print("🚨 피처 결식 → 스킵")
                time.sleep(60)
                continue

            for direction in ["long", "short"]:
                if not executors[direction].should_enter():
                    executors[direction].monitor_position()
                    continue

                # Debugging: print threshold values before prediction
                print(
                    f"LONG_THRESHOLD = {LONG_THRESHOLD}, SHORT_THRESHOLD = {SHORT_THRESHOLD}"
                )
                # 1. 실전용 확신도 필터 적용
                filtered_action, filtered_prob, _ = predictor.predict_filtered(
                    state.astype(float).values, direction=direction
                )

                # 2. 실전 진입 판단
                if filtered_action in ['long', 'short']:
                    executors[direction].cancel_existing_orders()
                    price = float(state['5m_close'])
                    executors[direction].enter_position(direction=direction, current_price=price)
                else:
                    print(
                        f"⛔ [{direction.upper()}] 진입 조건 불일치 → 생략 "
                        f"(action={filtered_action.upper()}, prob={filtered_prob:.3f})"
                    )

                executors[direction].monitor_position()

                # 3. 학습용: 확신도 조건 없는 원본 정책 행동 추출
                policy_action, log_prob, value = predictor.predict_policy(
                    state.astype(float).values, direction=direction
                )

                # 4. 환경에서 정책 행동 평가 (진입 여부와 무관)
                market_df = collector.get_recent_market_df(tf='5min')
                env = LivePPOEnv(market_df)
                obs = env.reset()
                _, reward, done, _ = env.step(policy_action)

                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action_idx = 0 if policy_action == direction else 1

                # ✅ 4.5 버퍼 경로 및 파일명 설정
                temp_path = PPO_BUFFER_PATHS[direction]
                dir_path = os.path.dirname(temp_path)
                prefix = os.path.splitext(os.path.basename(temp_path))[0]

                # ✅ 5. 버퍼 로드 또는 생성
                if os.path.exists(temp_path):
                    buffer = RolloutBuffer.load(temp_path)
                else:
                    buffer = RolloutBuffer(buffer_size=PPO_BUFFER_SIZE)

                # ✅ 6. 정책 행동 기준으로 버퍼에 저장
                buffer.add(
                    obs_tensor,
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

                # ✅ 7. 로그 출력: policy_action 기준
                emoji_map = {'hold': '⏸️', 'long': '⚡', 'short': '⛓️'}
                emoji = emoji_map.get(policy_action, '')
                print(
                    f"🧐 [{direction.upper()}] Action: {policy_action.upper()} {emoji} | "
                    f"Confidence(prob): {log_prob:.3f} | Buffer: {len(buffer)}/{PPO_BUFFER_SIZE}"
                )

                # ✅ 8. 학습 실행
                if buffer.is_ready():
                    print(f"🚀 PPO 학습 시작: {direction.upper()}")
                    train_ppo_live(
                        direction=direction,
                        buffer_path=indexed_path,
                        imitation_model_path=PPO_IMITATION_MODEL_PATHS[direction],
                        value_model_path=VALUE_PRETRAIN_OUTPUT_PATH[direction],
                        save_path=PPO_FINAL_MODEL_PATHS[direction],
                        total_epochs=PPO_EPOCHS
                    )
                    buffer.reset()
                    RolloutBuffer.delete(indexed_path)

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    main()
