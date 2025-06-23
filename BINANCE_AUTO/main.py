import time
import torch
import traceback
import os
from datetime import datetime
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
    PPO_INPUT_DIM,
)

def main():
    collector = RealTimeDataCollector()
    predictor = Predictor(input_dim=PPO_INPUT_DIM)
    executors = {
        "long": FuturesTradeExecutor(),
        "short": FuturesTradeExecutor()
    }
    buffers = {
        "long": RolloutBuffer(buffer_size=PPO_BUFFER_SIZE),
        "short": RolloutBuffer(buffer_size=PPO_BUFFER_SIZE)
    }

    while True:
        try:
            now = datetime.utcnow()
            if now.minute % 30 != 0:
                time.sleep(60)
                continue

            print(f"✅ {now.strftime('%H:%M')} → 매매 판단 시작")
            time.sleep(60)  # ✅ 같은 분기 중 중복 실행 방지

            state = collector.run()
            if state is None:
                print("🚨 피처 결측 → 스킵")
                time.sleep(60)
                continue

            for direction in ["long", "short"]:
                if not executors[direction].should_enter():
                    executors[direction].monitor_position()
                    continue

                action_str, log_prob, value = predictor.predict(state.astype(float).values, direction=direction)

                if action_str in ['long', 'short']:
                    executors[direction].cancel_existing_orders()
                    price = float(state['5m_close'])
                    executors[direction].enter_position(direction=direction, current_price=price)
                else:
                    print(
                        f"⛔ [{direction.upper()}] 진입 조건 불충족 → 생략 "
                        f"(action={action_str.upper()}, prob={log_prob:.3f})"
                    )

                executors[direction].monitor_position()

                market_df = collector.get_recent_market_df(tf='5min')
                env = LivePPOEnv(market_df)
                obs = env.reset()
                _, reward, done, _ = env.step(action_str)

                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action_idx = 1 if action_str == direction else 0

                buffers[direction].add(
                    obs_tensor,
                    action_idx,
                    reward,
                    done,
                    log_prob=log_prob,
                    value=value
                )

                # [1] 루프별 step 기록 파일 저장 (시간별로 분리됨)
                step_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                step_log_path = PPO_BUFFER_PATHS[direction].replace('.pkl', f'_step_{step_timestamp}.pkl')
                buffers[direction].save(step_log_path)

                # [2] 누적 학습용 버퍼에 append 저장
                buffer_path = PPO_BUFFER_PATHS[direction]
                if os.path.exists(buffer_path):
                    existing = RolloutBuffer.load(buffer_path)
                    existing.observations += buffers[direction].observations
                    existing.actions += buffers[direction].actions
                    existing.rewards += buffers[direction].rewards
                    existing.dones += buffers[direction].dones
                    existing.log_probs += buffers[direction].log_probs
                    existing.values += buffers[direction].values
                    existing.save(buffer_path)
                else:
                    buffers[direction].save(buffer_path)

                emoji_map = {'hold': '⏸️', 'long': '⚡', 'short': '⛓️'}
                emoji = emoji_map.get(action_str, '')
                print(
                    f"🧠 [{direction.upper()}] Action: {action_str.upper()} {emoji} | "
                    f"Confidence(prob): {log_prob:.3f} | Buffer: {len(buffers[direction])}/{PPO_BUFFER_SIZE}"
                )

                if buffers[direction].is_ready():
                    print(f"🚀 PPO 학습 시작: {direction.upper()}")
                    train_ppo_live(
                        direction=direction,
                        buffer_path=PPO_BUFFER_PATHS[direction],
                        imitation_model_path=PPO_IMITATION_MODEL_PATHS[direction],
                        value_model_path=VALUE_PRETRAIN_OUTPUT_PATH[direction],
                        save_path=PPO_FINAL_MODEL_PATHS[direction],
                        total_epochs=PPO_EPOCHS
                    )
                    buffers[direction].reset()
                    RolloutBuffer.delete(PPO_BUFFER_PATHS[direction])

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    main()
