import time
import torch
import traceback
import pandas as pd
from modules.trading_executor import TradeExecutor
from modules.ppo_runtime.predictor import Predictor
from modules.data_collector import RealTimeDataCollector
from modules.training.ppo.core.buffer import RolloutBuffer
from modules.ppo_runtime.env_live import LivePPOEnv
from modules.training.ppo.reinforce.train_ppo import train_ppo
from modules.config import (
    PPO_FINAL_MODEL_PATHS,
    TRAIN_LABEL_PATHS,
    PPO_IMITATION_MODEL_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH,
    PPO_BUFFER_PATHS,
    PPO_BUFFER_SIZE,
    PPO_EPOCHS,
    PPO_INPUT_DIM,
    TZ
)

def main():
    collector = RealTimeDataCollector()
    predictor = Predictor(input_dim=PPO_INPUT_DIM)
    executors = {
        "long": TradeExecutor(),
        "short": TradeExecutor()
    }
    buffers = {
        "long": RolloutBuffer(buffer_size=PPO_BUFFER_SIZE),
        "short": RolloutBuffer(buffer_size=PPO_BUFFER_SIZE)
    }

    immediate_first_run = True  # comment out to run strictly on schedule
    while True:
        try:
            if not immediate_first_run:
                now_ts = pd.Timestamp.now(tz=TZ)
                next_run = now_ts.floor("30min") + pd.Timedelta(minutes=30)
                sleep_sec = (next_run - now_ts).total_seconds()
                time.sleep(sleep_sec)
            immediate_first_run = False

            state = collector.run()
            if state is None:
                print("🚨 피처 결측 → 스킵")
                time.sleep(60)
                continue

            for direction in ["long", "short"]:
                action, log_prob, value = predictor.predict(state.values, direction=direction)
                print(f"[{direction.upper()}] 예측: {action.upper()} (conf={log_prob:.3f})")

                if action != 'hold':
                    price = float(state['5m_close'])  # 5m 기준 진입가 사용
                    executors[direction].enter_position(action, price)

                executors[direction].monitor_position()

                market_df = collector.get_recent_market_df(tf='5m')  # 5m 기준 명시
                env = LivePPOEnv(market_df)
                obs = env.reset()
                _, reward, done, _ = env.step(action)

                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action_map = {'long': 0, 'short': 1, 'hold': 2}
                action_idx = action_map[action]

                buffers[direction].add(
                    obs_tensor,
                    action_idx,
                    reward,
                    done,
                    log_prob=log_prob,
                    value=value
                )
                print(f"[{direction.upper()}] Buffer size: {len(buffers[direction])}")

                if buffers[direction].is_ready():
                    print(f"🚀 PPO 학습 시작: {direction.upper()}")
                    train_ppo(
                        direction=direction,
                        csv_path=TRAIN_LABEL_PATHS[direction],
                        imitation_model_path=PPO_IMITATION_MODEL_PATHS[direction],
                        value_model_path=VALUE_PRETRAIN_OUTPUT_PATH[direction],
                        save_path=PPO_FINAL_MODEL_PATHS[direction],
                        total_epochs=PPO_EPOCHS
                    )
                    buffers[direction].reset()

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            traceback.print_exc()
            time.sleep(60)

if __name__ == "__main__":
    main()
