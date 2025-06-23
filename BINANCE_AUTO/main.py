import time
import torch
import traceback
import pandas as pd
from datetime import datetime
from modules.trading_executor import FuturesTradeExecutor, calculate_futures_quantity
from modules.ppo_runtime.predictor import Predictor
from modules.data_collector import RealTimeDataCollector
from modules.ppo_runtime.rollout_updater import RolloutBuffer
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
    PPO_INPUT_DIM
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
                if executors[direction].position is not None:
                    print(f"⏸️ [{direction.upper()}] Position already open → skipping decision.")
                    executors[direction].monitor_position()
                    continue

                action, log_prob, value = predictor.predict(state.astype(float).values, direction=direction)
                action_map_rev = {0: 'hold', 1: direction}
                action_str = action_map_rev.get(action, 'unknown')

                if action_str != 'hold':
                    price = float(state['5m_close'])
                    balance = executors[direction].get_balance()
                    qty = calculate_futures_quantity(balance, price)
                    print(f"[DEBUG] balance={balance}, price={price}, notional={balance * 5}, qty={qty}")
                    side = 'BUY' if action_str == 'long' else 'SELL'
                    price = float(state['5m_close'])
                    executors[direction].enter_position(direction=direction, current_price=price)
                    continue

                executors[direction].monitor_position()

                market_df = collector.get_recent_market_df(tf='5min')
                env = LivePPOEnv(market_df)
                obs = env.reset()
                _, reward, done, _ = env.step(action_str)

                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                action_idx = action  # 이미 int형

                buffers[direction].add(
                    obs_tensor,
                    action_idx,
                    reward,
                    done,
                    log_prob=log_prob,
                    value=value
                )

                emoji_map = {'hold': '⏸️', 'long': '⚡', 'short': '⛓️'}
                emoji = emoji_map.get(action_str, '')
                print(
                    f"🧠 [{direction.upper()}] Action: {action_str.upper()} {emoji} | "
                    f"Confidence(prob): {log_prob:.3f} | Buffer: {len(buffers[direction])}/{PPO_BUFFER_SIZE}"
                )

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
