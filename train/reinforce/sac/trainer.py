# train/reinforce/sac/trainer.py

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque


class Trainer:
    """
    SAC 에이전트의 학습 전체 과정을 관리하는 클래스.
    """
    def __init__(self, agent, env, replay_buffer, config):
        self.agent = agent
        self.env = env
        self.replay_buffer = replay_buffer
        self.config = config

        # 설정값 추출
        self.total_steps = config.get("total_steps", 1_000_000)
        self.learning_starts = config.get("learning_starts", 20_000)
        self.batch_size = config.get("batch_size", 256)
        self.log_interval = config.get("log_interval", 1000)
        self.save_interval = config.get("save_interval", 50000)
        self.save_path = Path(config.get("save_path", "models/checkpoints"))

        self.save_path.mkdir(parents=True, exist_ok=True)

        # TensorBoard SummaryWriter
        logs_dir = Path("ai_binance/data/models/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(logs_dir)

    def train(self):
        """메인 학습 루프 실행"""
        obs = self.env.reset()
        total_reward = 0
        episode_reward = 0
        episode_count = 0

        W = 5000
        trade_events = deque(maxlen=W)
        flip_events  = deque(maxlen=W)
        turnover_hist = deque(maxlen=W)

        print(f"[Trainer] Starting training for {self.total_steps} steps.")

        for step in range(1, self.total_steps + 1):
            # 학습 초반 (learning_starts)에는 무작위 액션으로 버퍼 채우기
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.agent.select_action(obs)

            # 환경과 상호작용
            next_obs, reward, done, info = self.env.step(action)
            total_reward += reward
            episode_reward += reward

            # 리플레이 버퍼에 경험 저장
            self.replay_buffer.add(obs, action, reward, next_obs, done)

            # 슬라이딩 윈도우 로깅을 위한 정보 업데이트
            trade_events.append(1 if info.get("did_trade") else 0)
            flip_events.append(1 if info.get("did_flip") else 0)
            turnover_hist.append(info.get("turnover", 0.0))

            # 학습 시작 스텝 이후부터 에이전트 업데이트
            if step >= self.learning_starts:
                losses = self.agent.update(
                    self.replay_buffer, batch_size=self.batch_size, recent_reward=episode_reward
                )

                # TensorBoard에 학습 로그 기록
                if losses:
                    for k, v in losses.items():
                        self.writer.add_scalar(f"Loss/{k}", v, step)

            obs = next_obs
            if done:
                episode_count += 1
                portfolio_value = info.get("portfolio_value", 0)

                print(f"[Step {step}/{self.total_steps}] Episode {episode_count} finished. "
                      f"Reward: {episode_reward:.4f}, Final Portfolio: {portfolio_value:.2f}")

                # TensorBoard에 Episode 로그 기록
                self.writer.add_scalar("Episode/Reward", episode_reward, episode_count)
                self.writer.add_scalar("Episode/PortfolioValue", portfolio_value, episode_count)

                obs = self.env.reset()
                episode_reward = 0

            # 로그 출력
            if step % self.log_interval == 0:
                print(f"--- [Step {step}/{self.total_steps}] ---")
                self.writer.add_scalar("Train/RewardRunning", total_reward / step, step)

                # 슬라이딩 윈도우 지표 계산 및 로깅
                if len(trade_events) > 0:
                    trades_per_1k = 1000.0 * (sum(trade_events) / len(trade_events))
                    flip_rate_1k  = 1000.0 * (sum(flip_events) / len(flip_events))
                    turnover_win  = sum(turnover_hist) / max(1, len(turnover_hist))

                    self.writer.add_scalar("Trade/trades_per_1k", trades_per_1k, step)
                    self.writer.add_scalar("Trade/action_flip_rate_1k", flip_rate_1k, step)
                    self.writer.add_scalar("Trade/turnover", turnover_win, step)

            # 모델 체크포인트 저장
            if step % self.save_interval == 0:
                checkpoint_path = self.save_path / f"sac_lstm_step_{step}.pth"
                self.agent.save(checkpoint_path)
                print(f"✅ Model checkpoint saved to {checkpoint_path}")

        print("[Trainer] Training finished.")
        final_save_path = self.save_path / "sac_lstm_final.pth"
        self.agent.save(final_save_path)
        print(f"✅ Final model saved to {final_save_path}")
        self.writer.close()
