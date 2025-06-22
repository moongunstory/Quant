import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
sys.path.insert(0, PROJECT_ROOT)

from modules.config import (
    TRAIN_LABEL_PATHS,
    PPO_IMITATION_MODEL_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH,
    PPO_FINAL_MODEL_PATHS,
    TP_THRESHOLD,
    SL_THRESHOLD,
    LABEL_HORIZON,
)

from modules.training.ppo.core.env_train import PPOTradingEnv
from modules.training.ppo.core.model import PPOPolicyNetwork
from modules.training.ppo.core.buffer import RolloutBuffer
from modules.training.ppo.core.core import (
    compute_ppo_loss,
    compute_value_loss,
    compute_explained_variance,
)


def train_ppo(
    direction: str,
    csv_path: str,
    imitation_model_path: str,
    value_model_path: str,
    save_path: str,
    total_epochs=30,
    batch_size=64,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    lr=2.5e-4,
    max_steps=5000,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    logger.info(f"🔁 [{direction.upper()}] PPO 학습 시작")
    logger.info(f"📁 [{direction.upper()}] CSV 파일: {csv_path}")
    logger.info(f"🎯 [{direction.upper()}] epochs={total_epochs}, batch_size={batch_size}, lr={lr}, device={device}")
    logger.info(f"🧭 ENV 생성 직전: direction={direction}, csv_path={csv_path}")
    env = PPOTradingEnv(
        csv_path=csv_path,
        direction=direction,
        seq_len=32,
        tp_ratio=TP_THRESHOLD,
        sl_ratio=SL_THRESHOLD,
        horizon=LABEL_HORIZON,
    )
    input_dim = env.sequences.shape[2]
    logger.info(f"📊 [{direction.upper()}] 환경 생성 완료: input_dim={input_dim}, sequences={env.sequences.shape}")

    model = PPOPolicyNetwork(input_dim=input_dim, hidden_dim=256).to(device)

    # 정책 사전학습 로드
    logger.info(f"📦 [{direction.upper()}] 모방학습 모델 로딩: {imitation_model_path}")
    model.load_model(imitation_model_path)

    # 가치 사전학습 weight 적용
    #logger.info(f"🎯 [{direction.upper()}] 가치 사전학습 모델 로딩: {value_model_path}")
    #model.value_head.load_state_dict(torch.load(value_model_path, map_location=device))
    logger.info(f"🧹 [{direction.upper()}] value head를 사전학습 없이 랜덤 초기화합니다.")
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    model.value_head.apply(init_weights)

    logger.info(f"✅ [{direction.upper()}] 모델 초기화 완료")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    all_epoch_rewards = []

    for epoch in range(total_epochs):
        logger.info(f"📈 [{direction.upper()}] Epoch {epoch+1}/{total_epochs} 시작")
        
        buffer = RolloutBuffer(buffer_size=max_steps)
        obs = env.reset()
        done = False
        episode_rewards = []

        # 롤아웃 수집
        while not done and len(buffer.rewards) < max_steps:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            action, log_prob, value = model.get_action(obs_tensor)
            next_obs, reward, done, _ = env.step(action.item())

            buffer.add(
                obs_tensor.squeeze(0).cpu(),
                action.item(),
                reward,
                done,
                log_prob.item(),
                value.item()
            )
            
            episode_rewards.append(reward)
            obs = next_obs

        # 마지막 value 계산
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            _, _, last_value = model.get_action(obs_tensor)

        buffer.compute_returns_and_advantages(last_value.item(), gamma=gamma, lam=lam)

        if logger.isEnabledFor(logging.DEBUG):
            rewards_arr = np.array(episode_rewards)
            logger.debug(
                f"Reward dist → mean:{rewards_arr.mean():.3f}, "
                f"std:{rewards_arr.std():.3f}, min:{rewards_arr.min():.3f}, max:{rewards_arr.max():.3f}")
            logger.debug(
                f"Advantage dist → mean:{buffer.advantages.mean():.3f}, "
                f"std:{buffer.advantages.std():.3f}, min:{buffer.advantages.min():.3f}, max:{buffer.advantages.max():.3f}")
            logger.debug(
                f"Return dist → mean:{buffer.returns.mean():.3f}, "
                f"std:{buffer.returns.std():.3f}, min:{buffer.returns.min():.3f}, max:{buffer.returns.max():.3f}")
        
        # 롤아웃 통계
        avg_reward = np.mean(episode_rewards)
        total_reward = np.sum(episode_rewards)
        logger.info(f"🏆 [{direction.upper()}] 롤아웃: {len(episode_rewards)} steps, Total Reward: {total_reward:.3f}")

        # PPO 업데이트
        model.train()
        policy_losses = []
        value_losses = []
        entropies = []
        batch_advantages = []
        batch_values = []
        batch_returns = []
        batch_evs = []
        value_ranges = []

        last_values = None
        last_returns = None
        
        for obs_batch, action_batch, return_batch, adv_batch, old_logprob_batch in buffer.get_batches(batch_size):
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            return_batch = return_batch.to(device)
            adv_batch = adv_batch.to(device)
            old_logprob_batch = old_logprob_batch.to(device)

            log_probs, entropy, values = model.evaluate_action(obs_batch, action_batch)
            policy_loss = compute_ppo_loss(log_probs, old_logprob_batch, adv_batch, clip_eps)
            value_loss = compute_value_loss(values, return_batch)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.mean().item())
            batch_advantages.append(adv_batch.mean().item())
            batch_values.append(values.mean().item())
            batch_returns.append(return_batch.mean().item())
            batch_evs.append(compute_explained_variance(values.detach(), return_batch.detach()).item())
            value_ranges.append((values.min().item(), values.max().item()))

            last_values = values.detach().cpu()
            last_returns = return_batch.detach().cpu()

        # 에폭 통계
        avg_advantage = np.mean(batch_advantages)
        avg_value = np.mean(batch_values)
        avg_return = np.mean(batch_returns) if batch_returns else 0.0
        ev = np.mean(batch_evs)
        value_min = min(v[0] for v in value_ranges)
        value_max = max(v[1] for v in value_ranges)

        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropies)
        all_epoch_rewards.append(avg_reward)

        # 핵심 로그 출력
        logger.info(
            f"📊 [{direction.upper()} Epoch {epoch+1}/{total_epochs}] "
            f"Avg Reward: {avg_reward:.3f} | Avg Advantage: {avg_advantage:.3f} | "
            f"Avg Return: {avg_return:.3f} | Avg Value: {avg_value:.3f} | Explained Variance: {ev:.3f}"
        )
        
        logger.info(f"🔧 [{direction.upper()} Epoch {epoch+1}/{total_epochs}] "
                   f"Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f} | "
                   f"Entropy: {avg_entropy:.4f}")

        # 이상치 경고
        adv_std = np.std(batch_advantages)
        if adv_std > 5.0:
            logger.warning(f"⚠️ [{direction.upper()} Epoch {epoch+1}] High Variance in Advantage: std={adv_std:.3f}")
        if abs(avg_value) < 0.001:
            logger.warning(f"⚠️ [{direction.upper()} Epoch {epoch+1}] Value predictions too low: {avg_value:.6f}")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Value prediction range → min:{value_min:.3f}, max:{value_max:.3f}")

    # 학습 완료 통계
    final_avg_reward = np.mean(all_epoch_rewards[-3:]) if len(all_epoch_rewards) >= 3 else np.mean(all_epoch_rewards)
    logger.info(f"🎯 [{direction.upper()}] 학습 완료 - 최종 평균 보상: {final_avg_reward:.4f}")

    # 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    logger.info(f"✅ [{direction.upper()}] PPO 학습 완료 → 저장: {save_path}")

    if last_values is not None and last_returns is not None:
        np.save("debug_values.npy", last_values.numpy())
        np.save("debug_returns.npy", last_returns.numpy())
        logger.info("📝 Debug tensors saved: debug_values.npy, debug_returns.npy")
    
    return {
        'direction': direction,
        'final_avg_reward': final_avg_reward,
        'model_path': save_path
    }


if __name__ == "__main__":
    logger.info("🚀 PPO 강화학습 훈련 시작")
    logger.info("=" * 60)
    
    results = []
    
    for direction in ["long", "short"]:
        result = train_ppo(
            direction=direction,
            csv_path=TRAIN_LABEL_PATHS[direction],
            imitation_model_path=PPO_IMITATION_MODEL_PATHS[direction],
            value_model_path=VALUE_PRETRAIN_OUTPUT_PATH[direction],
            save_path=PPO_FINAL_MODEL_PATHS[direction],
            total_epochs=30
        )
        results.append(result)
    
    # 전체 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("🏁 전체 PPO 학습 결과 요약")
    logger.info("=" * 60)
    
    for result in results:
        logger.info(f"{result['direction'].upper()} 모델: "
                   f"최종 평균 보상 {result['final_avg_reward']:.4f} → {result['model_path']}")
    
    logger.info("🎉 모든 PPO 강화학습 훈련이 완료되었습니다!")
