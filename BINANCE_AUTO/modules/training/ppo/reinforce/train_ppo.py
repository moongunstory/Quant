import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import logging
from typing import Dict, Any, Tuple

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../../../"))
sys.path.insert(0, PROJECT_ROOT)

from modules.config import (
    TRAIN_PICKLE_PATHS,
    PPO_IMITATION_MODEL_PATHS,
    VALUE_PRETRAIN_OUTPUT_PATH,
    PPO_FINAL_MODEL_PATHS,
    TIMEFRAMES,
    SEQ_LEN,
    HIDDEN_DIM,
    PPO_MAX_STEPS,
    PPO_CONFIG,
)

from modules.training.ppo.core.env_train import PPOTradingEnv
from modules.training.ppo.core.model import PPOPolicyNetwork
from modules.training.ppo.core.buffer import RolloutBuffer
from modules.training.ppo.core.core import (
    compute_ppo_loss,
    compute_value_loss,
    compute_explained_variance,
)


def move_obs_to_device(
    obs: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """MTF 관찰값을 디바이스로 이동"""
    return {tf: obs_tensor.to(device) for tf, obs_tensor in obs.items()}


def convert_obs_to_tensor(
    obs: Dict[str, np.ndarray], device: torch.device
) -> Dict[str, torch.Tensor]:
    """MTF numpy 관찰값을 torch 텐서로 변환"""
    return {
        tf: torch.tensor(obs_data, dtype=torch.float32).unsqueeze(0).to(device)
        for tf, obs_data in obs.items()
    }


def squeeze_obs_dict(obs_tensor: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """MTF 관찰값 텐서에서 배치 차원 제거"""
    return {tf: tensor.squeeze(0).cpu() for tf, tensor in obs_tensor.items()}


def monitor_training_health(ev: float, entropy: float) -> bool:
    """Return True if training should stop due to collapse."""
    if ev < -2.0:
        logger.error("Value function collapsed (explained variance < -2.0)")
        return True
    if entropy < 0.01:
        logger.error("Policy collapsed to deterministic (entropy < 0.01)")
        return True
    return False


def init_value_head_weights(model: PPOPolicyNetwork):
    """가치 헤드 가중치 초기화 (MTF 지원)"""

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    # MTF 모델의 경우 각 타임프레임별 가치 헤드 초기화
    if hasattr(model, "value_heads"):
        # Multi-head value network
        for tf, value_head in model.value_heads.items():
            value_head.apply(init_weights)
        logger.info("🧹 MTF value heads 랜덤 초기화 완료")
    else:
        # Single value head
        model.value_head.apply(init_weights)
        logger.info("🧹 Single value head 랜덤 초기화 완료")


def train_ppo(
    direction: str,
    csv_path: str,
    imitation_model_path: str,
    value_model_path: str,
    save_path: str,
    total_epochs: int = PPO_CONFIG["epochs"],
    batch_size: int = PPO_CONFIG["batch_size"],
    gamma: float = PPO_CONFIG["gamma"],
    lam: float = PPO_CONFIG["lambda"],
    clip_eps: float = PPO_CONFIG["clip_eps"],
    value_coef: float = PPO_CONFIG["value_coef"],
    entropy_coef: float = PPO_CONFIG["entropy_coef"],
    lr: float = PPO_CONFIG["learning_rate"],
    max_steps: int = PPO_MAX_STEPS,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    logger.info(f"🔁 [{direction.upper()}] PPO 학습 시작 (MTF)")
    logger.info(f"📁 [{direction.upper()}] CSV 파일: {csv_path}")
    logger.info(
        f"🎯 [{direction.upper()}] epochs={total_epochs}, batch_size={batch_size}, lr={lr}, device={device}"
    )
    logger.info(f"🕒 [{direction.upper()}] Timeframes: {TIMEFRAMES}, seq_len={SEQ_LEN}")

    # MTF 환경 생성
    logger.info(f"🧭 ENV 생성 직전: direction={direction}, csv_path={csv_path}")
    env = PPOTradingEnv(data_path=csv_path, direction=direction, seq_len=SEQ_LEN)

    # MTF 입력 차원 정보 가져오기
    input_dims = env.get_input_dims()  # Returns Dict[str, int]
    logger.info(f"📊 [{direction.upper()}] 환경 생성 완료: input_dims={input_dims}")

    # MTF 지원 PPO 모델 초기화
    model = PPOPolicyNetwork(timeframe_dims=input_dims, hidden_dim=HIDDEN_DIM).to(
        device
    )

    # 정책 사전학습 모델 로드
    logger.info(f"📦 [{direction.upper()}] 모방학습 모델 로딩: {imitation_model_path}")
    model.load_model(imitation_model_path, allow_partial=True)

    # 가치 헤드 랜덤 초기화 (사전학습 없이)
    logger.info(
        f"🧹 [{direction.upper()}] value head를 사전학습 없이 랜덤 초기화합니다."
    )
    init_value_head_weights(model)

    logger.info(f"✅ [{direction.upper()}] 모델 초기화 완료")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    initial_entropy_coef = entropy_coef
    min_entropy_coef = 0.01  # Prevent collapse to fully deterministic policy
    all_epoch_rewards = []
    early_stop = False

    for epoch in range(total_epochs):
        entropy_coef = max(
            initial_entropy_coef * (1 - epoch / total_epochs), min_entropy_coef
        )
        logger.info(
            f"📈 [{direction.upper()}] Epoch {epoch+1}/{total_epochs} 시작 "
            f"(entropy_coef={entropy_coef:.4f})"
        )

        buffer = RolloutBuffer(buffer_size=max_steps)
        obs = env.reset()  # Returns Dict[str, np.ndarray]
        done = False
        episode_rewards = []

        # 롤아웃 수집
        while not done and len(buffer.rewards) < max_steps:
            # MTF 관찰값을 텐서로 변환
            obs_tensor = convert_obs_to_tensor(obs, device)

            action, log_prob, value, _ = model.get_action(obs_tensor)
            env_action = action.item()  # 액션 그대로 전달
            next_obs, reward, done, _ = env.step(env_action)

            # MTF 관찰값을 CPU로 이동하여 버퍼에 저장
            obs_cpu = squeeze_obs_dict(obs_tensor)

            buffer.add(
                obs_cpu,  # Dict[str, Tensor] 형태로 저장
                action.item(),
                reward,
                done,
                log_prob.item(),
                value.item(),
            )

            episode_rewards.append(reward)
            obs = next_obs

        # 마지막 value 계산
        with torch.no_grad():
            obs_tensor = convert_obs_to_tensor(obs, device)
            _, _, last_value, _ = model.get_action(obs_tensor)

        buffer.compute_returns_and_advantages(last_value.item(), gamma=gamma, lam=lam)

        if logger.isEnabledFor(logging.DEBUG):
            rewards_arr = np.array(episode_rewards)
            logger.debug(
                f"Reward dist → mean:{rewards_arr.mean():.3f}, "
                f"std:{rewards_arr.std():.3f}, min:{rewards_arr.min():.3f}, max:{rewards_arr.max():.3f}"
            )
            logger.debug(
                f"Advantage dist → mean:{buffer.advantages.mean():.3f}, "
                f"std:{buffer.advantages.std():.3f}, min:{buffer.advantages.min():.3f}, max:{buffer.advantages.max():.3f}"
            )
            logger.debug(
                f"Return dist → mean:{buffer.returns.mean():.3f}, "
                f"std:{buffer.returns.std():.3f}, min:{buffer.returns.min():.3f}, max:{buffer.returns.max():.3f}"
            )

        # 롤아웃 통계
        avg_reward = np.mean(episode_rewards)
        total_reward = np.sum(episode_rewards)
        logger.info(
            f"🏆 [{direction.upper()}] 롤아웃: {len(episode_rewards)} steps, Total Reward: {total_reward:.3f}"
        )

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

        for (
            obs_batch,
            action_batch,
            return_batch,
            adv_batch,
            old_logprob_batch,
        ) in buffer.get_batches(batch_size):
            # MTF 관찰값 배치를 디바이스로 이동
            obs_batch = move_obs_to_device(obs_batch, device)
            action_batch = action_batch.to(device)
            return_batch = return_batch.to(device)
            adv_batch = adv_batch.to(device)
            old_logprob_batch = old_logprob_batch.to(device)

            log_probs, entropy, values = model.evaluate_action(obs_batch, action_batch)
            entropy = torch.clamp(entropy, min=0.005, max=2.0)  # 최대값도 제한
            policy_loss = compute_ppo_loss(
                log_probs, old_logprob_batch, adv_batch, clip_eps
            )
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
            batch_evs.append(
                compute_explained_variance(
                    values.detach(), return_batch.detach()
                ).item()
            )
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

        logger.info(
            f"🔧 [{direction.upper()} Epoch {epoch+1}/{total_epochs}] "
            f"Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f} | "
            f"Entropy: {avg_entropy:.4f} | Entropy Coef: {entropy_coef:.4f}"
        )

        if monitor_training_health(ev, avg_entropy):
            early_stop = True
            break

        # 이상치 경고
        adv_std = np.std(batch_advantages)
        if adv_std > 5.0:
            logger.warning(
                f"⚠️ [{direction.upper()} Epoch {epoch+1}] High Variance in Advantage: std={adv_std:.3f}"
            )
        if abs(avg_value) < 0.001:
            logger.warning(
                f"⚠️ [{direction.upper()} Epoch {epoch+1}] Value predictions too low: {avg_value:.6f}"
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Value prediction range → min:{value_min:.3f}, max:{value_max:.3f}"
            )

    if early_stop:
        logger.warning(
            f"⚠️ [{direction.upper()}] Early stopping triggered at epoch {epoch+1}"
        )

    # 학습 완료 통계
    final_avg_reward = (
        np.mean(all_epoch_rewards[-3:])
        if len(all_epoch_rewards) >= 3
        else np.mean(all_epoch_rewards)
    )
    logger.info(
        f"🎯 [{direction.upper()}] 학습 완료 - 최종 평균 보상: {final_avg_reward:.4f}"
    )

    # 모델 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    logger.info(f"✅ [{direction.upper()}] PPO 학습 완료 (MTF) → 저장: {save_path}")

    return {
        "direction": direction,
        "final_avg_reward": final_avg_reward,
        "model_path": save_path,
        "input_dims": input_dims,
    }


if __name__ == "__main__":
    logger.info("🚀 PPO 강화학습 훈련 시작 (MTF)")
    logger.info("=" * 60)

    results = []

    for direction in ["long", "short"]:
        result = train_ppo(
            direction=direction,
            csv_path=TRAIN_PICKLE_PATHS[direction],
            imitation_model_path=PPO_IMITATION_MODEL_PATHS[direction],
            value_model_path=VALUE_PRETRAIN_OUTPUT_PATH[direction],
            save_path=PPO_FINAL_MODEL_PATHS[direction],
            total_epochs=PPO_CONFIG["epochs"],
        )
        results.append(result)

    # 전체 결과 요약
    logger.info("\n" + "=" * 60)
    logger.info("🏁 전체 PPO 학습 결과 요약 (MTF)")
    logger.info("=" * 60)

    for result in results:
        logger.info(
            f"{result['direction'].upper()} 모델: "
            f"최종 평균 보상 {result['final_avg_reward']:.4f} → {result['model_path']}"
        )
        logger.info(f"  📐 입력 차원: {result['input_dims']}")

    logger.info("🎉 모든 PPO 강화학습 훈련이 완료되었습니다! (MTF)")
    logger.info(f"🕒 사용된 Timeframes: {TIMEFRAMES}")
    logger.info(f"📏 Sequence Length: {SEQ_LEN}")
    logger.info(f"🧠 Hidden Dimension: {HIDDEN_DIM}")
