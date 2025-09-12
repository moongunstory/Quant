import torch as th
import numpy as np
from typing import Dict, Any, Optional
import optuna
from buffer import RolloutBuffer

class LinearDecayLR:
    def __init__(self, optimizer, total_steps):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1
        lr_scale = max(1.0 - self.step_count / self.total_steps, 0.1)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * lr_scale

def train_with_config(
    env,
    eval_env,
    policy,
    config: Dict[str, Any],
    train_steps: int,
    learning_rate: float,
    trial: Optional[optuna.trial.Trial] = None,
) -> Dict[str, float]:
    device = policy.device
    policy.train()

    optimizer = th.optim.Adam(policy.parameters(), lr=learning_rate)
    for g in optimizer.param_groups:
        g["initial_lr"] = learning_rate

    if config.get("lr_scheduler") == "linear":
        scheduler = LinearDecayLR(optimizer, total_steps=train_steps)
    else:
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=learning_rate * 0.1)

    gamma = config.get("gamma", 0.99)
    lam = config.get("gae_lambda", 0.95)
    rollout_steps = config.get("n_steps", 2048)
    batch_size = config["batch_size"]
    ent_coef = config["ent_coef"]
    vf_coef = config.get("vf_coef", 0.5)
    clip_range = config.get("clip_range", 0.2)
    n_epochs = config.get("n_epochs", 10)

    buffer = RolloutBuffer(rollout_steps, env.observation_space, device=device)
    obs, _ = env.reset()

    for step in range(0, train_steps, rollout_steps):
        episodic_rewards = []

        for _ in range(rollout_steps):
            obs_tensor = {tf: th.tensor(obs[tf], dtype=th.float32, device=device).unsqueeze(0) for tf in obs}
            action_mask = th.tensor(env._action_mask(), dtype=th.bool, device=device).unsqueeze(0)

            action, log_prob, entropy, value = policy.get_action(obs_tensor, action_mask=action_mask)
            next_obs, reward, done, truncated, info = env.step(action.item())

            buffer.add(obs, action.item(), reward, done, log_prob.item(), value.item(), info)
            episodic_rewards.append(reward)

            obs = next_obs
            if done or truncated:
                obs, _ = env.reset()

        print(f"[Step {step}] Mean Reward: {np.mean(episodic_rewards):.4f}, Max: {np.max(episodic_rewards):.4f}, Min: {np.min(episodic_rewards):.4f}")

        buffer.compute_returns_and_advantages(gamma=gamma, lam=lam)

        for epoch in range(n_epochs):
            for batch in buffer.get_batches(batch_size):
                ppo_update(policy, optimizer, batch, clip_range, ent_coef, vf_coef, log=False)

        scheduler.step()

        if hasattr(policy, "aux_train_step"):
            obs_1h = buffer.obs["1h"]
            obs_4h = buffer.obs["4h"]
            labels = th.tensor([info.get("trend_label", 1) for info in buffer.infos], dtype=th.long, device=device)
            policy.aux_train_step(obs_1h, obs_4h, labels)

        if trial is not None:
            sharpe, mdd, total_return = evaluate(eval_env, policy, n_steps=288*7)
            # Early stopping은 필요할 때만 사용 (예: performance threshold 기준)
            trial.report(sharpe, step)
            return {"sharpe": sharpe, "mdd": mdd, "return": total_return}

        buffer.clear()

    sharpe, mdd, total_return = evaluate(eval_env, policy)
    return {"sharpe": sharpe, "mdd": mdd, "return": total_return}

def ppo_update(policy, optimizer, batch, clip_range, ent_coef, vf_coef, log=False):
    obs = batch["obs"]
    actions = batch["actions"]
    old_log_probs = batch["log_probs"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    values = batch["values"]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_inputs = {f"obs_{tf}": obs[tf].to(policy.device) for tf in obs}
    logits, new_values = policy.forward(
        obs_inputs["obs_5m"], obs_inputs["obs_15m"], obs_inputs["obs_1h"], obs_inputs["obs_4h"]
    )

    dist = th.distributions.Categorical(logits=logits)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy()

    ratio = th.exp(new_log_probs - old_log_probs)
    clipped_ratio = th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -th.min(ratio * advantages, clipped_ratio * advantages).mean()

    value_pred_clipped = values + (new_values - values).clamp(-clip_range, clip_range)
    value_loss = 0.5 * th.max((new_values - returns).pow(2), (value_pred_clipped - returns).pow(2)).mean()

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()

    if log:
        print(f"[PPO] loss={loss.item():.4f}, policy={policy_loss.item():.4f}, value={value_loss.item():.4f}, entropy={entropy.mean().item():.4f}, value_mean={new_values.mean().item():.4f}")

def evaluate(env, policy, n_steps: int | None = None) -> tuple:
    if n_steps is None:
        n_steps = len(env.common_index) - max(env.seq_lens.values()) - 1
    obs, _ = env.reset()
    device = policy.device
    rewards, equity_curve, actions = [], [], []
    initial_eq = env.portfolio.initial_equity

    for _ in range(n_steps):
        obs_tensor = {tf: th.tensor(obs[tf], dtype=th.float32, device=device).unsqueeze(0) for tf in obs}
        action_mask = th.tensor(env._action_mask(), dtype=th.bool, device=device).unsqueeze(0)
        action, _, _, _ = policy.get_action(obs_tensor, deterministic=True, action_mask=action_mask)
        obs, reward, done, truncated, info = env.step(action.item())

        rewards.append(reward)
        equity_curve.append(info.get("equity", 0.0) / initial_eq)
        actions.append(action.item())

        if done or truncated:
            obs, _ = env.reset()

    rewards = np.nan_to_num(np.array(rewards, dtype=np.float32))
    equity_curve = np.nan_to_num(np.array(equity_curve, dtype=np.float32))
    total_return = equity_curve[-1] - 1.0 if equity_curve.size > 0 else 0.0

    sharpe = np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(288 * 252)
    mdd = max_drawdown(equity_curve)
    return sharpe, mdd, total_return

def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-9)
    return float(-dd.min())
