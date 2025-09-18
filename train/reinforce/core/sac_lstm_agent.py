# train/reinforce/core/sac_lstm_agent.py

import torch
import torch.nn.functional as F
from ai_binance.train.reinforce.core.lstm_actor_critic import LSTMActor, LSTMCritic
from copy import deepcopy


class SACLSTMAgent:
    def __init__(self, input_dims: dict, action_dim, device="cpu", hidden_dim=128,
                 actor_lr=3e-4, critic_lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = LSTMActor(input_dims, action_dim, hidden_dim).to(device)
        self.critic_1 = LSTMCritic(input_dims, action_dim, hidden_dim).to(device)
        self.critic_2 = LSTMCritic(input_dims, action_dim, hidden_dim).to(device)

        self.critic_target_1 = deepcopy(self.critic_1).to(device)
        self.critic_target_2 = deepcopy(self.critic_2).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)
        self.critic_opt_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

    def _to_tensor(self, batch_dict):
        return {k: torch.FloatTensor(v).to(self.device) for k, v in batch_dict.items()}

    def select_action(self, state_seq_dict):
        self.actor.eval()
        with torch.no_grad():
            state_seq_tensor = self._to_tensor({k: v[None, ...] for k, v in state_seq_dict.items()})
            mu, log_std, _ = self.actor(state_seq_tensor)
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    def update(self, replay_buffer, batch_size):
        state_seq, action_seq, reward_seq, next_state_seq, done_seq = replay_buffer.sample(batch_size)

        state_seq = self._to_tensor(state_seq)
        next_state_seq = self._to_tensor(next_state_seq)
        action_seq = torch.FloatTensor(action_seq).to(self.device)
        reward_seq = torch.FloatTensor(reward_seq).to(self.device)
        done_seq = torch.FloatTensor(done_seq).to(self.device)

        # SAC는 단일 transition을 사용하므로 마지막 시점만 추출
        if len(reward_seq.shape) > 1:  # 시퀀스 형태라면
            reward = reward_seq[:, -1:]  # (batch_size, 1, 1) or (batch_size, 1)
            done = done_seq[:, -1:]      # (batch_size, 1, 1) or (batch_size, 1)
        else:  # 이미 단일 값이라면
            reward = reward_seq.unsqueeze(-1) # (batch_size, 1)
            done = done_seq.unsqueeze(-1)     # (batch_size, 1)

        # Ensure reward and done are (batch_size, 1)
        if reward.dim() == 3: # If it's (batch_size, 1, 1)
            reward = reward.squeeze(1) # -> (batch_size, 1)
        if done.dim() == 3: # If it's (batch_size, 1, 1)
            done = done.squeeze(1) # -> (batch_size, 1)

        # 액션도 시퀀스라면 마지막 시점만 추출
        if len(action_seq.shape) > 2:  # (batch, seq, action_dim)
            action = action_seq[:, -1, :]  # 마지막 시점만
        else:
            action = action_seq

        # Critic update
        with torch.no_grad():
            next_mu, next_log_std, _ = self.actor(next_state_seq)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mu, next_std)
            next_action = next_dist.rsample()
            next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)

            target_q1, _ = self.critic_target_1(next_state_seq, next_action)
            target_q2, _ = self.critic_target_2(next_state_seq, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target = reward + (1 - done) * self.gamma * target_q

        current_q1, _ = self.critic_1(state_seq, action)
        current_q2, _ = self.critic_2(state_seq, action)

        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        self.critic_opt_1.zero_grad()
        self.critic_opt_2.zero_grad()
        critic_loss.backward()
        self.critic_opt_1.step()
        self.critic_opt_2.step()

        # Actor update
        mu, log_std, _ = self.actor(state_seq)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        new_action = dist.rsample()
        log_prob = dist.log_prob(new_action).sum(-1, keepdim=True)

        q1, _ = self.critic_1(state_seq, new_action)
        q2, _ = self.critic_2(state_seq, new_action)
        q = torch.min(q1, q2)

        actor_loss = (self.alpha * log_prob - q).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.critic_target_1, self.critic_1)
        self.soft_update(self.critic_target_2, self.critic_2)

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)