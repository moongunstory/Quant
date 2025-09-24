# train/reinforce/core/sac_lstm_agent.py
import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from ai_binance.train.reinforce.core.lstm_actor_critic import LSTMActor, LSTMCritic


class SACLSTMAgent:
    def __init__(self,
                 input_dims: dict,
                 action_dim,
                 device="cpu",
                 hidden_dim=128,
                 lstm_layers=1,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 tau=0.01,
                 alpha=0.2,
                 total_steps=1_000_000,
                 use_scheduler=True,
                 eta_min=3e-5,
                 target_entropy_scale = 1.0,     
                 alpha_min = 0.02,
                 alpha_max = 0.12,          
                 clip_grad=5.0,
                 use_fixed_alpha=False,
                 fixed_alpha=0.10):
        self.action_dim = int(action_dim)
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.total_steps = int(total_steps)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.clip_grad = float(clip_grad)
        self.use_fixed_alpha = bool(use_fixed_alpha)
        self.fixed_alpha = float(fixed_alpha)

        # --- 네트워크 ---
        self.actor = LSTMActor(input_dims, action_dim, hidden_dim, lstm_layers).to(device)
        self.critic_1 = LSTMCritic(input_dims, action_dim, hidden_dim, lstm_layers).to(device)
        self.critic_2 = LSTMCritic(input_dims, action_dim, hidden_dim, lstm_layers).to(device)
        self.critic_target_1 = deepcopy(self.critic_1).to(device)
        self.critic_target_2 = deepcopy(self.critic_2).to(device)

        # --- Optimizers ---
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opts = [
            torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr),
            torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr),
        ]

        # --- scheduler: keep it but don't kill learning too early ---
        self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_opt, T_max=4*self.total_steps, eta_min=1e-4  # 더 천천히 감소
        )
        self.critic_schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2*self.total_steps, eta_min=1e-4)
            for opt in getattr(self, "critic_opts", [])
        ]

        # --- reward running stats (for TD only) ---
        self.r_mu  = 0.0
        self.r_std = 1.0
        self.reward_scale = 200.0  # gradient magnitude only, optimal policy 불변

        # --- entropy target & alpha autotune ---
        self.target_entropy = -0.5 * action_dim  # 더 높은 엔트로피로 탐험 장려
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(0.2, device=self.device)))
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)

    # ===== Alpha helpers =====
    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().clamp(0.05, 0.3))

    def _alpha_value(self) -> torch.Tensor:
        if self.use_fixed_alpha:
            return torch.tensor(self.fixed_alpha, device=self.device)
        return self.log_alpha.exp().clamp(0.05, 0.3)  # 통일된 범위

    # ===== Utils =====
    def _to_tensor(self, batch_dict):
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch_dict.items()}

    @torch.no_grad()
    def select_action(self, state_seq_dict, deterministic: bool = False):
        """정책에서 행동 샘플. tanh-squash로 [-1,1] 보장."""
        self.actor.eval()
        state_seq_tensor = self._to_tensor({k: v[None, ...] for k, v in state_seq_dict.items()})
        mu, log_std, _ = self.actor(state_seq_tensor)
        if deterministic:
            action = torch.tanh(mu)  # 결정론도 squash
        else:
            std = log_std.exp()
            dist = torch.distributions.Normal(mu, std)
            z = dist.sample()
            action = torch.tanh(z)
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    # ===== Update =====
    def update(self, replay_buffer, batch_size, recent_reward=None):
        state_seq, action_seq, reward_seq, next_state_seq, done_seq = replay_buffer.sample(batch_size)

        state_seq = self._to_tensor(state_seq)
        next_state_seq = self._to_tensor(next_state_seq)
        action_seq = torch.as_tensor(action_seq, dtype=torch.float32, device=self.device)
        reward_seq = torch.as_tensor(reward_seq, dtype=torch.float32, device=self.device)
        done_seq = torch.as_tensor(done_seq, dtype=torch.float32, device=self.device)

        # 마지막 스텝만 사용
        if reward_seq.ndim > 1:
            reward = reward_seq[:, -1:]
            done = done_seq[:, -1:]
        else:
            reward = reward_seq.unsqueeze(-1)
            done = done_seq.unsqueeze(-1)

        if action_seq.ndim > 2:
            action = action_seq[:, -1, :]
        else:
            action = action_seq

        # --- Critic update ---
        with torch.no_grad():
            next_mu, next_log_std, _ = self.actor(next_state_seq)
            next_std = next_log_std.exp()
            next_dist = torch.distributions.Normal(next_mu, next_std)
            # reparameterize + tanh squash
            z_next = next_dist.rsample()
            next_action = torch.tanh(z_next)

            # tanh 로그보정: log_prob(z) - sum log(1 - tanh(z)^2)
            LOG_EPS = 1e-6
            next_log_prob = next_dist.log_prob(z_next).sum(-1, keepdim=True) - torch.log(
                1 - next_action.pow(2) + LOG_EPS
            ).sum(-1, keepdim=True)

            alpha_t = self._alpha_value()

            target_q1, _ = self.critic_target_1(next_state_seq, next_action)
            target_q2, _ = self.critic_target_2(next_state_seq, next_action)
            target_q = torch.min(target_q1, target_q2) - alpha_t * next_log_prob

            # r: raw env reward (그대로 받아옴)
            r = reward

            # 보상 정규화 완전 제거 - 원본 보상 그대로 사용
            r = reward
            target = r + (1.0 - done) * self.gamma * target_q

            target = r + (1.0 - done) * self.gamma * target_q

        current_q1, _ = self.critic_1(state_seq, action)
        current_q2, _ = self.critic_2(state_seq, action)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        for opt in self.critic_opts:
            opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        clip_grad_norm_(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), max_norm=5.0)
        for opt in self.critic_opts:
            opt.step()

        # --- Actor update ---
        mu, log_std, _ = self.actor(state_seq)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        new_action = torch.tanh(z)

        LOG_EPS = 1e-6
        log_prob = dist.log_prob(z).sum(-1, keepdim=True) - torch.log(
            1 - new_action.pow(2) + LOG_EPS
        ).sum(-1, keepdim=True)

        q1, _ = self.critic_1(state_seq, new_action)
        q2, _ = self.critic_2(state_seq, new_action)
        q = torch.min(q1, q2)

        alpha_t = self._alpha_value()
        actor_loss = (alpha_t * log_prob - q).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
        self.actor_opt.step()

        # --- Alpha 자동 튜닝(고정 모드면 skip) ---
        if not self.use_fixed_alpha:
            entropy = -log_prob
            alpha_loss = -(self.log_alpha * (entropy.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.tensor(0.0, device=self.device)

        # --- Soft update ---
        self.soft_update(self.critic_target_1, self.critic_1)
        self.soft_update(self.critic_target_2, self.critic_2)

        # --- Scheduler step (옵션) ---
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        for sched in self.critic_schedulers:
            sched.step()

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha": float(alpha_t.detach().cpu()),
            "policy_entropy": float((-log_prob).detach().mean().cpu()),
            "log_std_mean": float(log_std.detach().mean().cpu()),
        }

    def soft_update(self, target_net, source_net):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def set_target_entropy_scale(self, scale: float):
        """Scale ∈ [0.6, 1.0]; target entropy = base * scale."""
        self.target_entropy_scale = float(np.clip(scale, 0.6, 1.0))

    # 런타임에서 정책 분산 범위를 조절할 수 있게 훅 제공
    @torch.no_grad()
    def set_log_std_bounds(self, min_v: float, max_v: float):
        if hasattr(self.actor, "set_log_std_bounds"):
            self.actor.set_log_std_bounds(min_v, max_v)
