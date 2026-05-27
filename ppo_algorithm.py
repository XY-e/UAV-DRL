# PPO algorithm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

# Algorithm
## Actor-Critic Neural Network
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # self.log_std_min = -5.0
        # self.log_std_max = 1.0

        # holonomic control
        self.log_std_min = -3.5
        self.log_std_max = -0.3

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.mu = nn.Linear(256, action_dim)
        self.sigma = nn.Linear(256, action_dim)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        """Inference / TorchScript export path — returns deterministic mean action."""
        x = self.actor(state)
        mu = self.mu(x)
        return mu

    def act(self, state, deterministic=False):
        x = self.actor(state)

        mu = self.mu(x)
        sigma = torch.clamp(self.sigma(x), self.log_std_min, self.log_std_max).exp()

        dist = Normal(mu, sigma)
        action = mu if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)

        return action, log_prob

    def evaluate(self, state, action):
        x = self.actor(state)
        mu = self.mu(x)
        sigma = torch.clamp(self.sigma(x), self.log_std_min, self.log_std_max).exp()
        dist = Normal(mu, sigma)
        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(state)
        return log_prob, entropy, value

## PPO Agent
class PPOAgent:

    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.lr = 3e-4

        # self.K_epochs = 6
        
        # holonomic control
        self.K_epochs = 4

        # self.entropy_coef_start = 0.05
        # self.entropy_coef_end = 0.02
        
        # holonomic control
        self.entropy_coef_start = 0.02
        self.entropy_coef_end = 0.005

        self.entropy_coef = self.entropy_coef_start

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.max_grad_norm = 0.5

    def set_training_progress(self, progress):
        """Linearly decay entropy coefficient as training progresses."""
        p = float(np.clip(progress, 0.0, 1.0))
        self.entropy_coef = self.entropy_coef_start + (
            self.entropy_coef_end - self.entropy_coef_start
        ) * p

    def update(self, buffer, mini_batch_size=256):

        states = torch.from_numpy(
            np.stack(buffer.states, axis=0).astype(np.float32, copy=False)
        ).to(self.device)
        actions = torch.from_numpy(
            np.stack(buffer.actions, axis=0).astype(np.float32, copy=False)
        ).to(self.device)
        old_log_probs = torch.stack(buffer.logprobs).to(self.device)

        rewards = buffer.rewards
        dones = buffer.dones

        with torch.no_grad():
            _, _, old_val = self.policy_old.evaluate(states, actions)
            old_values = old_val.squeeze(-1).detach()

        old_values_list = old_values.cpu().numpy().reshape(-1).tolist()

        advantages_np = compute_gae(rewards, old_values_list, dones)
        advantages = torch.as_tensor(
            advantages_np, dtype=torch.float32, device=self.device
        )
        returns = advantages + old_values

        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

        n = states.shape[0]

        for _ in range(self.K_epochs):
            indices = torch.randperm(n, device=self.device)

            for start in range(0, n, mini_batch_size):
                mb_idx = indices[start : start + mini_batch_size]

                mb_states      = states[mb_idx]
                mb_actions     = actions[mb_idx]
                mb_old_lp      = old_log_probs[mb_idx].detach()
                mb_advantages  = advantages[mb_idx]
                mb_returns     = returns[mb_idx].detach()

                log_probs, entropy, state_values = self.policy.evaluate(mb_states, mb_actions)

                ratios = torch.exp(log_probs - mb_old_lp)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages

                value_pred = state_values.squeeze(-1)
                value_loss = self.MseLoss(value_pred, mb_returns)

                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

## State Space Normalization
def normalize_state(state, state_mean, state_std):
    return (state - state_mean) / (state_std + 1e-8)

## Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = np.append(np.array(values), 0)

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.append(gae)
    return np.array(advantages[::-1])
