import torch
import torch.nn as nn
import torch.optim as optim
from actor import Actor
from critic import Critic


class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device="cpu",
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2
    ):

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action

        # Actor
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critics
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)

        self.critic1_target = Critic(state_dim, action_dim).to(device)
        self.critic2_target = Critic(state_dim, action_dim).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=1e-3
        )


        self.total_it = 0

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action


    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


    def train(self, replay_buffer, batch_size=256):

        self.total_it += 1

        # Sample replay buffer
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Target Policy Smoothing
        with torch.no_grad():
            # Add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)


            # Compute Target Q

            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)

            # TD3 twin critic min
            target_Q = torch.min(target_Q1, target_Q2)

            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get Current Q Estimates

        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)

        # Critic Loss

        critic_loss = (
                nn.MSELoss()(current_Q1, target_Q) +
                nn.MSELoss()(current_Q2, target_Q)
        )

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # Delayed Actor Updates

        if self.total_it % self.policy_delay == 0:
            # Actor loss (maximize Q)
            actor_loss = -self.critic1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


            # Soft Update Targets

            self.soft_update(self.critic1, self.critic1_target)
            self.soft_update(self.critic2, self.critic2_target)
            self.soft_update(self.actor, self.actor_target)