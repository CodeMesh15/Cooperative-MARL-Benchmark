
import torch
import torch.nn as nn
import torch.optim as optim

class Actor(nn.Module):
    """Simple MLP Actor network."""
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1) # For discrete actions
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    """
    Simple MLP Critic network. Takes concatenated observations and
    actions of all agents as input.
    """
    def __init__(self, full_obs_dim, full_action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(full_obs_dim + full_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, obs, actions):
        x = torch.cat([obs, actions], dim=1)
        return self.net(x)

class MADDPGAgent:
    """The main agent class for MADDPG."""
    def __init__(self, obs_dim, action_dim, full_obs_dim, full_action_dim,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.01):
        
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(full_obs_dim, full_action_dim)
        self.target_actor = Actor(obs_dim, action_dim)
        self.target_critic = Critic(full_obs_dim, full_action_dim)
        
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.tau = tau

    def choose_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        action_probs = self.actor(obs)
        action = torch.multinomial(action_probs, 1).item()
        return action
        
    def update_target_networks(self):
        """Soft update for target networks."""
        for target, source in zip(self.target_actor.parameters(), self.actor.parameters()):
            target.data.copy_(self.tau * source.data + (1.0 - self.tau) * target.data)
        for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
            target.data.copy_(self.tau * source.data + (1.0 - self.tau) * target.data)
