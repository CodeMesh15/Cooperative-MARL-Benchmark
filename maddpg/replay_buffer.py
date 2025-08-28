
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dims, action_dims, num_agents):
        self.capacity = capacity
        self.num_agents = num_agents
        self.pointer = 0
        self.size = 0

        self.obs = np.zeros((capacity, num_agents, obs_dims))
        self.actions = np.zeros((capacity, num_agents, action_dims))
        self.rewards = np.zeros((capacity, 1))
        self.next_obs = np.zeros((capacity, num_agents, obs_dims))
        self.dones = np.zeros((capacity, 1))

    def store_transition(self, obs, actions, reward, next_obs, done):
        idx = self.pointer % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = done
        
        self.pointer += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_obs=self.next_obs[idxs],
            dones=self.dones[idxs]
        )
