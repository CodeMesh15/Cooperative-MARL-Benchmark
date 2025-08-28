
import numpy as np

class CooperativePushBlockEnv:
    """
    A simple grid-world environment for a cooperative, heterogeneous MARL task.
    - Agent 0 (Pusher): Can push the block, cannot see the target.
    - Agent 1 (Scout): Can see the target, cannot push the block.
    """
    def __init__(self, size=10, max_steps=100):
        self.size = size
        self.max_steps = max_steps
        self.action_space = [0, 1, 2, 3, 4] # 0:Up, 1:Down, 2:Left, 3:Right, 4:Stay
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}

    def reset(self):
        """Resets the environment to a random initial state."""
        self.step_count = 0
        
        # Place entities randomly, ensuring no overlap
        positions = np.random.choice(self.size * self.size, 4, replace=False)
        self.pusher_pos = np.array([positions[0] // self.size, positions[0] % self.size])
        self.scout_pos = np.array([positions[1] // self.size, positions[1] % self.size])
        self.block_pos = np.array([positions[2] // self.size, positions[2] % self.size])
        self.target_pos = np.array([positions[3] // self.size, positions[3] % self.size])
        
        return self._get_obs()

    def _get_obs(self):
        """Returns observations for each agent."""
        # Pusher's observation: its own pos, and relative pos of scout and block
        pusher_obs = np.concatenate([
            self.pusher_pos,
            self.scout_pos - self.pusher_pos,
            self.block_pos - self.pusher_pos
        ])
        
        # Scout's observation: its own pos, and relative pos of pusher, block, and target
        scout_obs = np.concatenate([
            self.scout_pos,
            self.pusher_pos - self.scout_pos,
            self.block_pos - self.scout_pos,
            self.target_pos - self.scout_pos
        ])
        
        return [pusher_obs, scout_obs]

    def step(self, actions):
        """Executes a step for both agents."""
        self.step_count += 1
        
        # --- 1. Move Scout ---
        scout_move = self.action_map[actions[1]]
        new_scout_pos = np.clip(self.scout_pos + scout_move, 0, self.size - 1)
        # Scout cannot occupy the same space as the block or pusher
        if not np.array_equal(new_scout_pos, self.block_pos) and not np.array_equal(new_scout_pos, self.pusher_pos):
            self.scout_pos = new_scout_pos
            
        # --- 2. Move Pusher and Block ---
        pusher_move = self.action_map[actions[0]]
        new_pusher_pos = np.clip(self.pusher_pos + pusher_move, 0, self.size - 1)
        
        # Check if pusher is trying to push the block
        if np.array_equal(new_pusher_pos, self.block_pos):
            new_block_pos = np.clip(self.block_pos + pusher_move, 0, self.size - 1)
            # Push is successful if the new block position is valid and not occupied
            if not np.array_equal(new_block_pos, self.scout_pos):
                self.block_pos = new_block_pos
                self.pusher_pos = new_pusher_pos # Pusher moves with the block
        elif not np.array_equal(new_pusher_pos, self.scout_pos) and not np.array_equal(new_pusher_pos, self.block_pos):
            # Normal move if not pushing and not colliding
            self.pusher_pos = new_pusher_pos

        # --- 3. Calculate Reward ---
        dist_to_target = np.linalg.norm(self.block_pos - self.target_pos)
        reward = -dist_to_target  # Negative reward for distance
        
        # --- 4. Check for Done ---
        done = False
        if np.array_equal(self.block_pos, self.target_pos):
            reward += 100  # Large reward for reaching the target
            done = True
        elif self.step_count >= self.max_steps:
            done = True
            
        return self._get_obs(), reward, done, {}
