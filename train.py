
import torch
import numpy as np
import argparse
from tqdm import tqdm
import os

from environment.grid_world import CooperativePushBlockEnv
from maddpg.replay_buffer import ReplayBuffer
from maddpg.agent import MADDPGAgent, Actor, Critic # Assuming MADDPGAgent holds networks and optimizers

def update_agents(replay_buffer, batch_size, agents, gamma, device):
    """
    Samples a batch from the replay buffer and updates all agents.
    """
    batch = replay_buffer.sample(batch_size)
    
    obs_batch = torch.FloatTensor(batch['obs']).to(device)
    action_batch = torch.FloatTensor(batch['actions']).to(device)
    reward_batch = torch.FloatTensor(batch['rewards']).to(device)
    next_obs_batch = torch.FloatTensor(batch['next_obs']).to(device)
    done_batch = torch.FloatTensor(batch['dones']).to(device)

    # The full observation and action tensors are needed for the centralized critic
    full_obs_batch = obs_batch.view(batch_size, -1)
    full_action_batch = action_batch.view(batch_size, -1)
    full_next_obs_batch = next_obs_batch.view(batch_size, -1)
    
    for agent_idx, agent in enumerate(agents):
        # --- Critic Update ---
        agent.critic_optimizer.zero_grad()
        
        # Calculate target Q-value
        with torch.no_grad():
            # Get next actions from target actors
            next_actions = torch.cat([agents[i].target_actor(next_obs_batch[:, i, :]) for i in range(len(agents))], dim=1)
            target_q = agent.target_critic(full_next_obs_batch, next_actions)
            target_q = reward_batch + gamma * (1 - done_batch) * target_q
        
        current_q = agent.critic(full_obs_batch, full_action_batch)
        
        critic_loss = torch.nn.functional.mse_loss(current_q, target_q)
        critic_loss.backward()
        agent.critic_optimizer.step()
        
        # --- Actor Update ---
        agent.actor_optimizer.zero_grad()
        
        # Get actions for current state from regular actors
        current_actions = torch.cat([agents[i].actor(obs_batch[:, i, :]) for i in range(len(agents))], dim=1)
        actor_loss = -agent.critic(full_obs_batch, current_actions).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

    # --- Soft update for all target networks ---
    for agent in agents:
        agent.update_target_networks()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- 1. Initialization ---
    env = CooperativePushBlockEnv()
    num_agents = 2
    
    # Pusher (Agent 0): 6 obs dims (pos, rel_scout, rel_block)
    # Scout (Agent 1): 8 obs dims (pos, rel_pusher, rel_block, rel_target)
    obs_dims = [6, 8]
    action_dims = 5 # Discrete actions
    
    agents = [MADDPGAgent(obs_dim=obs_dims[i], action_dim=action_dims,
                          full_obs_dim=sum(obs_dims), full_action_dim=action_dims*num_agents) 
              for i in range(num_agents)]
    
    for agent in agents:
        agent.actor.to(device)
        agent.critic.to(device)
        agent.target_actor.to(device)
        agent.target_critic.to(device)

    buffer = ReplayBuffer(capacity=1_000_000, obs_dims=max(obs_dims), action_dims=1, num_agents=num_agents)

    # --- 2. Training Loop ---
    print("--- Starting Training ---")
    for episode in tqdm(range(args.num_episodes)):
        obs = env.reset()
        for step in range(env.max_steps):
            # Choose actions
            actions = [agents[i].choose_action(obs[i]) for i in range(num_agents)]
            
            # Step in environment
            next_obs, reward, done, _ = env.step(actions)
            
            # Store transition in replay buffer
            # Reshape actions for buffer
            actions_reshaped = [[a] for a in actions]
            buffer.store_transition(obs, actions_reshaped, reward, next_obs, done)
            
            # Update agents
            if buffer.size >= args.batch_size:
                update_agents(buffer, args.batch_size, agents, args.gamma, device)
                
            obs = next_obs
            
            if done:
                break
                
    # --- 3. Save Models ---
    os.makedirs(args.model_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(), os.path.join(args.model_dir, f'actor_{i}.pth'))
    print(f"Models saved to '{args.model_dir}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MADDPG agents.")
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--model_dir', type=str, default='models')
    args = parser.parse_args()
    main(args)
