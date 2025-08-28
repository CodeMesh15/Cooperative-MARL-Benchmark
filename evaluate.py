import torch
import argparse
import time
import pygame

from environment.grid_world import CooperativePushBlockEnv
from environment.visualizer import Visualizer
from maddpg.agent import MADDPGAgent

def main(args):
    # --- 1. Initialization ---
    env = CooperativePushBlockEnv()
    visualizer = Visualizer(grid_size=env.size)
    num_agents = 2
    
    obs_dims = [6, 8]
    action_dims = 5
    
    agents = [MADDPGAgent(obs_dim=obs_dims[i], action_dim=action_dims,
                          full_obs_dim=0, full_action_dim=0) # Dims not needed for eval
              for i in range(num_agents)]

    # Load trained actor models
    try:
        for i, agent in enumerate(agents):
            agent.actor.load_state_dict(torch.load(args.model_paths[i]))
            agent.actor.eval() # Set to evaluation mode
        print("Trained models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}. Please check model paths.")
        return

    # --- 2. Run Visual Demo ---
    print("--- Starting Visual Demostration ---")
    for episode in range(args.num_episodes):
        obs = env.reset()
        done = False
        
        while not done:
            # Handle Pygame events (like closing the window)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # Choose actions greedily (no exploration)
            actions = [agent.choose_action(obs[i]) for i in range(num_agents)]
            
            # Step in environment
            next_obs, _, done, _ = env.step(actions)
            obs = next_obs
            
            # Draw the current state
            env_state = [env.pusher_pos, env.scout_pos, env.block_pos, env.target_pos]
            visualizer.draw(env_state)
            
            time.sleep(0.1) # Slow down for visualization
            
        print(f"Episode {episode + 1} finished.")
        time.sleep(1) # Pause between episodes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained MADDPG agents.")
    parser.add_argument('--model_paths', type=str, nargs='+', required=True,
                        help="Paths to the trained actor models (e.g., models/actor_0.pth models/actor_1.pth).")
    parser.add_argument('--num_episodes', type=int, default=10,
                        help="Number of demo episodes to run.")
    args = parser.parse_args()
    main(args)
