# Cooperative-MARL-Benchmark

An implementation of a cooperative, heterogeneous multi-agent reinforcement learning (MARL) platform. This project provides an environment and benchmark models for training multiple, different agents to collaborate on a shared task, inspired by work at Amazon Alexa.

---

## 1. Project Overview

This project builds a benchmark for **cooperative, heterogeneous multi-agent reinforcement learning**. In simple terms, we're creating a simulated environment where multiple AI agents with different abilities must learn to work together to achieve a common goal. This is a complex task that goes beyond single-agent RL, as agents must learn to communicate or infer the intentions of others to succeed. The goal is to replicate the process of creating a platform for studying this kind of emergent cooperation.

---

## 2. Core Objectives

-   To create a simple but effective multi-agent grid-world environment.
-   To define a cooperative task that requires agents with **heterogeneous** (different) abilities to collaborate.
-   To implement a state-of-the-art multi-agent reinforcement learning algorithm, such as **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**.
-   To train the agents and visualize their learned cooperative behaviors.

---

## 3. Methodology

#### Phase 1: The Multi-Agent Environment

1.  **Environment Design**: We will build a simple 2D grid-world environment using a library like `Pygame` or `PettingZoo`. The environment will contain:
    -   Two or more agents.
    -   Various objects or landmarks.
    -   A specific goal state.
2.  **The Cooperative Heterogeneous Task**: A classic task that fits this description is "cooperative push-block."
    -   **Goal**: Push a heavy block to a target location.
    -   **Heterogeneity**:
        -   **Agent 1 ("Pusher")**: This agent is strong and is the only one that can push the heavy block. However, it is "blind" and cannot see the target location.
        -   **Agent 2 ("Scout")**: This agent is weak and cannot push the block. However, it can see the entire environment and knows the location of the target.
    -   **Cooperation**: The Pusher must learn to move the block based on the Scout's position or signals, and the Scout must learn to position itself to guide the Pusher to the goal.

#### Phase 2: The MARL Algorithm (MADDPG)

Standard RL algorithms like Q-Learning or PPO fail in multi-agent settings because the environment becomes non-stationary from each agent's perspective. We'll implement **MADDPG**, a popular algorithm for this.

1.  **Architecture**: Each agent will have its own **Actor** network (which decides on an action) and a **Critic** network (which evaluates how good that action is).
2.  **Centralized Critic, Decentralized Actor**:
    -   **Decentralized Actor (Execution)**: During execution, each agent only uses its own Actor network and its local observations to choose an action.
    -   **Centralized Critic (Training)**: During training, the Critic for each agent gets to see the observations and actions of *all* other agents. This allows the Critic to learn a stable value function, which in turn helps the Actor learn a better policy.

#### Phase 3: Training and Visualization

1.  **Training Loop**:
    -   At each step, all agents observe the environment and choose an action using their Actors.
    -   The environment executes the actions and returns a new state and a shared reward (e.g., a reward is given to all agents if the block gets closer to the goal).
    -   The `(state, action, reward, next_state)` tuples for all agents are stored in a replay buffer.
    -   Periodically, a batch of experiences is sampled, and the Actor and Critic networks for all agents are updated.
2.  **Visualization**: We will use `Pygame` to render the grid-world, showing the agents, the block, and the target, allowing us to visually inspect the learned cooperative policies.

---

## 4. Project Structure
```text
/cooperative-marl-benchmark
|
|-- /environment/
|   |-- grid_world.py           # The main environment class (handles state, actions, rewards)
|   |-- visualizer.py           # Pygame-based renderer for the environment
|
|-- /maddpg/
|   |-- agent.py                # Class defining a single MADDPG agent (with an Actor and Critic)
|   |-- replay_buffer.py        # Buffer to store experiences
|
|-- train.py                      # Main script to run the training loop
|-- evaluate.py                   # Script to load trained models and run a visual demo
|
|-- requirements.txt
|-- README.md
```
