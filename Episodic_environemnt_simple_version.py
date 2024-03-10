import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym import spaces
from copy import deepcopy
from gymnasium.spaces import Tuple, Discrete, Box
from sklearn.manifold import MDS

class EpisodicEnv(gym.Env):
    def __init__(self, states: pd.DataFrame,
                 semantic_similarities: pd.DataFrame,
                 spatial_similarities: pd.DataFrame,
                 goal_states: pd.DataFrame,
                 k=1, m=1, n=1, o=1):
        super(EpisodicEnv, self).__init__()
        # Initialize environment parameters
        self.states = states
        self.n_states = len(states)
        self.semantic_similarities = semantic_similarities
        self.spatial_similarities = spatial_similarities
        self.k = k
        self.m = m
        self.n = n
        self.o = o
        self.goal_states = goal_states
        self.set_reward(self.goal_states)
        self.goal_state = self.goal_function()
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.n_states)
        self.observation_space = spaces.Discrete(self.n_states)
        # Initialize environment
        self.reset()

    def reset(self, initial_state=None):
        # Reset environment state
        if initial_state is None:
            initial_state_index = np.random.randint(self.n_states)
        else:
            initial_state_index = initial_state
        self.current_state = initial_state_index
        self.current_state_details = self.states.iloc[initial_state_index]
        self.current_step = 0
        self.visited_states = set()
        return self.current_state

    def step(self, action):
        # Perform an environment step
        assert self.action_space.contains(action), f"Invalid action {action}"
        next_state = self.states.iloc[action]
        next_state_index = action
        reward = self.given_reward(self.states.iloc[next_state_index])
        self.current_state = next_state_index
        self.visited_states.add(next_state_index)
        done = next_state_index == self.goal_state
        info = {}  # Additional information
        return next_state_index, reward, done, info

    def given_reward(self, state):
        # Calculate reward for a given state
        if state.name == self.goal_state:
            return 10.0  # Higher reward for reaching the goal
        else:
            return 0.0  # No reward for non-goal states

    def set_reward(self, goal_states):
        # Set rewards for each state based on goal states
        num_states = len(self.states)
        self.rewards = np.zeros(num_states, dtype=np.float32)
        for i, state in self.states.iterrows():
            _, _, rewards_goal = self._compute_reward(state, goal_states)
            self.rewards[i] = rewards_goal

    def goal_function(self):
        # Determine the goal state based on maximum reward
        max_reward_state_index = np.argmax(self.rewards)
        return max_reward_state_index

    def _compute_reward(self, state, goal_states):
        # Compute reward for a given state
        location_rewards = []
        semantic_rewards = []
        for i, row in goal_states.iterrows():
            semantic_sim = self.semantic_similarities.loc[state["word"]][row["word"]]
            semantic_rewards.append(semantic_sim)
            spatial_sim = self.spatial_similarities.loc[state["location"]][row["location"]]
            location_rewards.append(spatial_sim)
        rewards_goal = sum(semantic_rewards) + sum(location_rewards)
        return semantic_rewards, location_rewards, rewards_goal

class DynaQAgent:
    def __init__(self, env, n, alpha=0.1, gamma=0.9, epsilon=0.1):
        # Agent initialization
        self.env = env
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = np.zeros((env.n_states, env.action_space.n))
        self.episode_rewards = []
        self.tot_reward =[]
        self.steps_per_episode = []
        self.accumulated_reward_per_episode = []

    # Methods for agent training
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[state])

    def update_q_values(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_values[next_state])
        self.q_values[state][action] += self.alpha * (reward + self.gamma * self.q_values[next_state][best_next_action] - self.q_values[state][action])

    def planning_step(self, state):
        for _ in range(self.n):
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.update_q_values(state, action, reward, next_state)

    def train(self, num_episodes):
        tot_rew = 0
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_q_values(state, action, reward, next_state)
                self.planning_step(state)
                total_reward += reward
                tot_rew += reward
                self.tot_reward.append(tot_rew)
                state = next_state
                steps += 1
                accumulated_reward_per_episode = total_reward if steps > 0 else 0
                self.accumulated_reward_per_episode.append(accumulated_reward_per_episode)
                if done:
                    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Steps: {steps}")
                    break
            average_reward_per_episode = total_reward / steps if steps > 0 else 0
            self.episode_rewards.append(average_reward_per_episode)
            self.steps_per_episode.append(steps)

def run_simulation(num_runs=1, num_episodes=100):
    for _ in range(num_runs):
        agent = DynaQAgent(env, n=10)
        agent.train(num_episodes)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(agent.episode_rewards, label='Average Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Dyna-Q: Average Reward per Episode')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(agent.steps_per_episode, label='Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Dyna-Q: Steps per Episode')
        plt.legend()

        plt.tight_layout()
        plt.show()

        plt.plot(agent.tot_reward, label='Total Reward')
        plt.xlabel('Step')
        plt.ylabel('Total Reward')
        plt.title('Dyna-Q: Total Reward')
        plt.legend()
        plt.show()

def main():
    # Load data and initialize environment
    states = pd.read_csv('states.csv')
    semantic_similarities = pd.read_csv('semantic_similarities.csv')
    spatial_similarities = pd.read_csv('spatial_similarities.csv')
    goal_states = pd.read_csv('goal_states.csv')

    env = EpisodicEnv(states, semantic_similarities, spatial_similarities, goal_states)

    # Run simulation
    run_simulation(num_runs=1, num_episodes=100)

if __name__ == "__main__":
    main()

