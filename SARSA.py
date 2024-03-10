import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from NEW_environment import EpisodicEnv

class SARSA:
    def __init__(self, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
        self.lr = lr
        self.num_episodes = num_episodes
        self.eps = eps
        self.gamma = gamma
        self.eps_decay = eps_decay

    def preprocess_data(self, similarities_df):
        similarities_df[similarities_df < 0] = 0
        similarities_df[similarities_df > 1] = 1
        return similarities_df

    def perform_mds(self, similarities_df):
        mds = MDS(n_components=1, dissimilarity='precomputed', random_state=0)
        return mds.fit_transform(similarities_df).reshape(-1)

    def run_episodes(self, env, Q, num_episodes=100, to_print=False):
        # Implementation of run_episodes function here

    def eps_greedy(self, Q, s, eps=0.1):
        # Implementation of eps_greedy function here

    def greedy(self, Q, s):
        # Implementation of greedy function here

    def train(self, env):
        # Implementation of the training loop here
        return Q, games_reward, test_rewards

    def visualize_q_values(self, Q_values, states_to_visualize):
        # Implementation of visualize_q_values function here

    def visualize_rewards(self, test_rewards):
        # Implementation of visualize_rewards function here

    def visualize_rewards_cum(self, test_rewards):
        # Implementation of visualize_rewards_cum function here

def main():
    df_words_similarities = pd.read_csv('words_similarities.csv', index_col=0)
    df_locations_similarities = pd.read_csv('locations_similarities.csv', index_col=0)
    df_env_states = pd.read_csv('environment_states.csv', index_col=0)
    df_env_novel_states = pd.read_csv('environment_novel_states.csv', index_col=0)

    sarsa_agent = SARSA(lr=0.1, num_episodes=10, eps=0.1, gamma=0.95, eps_decay=0.00005)

    df_words_similarities = sarsa_agent.preprocess_data(df_words_similarities)
    df_locations_similarities = sarsa_agent.preprocess_data(df_locations_similarities)

    semantic_mds = sarsa_agent.perform_mds(df_words_similarities)
    spatial_mds = sarsa_agent.perform_mds(df_locations_similarities)

    df_env_states['time'] = df_env_states['time'] / df_env_states['time'].max()

    env = EpisodicEnv(states=df_env_states, semantic_similarities=df_words_similarities,
                      spatial_similarities=df_locations_similarities, goal_states=df_env_novel_states,
                      k=1, m=1, n=1)

    Q_sarsa, games_reward, test_rewards = sarsa_agent.train(env)

    sarsa_agent.visualize_rewards_cum(test_rewards)

if __name__ == "__main__":
    main()

