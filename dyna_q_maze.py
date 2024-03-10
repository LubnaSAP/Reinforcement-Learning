import numpy as np
import matplotlib.pyplot as plt
import maze

BLOCKING_MAZE1 = ['############',
                  '#          #',
                  '#          #',
                  '#          #',
                  '########## #',
                  '#          #',
                  '#   P      #',
                  '############']
BLOCKING_MAZE2 = ['############',
                  '#          #',
                  '#          #',
                  '#          #',
                  '# ##########',
                  '#          #',
                  '#   P      #',
                  '############']

class DynaQ:
    def __init__(self, game, n, alpha, gamma, epsilon, max_steps):
        self.game = game
        self.env = game.make(BLOCKING_MAZE1)
        self.q = np.zeros((self.env.observation_space.n, self.env.action_space.n))
        self.epsilon = epsilon
        self.model = Model(self.env.observation_space.n, self.env.action_space.n)
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.max_steps = max_steps

    def learn(self):
        s = self.env.reset()
        cum_reward = [0]

        for step in range(self.max_steps):
            if np.random.uniform() < self.epsilon:
                a = self.env.action_space.sample()
            else:
                a = np.random.choice(np.where(self.q[s] == np.max(self.q[s]))[0])

            s_prime, r, done, info = self.env.step(a)
            self.q[s, a] += self.alpha * (r + self.gamma * np.max(self.q[s_prime]) - self.q[s, a])
            self.model.add(s, a, s_prime, r)

            self.planning()

            s = s_prime

            if done:
                s = self.env.reset()

            if step == 1000:
                self.env = self.game.make(BLOCKING_MAZE2)
                s = self.env.reset()

            cum_reward.append(cum_reward[-1] + r)

        return np.array(cum_reward[1:])

    def planning(self):
        for _ in range(self.n):
            s, a = self.model.sample()
            s_prime, r = self.model.step(s, a)
            self.q[s, a] += self.alpha * (r + self.gamma * np.max(self.q[s_prime]) - self.q[s, a])

class Model:
    def __init__(self, n_states, n_actions):
        self.transitions = np.zeros((n_states, n_actions), dtype=np.uint8)
        self.rewards = np.zeros((n_states, n_actions))

    def add(self, s, a, s_prime, r):
        self.transitions[s, a] = s_prime
        self.rewards[s, a] = r

    def sample(self):
        s = np.random.choice(np.where(np.sum(self.transitions, axis=1) > 0)[0])
        a = np.random.choice(np.where(self.transitions[s] > 0)[0])
        return s, a

    def step(self, s, a):
        s_prime = self.transitions[s, a]
        r = self.rewards[s, a]
        return s_prime, r

def plot_data(y):
    x = np.arange(y.size)
    _ = plt.plot(x, y, '-')
    plt.show()

def multi_plot_data(data, names):
    x = np.arange(data[0].size)
    for i, y in enumerate(data):
        plt.plot(x, y, '-', markersize=2, label=names[i])
    plt.legend(loc='lower right', prop={'size': 16}, numpoints=5)
    plt.show()

def main():
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.3
    max_steps = 3000
    trials = 3

    dynaq_5_r = np.zeros((trials, max_steps))
    dynaq_50_r = np.zeros((trials, max_steps))
    qlearning_r = np.zeros((trials, max_steps))

    for t in range(trials):
        n = 5
        agent = DynaQ(maze, n, alpha, gamma, epsilon, max_steps)
        dynaq_5_r[t] = agent.learn()

        n = 50
        agent = DynaQ(maze, n, alpha, gamma, epsilon, max_steps)
        dynaq_50_r[t] = agent.learn()

        n = 0
        agent = DynaQ(maze, n, alpha, gamma, epsilon, max_steps)
        qlearning_r[t] = agent.learn()

    dynaq_5_r = np.mean(dynaq_5_r, axis=0)
    dynaq_50_r = np.mean(dynaq_50_r, axis=0)
    qlearning_r = np.mean(qlearning_r, axis=0)

    data = [dynaq_5_r, dynaq_50_r, qlearning_r]
    names = ["DynaQ, n=5", "DynaQ, n=50", "Q-Learning"]
    multi_plot_data(data, names)

if __name__ == "__main__":
    main()

