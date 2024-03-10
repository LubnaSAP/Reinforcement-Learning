import torch
from torch.distributions import Categorical


class Simulator:
    def __init__(self, propagator, no_dwell=True, mass=1):
        self.no_dwell = no_dwell
        if not isinstance(mass, int):
            raise ValueError("Mass must be an integer")
        self.mass = mass
        self.propagator = propagator

    def sample_state(self, rho, prev_states=None):
        if self.no_dwell and self.mass != 0 and prev_states is not None:
            except_states = prev_states.type(torch.int64)[-min(self.mass, len(prev_states)):]
            rho[except_states] = 0.
            rho /= rho.sum()

        m = Categorical(rho)
        state = m.sample()
        log_prob = m.log_prob(state)
        reward = self.propagator.generator.environment.rewards[state] if self.propagator.generator.environment.rewards is not None else 0
        return state, log_prob, reward

    @staticmethod
    def norm_density(V, beta=1.0, type="l1"):
        V[torch.isinf(V)] = 0.0
        if (V < 0).any():
            V -= V.min()
        if type == "l1":
            P = V / V.sum()
        else:
            raise ValueError("Unknown normalization requested.")
        return P

    @staticmethod
    def check_state_distribution(rho):
        rho[rho < 0] = 0
        rho[rho > 1] = 1
        rho /= rho.sum()
        return rho

    def process_rho(self, state, n_states):
        rho_inter = torch.zeros(n_states)
        rho_inter[state] = 1.0
        return rho_inter

    def evolve(self, rho_start):
        rho_stop = rho_start @ self.propagator.P
        rho_stop = self.check_state_distribution(rho_stop)
        rho_stop = self.norm_density(rho_stop, type='l1')
        return rho_stop

    def sample_states(self, num_states=100, rho_start=None):
        if rho_start is None:
            rho_start = torch.ones(len(self.propagator.generator.environment.states))
            rho_start /= rho_start.sum()

        state, log_prob, reward = self.sample_state(rho_start)
        states, log_probs, rewards = [state], [log_prob], [reward]
        rho_inter = self.process_rho(state, len(rho_start))

        for _ in range(1, num_states):
            rho_stop = self.evolve(rho_inter)
            state, log_prob, reward = self.sample_state(rho_stop, prev_states=torch.Tensor(states))
            states.append(state)
            log_probs.append(log_prob)
            rewards.append(reward)
            rho_inter = self.process_rho(state, len(rho_start))

        return states, log_probs, rewards

