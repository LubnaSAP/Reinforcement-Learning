import torch
from environment import EpisodicEnv


class Generator:
    def __init__(self, environment: EpisodicEnv, jump_rate=15.0, symmetrize=False):
        self.environment = environment
        self.n_states = self.environment.n_states
        self.jump_rate = jump_rate
        self.symmetrize = symmetrize

        self.states = None
        self.transitions = None
        self.T = self.environment.transition_matrix
        self.O = stochastic_to_generator(self.T, self.jump_rate)

        self.G, self.V, self.W = self.compute_eigenvectors()

    def compute_eigenvectors(self):
        eigenvalues, eigenvectors = torch.linalg.eig(self.O)
        eigenvectors = eigenvectors.detach()

        G = torch.real(eigenvectors)
        V = torch.real(eigenvalues)
        W = torch.real(eigenvectors.inverse())
        return G, V, W

    def spectral_matrix(self):
        return torch.einsum('ij, jk->jik', self.G, self.W)


def check_generator(Q: torch.Tensor):
    row_sums = Q.sum(dim=1)
    if not torch.allclose(row_sums, torch.zeros_like(row_sums)):
        raise ValueError("Row sums must be 0")

    if not torch.all(Q.diag() <= torch.zeros_like(Q.diag())):
        raise ValueError("Diagonal must be non-positive")

    if not torch.all(Q[~torch.eye(Q.shape[0], dtype=torch.bool)] >= torch.zeros_like(Q[~torch.eye(Q.shape[0], dtype=torch.bool)])):
        raise ValueError("Non-diagonal must be non-negative")

    return True


def stochastic_to_generator(transition_matrix: torch.Tensor, jump_rate=15.0):
    return jump_rate * (transition_matrix - torch.eye(len(transition_matrix)))


def generator_to_weighted(Q: torch.Tensor):
    W = Q.clone()
    W[W < 0] = 0
    return W

