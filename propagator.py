import torch
import numpy as np

from scipy.optimize import minimize
from functools import partial

from autocorrelation import acf_sum, constraints_stochmat


class Propagator:
    def __init__(self, generator, sigma=1.0, tau=1.0):
        self.generator = generator
        self.n_states = self.generator.n_states
        self.sigma = float(sigma)
        self.tau = float(tau)

        if self.sigma <= 0:
            raise ValueError("Sigma must be greater than 0")
        
        if self.tau <= 0:
            raise ValueError("Tau must be greater than 0")

        self.compute_kernels()

    def compute_kernels(self, power_spec=None):
        if power_spec is None:
            self.S = torch.diag(torch.exp(self.generator.V / self.tau))
        else:
            self.S = torch.diag(power_spec)

        self.P = self.generator.G @ self.S @ self.generator.W
        self.P = torch.real(self.P)

    def min_auto_cf(self, T=2, lags=(1, 2), rho_init=None, maxiter=1000):
        if isinstance(rho_init, int):
            rho0 = torch.zeros(self.n_states)
            rho0[rho_init] = 1.0
        elif rho_init.size == self.n_states:
            rho0 = rho_init
        else:
            raise ValueError("Unknown setting for initial distribution in acf calculation")

        x0 = np.diag(self.S)
        W = np.array(self.generator.spectral_matrix())
        fun = partial(acf_sum, W=W, T=T, deltaT=lags, rho=rho0)
        options = {"maxiter": maxiter}

        lc1_stochmat, lc2_stochmat = constraints_stochmat(W)

        opt = minimize(fun, x0, method=None, constraints=[lc1_stochmat, lc2_stochmat], tol=None, callback=None, options=options)

        s_opt = torch.Tensor(opt.x)

        self.compute_kernels(power_spec=s_opt)

