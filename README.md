This repository hosts a collection of Python scripts and classes aimed at various aspects of reinforcement learning, simulation, and related algorithms. 

To be able to fully run all the files you need to create and use a file containing the states/nodes of the initial memory events (location, semantic, temporal and episodic variables) and another one for the novel states/nodes. 

Contents:

    Simulator Class: The Simulator class provides functionality to simulate state transitions based on a given propagator. It allows for sampling states and computing state distributions.

    Propagator Class: The Propagator class computes propagator kernels based on a given generator and provides methods to minimize autocorrelation and compute power spectra.

    Generator Class: The Generator class generates spectral matrices based on a given environment and jump rate. It also provides methods to convert stochastic matrices to generator matrices.

    Environment Class: The Environment class represents an episodic environment for reinforcement learning tasks. It provides methods to interact with the environment, including resetting and stepping through episodes.

    DynaQ Implementation: This implementation of DynaQ is a reinforcement learning algorithm that combines Q-learning with planning using a learned model of the environment.

    SARSA Implementation: This implementation of SARSA is a reinforcement learning algorithm that learns a policy by iteratively updating Q-values based on state-action-reward-state-action transitions.
