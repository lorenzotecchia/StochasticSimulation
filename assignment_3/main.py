import numpy as np

# ----- Build the Protein -----
## We are assuming a bead-spring model here
## If we want to be more fancy, we can try and use the Gaussian-chain model

# let "r" denote the matrix of bead coordinates in 3D
# such that r[i] is the [x_i, y_i, z_i] position of bead i
# "r" is then a matrix of shape (n, 3) where n is the number of beads in a polymer chain

# all our functions will be based around this simple bead representation of polymers

# this a good high level introduction to polymer physics:
# https://en.wikipedia.org/wiki/Molecular_mechanics


def chain_bond_energy(r: np.ndarray) -> float:
    """Computes the bond energy of the polymer chain.
    The bond energy is the sum of harmonic and lennard jones bond potentials between all beads in the chain
    """
    return harmonic_bond_potential(r) + lennard_jones_potential(r)


def harmonic_bond_potential(r: np.ndarray, d0: float, k: float) -> float:
    """Comutes the harmonic bond potential between two beads.
    where d is the distance between the two beads, d0 is the equilibrium bond length,
    and k is the bond spring constant.

    Groning p. 17 gives this as
    U = log(1 - d^2/d0^2) ; d < do
    U = infinity ; d >= d0
    -------------
    https://www-thphys.physics.ox.ac.uk/people/ArdLouis/padding/PolymerDynamics_Padding.pdf
    Padding gives this as
    U = 0.5 * k * (d-d0)^2

    which is the standard harmonic potential form. (I think this is what we want to use here.)
    """
    e = 0.0
    return e


def lennard_jones_potential(r: np.ndarray, epsilon: float, sigma: float) -> float:
    """Computes the Lennard-Jones potential between all non-bonded beads in the chain.
    Each bead iteracts with every other bead via the Lennard-Jones potential that is not directly bonded
    https://en.wikipedia.org/wiki/Lennard-Jones_potential
    The Lennard-Jones potential is given by
    U = 4 * epsilon * ((sigma/d)^12 - (sigma/d)^6)
    where d is the distance between the two beads,
    epsilon is the depth of the potential well,
    and sigma is the disintace where the particle-particle potential is zero.
    """
    e = 0.0
    return e


# We have now defined the energy function for our polymer chain
# Next, we introduce overdamped Langevin dynamics to simulate the motion of the polymer chain


def langevin_dynamics_step(r: np.ndarray, gamma: float, T: float) -> np.ndarray:
    """A single step of overdamped Langevin dynamics for the polymer chain.
    Langevin dynamics consist of two components:
    1. The gradient of our energy function (deterministic force)
    2. A random noise term (stochastic force)
    If we study wikipedia, we find
    dX = - 1 / gamma * grad(u) * dt + sqrt(2) * sigma / gamma * dW
    where gamma is the friction coefficient,
    sigma is the noise strength,
    and dW is a Wiener process increment (Gaussian noise with mean 0 and variance dt)

    Note that sigma is given as
    sqrt(2 *k_b * T * gamma * M)
    whre k_b is the Boltzmann constant,
    T is the temperature,
    and M is the mass of the bead.

    I think we can assume a unit mass for the beads and this is anyways just a constant
    """
    return r


# this gives us the ability to siulate the motion of the polymer chain over time
# Next, we can implemment a markovian chain monte carlo t
# to perform a motion planning task for the polymer chain


def mcmc_step(r: np.ndarray, step_size: float, T: float) -> np.ndarray:
    """
    A single step of the Metropolis-Hastings MCMC algorithm for the polymer chain.
    https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm
    This looks quite straightforward to implement:
    1. Propose a new state r' by perturbing the current state r with Gaussian noise of standard deviation step_size
    2. Compute the energy difference delta_E = E(r') - E(r)
    3. Accept the new state with probability min(1, exp(-delta_E / (k_b * T)))
       where k_b is the Boltzmann constant and T is the temperature
    4. If the new state is accepted, return r', else return r
    """
    return r


# we now have a stepswise mcmc sim for our polymer chain
# next, we use simualted annealing to search for low-energy conformations of the polymer chain


def simulated_annealing(
    r: np.ndarray,
    initial_temp: float,
    cooling_rate: float,
    steps: int,
) -> list[np.ndarray, float]:
    """
    Simulated annealing for the polymer chain.
    1. Initialize the temperature T to initial_temp
    2. For each step:
        a. MCMC step at temperature t
        b. calculate energy of current state
        c. If energy is lower than previous best, update best state
        b. Decrease the temperature T according to the cooling schedule
           T = T * cooling_rate
    3. Return the best state found
    """
    e = 0.0
    return r, e


# all components in place - we can now run a full simulation
# 1. initialize a polymer chain
# 2. Get some random folding state by running multiple langevin dynamics steps
# 3. run simulated annealing to find low-energy conformations of that random polymer chain
