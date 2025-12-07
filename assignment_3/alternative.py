import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

# ----- Build the Protein -----
## We are assuming a bead-spring model here
## If we want to be more fancy, we can try and use the Gaussian-chain model

# let "r" denote the matrix of bead coordinates in 3D
# such that r[i] is the [x_i, y_i, z_i] position of bead i
# "r" is then a matrix of shape (n, 3) where n is the number of beads in a polymer chain

# all our functions will be based around this simple bead representation of polymers

# this a good high level introduction to polymer physics:
# https://en.wikipedia.org/wiki/Molecular_mechanics


def chain_bond_energy(r: np.ndarray, T: float) -> float:
    """Computes the bond energy of the polymer chain.
    The bond energy is the sum of harmonic and lennard jones bond potentials between all beads in the chain
    """
    return harmonic_bond_potential(r, T) + lennard_jones_potential(r)


def harmonic_bond_potential(r: np.ndarray, T: float) -> float:
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
    b_sqrd = kuhn_length(r)
    k = 3 * const.Boltzmann * T / b_sqrd  # spring constant
    for i in range(len(r) - 1):
        d = np.linalg.norm(r[i] - r[i + 1])
        e += 0.5 * k * d**2

    return e


def kuhn_length(r: np.ndarray) -> float:
    """Computes the Kuhn length of the polymer chain."""
    bond_vecs = r[1:] - r[:-1]
    R = bond_vecs.sum(axis=0)
    return np.dot(R, R)


def lennard_jones_potential(r: np.ndarray, phi: float = 1.0) -> float:
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
    epsilon = phi * 2 ** (1 / 6)  # depth of potential well
    n = len(r)
    for i in range(n):
        for j in range(i + 2, n):  # only non-bonded interactions
            d = np.linalg.norm(r[i] - r[j])
            e += 4 * epsilon * ((phi / d) ** 12 - (phi / d) ** 6)
    return e


def compute_forces(r: np.ndarray, T: float) -> np.ndarray:
    """Numerical forces F = -∇_r U using central differences."""
    F = np.zeros_like(r)
    h = 1e-5
    for i in range(len(r)):
        for d in range(r.shape[1]):
            rp = r.copy()
            rm = r.copy()
            rp[i, d] += h
            rm[i, d] -= h
            Up = chain_bond_energy(rp, T)
            Um = chain_bond_energy(rm, T)
            F[i, d] = -(Up - Um) / (2 * h)
    return F


# We have now defined the energy function for our polymer chain
# Next, we introduce overdamped Langevin dynamics to simulate the motion of the polymer chain


def langevin_dynamics_step(
    r: np.ndarray, gamma: float, T: float, dt: float
) -> np.ndarray:
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
    # sigma
    sigma = np.sqrt(2 * T * const.Boltzmann * dt / gamma)

    # random moise term
    wiener = np.random.normal(0, 1, r.shape)  # mean 0, variance 1
    stochastic_force = sigma * wiener
    F = compute_forces(r, T)

    dX = -1 / gamma * F * dt + stochastic_force
    r += dX
    return r


# this gives us the ability to siulate the motion of the polymer chain over time
# Next, we can implemment a markovian chain monte carlo t
# to perform a motion planning task for the polymer chain


def mcmc_step(r: np.ndarray, step_size: float, T: float = 300) -> np.ndarray:
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
    # propose new state
    r_proposed = langevin_dynamics_step(r, gamma=0.1, T=T, dt=step_size)
    delta_E = chain_bond_energy(r_proposed, T) - chain_bond_energy(r, T)
    acceptance_prob = min(1, np.exp(-delta_E / (const.Boltzmann * T)))
    if np.random.random() < acceptance_prob:
        return r_proposed
    return r


# we now have a stepswise mcmc sim for our polymer chain
# next, we use simualted annealing to search for low-energy conformations of the polymer chain


def simulated_annealing(
    r: np.ndarray,
    initial_temp_sa: float,
    T: float,
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
    # init temp
    t = initial_temp_sa
    r_current = r.copy()
    r_best = r.copy()
    e_best = chain_bond_energy(r_best, T=T)
    for step in range(steps):
        r_current = mcmc_step(r_current, step_size=0.1, T=T)
        e_current = chain_bond_energy(r_current, T=T)
        print("Current", e_current)
        print("Best", e_best)
        if np.abs(e_current) < np.abs(e_best):
            r_best = r_current.copy()
            e_best = e_current
        t *= cooling_rate

    return r_best, e_best


# all components in place - we can now run a full simulation
# 1. initialize a polymer chain
# 2. Get some random folding state by running multiple langevin dynamics steps
# 3. run simulated annealing to find low-energy conformations of that random polymer chain

if __name__ == "__main__":

    r = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0]])

    # plt.plot(r[:, 0], r[:, 1], marker="o", label="Initial Configuration")
    r_init = r.copy()
    e_init = chain_bond_energy(r_init, T=300)

    es = []
    for _ in range(100):
        r, e_new = simulated_annealing(
            r, initial_temp_sa=100.0, cooling_rate=0.99, steps=200, T=300.0
        )
        es.append(e_new)

    # plt.plot(r[:, 0], r[:, 1], marker="o", label="Final Configuration")

    ts = np.linspace(0, 1, len(es))
    plt.plot(ts, es)
    plt.title("Polymer Chain Simulated Annealing")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Initial Energy:", e_init)
    print("Final Energy:", e_new)
