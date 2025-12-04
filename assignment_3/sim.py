# TODO: placeholder file
import numpy as np

# TODO: compile c++ code and import it in here
# import polymer_engine  # This is your compiled C++ .so file

# 1. Setup Data
n_beads = 10
# Initialize a straight chain along X axis
positions = np.zeros((n_beads, 3), dtype=np.float64)
positions[:, 0] = np.linspace(0, 10, n_beads)

# Create an empty force array (C++ will fill this)
forces = np.zeros_like(positions)

# Constants
k_spring = 100.0
d0 = 0.5  # Equilibrium length

print(f"Initial X positions:\n{positions[:,0]}")

# 2. Call C++ (The Magic Moment)
# Note: The chain is stretched (dist=1.1, d0=0.5), so forces should pull it back.

# placeholder
# polymer_engine.compute_harmonic_forces(positions, forces, k_spring, d0)

print(f"\nForces computed by C++:\n{forces}")
