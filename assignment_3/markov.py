import matplotlib.pyplot as plt
import numpy as np

# transition matrix
P = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2], [0.3, 0.4, 0.3]])

# initial state in A
initial_state = np.array([1, 0, 0], dtype=float)


def simulate_markov_chain(
    P: np.ndarray, initial_state: np.ndarray, steps
) -> np.ndarray:
    state = initial_state
    history = [state]
    for _ in range(steps):
        state = state @ P
        history.append(state)
    return np.stack(history)


steps = 5
history = simulate_markov_chain(P, initial_state, steps)

print("history of states over time\n")
print(history)
plt.figure(figsize=(10, 6))
for i, state_prob in enumerate(history.T):
    plt.plot(state_prob, label=f"State {chr(i + 65)}")

plt.xlabel("Time Step")
plt.ylabel("Probability")
plt.title("Markov Chain Simulation")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

"""
1. Find the steady_state by solving πP = π
2. select the eigen vector corresponding to eigenvalue 1
3. Convert ot a 1-D real vector to normalize to sum 1
"""
eigen, eigvec = np.linalg.eig(P.T)
steady_state = eigvec[:, np.isclose(eigen, 1.0)]

steady_state = steady_state[:, 0].real
steady_state = steady_state / steady_state.sum()

print("steady_state:")
print(steady_state)
