import matplotlib.pyplot as plt
import numpy as np


# Objective function: Rastrigin function using NumPy
def objective_function(x):
    x = np.array(x)
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


# Neighbor function: small random change
def get_neighbor(x, step_size=0.1):
    neighbor = np.copy(x)
    index = np.random.randint(0, len(x))
    neighbor[index] += np.random.uniform(-step_size, step_size)
    return neighbor


# Simulated Annealing function
def simulated_annealing(objective, bounds, n_iterations, step_size, temp):
    best = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
    best_eval = objective(best)
    current, current_eval = best.copy(), best_eval
    scores = [best_eval]

    for i in range(n_iterations):
        t = temp / float(i + 1)
        candidate = get_neighbor(current, step_size)
        candidate_eval = objective(candidate)

        if candidate_eval < best_eval or np.random.random() < np.exp(
            (current_eval - candidate_eval) / t
        ):
            current, current_eval = candidate, candidate_eval
            if candidate_eval < best_eval:
                best, best_eval = candidate, candidate_eval
                scores.append(best_eval)

        if i % 100 == 0:
            print(
                f"Iteration {i}, Temperature {t:.3f}, Best Evaluation {best_eval:.5f}"
            )

    return best, best_eval, scores


# Define problem domain
bounds = [(-5.0, 5.0) for _ in range(2)]
n_iterations = 1000
step_size = 0.1
temp = 10

# Run simulated annealing
best, score, scores = simulated_annealing(
    objective_function, bounds, n_iterations, step_size, temp
)

print("Best Solution:", best)
print("Best Score:", score)

# Plotting optimization progress
plt.figure()
plt.plot(scores)
plt.xlabel("Improvement Step")
plt.ylabel("Best Score")
plt.title("Simulated Annealing Optimization Progress")
plt.show()

# Intuitively multiple execution of the function will yield different results
