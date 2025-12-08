import matplotlib.pyplot as plt
import numpy as np
from polymer_engine import simulate


def plot_polymer(position: np.ndarray, save_path: str = ""):
    coords = position

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], "-o")
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
