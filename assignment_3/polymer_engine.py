import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


# TODO: Implement simulated annealing and MCMC techinques


# Logging system
class Logger:
    LEVELS = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}

    def __init__(
        self,
        level: str = "INFO",
        to_file: str | None = None,
        keep_buffer: bool = False,
    ):
        self.level = self.LEVELS.get(level, 2)
        self.to_file = to_file
        self.keep_buffer = keep_buffer
        self.buffer = []
        self.file = open(to_file, "w") if to_file is not None else None

    def log(self, level: str, msg: str, tag: str | None = None):
        lvl = self.LEVELS.get(level, 2)
        if lvl <= self.level:
            prefix = f"[{level}]"
            if tag:
                prefix += f"[{tag}]"
            line = f"{prefix} {msg}"
            print(line)
            if self.file is not None:
                self.file.write(line + "\n")
                # Flush to ensure logs are persisted even on crashes
                self.file.flush()
            if self.keep_buffer:
                self.buffer.append(line)

    def set_level(self, level: str):
        self.level = self.LEVELS.get(level, self.level)

    def set_output(self, filename: str | None = None):
        if self.file is not None:
            self.file.close()
        self.to_file = filename
        self.file = open(filename, "w") if filename is not None else None

    def close(self):
        if self.file is not None:
            self.file.close()


LOGGER = Logger(level="WARN")


def set_log_level(level: str):
    LOGGER.set_level(level)


def set_log_output(filename: str | None = None):
    LOGGER.set_output(filename)


# Validation helpers
def ensure_positions(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise TypeError("positions must be a numpy.ndarray")
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if arr.dtype != np.float64:
        # Enforce float64 for stable Numba behavior and predictable strides
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    elif not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def ensure_forces_buffer_like(
    positions: np.ndarray, buf: np.ndarray | None = None
) -> np.ndarray:
    if buf is None:
        return np.zeros_like(positions, dtype=np.float64)
    if buf.shape != positions.shape or buf.dtype != np.float64:
        raise ValueError("forces buffer must match positions shape and be float64")
    if not buf.flags["C_CONTIGUOUS"]:
        raise ValueError("forces buffer must be C-contiguous")
    return buf


def check_nonnegative(name: str, v: float):
    if not np.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be finite and >= 0; got {v}")


def check_positive(name: str, v: float):
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {v}")


# Numba kernels
@njit(fastmath=True)
def _harmonic_kernel(
    positions: np.ndarray, forces: np.ndarray, k: float, d0: float, energy: float = 0.0
):
    n = positions.shape[0]
    # Caller ensures forces is zeroed
    for i in range(n - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        dz = positions[i + 1, 2] - positions[i, 2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        if r < 1e-12:  # why r and not r-d0
            continue
        fmag = -k * (r - d0)
        inv_r = 1.0 / r
        fx = fmag * dx * inv_r
        fy = fmag * dy * inv_r
        fz = fmag * dz * inv_r
        forces[i, 0] -= fx
        forces[i, 1] -= fy
        forces[i, 2] -= fz
        forces[i + 1, 0] += fx
        forces[i + 1, 1] += fy
        forces[i + 1, 2] += fz
        energy += 0.5 * k * (r - d0) ** 2


@njit(fastmath=True)
def _lj_kernel(
    positions: np.ndarray,
    forces: np.ndarray,
    epsilon: float,
    sigma: float,
    skip_bonded: int,
    cutoff: float,
    energy: float = 0.0,
):
    n = positions.shape[0]
    sig6 = sigma**6
    sig12 = sig6 * sig6
    cutoff2 = cutoff * cutoff
    start_offset = 2 if skip_bonded != 0 else 1  # start_offset used to avoid

    for i in range(n):  # double counting
        for j in range(i + start_offset, n):
            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 > cutoff2 or r2 < 1e-12:  # to change cut off if we observe
                continue  # some sensitivity, my simulation so far don't show any
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            f_over_r = 24.0 * epsilon * (2.0 * sig12 * inv_r12 - sig6 * inv_r6) * inv_r2
            fx = f_over_r * dx
            fy = f_over_r * dy
            fz = f_over_r * dz
            forces[i, 0] -= fx
            forces[i, 1] -= fy
            forces[i, 2] -= fz
            forces[j, 0] += fx
            forces[j, 1] += fy
            forces[j, 2] += fz
            energy += 4.0 * epsilon * (sig12 * inv_r12 - sig6 * inv_r6)


@njit(fastmath=True)
def _langevin_kernel(
    positions: np.ndarray,
    forces: np.ndarray,
    gamma: float,
    k_B: float,
    T: float,
    dt: float,
    rng_vals: np.ndarray,
):
    n = positions.shape[0]
    mobility = dt / gamma
    noise_strength = np.sqrt(2.0 * k_B * T * dt / gamma)
    idx = 0
    for i in range(n):
        dx = mobility * forces[i, 0] + noise_strength * rng_vals[idx]
        idx += 1
        dy = mobility * forces[i, 1] + noise_strength * rng_vals[idx]
        idx += 1
        dz = mobility * forces[i, 2] + noise_strength * rng_vals[idx]
        idx += 1
        positions[i, 0] += dx
        positions[i, 1] += dy
        positions[i, 2] += dz


# Functions to be called
def compute_harmonic_forces(
    positions: np.ndarray,
    k: float,
    d0: float,
    out: np.ndarray | None = None,
    energy: float = 0.0,
) -> np.ndarray:
    positions = ensure_positions(positions)
    if not np.isfinite(k) or not np.isfinite(d0):
        raise ValueError("k and d0 must be finite")
    out = ensure_forces_buffer_like(positions, out)
    out.fill(0.0)
    LOGGER.log("DEBUG", "Computing harmonic forces", tag="harmonic")
    _harmonic_kernel(positions, out, k, d0, energy=energy)
    LOGGER.log("DEBUG", f"Forces sample: {out[:3]}", tag="harmonic")
    LOGGER.log("DEBUG", f"Energy sample: {energy}", tag="harmonic")

    return out, energy


def compute_lj_forces(
    positions: np.ndarray,
    epsilon: float,
    sigma: float,
    skip_bonded: bool = True,
    cutoff_factor: float = 2.5,  # for efficiency how far the LJ interaction is computed
    out: np.ndarray | None = None,
    energy: float = 0.0,
) -> np.ndarray:
    positions = ensure_positions(positions)
    check_nonnegative("epsilon", epsilon)
    check_nonnegative("sigma", sigma)
    cutoff = cutoff_factor * sigma  # this way we ignore distant pairs saving some work
    out = ensure_forces_buffer_like(positions, out)
    out.fill(0.0)
    LOGGER.log(
        "DEBUG",
        f"Computing LJ forces (epsilon={epsilon}, sigma={sigma}, cutoff={cutoff:.3f}, skip_bonded={skip_bonded})",
        tag="lj",
    )
    _lj_kernel(
        positions, out, epsilon, sigma, 1 if skip_bonded else 0, cutoff, energy=energy
    )
    LOGGER.log("DEBUG", f"LJ sample: {out[:3]}", tag="lj")
    LOGGER.log("DEBUG", f"energy sample: {energy}", tag="lj")
    return out, energy


def langevin_step(
    positions: np.ndarray,
    forces: np.ndarray,
    gamma: float,
    k_B: float,
    T: float,
    dt: float,
    rng: np.random.Generator | None = None,
):
    positions = ensure_positions(positions)
    forces = ensure_positions(
        forces
    )  # shape/dtype check; we don't require writability on forces
    check_positive("gamma", gamma)
    check_nonnegative("k_B", k_B)
    check_nonnegative("T", T)
    check_positive("dt", dt)
    if positions.shape != forces.shape:
        raise ValueError("positions and forces must have the same shape")

    LOGGER.log("DEBUG", "Langevin step", tag="langevin")
    n = positions.shape[0]
    # Use a Generator to avoid global RNG state; sample in one call (fast)
    if rng is None:
        rng = np.random.default_rng()
    rng_vals = rng.normal(0.0, 1.0, 3 * n).astype(np.float64)
    _langevin_kernel(positions, forces, gamma, k_B, T, dt, rng_vals)
    LOGGER.log("DEBUG", f"Positions sample: {positions[:3]}", tag="langevin")


# High-level simulation
def simulate(
    positions: np.ndarray,
    steps: int,
    k: float,
    d0: float,
    epsilon: float,
    sigma: float,
    gamma: float,
    k_B: float,
    T: float,
    dt: float,
    skip_bonded: bool = True,  # that's how we skip the alredy counted bead
    cutoff_factor: float = 2.5,  # cut off * sigma (ligma ahahhaha) sorry it's late
    rng: np.random.Generator | None = None,
    ideal_chain: bool = None,
) -> np.ndarray:
    positions = ensure_positions(positions)
    if rng is None:
        rng = np.random.default_rng()

    LOGGER.log("INFO", "Starting simulation", tag="sim")
    LOGGER.log("INFO", f"N={positions.shape[0]}, steps={steps}", tag="sim")

    # Preallocate forces to avoid per-step allocations
    f_h = np.zeros_like(positions, dtype=np.float64)
    f_lj = np.zeros_like(positions, dtype=np.float64)
    total_f = np.zeros_like(positions, dtype=np.float64)

    for step in range(steps):
        energy = np.zeros_like(positions, dtype=np.float64)

        if step % 100 == 0:
            LOGGER.log("INFO", f"[step {step}]", tag="sim")

        # Fill preallocated buffers
        compute_harmonic_forces(positions, k, d0, out=f_h, energy=energy[1])

        if not ideal_chain:
            compute_lj_forces(
                positions,
                epsilon,
                sigma,
                skip_bonded=skip_bonded,
                cutoff_factor=cutoff_factor,
                out=f_lj,
                energy=energy[2],
            )
        else:
            f_lj.fill(0.0)

        # Sum forces
        total_f[:] = f_h + f_lj

        # Integrate
        langevin_step(positions, total_f, gamma, k_B, T, dt, rng=rng)

    LOGGER.log("INFO", "Simulation finished", tag="sim")
    return positions


def end_to_end_radius2(positions: np.ndarray) -> float:
    m1 = positions[0]
    mn = positions[-1]

    dx = m1[0] - mn[0]
    dy = m1[1] - mn[1]
    dz = m1[2] - mn[2]
    R_ee2 = dx * dx + dy * dy + dz * dz

    return R_ee2


def gyration_radius2(positions: np.ndarray) -> float:
    n = positions.shape[0]
    R_cm = np.mean(positions, axis=0)
    R_g2 = 0.0
    for m in positions:
        dx = R_cm[0] - m[0]
        dy = R_cm[1] - m[1]
        dz = R_cm[2] - m[2]
        r2 = dx * dx + dy * dy + dz * dz

        R_g2 += r2
    return R_g2 / n


def plot_polymer(position: np.ndarray, save_path: str = ""):
    coords = position

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], "-o")
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


# ================================================================
# Example run
# ================================================================

if __name__ == "__main__":
    set_log_level("ERROR")  # ERROR, WARN, INFO, DEBUG
    set_log_output("sim.log")  # or None
    """
    N = 10
    pos = np.random.randn(N, 3).astype(np.float64)
    final = simulate(
        positions=pos,
        steps=1000,
        k=10.0,
        d0=1.0,
        epsilon=1.0,
        sigma=1.0,
        gamma=1.0,
        k_B=1.0,
        T=1.0,
        dt=0.01,
    )
    """
    # ================================================================
    # Validation - ideal chain
    # ================================================================

    # gyration radius
    N = np.linspace(10, 210, 40, dtype=int)
    r_ee2_collector = np.zeros_like(N)
    r_g2_collector = np.zeros_like(N)

    for i in tqdm(range(len(N))):
        for j in range(40):
            n = N[i]
            pos = np.random.randn(n, 3).astype(np.float64)
            final = simulate(
                positions=pos,
                steps=500,
                k=10.0,
                d0=1.0,
                epsilon=1.0,
                sigma=1.0,
                gamma=1.0,
                k_B=1.0,
                T=1.0,
                dt=0.01,
                ideal_chain=True,
            )

            r_ee2_collector[i] += end_to_end_radius2(final)
            r_g2_collector[i] += gyration_radius2(final)
    # plot_polymer(position=final, save_path="img/polymer_plot")

    plt.plot(N, r_ee2_collector / 40, label="end to end radius")
    plt.plot(N, r_g2_collector / 40, label="gyration radius")
    plt.plot(N, (N - 1) * (1 / 10 + 1), label="ideal end to end", ls="--")
    plt.plot(N, (N * N - 1) * (1 / 10 + 1) / 6 / N, label="ideal gyration", ls="--")

    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()

    # not validated :D
