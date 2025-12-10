import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from numba import njit
from tqdm import tqdm
from scipy import constants
import csv

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
    positions: np.ndarray,
    forces: np.ndarray,
    k: float,
    d0: float,
    energy: float | None = None,
):
    n = positions.shape[0]
    # Caller ensures forces is zeroed
    for i in range(n - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        dz = positions[i + 1, 2] - positions[i, 2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        if r < 1e-12:
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
        energy[0] += 0.5 * k * (r - d0) ** 2


@njit(fastmath=True)
def _lj_kernel(
    positions: np.ndarray,
    forces: np.ndarray,
    epsilon: float,
    sigma: float,
    skip_bonded: int,
    cutoff: float,
    energy: float | None = None,
):
    n = positions.shape[0]
    sig6 = sigma**6
    sig12 = sig6 * sig6
    cutoff2 = cutoff * cutoff
    start_offset = 3 if skip_bonded != 0 else 1  # start_offset used to avoid

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
            energy[0] += 4.0 * epsilon * (sig12 * inv_r12 - sig6 * inv_r6)


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
    if not np.isfinite(d0):
        raise ValueError("d0 must be finite")
    out = ensure_forces_buffer_like(positions, out)
    out.fill(0.0)
    LOGGER.log("DEBUG", "Computing harmonic forces", tag="harmonic")
    energy_arr = np.array([energy], dtype=np.float64)
    _harmonic_kernel(positions, out, k, d0, energy=energy_arr)
    energy = energy_arr[0]
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
    energy_arr = np.array([energy], dtype=np.float64)
    _lj_kernel(
        positions,
        out,
        epsilon,
        sigma,
        1 if skip_bonded else 0,
        cutoff,
        energy=energy_arr,
    )
    energy = energy_arr[0]
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
    d0: float,
    epsilon: float,
    sigma: float,
    gamma: float,
    k: float,
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
        energy = np.zeros(3, dtype=np.float64)

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
        # print("mean |F| =", np.mean(np.linalg.norm(total_f, axis=1)))
        # print(f"Harmonic Energy: {energy[1]}, LJ Energy: {energy[2]}")
        # Integrate
        langevin_step(positions, total_f, gamma, k_B, T, dt, rng=rng)
        print(energy)
    LOGGER.log("INFO", "Simulation finished", tag="sim")
    return positions


def mala_step(
    positions: np.ndarray,
    gamma: float,
    k_b: float,
    k: float,
    T: float,
    dt: float,
    epsilon: float,
    sigma: float,
    d0: float,
    skip_bonded: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Performs one MALA step for the polymer chain
    """

    # compute current forces and energy
    f_h = np.zeros_like(positions, dtype=np.float64)
    f_lj = np.zeros_like(positions, dtype=np.float64)

    _, e_h = compute_harmonic_forces(positions=positions, k=k, d0=d0, out=f_h)
    _, e_lj = compute_lj_forces(
        positions=positions,
        epsilon=epsilon,
        sigma=sigma,
        skip_bonded=skip_bonded,
        out=f_lj,
    )

    u_current = e_h + e_lj
    f_current = f_h + f_lj

    # save current state
    pos_current = positions.copy()

    # generate proposal
    langevin_step(positions, f_current, gamma, k_b, T, dt, rng=rng)

    pos_proposal = positions

    # compute forces and energy at proposal
    f_h_prop = np.zeros_like(pos_proposal, dtype=np.float64)
    f_lj_prop = np.zeros_like(pos_proposal, dtype=np.float64)
    _, e_h_prop = compute_harmonic_forces(
        positions=pos_proposal, k=k_b, d0=d0, out=f_h_prop
    )
    _, e_lj_prop = compute_lj_forces(
        positions=pos_proposal,
        epsilon=epsilon,
        sigma=sigma,
        skip_bonded=skip_bonded,
        out=f_lj_prop,
    )
    u_proposal = e_h_prop + e_lj_prop
    f_proposal = f_h_prop + f_lj_prop

    print(u_proposal, u_current)
    # hastings correction
    mu = dt / gamma
    D = k_b * T * dt / gamma

    # forward prob q(x'|x)
    diff_forward = pos_proposal - pos_current - mu * f_current
    log_q_forward = -np.sum(diff_forward**2) / (4 * D)

    # backward prob q(x|x')
    diff_backward = pos_current - pos_proposal - mu * f_proposal
    log_q_backward = -np.sum(diff_backward**2) / (4 * D)

    # acceptance probability
    log_alpha = (u_current - u_proposal) / (k_b * T) + log_q_backward - log_q_forward

    # accept or reject
    if np.random.rand() < np.exp(log_alpha):
        # accept
        return pos_proposal, [e_h_prop, e_lj_prop]
    else:
        # reject
        return pos_current, [e_h, e_lj]


def initial_positions(N: int, d0: float):
    positions = np.zeros((N, 3))
    for j in range(1, N):
        displacement = np.random.randn(3)
        positions[j] = positions[j - 1] + d0 * displacement / np.linalg.norm(
            displacement
        )
    positions -= positions.mean(axis=0)
    return positions


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


def end_to_end_radius2(positions: np.ndarray) -> float:
    m1 = positions[0]
    mn = positions[-1]

    dx = m1[0] - mn[0]
    dy = m1[1] - mn[1]
    dz = m1[2] - mn[2]
    R_ee2 = dx * dx + dy * dy + dz * dz

    return R_ee2


def bond_length_var(positions: np.ndarray, d0) -> np.ndarray:
    r2 = np.zeros(positions.shape[0] - 1)
    for i in range(len(positions) - 1):
        m0 = positions[i]
        m1 = positions[i + 1]

        dx = m0[0] - m1[0]
        dy = m0[1] - m1[1]
        dz = m0[2] - m1[2]
        r2[i] = np.sqrt(dx * dx + dy * dy + dz * dz) - d0

    return r2


def bond_length(positions: np.ndarray) -> np.ndarray:
    r2 = np.zeros(positions.shape[0] - 1)
    for i in range(len(positions) - 1):
        m0 = positions[i]
        m1 = positions[i + 1]

        dx = m0[0] - m1[0]
        dy = m0[1] - m1[1]
        dz = m0[2] - m1[2]
        r2[i] = dx * dx + dy * dy + dz * dz

    return r2


def MC_MSD(pos_start: np.ndarray, pos_t: np.ndarray) -> float:
    MC_start = np.mean(pos_start, axis=0)
    MC_t = np.mean(pos_t, axis=0)
    dx = MC_start[0] - MC_t[0]
    dy = MC_start[1] - MC_t[1]
    dz = MC_start[2] - MC_t[2]
    return dx * dx + dy * dy + dz * dz


def two_sides_test(sample: list, theoretical_value: float, alpha: float = 0.05) -> bool:
    t_crit = stats.t.ppf(q=1 - alpha / 2, df=len(sample) - 1)
    t_stat = (np.mean(sample) - theoretical_value) / (
        np.std(sample) / np.sqrt(len(sample))
    )
    return abs(t_stat) < t_crit


def diffusion_fit_plot(
    MSD_all: np.ndarray,
    N: int,
    dt: float,
    plot: bool = True,
    gamma: float = 1,
    k_B: float = 1,
    T: float = 1,
):
    reps = MSD_all.shape[0]
    n_intervals = MSD_all.shape[1]
    t = np.arange(n_intervals) * dt

    MSD_mean = MSD_all.mean(axis=0)
    MSD_sem = MSD_all.std(axis=0, ddof=1) / np.sqrt(reps)
    MSD_theory = 6 * (k_B * T) / (N * gamma) * t

    # 95% confidence interval
    t_crit = stats.t.ppf(0.975, reps - 1)
    CI_low = MSD_mean - t_crit * MSD_sem
    CI_high = MSD_mean + t_crit * MSD_sem

    inside = np.logical_and(MSD_theory >= CI_low, MSD_theory <= CI_high)
    inside_all = inside.all()
    print(f"Is theoretical MSD within 95% CI at all times? {inside_all}")

    if plot:
        plt.plot(t, MSD_mean, label="simulated MSD", markersize=7)
        plt.fill_between(
            t, CI_low, CI_high, color="gray", alpha=0.3, label="95% Confidence Interval"
        )
        plt.plot(t, MSD_theory, label="theoretical MSD", ls="--")
        plt.xlabel("Time (t)")
        plt.ylabel("Mean Squared Displacement (MSD)")
        plt.title("Center of Mass MSD over Time")
        plt.tight_layout()

        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/diff_validation.png", dpi=300)
        plt.close()

    slopes_sample = np.zeros(reps)
    for r in range(reps):
        slopes_sample[r], _, r_value, _, stderr = stats.linregress(t, MSD_all[r, :])
    NOT_REJECT = two_sides_test(slopes_sample, 6 * k_B * T / (N * gamma))
    print(f"not rejected: {NOT_REJECT}")


def plot_diffusion_3D(x: np.ndarray, y: np.ndarray, z: np.ndarray, steps: int):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(
        x,
        y,
        z,
        c=np.arange(steps),
        cmap="viridis",
        s=5,
    )

    fig.colorbar(p, ax=ax, label="Time step")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Diffusion Trajectory")
    plt.savefig("img/diffusion_3D.png", dpi=300)
    plt.close()

    # this moves:
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                marker=dict(size=2, color=np.arange(steps), colorscale="Viridis"),
            )
        ]
    )
    fig.update_layout(
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        title="3D Diffusion Trajectory",
    )

    fig.write_html("img/diffusion_plot.html")


def plot_diffusion_XYZ(
    dt: float,
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
    z_coordinates: np.ndarray,
):
    reps = x_coordinates.shape[0]
    n_intervals = x_coordinates.shape[1]
    t = np.arange(n_intervals) * dt

    labels = ["x", "y", "z"]
    coordinates = [x_coordinates, y_coordinates, z_coordinates]

    for i in range(3):
        for r in range(reps):
            plt.plot(
                t,
                coordinates[i][r, :],
                alpha=0.7,
                lw=0.5,
            )
        plt.xlabel("Time (t)")
        plt.ylabel(f"{labels[i]} coordinate")
        plt.tight_layout()
        plt.grid(alpha=0.5)
        plt.savefig(f"img/diff_spread_{labels[i]}.png", dpi=300)
        plt.close()


def diffusion_fit_plot(
    MSD_all: np.ndarray,
    N: int,
    dt: float,
    plot: bool = True,
    gamma: float = 1,
    k_B: float = 1,
    T: float = 1,
):
    reps = MSD_all.shape[0]
    n_intervals = MSD_all.shape[1]
    t = np.arange(n_intervals) * dt

    MSD_mean = MSD_all.mean(axis=0)
    MSD_sem = MSD_all.std(axis=0, ddof=1) / np.sqrt(reps)
    MSD_theory = 6 * (k_B * T) / (N * gamma) * t

    # 95% confidence interval
    t_crit = stats.t.ppf(0.975, reps - 1)
    CI_low = MSD_mean - t_crit * MSD_sem
    CI_high = MSD_mean + t_crit * MSD_sem

    inside = np.logical_and(MSD_theory >= CI_low, MSD_theory <= CI_high)
    inside_all = inside.all()
    print(f"Is theoretical MSD within 95% CI at all times? {inside_all}")

    if plot:
        plt.plot(t, MSD_mean, label="simulated MSD", markersize=7)
        plt.fill_between(
            t, CI_low, CI_high, color="gray", alpha=0.3, label="95% Confidence Interval"
        )
        plt.plot(t, MSD_theory, label="theoretical MSD", ls="--")
        plt.xlabel("Time (t)")
        plt.ylabel("Mean Squared Displacement (MSD)")
        plt.title("Center of Mass MSD over Time")
        plt.tight_layout()

        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/diff_validation.png", dpi=300)
        plt.close()

    slopes_sample = np.zeros(reps)
    for r in range(reps):
        slopes_sample[r], _, r_value, _, stderr = stats.linregress(t, MSD_all[r, :])
    NOT_REJECT = two_sides_test(slopes_sample, 6 * k_B * T / (N * gamma))
    print(f"not rejected: {NOT_REJECT}")


def diffusion(
    N: int,
    steps: int,
    reps: int,
    fit: bool,
    plot3D: bool,
    plotXYZ: bool = False,
    sigma: float = 1,
    d0: float = 1,
    gamma: float = 1,
    k_B: float = 1,
    epsilon: float = 1,
    T: float = 1,
    k: float = 30,
    dt: float = 1e-3,
):

    # data collectors
    MSD_collector = np.zeros((reps, steps - 1))
    CMx_trace_collector = np.zeros((reps, steps))
    CMy_trace_collector = np.zeros((reps, steps))
    CMz_trace_collector = np.zeros((reps, steps))

    for r in tqdm(range(reps)):

        # Initialize chain:
        positions = initial_positions(N, d0)
        pos_start = positions.copy()

        # evolve positions
        for step in range(steps - 1):
            positions = simulate(
                positions=positions,
                steps=1,
                d0=d0,
                k=k,
                epsilon=epsilon,
                sigma=sigma,
                gamma=gamma,
                k_B=k_B,
                T=T,
                dt=dt,
                ideal_chain=False,
            )

            CMx_trace_collector[r, step] = np.mean(positions, axis=0)[0]
            CMy_trace_collector[r, step] = np.mean(positions, axis=0)[1]
            CMz_trace_collector[r, step] = np.mean(positions, axis=0)[2]
            MSD_collector[r, step] = MC_MSD(pos_start, positions)

    if fit:
        diffusion_fit_plot(MSD_collector, N, dt, steps)

    if plot3D:
        plot_diffusion_3D(
            CMx_trace_collector[0],
            CMy_trace_collector[0],
            CMz_trace_collector[0],
            steps,
        )

    if plotXYZ:
        plot_diffusion_XYZ(
            dt,
            CMx_trace_collector,
            CMy_trace_collector,
            CMz_trace_collector,
        )


def warm_up(
    N: int,
    steps: int,
    reps: int,
    plot: bool = True,
    sigma: float = 1,
    d0: float = 1,
    gamma: float = 1,
    k_B: float = 1,
    epsilon: float = 1,
    T: float = 1,
    k: float = 30,
    dt: float = 1e-3,
):

    r_ee_collector = np.zeros((reps, steps))
    for r in tqdm(range(reps)):
        positions = initial_positions(N, d0)

        for s in range(steps):
            positions = simulate(
                positions,
                steps=1,
                d0=d0,
                epsilon=epsilon,
                sigma=sigma,
                gamma=gamma,
                k=k,
                k_B=k_B,
                T=T,
                dt=dt,
                ideal_chain=True,
            )
            r_ee_collector[r, s] = np.sqrt(end_to_end_radius2(positions))

    t = np.linspace(dt, dt * steps, steps)

    if plot:
        for i in range(20):
            plt.plot(t, r_ee_collector[i, :], alpha=0.5, lw=0.5)
        plt.plot(
            t,
            np.mean(r_ee_collector, axis=0),
            label="average end to end radius",
            color="black",
        )
        plt.xlabel("Time (t)")
        plt.ylabel("End to end radius")
        plt.title("End to End Radius over Time")
        plt.plot(t, np.sqrt(N) * np.ones_like(t), ls="--", c="r")
        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/end_to_end_over_time.png", dpi=300)
        plt.close()


def data_varying_N(
    N_list: np.ndarray,
    steps: int,
    reps: int,
    ideal_chain: bool,
    plot_diff_N: bool = True,
    plot_Ree_N: bool = True,
    verbose: bool = False,
    sigma: float = 1,
    d0: float = 1,
    gamma: float = 1,
    k_B: float = 1,
    epsilon: float = 1,
    T: float = 1,
    k: float = 30,
    dt: float = 1e-3,
):
    # data collectors for slope
    slopes_collector = np.zeros((len(N_list), reps))
    slope_CI_low = np.zeros(len(N_list))
    slope_CI_high = np.zeros(len(N_list))
    slope_test_result = np.zeros(len(N_list), dtype=bool)
    slope_mean = np.zeros(len(N_list))

    # data collector for end to end radius
    r_ee2_collector = np.zeros((len(N_list), reps))
    r_ee2_CI_low = np.zeros(len(N_list))
    r_ee2_CI_high = np.zeros(len(N_list))
    r_ee2_test_result = np.zeros(len(N_list), dtype=bool)
    r_ee2_mean = np.zeros(len(N_list))

    # time
    t = np.arange(1, steps) * dt

    for i in tqdm(range(len(N_list))):
        n = N_list[i]

        for r in tqdm(range(reps)):
            positions = initial_positions(n, d0)
            start_pos = positions.copy()

            MSD_collector = np.zeros(steps - 1)
            r_ee2 = np.zeros(steps - 1)

            for step in range(steps - 1):
                positions = simulate(
                    positions,
                    steps=1,
                    d0=d0,
                    k=k,
                    epsilon=epsilon,
                    sigma=sigma,
                    gamma=gamma,
                    k_B=k_B,
                    T=T,
                    dt=dt,
                    ideal_chain=ideal_chain,
                )
                MSD_collector[step] = MC_MSD(start_pos, positions)
                r_ee2[step] = end_to_end_radius2(positions)

            slopes_collector[i, r], _, r_value, _, stderr = stats.linregress(
                t, MSD_collector
            )
            r_ee2_collector[i, r] = np.mean(r_ee2)

        # compute statistics for this N
        slope_mean[i] = slopes_collector[i, :].mean()
        r_ee2_mean[i] = r_ee2_collector[i, :].mean()

        # 95% CI for slopes
        sem = slopes_collector[i, :].std(ddof=1) / np.sqrt(reps)
        t_crit = stats.t.ppf(0.975, reps - 1)
        slope_CI_low[i] = slope_mean[i] - t_crit * sem
        slope_CI_high[i] = slope_mean[i] + t_crit * sem

        # 95% CI for r_ee
        sem = r_ee2_collector[i, :].std(ddof=1) / np.sqrt(reps)
        t_crit = stats.t.ppf(0.975, reps - 1)
        r_ee2_CI_low[i] = r_ee2_mean[i] - t_crit * sem
        r_ee2_CI_high[i] = r_ee2_mean[i] + t_crit * sem

        # test theoretical slope
        slope_theory = 6 * k_B * T / (n * gamma)
        slope_test_result[i] = two_sides_test(slopes_collector[i, :], slope_theory)

        if verbose:
            print(
                f"N = {n:3d} | slope mean = {slope_mean[i]:.4f} | "
                f"theory = {slope_theory:.4f} | test: {slope_test_result[i]}"
            )

        # test theoretical end to end radius
        r_ee2_theory = n
        r_ee2_test_result[i] = two_sides_test(r_ee2_collector[i, :], r_ee2_theory)

        if verbose:
            print(
                f"N = {n:3d} | slope mean = {r_ee2_mean[i]:.4f} | "
                f"theory = {r_ee2_theory:.4f} | test: {r_ee2_test_result[i]}"
            )

    tag = "_LJ"
    if ideal_chain:
        tag = "_ideal"

    if plot_Ree_N:
        plt.figure(figsize=(7, 5))

        plt.plot(
            N_list, r_ee2_mean, "o-", label="Mean simulated squared end to end radius"
        )
        plt.fill_between(
            N_list, r_ee2_CI_low, r_ee2_CI_high, alpha=0.3, color="gray", label="95% CI"
        )

        plt.plot(N_list, N_list, "--", label="Theoretical end to end radius")

        plt.xlabel("Polymer length N")
        plt.ylabel(r"$R$")
        plt.title("End to end radius vs N")
        plt.grid(alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"img/end_to_end_val{tag}.png", dpi=300)
        plt.close()

    if plot_diff_N:
        plt.figure(figsize=(7, 5))

        plt.plot(N_list, slope_mean, "o-", label="Mean simulated slope")
        plt.fill_between(
            N_list, slope_CI_low, slope_CI_high, alpha=0.3, color="gray", label="95% CI"
        )

        plt.plot(
            N_list, 6 * k_B * T / (N_list * gamma), "--", label="Theoretical slope"
        )

        plt.xlabel("Polymer length N")
        plt.ylabel("Slope of MSD(t)")
        plt.title("Slope of MSD vs N")
        plt.grid(True, alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"img/slope_vs_N{tag}.png", dpi=300)
        plt.close()


# ================================================================
# Example run
# ================================================================

if __name__ == "__main__":
    set_log_level("ERROR")  # ERROR, WARN, INFO, DEBUG
    set_log_output("sim.log")  # or None

    # ================================================================
    # Validation
    # ================================================================

    # to see if warm up is needed, end to end radius over time with N monomers
    # it plots ee radius over time, with theroetical value

    steps = 500
    reps = 100
    N = 50
    # warm_up(N, steps, reps)

    # for ideal chain: computes R_ee letting N vary, plus fit?
    N_list = np.linspace(10, 210, 5, dtype=int)
    # data_varying_N(N_list, steps, reps, ideal_chain=True)

    # introducing LJ potential, see how diffusion constant changes with N plus fit
    data_varying_N(N_list, steps, reps, ideal_chain=False)

    N = 20
    # diffusion(N, steps, reps, fit=True, plot3D=True, plotXYZ=True)

    # from here it can be removed
    BOND_VAL = False
    BOND_VAR_VAL = False
    FULL_SIM = True

    if FULL_SIM:
        sigma = 1
        d0 = 1  #
        gamma = 1  # boh
        k_B = 1
        epsilon = 1
        T = 1
        k = 30
        dt = 1e-3
        N = 5
        pos = np.random.randn(N, 3).astype(np.float64)
        pos = ensure_positions(pos)
        start_pos = pos.copy()

        es = []
        for i in range(50):
            pos, energy = mala_step(
                pos,
                gamma,
                k_B,
                k,
                T,
                dt,
                epsilon,
                sigma,
                d0,
                skip_bonded=True,
                rng=None,
            )
            es.append(energy)

        plt.plot(np.array(es)[:, 0], label="Harmonic Energy")
        plt.plot(np.array(es)[:, 1], label="LJ Energy")
        plt.legend()
        plt.show()

    if BOND_VAL:  # james
        steps = 500
        reps = 40

        N = np.linspace(10, 210, 50, dtype=int)
        r2_collector = np.zeros_like(N)

        for i in tqdm(range(len(N))):
            for j in range(reps):
                n = N[i]
                pos = np.random.randn(n, 3).astype(np.float64)
                final = simulate(
                    positions=pos,
                    steps=steps,
                    d0=d0,
                    epsilon=epsilon,
                    sigma=sigma,
                    gamma=gamma,
                    k_B=k_B,
                    k=k,
                    T=T,
                    dt=1e-3,
                    ideal_chain=True,
                )

                r2_collector[i] += np.mean(bond_length(final))

        plt.plot(N, r2_collector / reps, label="bond length")
        plt.plot(N, 3 * k_B * T / k / N, label="ideal bond length", ls="--")

        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/b_length_val.png", dpi=300)
        plt.show()
        plt.close()

    if BOND_VAR_VAL:
        steps = 500
        reps = 40

        N = np.linspace(10, 210, 50, dtype=int)
        r2_collector = np.zeros_like(N)

        for i in tqdm(range(len(N))):
            for j in range(reps):
                n = N[i]
                pos = np.random.randn(n, 3).astype(np.float64)
                final = simulate(
                    positions=pos,
                    steps=steps,
                    d0=d0,
                    k=k,
                    epsilon=epsilon,
                    sigma=sigma,
                    gamma=gamma,
                    k_B=k_B,
                    T=T,
                    dt=1e-3,
                    ideal_chain=True,
                )

                r2_collector[i] += np.mean(bond_length_var(final, d0))

        plt.plot(N, r2_collector / reps, label="bond variation")
        plt.plot(
            N,
            3 * k_B * T / k * np.ones_like(r2_collector),
            label="theoretical",
            ls="--",
        )

        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/b_var_val.png", dpi=300)
        plt.show()
        plt.close()
