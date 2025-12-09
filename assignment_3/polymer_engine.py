import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from numba import njit
from tqdm import tqdm
from scipy import constants


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


def linear_fit(x, a, b):
    return a * x + b


def plot_polymer(position: np.ndarray, save_path: str = ""):
    coords = position


def validation_simulation():

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
    plot_polymer(position=final, save_path="img/polymer_plot")

    plt.plot(N, r_ee2_collector / 40, label="end to end radius")
    plt.plot(N, r_g2_collector / 40, label="gyration radius")
    plt.plot(N, (N - 1) * (1 / 10 + 1), label="ideal end to end", ls="--")
    plt.plot(N, (N * N - 1) * (1 / 10 + 1) / 6 / N, label="ideal gyration", ls="--")

    plt.grid(alpha=0.5)
    plt.legend()
    plt.show()


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


# ================================================================
# Example run
# ================================================================

if __name__ == "__main__":
    set_log_level("ERROR")  # ERROR, WARN, INFO, DEBUG
    set_log_output("sim.log")  # or None

    # ================================================================
    # Validation - ideal chain
    # ================================================================

    DIFFUSION_VAL = False
    WARM_UP = False
    RADIUS_VAL = False
    BOND_VAL = False
    BOND_VAR_VAL = False
    FULL_SIM = True

    # sigma = 10**-9
    # d0 = 0.95  # boh
    # m= 1.66*10**-25
    # gamma = 0.75 #boh
    # k_B = constants.Boltzmann
    # epsilon = k_B * 300
    # T = 300
    # dt = 1e-3

    sigma = 1
    d0 = 1  #
    gamma = 1  # boh
    k_B = 1
    epsilon = 1
    T = 1
    k = 30
    dt = 1e-3

    if FULL_SIM:
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

    if DIFFUSION_VAL:
        steps = 10000
        reps = 1000

        N = 20
        pos = np.random.randn(N, 3).astype(np.float64)

        MSD_collector = np.zeros(steps - 1)
        CMx_trace_collector = np.zeros((reps, steps))
        CMy_trace_collector = np.zeros((reps, steps))
        CMz_trace_collector = np.zeros((reps, steps))

        for r in tqdm(range(reps)):
            positions = np.zeros((N, 3))
            for j in range(1, N):
                displacement = np.random.randn(3)
                positions[j] = positions[j - 1] + d0 * displacement / np.linalg.norm(
                    displacement
                )

            positions -= positions.mean(axis=0)
            pos_start = positions.copy()
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
                MSD_collector[step] += MC_MSD(pos_start, positions)

        t = np.cumsum(np.ones(steps - 1) * dt)

        popt, pcov = curve_fit(linear_fit, t, MSD_collector / reps)
        a_opt, b_opt = popt
        print("optimized parameters: a = {a_opt}, b = {b_opt}")

        plt.plot(t, MSD_collector / reps, label="MSD of center of mass", markersize=7)
        plt.plot(t, 6 * t * k_B * T / (N * gamma), label="theoretical MSD", ls="--")
        plt.plot(t, linear_fit(t, a_opt, b_opt), "r-", label="fitted curve")

        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/diff_validation.png", dpi=300)
        plt.close()

        for r in range(reps):
            plt.plot(
                t,
                CMx_trace_collector[r, :],
                label="trace of center of mass",
                alpha=0.7,
                lw=0.5,
            )

        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/diff_spread_x.png", dpi=300)
        plt.close()

        for r in range(reps):
            plt.plot(
                t,
                CMy_trace_collector[r, :],
                label="trace of center of mass",
                alpha=0.7,
                lw=0.5,
            )

        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/diff_spread_y.png", dpi=300)
        plt.close()

        for r in range(reps):
            plt.plot(
                t,
                CMz_trace_collector[r, :],
                label="trace of center of mass",
                alpha=0.7,
                lw=0.5,
            )

        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/diff_spread_z.png", dpi=300)
        plt.close()

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            CMx_trace_collector[0],
            CMy_trace_collector[0],
            CMz_trace_collector[0],
            lw=1.0,
            color="blue",
        )

        # color by time
        p = ax.scatter(
            CMx_trace_collector[0],
            CMy_trace_collector[0],
            CMz_trace_collector[0],
            c=np.arange(steps),
            cmap="viridis",
            s=5,
        )

        fig.colorbar(p, ax=ax, label="Time step")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Diffusion Trajectory")
        plt.savefig("diffusion_3D.png", dpi=300)
        plt.close()

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=CMx_trace_collector[0],
                    y=CMy_trace_collector[0],
                    z=CMz_trace_collector[0],
                    mode="lines+markers",
                    marker=dict(size=2, color=np.arange(steps), colorscale="Viridis"),
                    line=dict(color="blue"),
                )
            ]
        )

        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title="3D Diffusion Trajectory",
        )

        fig.show()
        fig.write_html("diffusion_plot.html")

    if WARM_UP:
        steps_sample = 5000
        reps = 1000

        r_ee_collector = np.zeros((reps, steps_sample))
        # r_g_collector = np.zeros((reps, steps_sample))

        N = 50
        for r in tqdm(range(reps)):
            positions = np.zeros((N, 3))
            for j in range(1, N):
                displacement = np.random.randn(3)
                positions[j] = positions[j - 1] + d0 * displacement / np.linalg.norm(
                    displacement
                )
            positions -= positions.mean(axis=0)

            # b2 = end_to_end_radius2(positions) / N

            # print("data collection")
            for s in range(steps_sample):
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
                # r_g_collector[r, s] = np.sqrt(gyration_radius2(positions))

        t = np.linspace(dt, dt * steps_sample, steps_sample)

        for i in range(20):
            plt.plot(t, r_ee_collector[i, :], alpha=0.5, lw=0.5)

        plt.plot(
            t,
            np.mean(r_ee_collector, axis=0),
            label="average end to end radius",
            color="black",
        )

        plt.plot(t, np.sqrt(N) * np.ones_like(t), ls="--", c="r")
        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/end_to_end_over_time.png", dpi=300)
        plt.close()

        # for i in range(20):
        #    plt.plot(t, r_g_collector[i, :], alpha=0.5, lw=0.5)
        #
        # plt.plot(
        #    t,
        #    np.mean(r_g_collector, axis=0),
        #    label="average gyration radius",
        #    color="black",
        # )
        #
        # plt.plot(t, np.sqrt(N * k_B * T / k / 6) * np.ones_like(t), ls="--", c="r")
        #
        # plt.grid(alpha=0.5)
        # plt.legend()
        # plt.savefig("img/gyration_over_time.png", dpi=300)
        # plt.close()

    if RADIUS_VAL:
        N = np.linspace(10, 510, 50, dtype=int)
        r_ee2_collector = np.zeros_like(N)
        MSD_collector = np.zeros_like(N)
        # r_g2_collector = np.zeros_like(N)

        steps_equil = 0
        steps_sample = 1000
        reps = 200

        for i in tqdm(range(len(N))):
            n = N[i]
            Ree2 = 0
            # Rg2 = 0

            for r in tqdm(range(reps)):
                positions = np.zeros((n, 3))
                MSD = 0.0

                for j in range(1, n):
                    displacement = np.random.randn(3)
                    positions[j] = positions[
                        j - 1
                    ] + d0 * displacement / np.linalg.norm(displacement)

                positions -= positions.mean(axis=0)
                start_pos = positions.copy()

                # b2 = end_to_end_radius2(positions) / n
                # equilibrium
                # positions = simulate(
                #     positions=positions,
                #     steps=steps_equil,
                #     d0=d0,
                #     k=k,
                #     epsilon=epsilon,
                #     sigma=sigma,
                #     gamma=gamma,
                #     k_B=k_B,
                #     T=T,
                #     dt=1e-3,
                #     ideal_chain=True,
                # )
                # positions -= np.mean(positions, axis=0)

                ree2_sum = 0.0
                # rg2_sum = 0.0

                # print("data collection")
                for _ in range(steps_sample):
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
                        ideal_chain=True,
                    )
                    # positions -= positions.mean(axis=0)
                    ree2_sum += end_to_end_radius2(positions)
                    # rg2_sum += gyration_radius2(positions)

                Ree2 += ree2_sum / steps_sample
                MSD += np.mean(positions, axis=0)
                # Rg2 += rg2_sum / steps_sample
            r_ee2_collector[i] = Ree2 / reps
            MSD_collector[i] = MSD / reps
            # r_g2_collector[i] = Rg2 / reps

        # plt.plot(N, np.sqrt(r_g2_collector), label="gyration radius")
        # plt.plot(N, np.sqrt(N), label="ideal gyration", ls="--")
        # plt.grid(alpha=0.5)
        # plt.legend()
        # plt.savefig("img/gyration_val.png", dpi=300)
        # plt.close()

        plt.plot(N, np.sqrt(r_ee2_collector), label="end to end radius")
        plt.plot(N, np.sqrt(N), label="ideal end to end", ls="--")

        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/end_to_end_val.png", dpi=300)
        plt.close()

        plt.plot(N, MSD_collector, label="MSD")
        plt.plot(
            N, k_B * T / gamma / N, label="theoretical diffusion constant", ls="--"
        )

        plt.grid(alpha=0.5)
        plt.legend()
        plt.savefig("img/diffusion_N.png", dpi=300)
        plt.close()

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
