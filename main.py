from itertools import combinations, product

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1 import host_subplot
from numba import njit
from scipy.stats import norm
from tqdm import tqdm


def box_volume(box: list) -> float:
    """
    Calculates the volume of a 3D box defined by its min and max coordinates along each axis.
    """
    return (box[0][1] - box[0][0]) * (box[1][1] - box[1][0]) * (box[2][1] - box[2][0])


def box_2d_area(box: list) -> float:
    """
    Calculates the area of a 2D box defined by its min and max coordinates along each axis.
    """
    return (box[0][1] - box[0][0]) * (box[1][1] - box[1][0])


def box_standard(
    r: float, R: float, origin: list[float] = [0.0, 0.0, 0.0]
) -> list[list]:
    """
    Box exactly around the torus in 3D
    """
    dy = R + r + origin[1]
    dx = R + r + origin[0]
    dz = r + origin[2]

    return [[-dx, dx], [-dy, dy], [-dz, dz]]


def box_onesided_2d(r: float, R: float, origin: list[float] = [0.0, 0.0]) -> list:
    """
    Box exactly around one side of the torus in 2D (x-z plane, positive x only)
    """
    dy = r + origin[1]
    dy_minus = -r + origin[1]
    dx = R + r + origin[0]
    dx_minus = R - r + origin[0]

    return [[dx_minus, dx], [dy_minus, dy]]


def box_sample(r: float, R: float, origin: list[float] = [0.0, 0.0, 0.0]) -> list:
    """
    Smaller box shifted towards one side of the torus in 3D
    """
    dy = R + r
    dx = R + r
    dz = r

    return [
        [-dx + origin[0], dx + origin[1]],
        [-dy + origin[1], dy + origin[1]],
        [-dz + origin[2], dz + origin[2]],
    ]


def torus(
    samples: np.ndarray,
    r: float,
    R: float,
    origin: list[float] = [0.0, 0.0, 0.0],
) -> np.ndarray:
    """
    Returns true if the given points (x, y, z) is inside the torus defined by the
    major radius R and the minor radius r.
    """
    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
    x0, y0, z0 = origin[0], origin[1], origin[2]
    val = (np.sqrt((x - x0) ** 2 + (y - y0) ** 2) - R) ** 2 + (z - z0) ** 2 - r**2
    mask = val <= 0
    return mask


def sphere(samples: np.ndarray, k: float) -> np.ndarray:
    """
    Returns true if the given points (x, y, z) is inside the sphere defined by the
    radius k.
    """
    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
    val = x**2 + y**2 + z**2 - k**2
    mask = val <= 0
    return mask


def samples(
    box: list[list[float]],
    n: int = 100_000,
):
    """
    Generates n random samples within the given 3D box.
    """

    min_x, min_y, min_z = box[0][0], box[1][0], box[2][0]
    max_x, max_y, max_z = box[0][1], box[1][1], box[2][1]

    samples = np.random.uniform(
        [min_x, min_y, min_z], [max_x, max_y, max_z], size=(n, 3)
    )
    return samples


def samples_2d(
    box: list[list[float]],
    n: int = 100_000,
):
    """
    Generates n random samples within the given 2D box.
    """
    min_x, min_z = box[0][0], box[1][0]
    max_x, max_z = box[0][1], box[1][1]

    samples = np.random.uniform([min_x, min_z], [max_x, max_z], size=(n, 2))
    return samples


def monte_carlo_2d(
    k: float, r: float, R: float, n: int = 100_000
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """
    Estimates area of intersection between circle (radius k) and torus (R, r).
    Uses the method of Pappus to convert area to volume.
    Returns estimated volume and the sampled points with in-mask.
    """
    box = box_onesided_2d(r, R)
    pts = samples_2d(box, n)
    pts_3d = np.hstack((pts[:, 0:1], np.zeros((pts.shape[0], 1)), pts[:, 1:2]))

    inside_sphere = sphere(pts_3d, k)
    inside_torus = torus(pts_3d, r, R)

    count = np.sum(inside_sphere & inside_torus)
    pts_in = inside_sphere & inside_torus
    area = box_2d_area(box) * count / n

    centroid_pos, _ = find_centroid(pts[pts_in])
    x_centroid = centroid_pos[0]

    volume = area * 2 * np.pi * x_centroid

    return volume, (pts, pts_in)


def monte_carlo_3d(
    k: float,
    r: float,
    R: float,
    n: int = 100_000,
    origin: list = [0.0, 0.0, 0.0],
    deterministic: bool = False,
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """
    Estimates volume of intersection between sphere (radius k) and torus (R, r).
    Returns estimated volume and the sampled points with in-mask.
    """
    box = box_standard(r, R, origin=origin)  # or box_sample(r, R)
    if deterministic:
        box = np.array(box, dtype=np.float64)
        pts = deterministic_sequence(np.random.uniform(0, 1, 3), box)
    else:
        pts = samples(box, n)  # deterministic_sequence(box, n)

    inside_sphere = sphere(pts, k)
    inside_torus = torus(pts, r, R, origin=origin)

    count = np.sum(inside_sphere & inside_torus)

    # in points boolean mask
    pts_in = inside_sphere & inside_torus

    return count / n * box_volume(box), (pts, pts_in)


def mixed_sampling(
    k: float,
    r: float,
    R: float,
    n: int = 100_000,
    p: float = 0.5,
    origin: list[float] | None = None,
) -> tuple[float, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]]:
    """
    Estimates the volume of intersection between a sphere (radius k) and a torus (R, r)
    using mixed Monte Carlo sampling.

    Each of the n samples is drawn from:
      - box1 (centered at 0) with probability `p`
      - box2 (centered at `origin`) with probability (1 - p)

    Returns:
        tuple:
            - result (float): Estimated intersection volume.
            - ((pts1, mask1), (pts2, mask2)): Points and inside-mask for both boxes.
    """
    if origin is None:
        origin = [0.0, 0.0, 0.0]

    # Boxes
    box1 = box_standard(r, R)
    box2 = box_sample(r, R, origin=origin)

    # --- Randomly assign which box each point will come from ---
    # 0 → box1, 1 → box2
    choices = np.random.choice([0, 1], size=n, p=[p, 1 - p])

    # Count how many per box (not deterministic anymore)
    n1 = np.count_nonzero(choices == 0)
    n2 = n - n1

    # --- Sample points ---
    pts1 = samples(box1, n1)
    pts2 = samples(box2, n2)

    # --- Evaluate intersection for each box ---
    mask1 = sphere(pts1, k) & torus(pts1, r, R)
    mask2 = sphere(pts2, k) & torus(pts2, r, R)

    # --- Volume estimates per box ---
    vol1 = box_volume(box1)
    vol2 = box_volume(box2)

    # Monte Carlo estimate: weighted by proportion of total points drawn from each box
    result = (np.sum(mask1) / n1 * vol1) * (n1 / n) + (np.sum(mask2) / n2 * vol2) * (
        n2 / n
    )

    return result, ((pts1, mask1), (pts2, mask2))


def surface_to_volume(d: float, surface_area: float) -> float:
    """
    Converts surface area to volume
    """
    volume = surface_area * 2 * np.pi * d * surface_area

    return volume


def generate_dataframe(
    k: float, r: float, R: float, n: int = 100_000
) -> tuple[pd.Series, pd.Series]:
    """
    Generates a dataframe with the results of multiple executions of different Monte Carlo methods.
    """

    results = []
    # 1. Run monte_carlo_2d
    results.append(pd.DataFrame(run_multi_executions(monte_carlo_2d, n, k, r, R)))
    # 2. Run monte_carlo_3d
    results.append(pd.DataFrame(run_multi_executions(monte_carlo_3d, n, k, r, R)))
    # 3. Run mixed_sampling
    results.append(pd.DataFrame(run_multi_executions(mixed_sampling, n, k, r, R)))
    # 4. Run deterministic_sequence
    results.append(
        pd.DataFrame(
            run_multi_executions(monte_carlo_3d, n, k, r, R, deterministic=True)
        )
    )
    df = pd.concat(results, axis=1)
    means = df.mean(axis=0)
    stds = df.std(axis=0)

    return means, stds


def generate_table(means: pd.Series, std: pd.Series, names_methods: list[str]):
    """
    Generates a LaTeX table with the results of the Monte Carlo methods.
    """

    table = pd.DataFrame(
        {
            "Method": names_methods,
            "Mean Volume": means.values,
            "Std Dev": std.values,
        }
    )
    table.to_latex(buf="img/table.tex", index=False)


def find_centroid(in_pts: np.ndarray) -> np.ndarray:
    """
    Takes all Monte Carlo sample points and the in-mask of those points to find the centroid of the surface area slice
    One for positive x and one for negative x - 2D only
    """
    # split in positive x and y
    centroid_x_plus = np.mean(in_pts[in_pts[:, 0] >= 0][:, 0])
    centroid_x_minus = np.mean(in_pts[in_pts[:, 0] < 0][:, 0])
    centroid_y = np.mean(in_pts[:, 1])

    return [[centroid_x_plus, centroid_y], [centroid_x_minus, centroid_y]]


@njit
def deterministic_sequence(
    points: np.ndarray, box: list[list], n: int = 100_000
) -> np.ndarray:
    """
    Generates a deterministic sequence of n points in 3D using the logistic map.
    """

    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)

    x[0], y[0], z[0] = points[0], points[1], points[2]

    for i in range(1, n):
        x[i] = 3.8 * x[i - 1] * (1 - x[i - 1])
        y[i] = 3.8 * y[i - 1] * (1 - y[i - 1])
        z[i] = 3.8 * z[i - 1] * (1 - z[i - 1])

    x = x * (box[0][1] - box[0][0]) + box[0][0]
    y = y * (box[1][1] - box[1][0]) + box[1][0]
    z = z * (box[2][1] - box[2][0]) + box[2][0]
    return np.stack((x, y, z), axis=1)


def plot_2d(
    k: float,
    r: float,
    R: float,
    origin_s: list = [0, 0],
    origin_t: list = [0, 0],
    pts: np.ndarray | None = None,
    pts_in: np.ndarray | None = None,
    centroid: np.ndarray | None = None,
    save_path: str = "",
    show: bool = False,
    title: str = "",
) -> Axes:
    """
    Plots the intersection between the sphere and the torus.
    k: radius of the sphere
    r: minor radius of the torus
    R: major radius of the torus
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    max_val = np.max([k, R + r])
    x_max = max_val * 1.2

    # sphere
    sphere = plt.Circle(origin_s, k, color="blue", fill=False)
    k_line = plt.Line2D(
        [origin_s[0] + 0],
        [origin_s[1], origin_s[1] + k],
        color="blue",
        ls="--",
        label="k",
    )

    # torus
    torus_l = plt.Circle((origin_t[0] + R, origin_t[1]), r, color="green", fill=False)
    r_line = plt.Line2D(
        [origin_t[0] + R, origin_t[0] + R + r],
        [origin_t[1], origin_t[1]],
        color="orange",
        ls="--",
        label="r",
    )
    R_line = plt.Line2D(
        [origin_t[0], origin_t[0] + R],
        [origin_t[1], origin_t[1]],
        color="green",
        ls="--",
        label="R",
    )
    torus_r = plt.Circle((origin_t[0] - R, origin_t[1]), r, color="green", fill=False)

    # box (only x and z)
    # from 3d:
    if pts.shape[1] == 3:

        box = box_standard(r, R, origin=[0, *origin_t])

        rect = mpatches.Rectangle(
            (box[0][0], box[2][0]),
            box[0][1] - box[0][0],
            box[2][1] - box[2][0],
            fill=False,
            ls="-.",
            color="darkred",
            lw=2,
            label="standard box",
        )
        ax.add_patch(rect)

    # if from 2d, use narrow box around torus
    if pts.shape[1] == 2:
        box = box_onesided_2d(r, R)

        rect = mpatches.Rectangle(
            (box[0][0], box[1][0]),
            box[0][1] - box[0][0],
            box[1][1] - box[1][0],
            fill=False,
            ls="-.",
            color="darkred",
            lw=2,
            label="standard box",
        )
        ax.add_patch(rect)

    # points from monte carlo sim
    s = 0.1
    if pts is not None:
        if pts_in is not None:
            # in and out samples

            # if from 3d points
            if pts.shape[1] == 3:
                # slice y = 0 plane
                delta = 0.35 * r
                slice_y = np.abs(pts[:, 1]) < delta
                pts_in_2d = pts_in & slice_y
                pts_out_2d = ~pts_in & slice_y
                # mask
                in_pts = pts[pts_in_2d][:, [0, -1]]
                # binary invert for out points
                out_pts = pts[pts_out_2d][:, [0, -1]]
            # points in 2d
            else:
                in_pts = pts[pts_in]
                # binary invert for out points
                out_pts = pts[~pts_in]

            ax.scatter(
                out_pts[:, 0],
                out_pts[:, 1],
                color="darkred",
                marker=".",
                s=s,
                alpha=0.3,
                label="out samples",
            )

            ax.scatter(
                in_pts[:, 0],
                in_pts[:, 1],
                color="darkblue",
                marker=".",
                s=s,
                alpha=0.3,
                label="in samples",
            )
        else:
            # only out samples
            pts = np.array(pts)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                color="darkred",
                marker=".",
                s=s,
                alpha=0.3,
                label="out samples",
            )

    if centroid is not None:
        plt.scatter(
            centroid[0][0],
            centroid[0][1],
            color="black",
            marker="o",
            s=25,
            label="centroid",
        )

        plt.scatter(centroid[1][0], centroid[1][1], color="darkblue", marker="o", s=25)

    # plot
    ax.add_artist(sphere)
    ax.add_artist(k_line)
    ax.add_artist(torus_l)
    ax.add_artist(torus_r)
    ax.add_artist(r_line)
    ax.add_artist(R_line)

    # plot origin
    plt.scatter(origin_s[0], origin_s[1], color="blue", marker="x")
    plt.scatter(origin_t[0], origin_t[1], color="green", marker="x")
    # limits
    ax.set_xlim([-x_max, x_max])
    ax.set_ylim([-x_max, x_max])
    if title:
        ax.set_title(title)
    else:
        ax.set_title("2D cross-section of Sphere and Torus")

    # nicer plot
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # show flag
    if show:
        plt.show()

    # save if path is there
    if save_path:
        fig.savefig(save_path, dpi=300)

    return ax


def plot_3d(
    k: float,
    r: float,
    R: float,
    pts: np.ndarray | None = None,
    pts_in: np.ndarray | None = None,
    save_path: str = "",
    show=False,
) -> Axes:
    """
    Plots the intersection between the sphere and the torus in 3D.
    k: radius of the sphere
    r: minor radius of the torus
    R: major radius of the torus
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Sphere
    phi, theta = np.mgrid[0.0 : np.pi : 100j, 0.0 : 2.0 * np.pi : 100j]
    xs = k * np.sin(phi) * np.cos(theta)
    ys = k * np.sin(phi) * np.sin(theta)
    zs = k * np.cos(phi)
    ax.plot_surface(xs, ys, zs, edgecolor="darkblue", linewidth=0.1, alpha=0.1)

    # Torus
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, 2 * np.pi, 100)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax.plot_surface(X, Y, Z, edgecolor="darkgreen", linewidth=0.1, alpha=0.1)

    # box
    box = box_standard(r, R)
    x_min, x_max = box[0]
    y_min, y_max = box[1]
    z_min, z_max = box[2]
    # vertices of the box
    box_vertices = np.array(
        list(product([x_min, x_max], [y_min, y_max], [z_min, z_max]))
    )
    # draw box edges
    for s, e in combinations(box_vertices, 2):
        # two vertices share an edge if they differ by exactly one coordinate
        if np.sum(np.abs(s - e) == 0) == 2:
            ax.plot3D(*zip(s, e), color="darkred", linestyle="-", linewidth=0.9)

    # draw monte carlo points if available
    s = 0.05
    if pts is not None:
        if pts_in is not None:
            # plot out samples
            in_pts = pts[pts_in]
            # out points (bitwise invert)
            out_pts = pts[~pts_in]

            ax.scatter(
                out_pts[:, 0],
                out_pts[:, 1],
                out_pts[:, 2],
                color="darkred",
                marker=".",
                s=s,
                alpha=0.3,
            )

            # plot in samples
            ax.scatter(
                in_pts[:, 0],
                in_pts[:, 1],
                in_pts[:, 2],
                color="darkblue",
                marker=".",
                s=s,
                alpha=0.3,
            )
        else:
            # plot out samples
            pts = np.array(pts)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                color="darkred",
                marker=".",
                s=s,
                alpha=0.3,
            )

    ax.set_title("3D view of Sphere and Torus")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.tight_layout()
    if show:
        plt.show()

    # save if path is there
    if save_path:
        fig.savefig(save_path, dpi=300)
    return ax


def run_multi_executions(
    sim: callable,
    runs: int,
    k: float,
    r: float,
    R: float,
    deterministic: bool = False,
) -> np.ndarray:
    """
    Runs multiple executions of the given simulation function and returns the results.
    sim: simulation function to run
    runs: number of runs to execute
    """
    results = np.empty(runs)
    for i in tqdm(range(runs), desc=f"Running simulations number", total=runs):
        if deterministic:
            result, _ = sim(k, r, R, n, deterministic=True)
        else:
            result, _ = sim(k, r, R, n)
        results[i] = result
    return results


def plot_mix_2d(
    k: float,
    r: float,
    R: float,
    origin_s: list = [0, 0, 0],
    origin_t: list = [0, 0, 0],
    pts1: np.ndarray | None = None,
    pts1_in: np.ndarray | None = None,
    pts2: np.ndarray | None = None,
    pts2_in: np.ndarray | None = None,
    save_path: str = "",
    show: bool = False,
    title: str = "",
) -> Axes:
    """
    Plots the intersection between the sphere and the torus.
    k: radius of the sphere
    r: minor radius of the torus
    R: major radius of the torus
    """

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    max_val = np.max([k, R + r])
    x_max = max_val * 1.2

    # sphere
    sphere = plt.Circle(origin_s, k, color="blue", fill=False)
    k_line = plt.Line2D(
        [origin_s[0] + 0],
        [origin_s[2], origin_s[2] + k],
        color="blue",
        ls="--",
        label="k",
    )

    # torus
    torus_l = plt.Circle((origin_t[0] + R, origin_t[2]), r, color="green", fill=False)
    r_line = plt.Line2D(
        [origin_t[0] + R, origin_t[0] + R + r],
        [origin_t[2], origin_t[2]],
        color="orange",
        ls="--",
        label="r",
    )
    R_line = plt.Line2D(
        [origin_t[0], origin_t[0] + R],
        [origin_t[2], origin_t[2]],
        color="green",
        ls="--",
        label="R",
    )
    torus_r = plt.Circle((origin_t[0] - R, origin_t[2]), r, color="green", fill=False)

    # box (only x and z)
    # from 3d:

    box1 = box_standard(r, R, origin=origin_t)
    box2 = box_sample(r, R, origin=origin_t)

    if pts1.shape[1] == 3:

        rect1 = mpatches.Rectangle(
            (box1[0][0], box1[2][0]),
            box1[0][1] - box1[0][0],
            box1[2][1] - box1[2][0],
            fill=False,
            ls="-.",
            color="purple",
            lw=2,
            label="standard box",
        )

        rect2 = mpatches.Rectangle(
            (box2[0][0], box2[2][0]),
            box2[0][1] - box2[0][0],
            box2[2][1] - box2[2][0],
            fill=False,
            ls="-.",
            color="darkred",
            lw=2,
            label="standard box",
        )
        ax.add_patch(rect1)
        ax.add_patch(rect2)

    s = 0.1

    if pts1 is not None:
        if pts1_in is not None:
            # in and out samples

            # if from 3d points
            if pts1.shape[1] == 3:
                # slice y = 0 plane
                delta = 0.15 * r
                slice1_y = np.abs(pts1[:, 1]) < delta
                pts1_in_2d = pts1_in & slice1_y
                pts1_out_2d = ~pts1_in & slice1_y
                # mask
                in_pts1 = pts1[pts1_in_2d][:, [0, -1]]
                # binary invert for out points
                out_pts1 = pts1[pts1_out_2d][:, [0, -1]]
            # points in 2d
            else:
                in_pts1 = pts1[pts1_in]
                # binary invert for out points
                out_pts1 = pts1[~pts1_in]

            ax.scatter(
                out_pts1[:, 0],
                out_pts1[:, 1],
                color="purple",
                marker=".",
                s=s,
                alpha=0.7,
                label="out samples",
            )

            ax.scatter(
                in_pts1[:, 0],
                in_pts1[:, 1],
                color="blue",
                marker=".",
                s=s,
                alpha=0.3,
                label="in samples",
            )
        else:
            # only out samples
            pts1 = np.array(pts1)
            ax.scatter(
                pts1[:, 0],
                pts1[:, 1],
                color="purple",
                marker=".",
                s=s,
                alpha=0.7,
                label="out samples",
            )

    if pts2 is not None:
        if pts2_in is not None:
            # in and out samples

            # if from 3d points
            if pts2.shape[1] == 3:
                # slice y = 0 plane
                delta = 0.15 * r
                slice2_y = np.abs(pts2[:, 1]) < delta
                pts2_in_2d = pts2_in & slice2_y
                pts2_out_2d = ~pts2_in & slice2_y
                # mask
                in_pts2 = pts2[pts2_in_2d][:, [0, -1]]
                # binary invert for out points
                out_pts2 = pts2[pts2_out_2d][:, [0, -1]]
            # points in 2d
            else:
                in_pts2 = pts2[pts2_in]
                # binary invert for out points
                out_pts2 = pts2[~pts2_in]

            ax.scatter(
                out_pts2[:, 0],
                out_pts2[:, 1],
                color="orange",
                marker=".",
                s=s,
                alpha=0.5,
                label="out samples",
            )

            ax.scatter(
                in_pts2[:, 0],
                in_pts2[:, 1],
                color="darkblue",
                marker=".",
                s=s,
                alpha=0.3,
                label="in samples",
            )
        else:
            # only out samples
            pts2 = np.array(pts2)
            ax.scatter(
                pts2[:, 0],
                pts2[:, 1],
                color="orange",
                marker=".",
                s=s,
                alpha=0.5,
                label="out samples",
            )

    # plot
    ax.add_artist(sphere)
    ax.add_artist(k_line)
    ax.add_artist(torus_l)
    ax.add_artist(torus_r)
    ax.add_artist(r_line)
    ax.add_artist(R_line)

    # plot origin
    plt.scatter(origin_s[0], origin_s[2], color="blue", marker="x")
    plt.scatter(origin_t[0], origin_t[2], color="green", marker="x")
    # limits
    ax.set_xlim([-x_max, x_max])
    ax.set_ylim([-x_max, x_max])
    if title:
        ax.set_title(title)
    else:
        ax.set_title("2D cross-section of Sphere and Torus")

    # nicer plot
    plt.legend()
    plt.tight_layout()
    plt.grid()

    # show flag
    if show:
        plt.show()

    # save if path is there
    if save_path:
        fig.savefig(save_path, dpi=300)

    return ax


def plot_volume_histogram(
    list_of_volumes: list[list[float]], case_labels: list[str], save_path: str
):
    """
    Plots histograms of volume estimates with Gaussian fits for multiple cases.
    """
    rows = int(len(list_of_volumes) // 2)
    cols = 2
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        sharex="col",
    )
    axs = np.atleast_2d(axs)
    n = len(list_of_volumes[0])
    for i, volumes in enumerate(list_of_volumes):
        # histogram
        axs[i // 2, i % 2].hist(volumes, bins=100, color="skyblue", density=True)

        # fit gaussian
        mu = np.mean(volumes)
        std = np.std(volumes)
        xmin = min(volumes)
        xmax = max(volumes)
        x = np.linspace(xmin, xmax, 100)

        # normed to bin width
        y = norm.pdf(x, mu, std)

        axs[i // 2, i % 2].plot(
            x,
            y,
            "r--",
            linewidth=2,
        )

        # plot one sigma lines only until it hits the normal curve
        axs[i // 2, i % 2].vlines(
            mu,
            ymin=0,
            ymax=norm.pdf(mu, mu, std),
            color="darkblue",
            linestyle="--",
            label=r"$\mu$",
        )
        axs[i // 2, i % 2].vlines(
            mu + std,
            ymin=0,
            ymax=norm.pdf(mu + std, mu, std),
            color="darkgreen",
            linestyle="--",
            label=r"$+\sigma$",
        )
        axs[i // 2, i % 2].vlines(
            mu - std,
            ymin=0,
            ymax=norm.pdf(mu - std, mu, std),
            color="darkgreen",
            linestyle="--",
            label=r"$-\sigma$",
        )
        axs[i // 2, i % 2].set_title(f"{case_labels[i]}")
        axs[i // 2, i % 2].set_xlabel("Estimated Volume")
        axs[i // 2, i % 2].set_ylabel("Frequency")

    axs[0, 0].legend()
    plt.suptitle(f"Histograms of Volume Estimates over {n} runs", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Parameters

    k = 1.0  # sphere radius
    r_a = 0.4  # torus minor radius
    R_a = 0.75  # torus major radius
    r_b = 0.5
    R_b = 0.5
    n = 100_000  # Monte Carlo samples
    origin_shift = [0.0, 0.0, 0.1]
    n_runs = 1000

    Q1 = False
    Q2 = False
    Q3 = False
    PSWEEP = True

    # Question 1: volume of intersection between sphere and torus.
    if Q1:
        # case a:
        volumes_a = run_multi_executions(monte_carlo_3d, n_runs - 1, k, r_a, R_a)
        estimated_volume_a_3d, (pts, pts_in) = monte_carlo_3d(k, r_a, R_a, n)
        volumes_a = np.append(volumes_a, estimated_volume_a_3d)
        avg_volume_a = np.mean(volumes_a)
        std_volume_a = np.std(volumes_a)

        plot_2d(k, r_a, R_a, pts=pts, pts_in=pts_in, save_path="img/case_a_2d.png")
        plot_3d(k, r_a, R_a, pts=pts, pts_in=pts_in, save_path="img/case_a_3d.png")

        # case b:
        volumes_b = run_multi_executions(monte_carlo_3d, n_runs - 1, k, r_b, R_b)
        estimated_volume_b_3d, (pts, pts_in) = monte_carlo_3d(k, r_b, R_b, n)
        volumes_b = np.append(volumes_b, estimated_volume_b_3d)
        avg_volume_b = np.mean(volumes_b)
        std_volume_b = np.std(volumes_b)

        plot_2d(k, r_b, R_b, pts=pts, pts_in=pts_in, save_path="img/case_b_2d.png")
        plot_3d(k, r_b, R_b, pts=pts, pts_in=pts_in, save_path="img/case_b_3d.png")

        # case a but with 2d MC approximation
        volumes_a_2d = run_multi_executions(monte_carlo_2d, n_runs - 1, k, r_a, R_a)
        estimated_volume_a_2d, (pts_2d, pts_in_2d) = monte_carlo_2d(k, r_a, R_a, n)
        volumes_a_2d = np.append(volumes_a_2d, estimated_volume_a_2d)
        avg_volume_a_2d = np.mean(volumes_a_2d)
        std_volume_a_2d = np.std(volumes_a_2d)

        plot_2d(
            k,
            r_a,
            R_a,
            pts=pts_2d,
            pts_in=pts_in_2d,
            save_path="img/case_a_2d_from_2d.png",
        )

        # case b but with 2d MC approximation
        volumbes_b_2d = run_multi_executions(monte_carlo_2d, n_runs - 1, k, r_b, R_b)
        estimated_volume_b_2d, (pts_2d, pts_in_2d) = monte_carlo_2d(k, r_b, R_b, n)
        volumbes_b_2d = np.append(volumbes_b_2d, estimated_volume_b_2d)
        avg_volume_b_2d = np.mean(volumbes_b_2d)
        std_volume_b_2d = np.std(volumbes_b_2d)

        plot_2d(
            k,
            r_b,
            R_b,
            pts=pts_2d,
            pts_in=pts_in_2d,
            save_path="img/case_b_2d_from_2d.png",
        )

        print(f"----- Monte Carlo Volume Estimation Results over {n_runs} runs -----")
        print(
            f"Q1, Case A 3D || Mean: {avg_volume_a:.9f} || Standard Deviation: {std_volume_a:.9f}"
        )
        print(
            f"Q1, Case B 3D || Mean: {avg_volume_b:.9f} || Standard Deviation: {std_volume_b:.9f}"
        )
        print(
            f"Q1, Case A 2D || Mean: {avg_volume_a_2d:.9f} || Standard Deviation: {std_volume_a_2d:.9f}"
        )
        print(
            f"Q1, Case B 2D || Mean: {avg_volume_b_2d:.9f} || Standard Deviation: {std_volume_b_2d:.9f}"
        )

        plot_volume_histogram(
            [
                volumes_a,
                volumes_b,
                volumes_a_2d,
                volumbes_b_2d,
            ],
            [
                "Case A 3D",
                "Case B 3D",
                "Case A 2D",
                "Case B 2D",
            ],
            "img/q1_volume_histograms.png",
        )

    # Question 2: change in estimate and error for determinstic sequence
    if Q2:
        volumes_deterministic_a = run_multi_executions(
            monte_carlo_3d, n_runs - 1, k, r_a, R_a, deterministic=True
        )
        estimated_volume_deterministic_a, (pts, pts_in) = monte_carlo_3d(
            k, r_a, R_a, n, deterministic=True
        )
        volumes_deterministic_a = np.append(
            volumes_deterministic_a, estimated_volume_deterministic_a
        )

        avg_volume_deterministic_a = np.mean(volumes_deterministic_a)
        std_volume_deterministic_a = np.std(volumes_deterministic_a)

        plot_2d(
            k, r_a, R_a, pts=pts, pts_in=pts_in, save_path="img/deterministic_a_2d.png"
        )
        plot_3d(
            k, r_a, R_a, pts=pts, pts_in=pts_in, save_path="img/deterministic_a_3d.png"
        )
        print(
            f"Q2, Deterministic Case A || Mean: {avg_volume_deterministic_a:.9f} || Standard Deviation: {std_volume_deterministic_a:.9f}"
        )

        volumes_deterministic_b = run_multi_executions(
            monte_carlo_3d, n_runs - 1, k, r_b, R_b, deterministic=True
        )
        estimated_volume_deterministic_b, (pts, pts_in) = monte_carlo_3d(
            k, r_b, R_b, n, deterministic=True
        )
        volumes_deterministic_b = np.append(
            volumes_deterministic_b, estimated_volume_deterministic_b
        )
        avg_volume_deterministic_b = np.mean(volumes_deterministic_b)
        std_volume_deterministic_b = np.std(volumes_deterministic_b)

        plot_2d(
            k, r_b, R_b, pts=pts, pts_in=pts_in, save_path="img/deterministic_b_2d.png"
        )
        plot_3d(
            k, r_b, R_b, pts=pts, pts_in=pts_in, save_path="img/deterministic_b_3d.png"
        )
        print(
            f"Q2, Deterministic Case B || Mean: {avg_volume_deterministic_b:.9f} || Standard Deviation: {std_volume_deterministic_b:.9f}"
        )

        plot_volume_histogram(
            [
                volumes_deterministic_a,
                volumes_deterministic_b,
            ],
            ["Case A Deterministic", "Case B Deterministic"],
            "img/q2_volume_histograms.png",
        )

    # Question 3: Off center torus
    # a) estimate volume

    if Q3:
        volumes_offcenter = np.empty(0)
        volume_offcenter, (pts, pts_in) = monte_carlo_3d(
            k, r_a, R_a, n, origin=origin_shift
        )
        volumes_offcenter = np.append(volumes_offcenter, volume_offcenter)
        for i in range(n_runs - 1):
            volume_offcenter, _ = monte_carlo_3d(k, r_a, R_a, n, origin=origin_shift)
            volumes_offcenter = np.append(volumes_offcenter, volume_offcenter)

        avg_volume_offcenter = np.mean(volumes_offcenter)
        std_volume_offcenter = np.std(volumes_offcenter)

        plot_2d(
            k,
            r_a,
            R_a,
            origin_t=origin_shift[-2:],
            pts=pts,
            pts_in=pts_in,
            save_path="img/q3a_2d.png",
        )
        # plot_3d(k, r_a, R_a, pts=pts, pts_in=pts_in, save_path="img/q3a_3d.png")
        print(
            f"Q3, a) || Mean: {avg_volume_offcenter:.9f} || Standard Deviation: {std_volume_offcenter:.9f}"
        )
        # b) mixed sampling
        p = 0.5
        volumes_mixed = np.empty(0)
        volume_mixed, ((pts1, pts1_in), (pts2, pts2_in)) = mixed_sampling(
            k, r_a, R_a, n=n, p=p, origin=origin_shift
        )
        volumes_mixed = np.append(volumes_mixed, volume_mixed)
        for i in range(n_runs - 1):
            volume_mixed, _ = mixed_sampling(k, r_a, R_a, n=n, p=p, origin=origin_shift)
            volumes_mixed = np.append(volumes_mixed, volume_mixed)

        avg_volume_mixed = np.mean(volumes_mixed)
        std_volume_mixed = np.std(volumes_mixed)

        plot_mix_2d(
            k,
            r_a,
            R_a,
            origin_t=origin_shift,
            save_path="img/q3b_2d.png",
            pts1=pts1,
            pts1_in=pts1_in,
            pts2=pts2,
            pts2_in=pts2_in,
        )

        print(
            f"Q3, b) || Mean: {avg_volume_mixed:.9f} || Standard Deviation: {std_volume_mixed:.9f}"
        )

    if PSWEEP:
        # p-sweep
        p_range = np.linspace(0.00001, 0.999, 100)
        volumes_p_sweep = np.empty([len(p_range), n_runs])
        for i, p in enumerate(tqdm(p_range, desc="Running p-sweep")):
            for j in range(n_runs):
                volume_p, _ = mixed_sampling(k, r_a, R_a, n=n, p=p, origin=origin_shift)
                volumes_p_sweep[i, j] = volume_p

        # plot p-sweep
        ax1 = host_subplot(111)
        ax2 = ax1.twinx()
        mean = np.mean(volumes_p_sweep, axis=1)
        std = np.std(volumes_p_sweep, axis=1)

        ax1.plot(p_range, mean, color="green", lw=1, label="Mean")
        # on second y axis, plot relative std dev
        ax2.plot(
            p_range,
            std / mean,
            color="blue",
            lw=1,
            label="Std Dev",
        )

        ax1.set_xlabel(r"Mixing Probability $p$")
        ax1.set_ylabel("Estimated Volume Mean")
        ax2.set_ylabel("Estimated Volume Std Dev")

        ax1.set_title("Effect of Mixing Probability on Volume Estimate")
        ax1.legend()
        plt.tight_layout()
        plt.savefig("img/p_sweep.png", dpi=300)
        plt.close()

    # --- Generate results table ---
    # names_methods = ["Mixed Sampling", "2D MC", "3D MC", "Deterministic Sequence"]
    # means, std = generate_dataframe(k, r_a, R_a, n=100)
    # generate_table(means, std, names_methods)
    # count_monte_det, (pts_monte_det, pts_in_monte_det) = monte_carlo_3d(
    #    k, r_a, R_a, deterministic=True
    # )
