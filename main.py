from itertools import combinations, product

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.random import rand
from tqdm import tqdm

BBOX_STANDARD = []
BBOX_SAMPLE = []


def box_volume(box: list) -> float:
    return (box[0][1] - box[0][0]) * (box[1][1] - box[1][0]) * (box[2][1] - box[2][0])


def box_2d_area(box: list) -> float:
    return (box[0][1] - box[0][0]) * (box[1][1] - box[1][0])


def box_standard(r: float, R: float, origin: list[float] = [0.0, 0.0, 0.0]) -> list:
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
    min_x, min_z = box[0][0], box[1][0]
    max_x, max_z = box[0][1], box[1][1]

    samples = np.random.uniform([min_x, min_z], [max_x, max_z], size=(n, 2))
    return samples


def monte_carlo_2d(
    k: float, r: float, R: float, n: int = 100_000
) -> tuple[float, tuple[np.ndarray, np.ndarray]]:
    """
    Estimates area of intersection between circle (radius k) and torus (R, r).
    """
    # box in x and z only (new x-y sytem defined here)
    box = box_onesided_2d(r, R)

    # pts in 2d
    pts = samples_2d(box, n)

    # add zero z cooridinate for sphere and torus function compatibility
    pts_3d = np.hstack((pts[:, 0:1], np.zeros((pts.shape[0], 1)), pts[:, 1:2]))

    inside_sphere = sphere(pts_3d, k)
    inside_torus = torus(pts_3d, r, R)

    # add points inside
    count = np.sum(inside_sphere & inside_torus)

    # in points boolean mask
    pts_in = inside_sphere & inside_torus

    # return area
    area = box_2d_area(box) * count / n

    # find centroid of area
    centroid_pos, centroid_neg = find_centroid(pts[pts_in])
    # find x for centroid
    x_centroid = centroid_pos[0]

    # rotate area around centroid to get volume
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
    """
    box = box_standard(r, R, origin=origin)  # or box_sample(r, R)
    if deterministic:
        pts = deterministic_sequence(np.mean(box, axis=1), n)
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

    With probability `p`, samples are drawn from a standard box centered at (0, 0, 0),
    and with probability (1 - p), from a smaller box centered at a shifted `origin`.
    """
    if origin is None:
        origin = [0.0, 0.0, 0.0]

    # --- Vectorized sampling ---
    n1 = int(p * n)
    n2 = n - n1

    box1 = box_standard(r, R, origin=origin)
    box2 = box_sample(r, R, origin=origin)

    pts1 = samples(box1, n1)
    pts2 = samples(box2, n2)

    # --- Evaluate intersection for box1 ---
    mask1 = sphere(pts1, k) & torus(pts1, r, R)
    count1 = np.count_nonzero(mask1)

    # --- Evaluate intersection for box2 ---
    mask2 = sphere(pts2, k) & torus(pts2, r, R)
    count2 = np.count_nonzero(mask2)

    # --- Combine estimates (weighted average) ---
    vol1 = box_volume(box1)
    vol2 = box_volume(box2)

    result = p * (count1 / n1 * vol1) + (1 - p) * (count2 / n2 * vol2)

    return result, ((pts1, mask1), (pts2, mask2))


def surface_to_volume(R: float, surface_area: float) -> float:
    """
    Converts surface area to volume
    """
    volume = surface_area * 2 * np.pi * R * surface_area

    return volume


def generate_dataframe(
    k: float, r: float, R: float, n: int = 100_000
) -> tuple[pd.Series, pd.Series]:
    # 1. Run monte_carlo_2d
    # 2. Run monte_carlo_3d
    # 3. Run mixed_sampling
    # 4. Run deterministic_sequence

    results = []
    results.append(pd.DataFrame(run_multi_executions(monte_carlo_2d, n, k, r, R)))
    results.append(pd.DataFrame(run_multi_executions(monte_carlo_3d, n, k, r, R)))
    results.append(pd.DataFrame(run_multi_executions(mixed_sampling, n, k, r, R)))
    results.append(
        pd.DataFrame(
            run_multi_executions(monte_carlo_3d, n, k, r, R, deterministic=True)
        )
    )
    df = pd.concat(results, axis=1)
    means = df.mean(axis=0)
    stds = df.std(axis=0)

    return means, stds


def generate_table(means: pd.DataFrame, std: pd.DataFrame, names_methods: list[str]):

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


def deterministic_sequence(points: np.ndarray, n: int = 100_000) -> np.ndarray:
    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)

    x[0], y[0], z[0] = points[:3]
    for i in range(1, n):
        x[i] = 3.8 * x[i - 1] * (1 - x[i - 1])
        y[i] = 3.8 * y[i - 1] * (1 - y[i - 1])
        z[i] = 3.8 * z[i - 1] * (1 - z[i - 1])

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

        box = box_standard(r, R)

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
    plt.show()

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
    for i in tqdm(range(runs), desc="Running simulations"):
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

    if pts.shape[1] == 3:

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


if __name__ == "__main__":
    # Parameters
    seed_deterministic = np.random.rand(3)
    k = 1.0  # sphere radius
    r = 0.4  # torus minor radius
    R = 0.75  # torus major radius
    n = 100_000  # Monte Carlo samples
    origin_shift = [0.0, 0.0, 0.1]

    # --- Monte Carlo volume estimation with mixed sampling ---

    estimated_volume, (pts, pts_in) = monte_carlo_3d(k, r, R, n, origin=origin_shift)
    print(
        f"Estimated intersection volume (sphere ∩ torus) with torus' origin shifted: {estimated_volume:.9f}"
    )

    estimated_volume_mix, ((pts1, pts1_in), (pts2, pts2_in)) = mixed_sampling(
        k, r, R, n, p=0.5, origin=origin_shift
    )
    print(
        f"shape of pt1, pt1_in: {pts1.shape, pts1_in.shape}, shape of pt2, pt2_in: { pts2.shape, pts2_in.shape}"
    )
    print(
        f"Estimated intersection volume (sphere ∩ torus) with mixed sampling and torus' origin shifted: {estimated_volume_mix:.9f}"
    )

    plot_mix_2d(
        k,
        r,
        R,
        origin_t=origin_shift,
        pts1=pts1,
        pts1_in=pts1_in,
        pts2=pts2,
        pts2_in=pts2_in,
        show=True,
        save_path="img/mix_sampling_2d.png",
        title="2D cross-section of miixture sampling",
    )

    # --- Generate results table ---
    names_methods = ["Mixed Sampling", "2D MC", "3D MC", "Deterministic Sequence"]
    means, std = generate_dataframe(k, r, R, n=1_000)
    generate_table(means, std, names_methods)
