import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand

BBOX_STANDARD = []
BBOX_SAMPLE = []


def box_volume(box: list) -> float:
    return (box[0][1] - box[0][0]) * (box[1][1] - box[1][0]) * (box[2][1] - box[2][0])


def box_standard(r: float, R: float, origin: list[float] = [0.0, 0.0, 0.0]) -> list:
    dy = r + origin[1]
    dx = R + r + origin[0]
    dz = r + origin[2]

    return [[-dx, dx], [-dy, dy], [-dz, dz]]


def box_sample(r: float, R: float, origin: list[float] = [0.0, 0.0, 0.0]) -> list:
    dy = r
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


# TODO: this ok ?
def torus2D(x: float, y: float, r: float, R: float, origin: list = [0, 0]) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the torus defined by the
    major radius R and the minor radius r.
    """
    result = ((x - origin[0] - R) ** 2 + (y - origin[1]) ** 2 - r**2 <= 0) or (
        (x - origin[0] + R) ** 2 + (y - origin[1]) ** 2 - r**2 <= 0
    )
    return result


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


def monte_carlo_3d(k: float, r: float, R: float, n: int = 100_000) -> float:
    """
    Estimates volume of intersection between sphere (radius k) and torus (R, r).
    """
    box = box_standard(r, R)  # or box_sample(r, R)

    pts = samples(box, n)  # or deterministic_sequence(box, n)

    inside_sphere = sphere(pts, k)
    inside_torus = torus(pts, r, R)

    count = np.sum(inside_sphere & inside_torus)

    return count / n * box_volume(box)


def mixed_sampling(
    k: float,
    r: float,
    R: float,
    p: float,
    n: int = 100_000,
    origin: list[float] = [0.0, 0.0, 0.0],
) -> float:
    """ """
    # first box
    total1 = 0
    box1 = box_standard(r, R)

    # second box
    total2 = 0
    box2 = box_sample(r, R, origin=[0.0, 0.0, 0.1])

    for _ in range(n):
        rnd = rand()
        if rnd <= p:
            total1 = one_sample(k, r, R, box1, total1)

        else:
            total2 = one_sample(k, r, R, box2, total2)

    surface_area1 = box_volume(box1)
    surface_area2 = box_volume(box2)

    return p * surface_area1 + (1 - p) * surface_area2


def surface_to_volume(R: float, surface_area: float) -> float:
    """
    Converts surface area to volume
    """
    volume = surface_area * 2 * np.pi * R * surface_area

    return volume


# TODO: to change with ndarray
def find_centroid_2d(mc_samples: list, mc_in_mask: list) -> list:
    """
    Takes all Monte Carlo sample points and the in-mask of those points to find the centroid of the surface area slice
    """

    # mc_in_mask contains both sides of the intersection
    # use only points with positive x

    in_samples = np.array(
        [
            mc_samples[i]
            for i in range(len(mc_samples))
            if mc_in_mask[i][0] and mc_in_mask[i][1] and mc_samples[i][0] >= 0
        ]
    )

    # the centroid is the mean of all points
    centroid_x = np.mean(in_samples[:, 0])
    centroid_y = np.mean(in_samples[:, 1])

    return [centroid_x, centroid_y]


# TODO: bottle neck
def deterministic_sequence(points: np.ndarray, n: int = 100_000) -> np.ndarray:
    """
    Generates a deterministic sequence of numbers
    based on the seed. Must be between 0 and 1.
    """
    x_0, y_0, z_0 = points[1], points[1], points[2]

    samples = np.random.uniform(0, 0, size=(n, 3))
    for i in range(n):
        x_0 = (x_0 * 3.8 * (1 - x_0)) % 1
        y_0 = (y_0 * 3.8 * (1 - y_0)) % 1
        z_0 = (z_0 * 3.8 * (1 - z_0)) % 1
        samples[i, 0] = x_0
        samples[i, 1] = y_0
        samples[i, 2] = z_0

    return samples


def _fill_between2D(
    ax, r: float, R: float, k: float, origin_s: list, origin_t: list, alpha=0.3
):
    """
    Fills intersection area betweeen sphere and torus
    """

    # circles (2D cuts)
    torus_R = mpatches.Circle(
        (origin_t[0] + R, origin_t[1]), r, fill=False, ec=None, lw=2
    )
    torus_L = mpatches.Circle(
        (origin_t[0] - R, origin_t[1]), r, fill=False, ec=None, lw=2
    )
    sphere = mpatches.Circle((origin_s[0], origin_s[1]), k, fill=False, ec=None, lw=2)

    ax.add_patch(torus_R)
    ax.add_patch(torus_L)
    ax.add_patch(sphere)

    # --- overlap fills ---
    alpha = 0.35

    # right overlap: (sphere ∩ torus_R)
    over_R = mpatches.Circle(
        (origin_s[0], origin_s[1]), k, fc="gray", ec="none", alpha=alpha
    )
    over_R.set_clip_path(torus_R)  # clip the sphere by the torus circle
    ax.add_patch(over_R)

    # left overlap: (sphere ∩ torus_L)
    over_L = mpatches.Circle(
        (origin_s[0], origin_s[1]), k, fc="gray", ec="none", alpha=alpha
    )
    over_L.set_clip_path(torus_L)
    ax.add_patch(over_L)


def plot_2d(
    k: float,
    r: float,
    R: float,
    origin_s: list = [0, 0],
    origin_t: list = [0, 0],
    samples: list = None,
    mc_in_mask: list = None,
    box: list = None,
    save_path: str = "",
    show: bool = False,
) -> plt.Axes:
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

    # fill intersection
    # _fill_between2D(ax, r, R, k, origin_s, origin_t)

    # print box if available
    if box is not None:
        # box
        rect = mpatches.Rectangle(
            (box[0][0], box[1][0]),
            box[0][1] - box[0][0],
            box[1][1] - box[1][0],
            fill=False,
            ls="-.",
            color="darkred",
            lw=2,
            label="sampling box",
        )
        ax.add_patch(rect)

    # mc samples
    if samples is not None:
        if mc_in_mask is not None:
            # mask in samples
            in_samples = np.array(
                [
                    samples[i]
                    for i in range(len(samples))
                    if mc_in_mask[i][0] and mc_in_mask[i][1]
                ]
            )
            out_samples = np.array(
                [
                    samples[i]
                    for i in range(len(samples))
                    if not (mc_in_mask[i][0] and mc_in_mask[i][1])
                ]
            )

            # plot out samples
            plt.scatter(
                out_samples[:, 0],
                out_samples[:, 1],
                color="darkred",
                marker=".",
                s=0.2,
                alpha=0.3,
            )

            # plot in samples
            plt.scatter(
                in_samples[:, 0],
                in_samples[:, 1],
                color="darkblue",
                marker=".",
                s=0.2,
                alpha=0.3,
            )
        else:
            # plot out samples
            samples = np.array(samples)
            plt.scatter(
                samples[:, 0],
                samples[:, 1],
                color="darkred",
                marker=".",
                s=0.2,
                alpha=0.3,
            )

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
    if np.sum(origin_t) != 0:
        ax.set_title("2D cross-section of Sphere and Torus (off-center)")
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
    r = 0.5  # torus minor radius
    R = 0.5  # torus major radius
    n = 100_000  # Monte Carlo samples

    sequence = deterministic_sequence(seed_deterministic, n)
    print(sequence)
    # --- Monte Carlo volume estimation ---
    estimated_volume = monte_carlo_3d(k, r, R, n)
    print(f"Estimated intersection volume (sphere ∩ torus): {estimated_volume:.4f}")
