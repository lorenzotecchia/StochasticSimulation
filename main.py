import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


BBOX_STANDARD = []
BBOX_SAMPLE = []


def torus(
    x: float, y: float, z: float, r: float, R: float, origin: list = [0, 0, 0]
) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the torus defined by the
    major radius R and the minor radius r.
    """
    result = (np.sqrt((x - origin[0]) ^ 2 + (y - origin[1]) ^ 2) - R) ^ 2 + (
        z - origin[2]
    ) ^ 2 - r ^ 2 <= 0
    return result


def sphere(x: float, y: float, z: float, k: float) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the sphere defined by the
    radius k.
    """
    result = x ^ 2 + y ^ 2 + z ^ 2 - k ^ 2 <= 0
    return result


def torus2D(x: float, y: float, r: float, R: float, origin: list = [0, 0]) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the torus defined by the
    major radius R and the minor radius r.
    """
    result = (x - origin[0] - R) ** 2 + (y - origin[1]) ** 2 - r**2 <= 0
    return result


def sphere2D(x: float, y: float, k: float) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the sphere defined by the
    radius k.
    """
    result = x**2 + y**2 - k**2 <= 0
    return result


def box_standard(k: float, r: float, R: float, origin: list = [0, 0]) -> list:
    dy = r + origin[1]
    dx = R + r + origin[0]

    return [[-dx, dx], [-dy, dy]]


def box_sample(k: float, r: float, R: float, origin: list = [0, 0]) -> list:
    dy = r
    dx = R + r

    return [[-dx + origin[0], dx + origin[0]], [-dy + origin[1], dy + origin[1]]]


def box_area(box: list) -> float:
    return (box[0][1] - box[0][0]) * (box[1][1] - box[1][0])


def one_sample(k: float, r: float, R: float, box: list, total: int) -> int:
    min_x = box[0][0]
    max_x = box[0][1]
    min_y = box[1][0]
    max_y = box[1][1]

    num_x = random.random()
    num_y = random.random()

    x = num_x * (max_x - min_x) + min_x
    y = num_y * (max_y - min_y) + min_y

    x_in = sphere2D(x, y, k)
    y_in = torus2D(x, y, r, R)

    total = total + (x_in and y_in)

    return total, [x, y], [x_in, y_in]


def monte_carlo_2d(k: float, r: float, R: float, n: int = 100_000) -> float:
    """
    Returns fraction of surface area of intersection between sphere and torus
    """
    total = 0
    box = box_standard(k, r, R)
    mc_samples = []
    mc_in_mask = []
    for i in range(n):
        total, (x, y), (x_in, y_in) = one_sample(k, r, R, box, total)
        mc_samples.append([x, y])
        mc_in_mask.append([x_in, y_in])

    return total * box_area(box) / n, mc_samples, mc_in_mask


def mixed_sampling(
    k: float, r: float, R: float, p: float, n: int = 100_000, origin=[0, 0]
) -> float:
    """ """
    # first box
    total1 = 0
    box1 = box_standard(k, r, R)

    # second box
    total2 = 0
    box2 = box_sample(k, r, R, origin=origin)

    for i in range(n):
        rnd = random.random()
        if rnd <= p:
            total1, _, _ = one_sample(k, r, R, box1, total1)

        else:
            total2, _, _ = one_sample(k, r, R, box2, total2)

    surface_area1 = box_area(box1)
    surface_area2 = box_area(box2)

    return p * surface_area1 + (1 - p) * surface_area2


def surface_to_volume(R: float, surface_area: float) -> float:
    """
    Converts surface area to volume
    """
    volume = surface_area * 2 * np.pi * surface_area

    return volume


# neeeded?
def PRNG():
    pass


def deterministic_sequence(seed: float) -> float:
    """
    Generates a deterministic sequence of numbers based on the seed.
    """
    pass


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
    _fill_between2D(ax, r, R, k, origin_s, origin_t)

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
                label="out samples",
            )

            # plot in samples
            plt.scatter(
                in_samples[:, 0],
                in_samples[:, 1],
                color="darkblue",
                marker=".",
                s=0.2,
                alpha=0.3,
                label="in samples",
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
                label="samples",
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
    print("Bombaclat!")

    # case a:
    k = 1
    R = 0.75
    r = 0.4

    area, mc_samples, mc_in_mask = monte_carlo_2d(k, r, R, n=10000)
    volume = surface_to_volume(R, area)
    volume_toro = 2 * np.pi * R * np.pi * r**2

    print(f"areozza = {area} e volumozzo = {volume}, invece che {volume_toro}")

    area = mixed_sampling(k, r, R, p=0.5, n=10000, origin=[0, 0.1])
    volume = surface_to_volume(R, area)

    print(f"areozza = {area} e volumozzo = {volume}, invece che {volume_toro}")

    ax_sample = plot_2d(
        k,
        r,
        R,
        samples=mc_samples,
        mc_in_mask=mc_in_mask,
        box=box_sample(k, r, R, origin=[0, 0]),
        save_path="img/2d.png",
    )
