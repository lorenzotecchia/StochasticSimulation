BBOX_STANDARAD = []
BBOX_SAMPLE = []


def torus(
    x: float, y: float, z: float, r: float, R: float, origin: list = [0, 0, 0]
) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the torus defined by the
    major radius R and the minor radius r.
    """

    pass


def sphere(x: float, y: float, z: float, k: float) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the sphere defined by the
    radius k.
    """
    pass


def monte_carlo_2d(n: int = 100_000) -> float:
    """
    Returns fraction of surface area of intersection between sphere and torus
    """

    pass


def mixed_sampling(p: float, n: int = 100_000) -> float:
    """ """

    pass


def surface_to_volume(surface_area: float) -> float:
    """
    Converts surface area to volume
    """

    pass


# neeeded?
def PRNG():
    pass


def deterministic_sequence(seed: float) -> float:
    """
    Generates a deterministic sequence of numbers based on the seed.
    """
    pass


def plot_intersection():
    """
    Plots the intersection between the sphere and the torus.
    """
    pass


if __name__ == "__main__":
    print("Bombaclat!")
