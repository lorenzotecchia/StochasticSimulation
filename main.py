import numpy as np
import random
import matplotlib.pyplot as plt

BBOX_STANDARD = []
BBOX_SAMPLE = []


def torus(
    x: float, y: float, z: float, r: float, R: float, origin: list = [0, 0, 0]
) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the torus defined by the
    major radius R and the minor radius r.
    """
    result = ((np.sqrt((x - origin[0])^2 + (y - origin[1])^2) - R)^2 + (z-origin[2])^2 - r^2 <= 0)
    return result

def sphere(x: float, y: float, z: float, k: float) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the sphere defined by the
    radius k.
    """
    result = (x^2 + y^2 + z^2 - k^2 <= 0)
    return result

def torus2D(
    x: float, y: float, r: float, R: float, origin: list = [0, 0]
) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the torus defined by the
    major radius R and the minor radius r.
    """
    result = ((x - origin[0] - R)**2 + (y - origin[1])**2 - r**2 <= 0)
    return result

def sphere2D(x: float, y: float, k: float) -> bool:
    """
    Returns true if the given points (x, y, z) is inside the sphere defined by the
    radius k.
    """
    result = (x**2 + y**2 - k**2 <= 0)
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

    total = total + (sphere2D(x, y, k) and torus2D(x, y, r, R))

    return total

def monte_carlo_2d(k: float, r: float, R: float, n: int = 100_000) -> float:
    """
    Returns fraction of surface area of intersection between sphere and torus
    """
    total = 0
    box = box_standard(k, r, R)
    for i in range(n):
        total = one_sample(k, r, R, box, total)


    return total * box_area(box) / n

def mixed_sampling(k: float, r: float, R: float, p: float, n: int = 100_000, origin = [0, 0]) -> float:
    """ """
    #first box
    total1 = 0
    box1 = box_standard(k, r, R)

    #second box    
    total2 = 0
    box2 = box_sample(k, r, R, origin = origin)

    for i in range(n):
        rnd = random.random()
        if rnd <= p:
            total1 = one_sample(k, r, R, box1, total1)

        else:
            total2 = one_sample(k, r, R, box2, total2)


    surface_area1 = box_area(box1)
    surface_area2 = box_area(box2)

    return p*surface_area1 + (1-p)*surface_area2


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


def plot_intersection():
    """
    Plots the intersection between the sphere and the torus.
    """
    pass


if __name__ == "__main__":
    print("Bombaclat!")

    #case a:
    k = 1
    R = 0.75
    r = 0.4

    area = monte_carlo_2d(k, r, R, n = 10000)
    volume = surface_to_volume(R, area)
    volume_toro = 2 * np.pi * R * np.pi * r ** 2

    print(f"areozza = {area} e volumozzo = {volume}, invece che {volume_toro}")

    area = mixed_sampling(k, r, R, p =0.5, n = 10000, origin = [0, 0.1])
    volume = surface_to_volume(R, area)

    print(f"areozza = {area} e volumozzo = {volume}, invece che {volume_toro}")

