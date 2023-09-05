import numpy as np
from spaudiopy.utils import cart2sph


def has_hole_at_the_bottom(col: np.ndarray = None, xyz: np.ndarray = None) -> bool:
    """
    Args:
        col: (N,)
        xyz: (N, 3)
    """
    if col is None:
        col = cart2sph(xyz[:, 0], xyz[:, 1], xyz[:, 2])[1]
    col = col / np.pi * 180
    return not np.any(col > 178)
