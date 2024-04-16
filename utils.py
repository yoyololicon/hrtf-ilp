import numpy as np
from spaudiopy.utils import cart2sph
from spaudiopy.sph import sh_matrix


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


def sht_lstsq_reg(f, N_sph, azi, colat, sh_type, Y_nm=None, eps=1):
    if f.ndim == 1:
        f = f[:, np.newaxis]  # upgrade to handle 1D arrays
    if Y_nm is None:
        Y_nm = sh_matrix(N_sph, azi, colat, sh_type)
    reg = sum(([1 + n * (n + 1)] * (2 * n + 1) for n in range(N_sph + 1)), start=[])
    return np.linalg.solve(Y_nm.T.conj() @ Y_nm + eps * np.diag(reg), Y_nm.T.conj() @ f)


def lsd(pred, target):
    pred = np.abs(pred) + 1e-16
    target = np.abs(target) + 1e-16
    return 20 * np.abs(np.log10(pred) - np.log10(target))
