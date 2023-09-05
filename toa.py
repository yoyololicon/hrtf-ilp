import networkx as nx
import numpy as np
from numba import njit, prange
import numba as nb
from soxr import resample
from scipy.spatial import ConvexHull, Delaunay
from scipy import sparse as sp
from typing import Tuple, Iterable

from rigid import hrtf_toa
from graph import stereographic_projection
from linprog import solve_linprog
from utils import has_hole_at_the_bottom


def simplices2edges(simplices: Iterable[Iterable[int]]) -> np.ndarray:
    edges = set()
    for simplex in simplices:
        u = simplex[-1]
        for v in simplex:
            e = (u, v) if u < v else (v, u)
            edges.add(e)
            u = v
    return np.array(list(edges))


@njit(
    nb.types.Tuple((nb.int64[:], nb.float64[:]))(
        nb.float64[:, :], nb.float64[:, :], nb.int64
    ),
    parallel=True,
)
def get_max_cross_correlation_index(
    hrir_s: np.ndarray, hrir_t: np.ndarray, tol: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        hrir_s: (N, T)
        hrir_t: (N, T)
        tol: int
    """
    N, T = hrir_s.shape
    if tol is None:
        tol = T // 2

    corr = np.zeros((N, 2 * tol + 1))
    for i in prange(2 * tol + 1):
        for j in prange(N):
            x = hrir_s[j, max(0, tol - i) : min(T, tol - i + T)]
            y = hrir_t[j, max(0, i - tol) : min(T, i - tol + T)]
            corr[j, i] = x @ y

    argmax = np.empty(N, dtype=np.int64)
    max_corr = np.empty(N, dtype=np.float64)
    for i in prange(N):
        argmax[i] = np.argmax(corr[i])
        max_corr[i] = corr[i, argmax[i]]
    return (argmax - tol), max_corr


def smooth_toa_l2_core(
    edges: np.ndarray,
    differences: np.ndarray,
    weights: np.ndarray,
    naive_toa: np.ndarray = None,
    lda: float = 1.0,
) -> np.ndarray:
    assert np.all(edges >= 0), "negative edge index detected"
    assert np.all(edges[:, 0] < edges[:, 1]), "edge index is not sorted"

    if naive_toa is not None:
        N = naive_toa.shape[0]
    else:
        N = np.max(edges) + 1

    Gamma_triu = sp.csr_matrix(
        (differences, (edges[:, 0], edges[:, 1])),
        shape=(N, N),
    )
    Gamma = -Gamma_triu + Gamma_triu.T

    W_triu = sp.csr_matrix(
        (weights, (edges[:, 0], edges[:, 1])),
        shape=(N, N),
    )
    W = W_triu + W_triu.T

    A = sp.diags(W.sum(1).A1) - W
    if naive_toa is not None:
        A = A + sp.eye(N) * lda
        B = (W @ Gamma).diagonal() - naive_toa * lda
        solver = sp.linalg.spsolve
    else:
        B = (W @ Gamma).diagonal() + lda

        rows, cols, vals = sp.find(A)
        vals = np.concatenate((vals, np.ones(N)))
        rows = np.concatenate((rows, np.full(N, N)))
        cols = np.concatenate((cols, np.arange(N)))
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N + 1, N))
        B = np.concatenate((B, np.array([0])), axis=0)

        solver = lambda A, B: sp.linalg.lsqr(A, B)[0]

    toa = solver(A, -B)

    print(f"Sparseness: {len(sp.find(A)[0]) / (N * N)}")
    return toa


def angle2weight(angle: np.ndarray, std: float = 8) -> np.ndarray:
    return np.exp(-angle / std)


def dot2angle(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    dot = np.clip(np.sum(vec1 * vec2, axis=-1), -1, 1)
    return np.arccos(dot) / np.pi * 180


def dot2weight(vec1: np.ndarray, vec2: np.ndarray, std: float = 8) -> np.ndarray:
    return angle2weight(dot2angle(vec1, vec2), std)


def smooth_toa(
    hrir: np.ndarray,
    xyz: np.ndarray,
    sr: int,
    method: str = "ilp",
    stereo_proj: bool = False,
    oversampling: int = 1,
    max_grid_ms: float = 0.05,
    max_cross_ms: float = 1.0,
    ignore_toa: bool = False,
    weighted: bool = False,
    weighting_method: str = "angle",
    toa_weight: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    if oversampling > 1:
        hrir = resample(
            hrir.reshape(-1, hrir.shape[-1]).T, sr, sr * oversampling
        ).T.reshape(hrir.shape[0], hrir.shape[1], -1)
        sr = sr * oversampling

    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    hrir = hrir / np.linalg.norm(hrir, axis=2, keepdims=True)

    N, _, T = hrir.shape

    stereo_proj |= has_hole_at_the_bottom(xyz=xyz)

    if stereo_proj:
        hull = Delaunay(stereographic_projection(-xyz))
        hull_simplices = hull.simplices

        # add the simplice at the bottom, which is represented as -1 in the neighbor simplices
        mask = hull.neighbors == -1
        simplex_edges = np.stack(
            (hull_simplices, np.roll(hull_simplices, -1, 1)), axis=2
        )[np.roll(mask, 1, 1)]
        G = nx.Graph(simplex_edges.tolist())
        cycles = nx.cycle_basis(G)
        assert len(cycles) == 1, "more than one cycle detected"
        bottom_simplex = cycles[0]
        assert len(bottom_simplex) == len(G.nodes), "bottom simplex is not complete"
        print("Size of the bottom simplex:", len(bottom_simplex))

        hull_simplices = hull_simplices.tolist()
        hull_simplices.append(bottom_simplex)
    else:
        hull = ConvexHull(xyz)
        hull_simplices = hull.simplices.tolist()

    sphere_edges = simplices2edges(hull_simplices)

    if verbose:
        print(f"Number of sphere simplices: {len(hull_simplices)}")
        print(f"Number of sphere edges: {len(sphere_edges)}")

    naive_toa, naive_toa_max_corr = hrtf_toa(hrir)
    if verbose:
        print(
            f"TOA correlation: max={naive_toa_max_corr.max()}, min={naive_toa_max_corr.min()}"
        )
        print(
            f"TOA delay: max={naive_toa.max(0) / sr * 1000} ms, min={naive_toa.min(0) / sr * 1000} ms"
        )

    # compute cross correlation
    max_cross_samples = int(max_cross_ms * sr / 1000)
    cross_diff, cross_diff_max_corr = get_max_cross_correlation_index(
        hrir[:, 0],
        hrir[:, 1],
        tol=max_cross_samples,
    )
    assert (
        np.count_nonzero(np.abs(cross_diff) == max_cross_samples) < 2
    ), "more than one maximum correlation at the edge of the window, consider increasing 'max_cross_ms'"
    if verbose:
        print(
            f"Cross correlation: max={cross_diff_max_corr.max()}, min={cross_diff_max_corr.min()}"
        )
        print(
            f"Cross delay: max={cross_diff.max() / sr * 1000} ms, min={cross_diff.min() / sr * 1000} ms"
        )

    # compute grid correlation
    max_grid_samples = int(max_grid_ms * sr / 1000)
    left_grid_diff, left_grid_diff_max_corr = get_max_cross_correlation_index(
        hrir[:, 0][sphere_edges[:, 0]],
        hrir[:, 0][sphere_edges[:, 1]],
        tol=max_grid_samples,
    )
    assert (
        np.count_nonzero(np.abs(left_grid_diff) == max_grid_samples) < 2
    ), "more than one maximum correlation at the edge of the window, consider increasing 'max_grid_ms'"
    if verbose:
        print(
            f"Left grid correlation: max={left_grid_diff_max_corr.max()}, min={left_grid_diff_max_corr.min()}"
        )
        print(
            f"Left grid delay: max={left_grid_diff.max() / sr * 1000} ms, min={left_grid_diff.min() / sr * 1000} ms"
        )

    right_grid_diff, right_grid_diff_max_corr = get_max_cross_correlation_index(
        hrir[:, 1][sphere_edges[:, 0]],
        hrir[:, 1][sphere_edges[:, 1]],
        tol=max_grid_samples,
    )
    assert (
        np.count_nonzero(np.abs(right_grid_diff) == max_grid_samples) < 2
    ), "more than one maximum correlation at the edge of the window, consider increasing 'max_grid_ms'"
    if verbose:
        print(
            f"Right grid correlation: max={right_grid_diff_max_corr.max()}, min={right_grid_diff_max_corr.min()}"
        )
        print(
            f"Right grid delay: max={right_grid_diff.max() / sr * 1000} ms, min={right_grid_diff.min() / sr * 1000} ms"
        )

    if weighted:
        if weighting_method == "angle":
            cross_weights = dot2weight(xyz, xyz * np.array([1, -1, 1]))
            left_grid_weights = dot2weight(
                xyz[sphere_edges[:, 0]], xyz[sphere_edges[:, 1]]
            )
            right_grid_weights = left_grid_weights
            toa_weights = np.full(N * 2, toa_weight)
        elif weighting_method == "dot":
            cross_weights = cross_diff_max_corr
            left_grid_weights = left_grid_diff_max_corr
            right_grid_weights = right_grid_diff_max_corr
            toa_weights = naive_toa_max_corr.T.flatten()
        else:
            raise ValueError(f"Unknown weighting method: {weighting_method}")
    else:
        cross_weights = np.ones_like(cross_diff_max_corr)
        left_grid_weights = np.ones_like(left_grid_diff_max_corr)
        right_grid_weights = np.ones_like(right_grid_diff_max_corr)
        toa_weights = np.ones(N * 2)

    edges = np.concatenate(
        (
            np.array([[i, i + N] for i in range(N)]),
            sphere_edges,
            sphere_edges + N,
        ),
        axis=0,
    )
    differences = np.concatenate((cross_diff, left_grid_diff, right_grid_diff))
    simplices = (
        hull_simplices
        + [[x + N for x in simplex] for simplex in hull_simplices]
        + np.concatenate((np.flip(sphere_edges, 1), sphere_edges + N), axis=1).tolist()
    )
    weights = np.concatenate((cross_weights, left_grid_weights, right_grid_weights))

    if method == "ilp":
        if not ignore_toa:
            simplices.extend([[-1, u, v] for u, v in sphere_edges])
            simplices.extend([[-1, u, v] for u, v in sphere_edges + N])
            simplices.extend([[-1, i, i + N] for i in range(N)])

            differences = np.concatenate(
                (
                    naive_toa[:, 0],
                    naive_toa[:, 1],
                    differences,
                )
            )
            edges = np.concatenate(
                (
                    np.array([[-1, i] for i in range(2 * N)]),
                    edges,
                ),
                axis=0,
            )
            weights = np.concatenate(
                (
                    toa_weights,
                    weights,
                )
            )

        print(f"Number of simplices: {len(simplices)}")
        print(f"Number of edges: {len(edges)}")

        k = solve_linprog(
            edges, simplices, differences, c=weights if weighted else None
        )

        finer_G = nx.DiGraph()
        for i in range(edges.shape[0]):
            u, v = edges[i]
            w = k[i] + differences[i]
            finer_G.add_edge(u, v, weight=w)
            finer_G.add_edge(v, u, weight=-w)

        result = np.zeros(N * 2)
        root = 0 if ignore_toa else -1
        for u, v in nx.dfs_edges(finer_G, root):
            result[v] = result[u] + finer_G[u][v]["weight"]

        toa = result.reshape((2, N)).T

    elif method == "l2":
        toa = (
            smooth_toa_l2_core(
                edges,
                differences,
                weights,
                naive_toa=None if ignore_toa else naive_toa.T.flatten(),
                lda=toa_weight,
            )
            .reshape((2, N))
            .T
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    if ignore_toa:
        toa = toa - toa.mean() + naive_toa.mean()

    if oversampling > 1:
        toa /= oversampling

    return toa
