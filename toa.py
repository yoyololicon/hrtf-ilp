import networkx as nx
import numpy as np
from numba import njit, prange
import numba as nb
from soxr import resample
from scipy.spatial import ConvexHull, Delaunay
from typing import Tuple, Iterable

from rigid import hrtf_toa
from graph import stereographic_projection, points2graph
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


def smooth_toa(
    hrir: np.ndarray,
    xyz: np.ndarray,
    sr: int,
    stereo_proj: bool = False,
    oversampling: int = 1,
    max_grid_ms: float = 0.05,
    max_cross_ms: float = 1.0,
    ignore_toa: bool = False,
    weighted: bool = False,
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
        simplices = hull.simplices

        # add the simplice at the bottom, which is represented as -1 in the neighbor simplices
        mask = hull.neighbors == -1
        simplex_edges = np.stack((simplices, np.roll(simplices, -1, 1)), axis=2)[
            np.roll(mask, 1, 1)
        ]
        G = nx.Graph(simplex_edges.tolist())
        cycles = nx.cycle_basis(G)
        assert len(cycles) == 1, "more than one cycle detected"
        bottom_simplex = cycles[0]
        assert len(bottom_simplex) == len(G.nodes), "bottom simplex is not complete"
        print("Size of the bottom simplex:", len(bottom_simplex))

        simplices = simplices.tolist()
        simplices.append(bottom_simplex)
    else:
        hull = ConvexHull(xyz)
        simplices = hull.simplices.tolist()

    sphere_edges = simplices2edges(simplices)

    if verbose:
        print(f"Number of sphere simplices: {len(simplices)}")
        print(f"Number of sphere edges: {len(sphere_edges)}")

    naive_toa, naive_toa_max_corr = hrtf_toa(hrir)
    if verbose:
        print(f"Maximum correlation: {naive_toa_max_corr.max(0)}")
        print(f"Minimum correlation: {naive_toa_max_corr.min(0)}")
        print(f"Maximum delay: {naive_toa.max(0) / sr * 1000} ms")

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
            f"Maximum cross correlation: {cross_diff_max_corr.max()}, minimum cross correlation: {cross_diff_max_corr.min()}"
        )
        print(
            f"Maximum cross delay: {cross_diff.max() / sr * 1000} ms, minimum cross delay: {cross_diff.min() / sr * 1000} ms"
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
            f"Maximum grid correlation: {left_grid_diff_max_corr.max()}, minimum grid correlation: {left_grid_diff_max_corr.min()}"
        )
        print(
            f"Maximum grid delay: {left_grid_diff.max() / sr * 1000} ms, minimum grid delay: {left_grid_diff.min() / sr * 1000} ms"
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
            f"Maximum grid correlation: {right_grid_diff_max_corr.max()}, minimum grid correlation: {right_grid_diff_max_corr.min()}"
        )
        print(
            f"Maximum grid delay: {right_grid_diff.max() / sr * 1000} ms, minimum grid delay: {right_grid_diff.min() / sr * 1000} ms"
        )

    # arccos = (
    #     np.arccos(np.sum(xyz[unique_edges[:, 0]] * xyz[unique_edges[:, 1]], axis=-1))
    #     / np.pi
    #     * 180
    # )
    # weights = np.exp(-arccos / 8)
    # print(f"Minumum weight: {weights.min()}, maximum weight: {weights.max()}")
    # weights = np.round(weights * 100).astype(int)

    if ignore_toa:
        simplices = (
            simplices
            + [[x + N for x in simplex] for simplex in simplices]
            + np.concatenate(
                (np.flip(sphere_edges, 1), sphere_edges + N), axis=1
            ).tolist()
        )
        differences = np.concatenate((cross_diff, left_grid_diff, right_grid_diff))
        edges = np.concatenate(
            (
                np.array([[i, i + N] for i in range(N)]),
                sphere_edges,
                sphere_edges + N,
            ),
            axis=0,
        )
        weights = np.concatenate(
            (cross_diff_max_corr, left_grid_diff_max_corr, right_grid_diff_max_corr)
        )

    else:
        simplices = (
            simplices
            + [[x + N for x in simplex] for simplex in simplices]
            + [[-1, u, v] for u, v in sphere_edges]
            + [[-1, u, v] for u, v in sphere_edges + N]
            + [[-1, i, i + N] for i in range(N)]
            + np.concatenate(
                (np.flip(sphere_edges, 1), sphere_edges + N), axis=1
            ).tolist()
        )

        differences = np.concatenate(
            (
                naive_toa[:, 0],
                naive_toa[:, 1],
                cross_diff,
                left_grid_diff,
                right_grid_diff,
            )
        )
        edges = np.concatenate(
            (
                np.array([[-1, i] for i in range(2 * N)]),
                np.array([[i, i + N] for i in range(N)]),
                sphere_edges,
                sphere_edges + N,
            ),
            axis=0,
        )
        weights = np.concatenate(
            (
                naive_toa_max_corr[:, 0],
                naive_toa_max_corr[:, 1],
                cross_diff_max_corr,
                left_grid_diff_max_corr,
                right_grid_diff_max_corr,
            )
        )

    print(f"Number of simplices: {len(simplices)}")
    print(f"Number of edges: {len(edges)}")

    k = solve_linprog(edges, simplices, differences, c=weights if weighted else None)

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

    if ignore_toa:
        toa = toa - toa.mean(0) + naive_toa.mean(0)

    if oversampling > 1:
        toa /= oversampling

    return toa
