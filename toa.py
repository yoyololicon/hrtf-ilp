import networkx as nx
import numpy as np
from soxr import resample
from scipy.spatial import ConvexHull, Delaunay
from scipy import sparse as sp
from typing import Tuple, Iterable
import time

from rigid import hrtf_toa
from graph import stereographic_projection
from linprog import solve_linprog, solve_quadprog, solve_linprog_ez
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


def smooth_toa_l2_core(
    edges: np.ndarray,
    differences: np.ndarray,
    weights: np.ndarray,
    naive_toa: np.ndarray = None,
    lamb: float = 1.0,
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
    B = (W @ Gamma).diagonal()
    if naive_toa is not None:
        A = A + sp.eye(N) * lamb
        B = B - naive_toa * lamb
        solver = sp.linalg.spsolve
    else:
        B = B + lamb

        rows, cols, vals = sp.find(A)
        vals = np.concatenate((vals, np.ones(N)))
        rows = np.concatenate((rows, np.full(N, N)))
        cols = np.concatenate((cols, np.arange(N)))
        A = sp.csr_matrix((vals, (rows, cols)), shape=(N + 1, N))
        B = np.concatenate((B, np.array([0])), axis=0)

        solver = lambda A, B: sp.linalg.lsqr(A, B)[0]

    matrix_shape = A.shape

    start_time = time.time()
    toa = solver(A, -B)
    elapsed_time = time.time() - start_time

    # print(f"Sparseness: {len(sp.find(A)[0]) / (N * N)}")
    # print(f"Matrix shape: {matrix_shape}")
    # print(f"Elapsed time: {elapsed_time} s")
    # if naive_toa is None:
    #     toa, lambda_ = toa[:-1], toa[-1]
    #     print(f"Lambda: {lambda_}")
    return toa, matrix_shape, elapsed_time


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
    ignore_toa: bool = False,
    ignore_cross: bool = False,
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
        hull = Delaunay(stereographic_projection(-xyz), qhull_options="QJ")
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
        if verbose:
            print("Size of the bottom simplex:", len(bottom_simplex))

        hull_simplices = hull_simplices.tolist()
        hull_simplices.append(bottom_simplex)
    else:
        hull = ConvexHull(xyz)
        hull_simplices = hull.simplices.tolist()

    sphere_edges = simplices2edges(hull_simplices)

    num_unique_nodes = len(np.unique(sphere_edges.ravel()))
    assert num_unique_nodes == N, f"expected {N} nodes, but got {num_unique_nodes}"

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

    hrtf = np.fft.rfft(hrir, n=T * 2, axis=-1)
    cross_corr = np.fft.irfft(hrtf[:, 1] * hrtf[:, 0].conj(), axis=-1)
    cross_diff, cross_diff_max_corr = np.argmax(cross_corr, axis=-1), cross_corr.max(
        axis=-1
    )
    cross_diff = (cross_diff + T) % (2 * T) - T

    if verbose:
        print(
            f"Cross correlation: max={cross_diff_max_corr.max()}, min={cross_diff_max_corr.min()}"
        )
        print(
            f"Cross delay: max={cross_diff.max() / sr * 1000} ms, min={cross_diff.min() / sr * 1000} ms"
        )

    # compute grid correlation
    left_corr = np.fft.irfft(
        hrtf[sphere_edges[:, 1], 0] * hrtf.conj()[sphere_edges[:, 0], 0], axis=-1
    )
    left_grid_diff, left_grid_diff_max_corr = np.argmax(
        left_corr, axis=-1
    ), left_corr.max(axis=-1)
    left_grid_diff = (left_grid_diff + T) % (2 * T) - T

    if verbose:
        print(
            f"Left grid correlation: max={left_grid_diff_max_corr.max()}, min={left_grid_diff_max_corr.min()}"
        )
        print(
            f"Left grid delay: max={left_grid_diff.max() / sr * 1000} ms, min={left_grid_diff.min() / sr * 1000} ms"
        )

    right_corr = np.fft.irfft(
        hrtf[sphere_edges[:, 1], 1] * hrtf.conj()[sphere_edges[:, 0], 1], axis=-1
    )
    right_grid_diff, right_grid_diff_max_corr = np.argmax(
        right_corr, axis=-1
    ), right_corr.max(axis=-1)
    right_grid_diff = (right_grid_diff + T) % (2 * T) - T

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

    num_nodes = 2 * N + (0 if ignore_toa else 1)

    if ignore_cross:
        toa, *_ = _lr_separate_toa(
            sphere_edges,
            hull_simplices,
            left_grid_diff,
            right_grid_diff,
            left_grid_weights,
            right_grid_weights,
            naive_toa=None if ignore_toa else naive_toa,
            toa_weights=None if ignore_toa else toa_weights,
            method=method,
            lamb=toa_weight,
        )
        if oversampling > 1:
            toa = toa / oversampling

        num_edges = (sphere_edges.shape[0] + (0 if naive_toa is None else N)) * 2
        return [toa] + _ + [num_edges, num_nodes]

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

    if method == "ilp" or method == "qlp":
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

        if verbose:
            print(f"Number of simplices: {len(simplices)}")
            print(f"Number of edges: {len(edges)}")

        start_time = time.time()
        if method == "qlp":
            k = solve_quadprog(
                edges, simplices, differences, c=weights if weighted else None
            )
        else:
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
        t = time.time() - start_time
        matrix_shape = (len(simplices), len(edges))
        num_edges = len(edges)

    elif method == "edgelist":
        start_time = time.time()

        m = solve_linprog_ez(
            edges,
            differences,
            weights=weights if weighted else None,
            toa=naive_toa.T.flatten() if not ignore_toa else None,
            toa_weights=toa_weights if not ignore_toa else None,
        )

        toa = m.reshape((2, N)).T
        t = time.time() - start_time
        matrix_shape = (
            (len(edges), len(edges) + N * 2)
            if ignore_toa
            else (len(edges) + N * 2, len(edges) + N * 4)
        )
        num_edges = len(edges) + (0 if ignore_toa else N * 2)

    elif method == "l2":
        toa, matrix_shape, t = smooth_toa_l2_core(
            edges,
            differences,
            weights,
            naive_toa=None if ignore_toa else naive_toa.T.flatten(),
            lamb=toa_weight,
        )
        toa = toa.reshape((2, N)).T
        num_edges = len(edges) + (0 if ignore_toa else N * 2)
    else:
        raise ValueError(f"Unknown method: {method}")

    if ignore_toa:
        toa = toa - toa.mean()

    if oversampling > 1:
        toa = toa / oversampling

    return toa, matrix_shape, t, num_edges, num_nodes


def _lr_separate_toa(
    sphere_edges,
    sphere_simplices,
    left_grid_diff,
    right_grid_diff,
    left_grid_weights,
    right_grid_weights,
    naive_toa=None,
    toa_weights=None,
    method="ilp",
    lamb=0.1,
):
    N = np.max(sphere_edges) + 1

    if method == "ilp" or method == "qlp":
        if naive_toa is not None:
            edges = np.concatenate(
                (
                    np.array([[-1, i] for i in range(N)]),
                    sphere_edges,
                ),
                axis=0,
            )
            simplices = sphere_simplices + [[-1, u, v] for u, v in sphere_edges]
            left_weights = np.concatenate(
                (
                    toa_weights[:N],
                    left_grid_weights,
                )
            )
            left_diff = np.concatenate(
                (
                    naive_toa[:, 0],
                    left_grid_diff,
                )
            )
            right_weights = np.concatenate(
                (
                    toa_weights[N:],
                    right_grid_weights,
                )
            )
            right_diff = np.concatenate(
                (
                    naive_toa[:, 1],
                    right_grid_diff,
                )
            )
        else:
            edges = sphere_edges
            simplices = sphere_simplices
            left_weights = left_grid_weights
            right_weights = right_grid_weights
            left_diff = left_grid_diff
            right_diff = right_grid_diff

        start_time = time.time()
        if method == "qlp":
            left_k = solve_quadprog(edges, simplices, left_diff, c=left_weights)
            right_k = solve_quadprog(edges, simplices, right_diff, c=right_weights)
        else:
            left_k = solve_linprog(edges, simplices, left_diff, c=left_weights)
            right_k = solve_linprog(edges, simplices, right_diff, c=right_weights)

        finer_G_l = nx.DiGraph()
        finer_G_r = nx.DiGraph()
        for i in range(edges.shape[0]):
            u, v = edges[i]
            left_w = left_k[i] + left_diff[i]
            right_w = right_k[i] + right_diff[i]
            finer_G_l.add_edge(u, v, weight=left_w)
            finer_G_l.add_edge(v, u, weight=-left_w)
            finer_G_r.add_edge(u, v, weight=right_w)
            finer_G_r.add_edge(v, u, weight=-right_w)

        result = np.zeros((2, N))
        root = 0 if naive_toa is None else -1
        for u, v in nx.dfs_edges(finer_G_l, root):
            result[0, v] = result[0, u] + finer_G_l[u][v]["weight"]
            result[1, v] = result[1, u] + finer_G_r[u][v]["weight"]

        toa = result.T
        t = time.time() - start_time
        matrix_shape = (len(simplices), len(edges))

    elif method == "edgelist":
        start_time = time.time()

        left_m = solve_linprog_ez(
            sphere_edges,
            left_grid_diff,
            weights=left_grid_weights,
            toa=naive_toa[:, 0] if naive_toa is not None else None,
            toa_weights=toa_weights[:N] if toa_weights is not None else None,
        )
        right_m = solve_linprog_ez(
            sphere_edges,
            right_grid_diff,
            weights=right_grid_weights,
            toa=naive_toa[:, 1] if naive_toa is not None else None,
            toa_weights=toa_weights[N:] if toa_weights is not None else None,
        )

        toa = np.stack((left_m, right_m), axis=1)
        t = time.time() - start_time
        matrix_shape = (
            (len(sphere_edges), len(sphere_edges) + N)
            if naive_toa is None
            else (len(sphere_edges) + N, len(sphere_edges) + N * 2)
        )

    elif method == "l2":
        left_toa, matrix_shape, lt = smooth_toa_l2_core(
            sphere_edges,
            left_grid_diff,
            left_grid_weights,
            naive_toa=None if naive_toa is None else naive_toa[:, 0],
            lamb=lamb,
        )
        right_toa, matrix_shape, rt = smooth_toa_l2_core(
            sphere_edges,
            right_grid_diff,
            right_grid_weights,
            naive_toa=None if naive_toa is None else naive_toa[:, 1],
            lamb=lamb,
        )
        t = lt + rt
        toa = np.stack((left_toa, right_toa), axis=1)
    else:
        raise ValueError(f"Unknown method: {method}")

    if naive_toa is None:
        toa = toa - toa.mean(0)

    return toa, matrix_shape, t
