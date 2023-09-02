import networkx as nx
import numpy as np
from numba import njit, prange
from soxr import resample
from scipy.spatial import ConvexHull, Delaunay

from rigid import hrtf_toa
from graph import stereographic_projection, points2graph


@njit(parallel=True)
def get_max_cross_correlation_index(
    hrir_s: np.ndarray, hrir_t: np.ndarray, tol: int = None
):
    """
    Args:
        hrir_s: (N, T)
        hrir_t: (N, T)
        tol: int
    """
    N, T = hrir_s.shape
    if tol is None:
        tol = T // 2

    # hrir_t = np.pad(hrir_t, ((0, 0), (tol,) * 2), mode='constant')
    corr = np.zeros((N, 2 * tol + 1))
    for i in prange(2 * tol + 1):
        for j in prange(N):
            x = hrir_s[j, max(0, tol - i) : min(T, tol - i + T)]
            y = hrir_t[j, max(0, i - tol) : min(T, i - tol + T)]
            corr[j, i] = np.sum(x * y)
    return np.argmax(corr, 1) - tol


def smooth_toa(
    hrir: np.ndarray,
    xyz: np.ndarray,
    sr: int,
    stereo_proj: bool = False,
    oversampling: int = 1,
    max_offset_ms: float = 0.05,
    verbose: bool = True,
) -> dict:
    if oversampling > 1:
        hrir = resample(
            hrir.reshape(-1, hrir.shape[-1]).T, sr, sr * oversampling
        ).T.reshape(hrir.shape[0], hrir.shape[1], -1)
        sr = sr * oversampling

    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

    num_hrir, _, T = hrir.shape
    naive_toa = hrtf_toa(hrir)
    max_offset_samples = int(max_offset_ms * sr / 1000)

    if stereo_proj:
        hull = Delaunay(stereographic_projection(xyz))
    else:
        hull = ConvexHull(xyz)

    simplices = hull.simplices
    simplices_neighbors = hull.neighbors
    num_simplices = simplices.shape[0]

    if not stereo_proj:
        # make sure the simplices are oriented counter-clockwise
        vec1 = xyz[simplices[:, 1]] - xyz[simplices[:, 0]]
        vec2 = xyz[simplices[:, 2]] - xyz[simplices[:, 0]]
        norm_vec = np.cross(vec1, vec2)
        mask = np.sum(norm_vec * xyz[simplices[:, 0]], axis=-1) >= 0
        simplices[mask] = simplices[mask][:, ::-1]
        simplices_neighbors[mask] = simplices_neighbors[mask][:, ::-1]

    edges = np.stack((np.roll(simplices, 1, 1), simplices), axis=2).reshape(-1, 2)
    simplices_edges = np.stack(
        (
            np.broadcast_to(
                np.arange(num_simplices)[:, None], simplices_neighbors.shape
            ),
            np.roll(simplices_neighbors, -1, 1),
        ),
        axis=2,
    ).reshape(-1, 2)

    swap_mask = edges[:, 0] > edges[:, 1]
    sorted_edges = np.where(swap_mask[:, None], np.flip(edges, 1), edges)
    unique_edges, unique_edges_index, inverse_edges_index = np.unique(
        sorted_edges, axis=0, return_index=True, return_inverse=True
    )
    unique_simplices_edges = simplices_edges[unique_edges_index]

    if verbose:
        print(f"Number of edges: {unique_edges.shape[0]}")

    # arccos = (
    #     np.arccos(np.sum(xyz[unique_edges[:, 0]] * xyz[unique_edges[:, 1]], axis=-1))
    #     / np.pi
    #     * 180
    # )
    # weights = np.exp(-arccos / 8)
    # print(f"Minumum weight: {weights.min()}, maximum weight: {weights.max()}")
    # weights = np.round(weights * 100).astype(int)

    result = np.empty((num_hrir, 2))
    traverse_path = list(
        nx.algorithms.traversal.breadth_first_search.bfs_edges(
            nx.DiGraph(edges.tolist()), 0
        )
    )
    for i in range(2):
        normalised_hrir = hrir[:, i] / np.linalg.norm(hrir[:, i], axis=1, keepdims=True)
        corr_offsets = get_max_cross_correlation_index(
            normalised_hrir[unique_edges[:, 0]],
            normalised_hrir[unique_edges[:, 1]],
            tol=max_offset_samples,
        )
        assert (
            np.count_nonzero(np.abs(corr_offsets) == max_offset_samples) < 2
        ), "more than one edge has a maximum correlation at the edge of the window, consider increasing the window size"

        corr_offsets = corr_offsets[inverse_edges_index]
        corr_offsets[swap_mask] *= -1

        demands = -corr_offsets.reshape(-1, 3).sum(1)
        if not stereo_proj and demands.sum() != 0:
            raise ValueError("demands do not sum to zero")

        G = nx.Graph()
        G.add_nodes_from(zip(range(num_simplices), [{"demand": d} for d in demands]))
        if stereo_proj:
            G.add_node(-1, demand=-demands.sum())

        weights = np.ones(unique_edges.shape[0], dtype=int)
        weights[(demands[unique_simplices_edges[:, 0]] == 0) & (demands[unique_simplices_edges[:, 1]] == 0)] = 10
        # weights = np.ones(unique_simplices_edges.shape[0])
        G.add_weighted_edges_from(
            zip(unique_simplices_edges[:, 0], unique_simplices_edges[:, 1], weights)
        )
        cost, flowdict = nx.network_simplex(G.to_directed())
        if verbose:
            print(f"{'Left' if i == 0 else 'Right'} ear cost: {cost}")

        diff = corr_offsets.copy()
        # add the flow to their orthogonal edge
        for j, (u, v) in enumerate(simplices_edges):
            diff[j] += -flowdict[u][v] + flowdict[v][u]

        # construct temporary dict that hold all edge gradients from different direction
        diff_dict = {j: {} for j in range(hrir.shape[0])}
        for d, (u, v) in zip(diff, edges):
            diff_dict[u][v] = d

            # integrate the gradients
        for u, v in traverse_path:
            result[v, i] = result[u, i] + diff_dict[u][v]

    toa = result - result.mean(0) + naive_toa.mean(0)
    if oversampling > 1:
        toa /= oversampling

    return toa
