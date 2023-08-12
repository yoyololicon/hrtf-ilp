import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import networkx as nx
from typing import Callable, Tuple, Union, List, Dict, Any, Optional

__all__ = ["points2graph", "plus_freq_dim", "stereographic_projection"]


def stereographic_projection(points: np.ndarray):
    """Projects points on a unit sphere to a flat plane using stereographic projection.

    Parameters
    ----------
    points : ndarray
        N x 3 array of points.

    Returns
    -------
    points_proj : ndarray
        N x 2 array of projected points.
    """
    assert points.ndim == 2
    assert points.shape[1] == 3
    N, _ = points.shape

    # project points to a plane
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    points_proj = np.zeros((N, 2))
    points_proj[:, 0] = x / (1 - z)
    points_proj[:, 1] = y / (1 - z)

    return points_proj


def points2graph(points: np.ndarray, stereo_proj: bool = False):
    """Creates a graph from a set of points.

    Parameters
    ----------
    points : ndarray
        N x 3 array of points.

    Returns
    -------
    G : networkx.Graph
        Graph of points.
    """

    assert points.ndim == 2
    assert points.shape[1] == 3
    N, _ = points.shape

    R = np.linalg.norm(points, axis=1)
    points = points / R[:, None]

    # create graph
    if stereo_proj:
        z = points[:, 2]
        if not np.all(z < 1) and np.all(z > -1):
            points = -points
        else:
            raise ValueError(
                f"z values must be in (-1, 1), but got ({z.min()}, {z.max()})"
            )
        points_proj = stereographic_projection(points)
        hull = Delaunay(points_proj)
    else:
        hull = ConvexHull(points)
    edges = np.vstack(
        (hull.simplices[:, :2], hull.simplices[:, 1:], hull.simplices[:, ::2])
    )
    G = nx.Graph()
    G.add_edges_from(edges)
    G = G.to_undirected()
    return G


def plus_freq_dim(G: nx.Graph, f: int):
    assert not G.is_directed()

    num_nodes = G.number_of_nodes()
    edges = np.array(G.edges)
    extended_edges = np.vstack([edges + i * num_nodes for i in range(f)])
    idxs = np.arange(num_nodes)
    extra_edges = np.vstack(
        [
            np.vstack([idxs + num_nodes * i, idxs + num_nodes * (i + 1)]).T
            for i in range(f - 1)
        ]
    )
    full_edges = np.vstack([extended_edges, extra_edges])
    new_G = nx.Graph()
    new_G.add_edges_from(full_edges)
    new_G = new_G.to_undirected()
    return new_G
