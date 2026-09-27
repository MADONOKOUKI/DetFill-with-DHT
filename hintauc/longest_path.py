"""Deterministic longest path on a binary skeleton (no FilFinder, no RNG).

The canonical DHT pipeline extracts the longest path of a region skeleton with
FilFinder2D, whose medial-axis step breaks pixel ties with an unseeded random
generator.  This module provides a dependency-free alternative,
``path_method="geodesic"``:

* the skeleton pixels form an 8-connected graph (orthogonal step = 1,
  diagonal step = sqrt(2)); a diagonal step is disallowed when the same
  neighbour is reachable through an orthogonal skeleton pixel (no corner
  cutting), so the path follows the traced skeleton through its corners;
* the longest path is the *geodesic diameter* of that graph -- the longest
  among all shortest paths between two skeleton pixels;
* every tie is broken by raster order (row-major pixel index), so the result
  depends only on the input skeleton.

The diameter is exact for every skeleton: for a tree-shaped component (the
common case after Zhang--Suen thinning) the ends of the diameter are
endpoints (degree-1 pixels), so only those are used as start points; for a
component that contains a cycle every pixel is a start point (a diameter can
end on the cycle, away from any endpoint).  Complexity is O(S * V log V) for S
start points and V skeleton pixels of a component, which is negligible at the
64x64 hint resolution.

Differences from the FilFinder path (kept deliberately simple; documented in
the README): no 3x3 dilation / medial-axis re-skeletonisation, and no pruning
of branches shorter than 3 px.  The returned path is a subset of the input
skeleton, so it always lies inside the region.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["geodesic_longest_path", "longest_path_pixels", "PathResult"]

_SQRT2 = float(np.sqrt(2.0))
# 8-neighbourhood offsets in a fixed (deterministic) order
_NEIGH = ((-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2),
          (0, -1, 1.0), (0, 1, 1.0),
          (1, -1, _SQRT2), (1, 0, 1.0), (1, 1, _SQRT2))


class PathResult(tuple):
    """(mask, length, start, end) with attribute access."""
    __slots__ = ()

    def __new__(cls, mask, length, start, end):
        return tuple.__new__(cls, (mask, length, start, end))

    mask = property(lambda self: self[0])      #: bool (H,W) path pixels
    length = property(lambda self: self[1])    #: geodesic length in pixels
    start = property(lambda self: self[2])     #: (row, col) or None
    end = property(lambda self: self[3])       #: (row, col) or None


def _build_graph(skel: np.ndarray):
    """Return (nodes, index, adjacency) for the True pixels of ``skel``."""
    rows, cols = np.nonzero(skel)
    nodes: List[Tuple[int, int]] = list(zip(rows.tolist(), cols.tolist()))  # raster order
    index: Dict[Tuple[int, int], int] = {p: i for i, p in enumerate(nodes)}
    adj: List[List[Tuple[int, float]]] = [[] for _ in nodes]
    for i, (r, c) in enumerate(nodes):
        for dr, dc, w in _NEIGH:
            j = index.get((r + dr, c + dc))
            if j is None:
                continue
            # no corner cutting: a diagonal step is not allowed when the same
            # neighbour is reachable through an orthogonal skeleton pixel, so the
            # path follows the traced skeleton through its corner pixels
            # (connectivity is preserved: the orthogonal route exists)
            if dr != 0 and dc != 0 and ((r + dr, c) in index or (r, c + dc) in index):
                continue
            adj[i].append((j, w))
    return nodes, index, adj


def _dijkstra(adj, src: int):
    n = len(adj)
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[src] = 0.0
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            # strict improvement, or equal distance from a smaller-index parent
            if nd < dist[v] - 1e-12 or (abs(nd - dist[v]) <= 1e-12 and prev[v] > u):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def _components(adj) -> List[List[int]]:
    n = len(adj)
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        comp, stack = [], [s]
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v, _ in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps


def geodesic_longest_path(skeleton: np.ndarray) -> PathResult:
    """Longest geodesic path on a binary skeleton.

    Parameters
    ----------
    skeleton : (H, W) array
        Non-zero pixels are skeleton pixels (any dtype).

    Returns
    -------
    PathResult
        ``mask`` is a bool array of the same shape marking the path pixels;
        ``length`` is the geodesic length (1 per orthogonal step, sqrt(2) per
        diagonal step; 0 for a single pixel); ``start``/``end`` are the path
        ends in (row, col), or ``None`` when the skeleton is empty.
    """
    skel = np.asarray(skeleton) != 0
    mask = np.zeros(skel.shape, dtype=bool)
    if not skel.any():
        return PathResult(mask, 0.0, None, None)

    nodes, _, adj = _build_graph(skel)
    best = None  # (length, start_idx, end_idx, prev_list)
    for comp in _components(adj):
        comp_set = set(comp)
        ends = [i for i in comp if len(adj[i]) <= 1]
        n_edges = sum(len(adj[i]) for i in comp) // 2
        cyclic = n_edges >= len(comp)                # a connected graph is a tree iff edges == nodes - 1
        candidates = comp if (cyclic or not ends) else ends   # exact: tree -> endpoints suffice; cycle -> all pixels
        for s in candidates:
            dist, prev = _dijkstra(adj, s)
            # farthest reachable node in this component; ties -> smallest index
            far, far_d = None, -1.0
            for t in comp:
                d = dist[t]
                if d > far_d + 1e-12:
                    far, far_d = t, d
            if far is None:
                continue
            key = (far_d, -s, -far)   # longer first; then smaller start, smaller end
            if best is None or key > best[0]:
                best = (key, s, far, prev)
    _, s, t, prev = best
    length = 0.0
    path = [t]
    while path[-1] != s:
        path.append(prev[path[-1]])
    for i in path:
        mask[nodes[i]] = True
    # recompute the geodesic length along the reconstructed path
    for a, b in zip(path[:-1], path[1:]):
        (r1, c1), (r2, c2) = nodes[a], nodes[b]
        length += _SQRT2 if (r1 != r2 and c1 != c2) else 1.0
    return PathResult(mask, float(length), nodes[s], nodes[t])


def longest_path_pixels(skeleton: np.ndarray, method: str = "geodesic") -> np.ndarray:
    """Convenience wrapper returning a {0,1} uint8 path mask."""
    if method != "geodesic":
        raise ValueError(f"unknown method {method!r}; this module implements 'geodesic'")
    return geodesic_longest_path(skeleton).mask.astype(np.uint8)
