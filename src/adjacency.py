"""
Adjacency helpers for PDE interface diagnostics.

This module builds tract-level adjacency graphs either from polygon boundaries
or by k-nearest-neighbor links between centroids.  It returns both the edge
list and the symmetric normalized Laplacian (I - D^{-1/2} A D^{-1/2}) encoded
as nested dictionaries for downstream computations.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import io as io_mod

try:  # Optional geometric backend
    from shapely.geometry import shape  # type: ignore
except Exception:  # pragma: no cover
    shape = None  # type: ignore


def _load_geojson(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"GeoJSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_geoid(feat: Dict) -> Optional[str]:
    props = feat.get("properties", {})
    for key in ("GEOID", "GEOID10", "geoid"):
        val = props.get(key)
        if val:
            return str(val)
    return None


def _bbox_overlap(b1: Sequence[float], b2: Sequence[float]) -> bool:
    return not (b1[2] < b2[0] or b1[0] > b2[2] or b1[3] < b2[1] or b1[1] > b2[3])


def _polygon_adjacency(features: List[Dict]) -> Optional[Set[Tuple[str, str]]]:
    if shape is None:
        return None
    geometries: List[Tuple[str, object, Sequence[float]]] = []
    for feat in features:
        geoid = _get_geoid(feat)
        geom = feat.get("geometry")
        if not geoid or not geom:
            continue
        try:
            poly = shape(geom)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            geometries.append((geoid, poly, poly.bounds))
        except Exception:
            continue
    if not geometries:
        return None
    edges: Set[Tuple[str, str]] = set()
    n = len(geometries)
    for i in range(n):
        geoid_i, poly_i, bounds_i = geometries[i]
        for j in range(i + 1, n):
            geoid_j, poly_j, bounds_j = geometries[j]
            if not _bbox_overlap(bounds_i, bounds_j):
                continue
            try:
                # touches implies boundary contact; intersects covers small overlaps
                if poly_i.touches(poly_j) or poly_i.intersects(poly_j):
                    edge = tuple(sorted((geoid_i, geoid_j)))
                    edges.add(edge)
            except Exception:
                continue
    return edges if edges else None


def _knn_adjacency(
    centroids: Dict[str, Tuple[Optional[float], Optional[float]]],
    k: int,
) -> Set[Tuple[str, str]]:
    nodes = sorted(centroids.keys())
    edges: Set[Tuple[str, str]] = set()
    for idx, node in enumerate(nodes):
        latlon = centroids.get(node)
        if not latlon or None in latlon:
            continue
        lat_i, lon_i = latlon
        dists: List[Tuple[float, str]] = []
        for jdx, other in enumerate(nodes):
            if other == node:
                continue
            latlon_o = centroids.get(other)
            if not latlon_o or None in latlon_o:
                continue
            dist = io_mod.haversine_km(lat_i, lon_i, latlon_o[0], latlon_o[1])
            dists.append((dist, other))
        if not dists:
            continue
        dists.sort(key=lambda t: t[0])
        limit = min(max(int(k), 1), len(dists))
        for _, nbr in dists[:limit]:
            edges.add(tuple(sorted((node, nbr))))
    return edges


def _normalized_laplacian(nodes: Sequence[str], edges: Iterable[Tuple[str, str]]):
    node_set = set(nodes)
    neighbors: Dict[str, Set[str]] = {n: set() for n in node_set}
    for u, v in edges:
        if u == v:
            continue
        if u not in neighbors:
            neighbors[u] = set()
            node_set.add(u)
        if v not in neighbors:
            neighbors[v] = set()
            node_set.add(v)
        neighbors[u].add(v)
        neighbors[v].add(u)
    degrees = {n: len(neighbors.get(n, ())) for n in neighbors.keys()}
    laplacian: Dict[str, Dict[str, float]] = {}
    for n in sorted(node_set):
        deg = degrees.get(n, 0)
        row: Dict[str, float] = {}
        if deg == 0:
            row[n] = 0.0
        else:
            row[n] = 1.0
            for nbr in neighbors[n]:
                deg_j = degrees.get(nbr, 0)
                if deg_j == 0:
                    continue
                val = -1.0 / (math.sqrt(deg) * math.sqrt(deg_j))
                row[nbr] = val
        laplacian[n] = row
    return laplacian


def build_adjacency(
    tracts_geojson: str = os.path.join("project", "data", "geo", "tracts.geojson"),
    method: str = "polygon|knn",
    k: int = 6,
):
    """Construct adjacency edges and normalized Laplacian from tract geometries.

    Parameters
    ----------
    tracts_geojson : str
        Path to census tract GeoJSON with GEOID + geometry/centroid properties.
    method : str
        Pipe-separated priority order (e.g., "polygon|knn") for adjacency
        construction.  Supports "polygon" and "knn".
    k : int
        Number of nearest neighbors for the kNN option (per node) when polygon adjacency is unavailable.

    Returns
    -------
    edges : List[Tuple[str, str]]
        Undirected edge list (u < v) reflecting adjacency.
    laplacian : Dict[str, Dict[str, float]]
        Symmetric normalized Laplacian entries for all tracts present in the
        GeoJSON centroids (including isolated nodes).
    """
    gj = _load_geojson(tracts_geojson)
    features = gj.get("features", [])
    centroids = io_mod.load_geojson_centroids(tracts_geojson)
    order = [tok.strip().lower() for tok in method.split("|") if tok.strip()]
    if not order:
        order = ["polygon", "knn"]
    edges: Optional[Set[Tuple[str, str]]] = None
    for mode in order:
        if mode == "polygon":
            edges = _polygon_adjacency(features)
        elif mode == "knn":
            edges = _knn_adjacency(centroids, k=k)
        if edges:
            break
    if edges is None:
        edges = set()
    edges_list = sorted(edges)
    node_ids = sorted(set(centroids.keys()) | {n for edge in edges_list for n in edge})
    laplacian = _normalized_laplacian(node_ids, edges_list)
    return edges_list, laplacian
