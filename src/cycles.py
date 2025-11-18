"""
Utilities for identifying cycle hotspots and approximate loop structures.

These helpers operate on the diagnostics exported by `potential_hodge_glm`
(`results/diagnostics/edges_decomp_glm.*`).  They are written to work even when
`pandas` is not installed: in that case the functions return a lightweight
table object that mimics the subset of the DataFrame interface we rely on.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from . import io as io_mod

@dataclass
class SimpleTable:
    """Minimal table used when pandas is unavailable.

    Only implements helpers that downstream code relies on: iteration, len(),
    truthiness, `.to_dict("records")`, and `.to_csv(...)`.
    """

    rows: List[Tuple]
    columns: Sequence[str]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        return bool(self.rows)

    @property
    def empty(self) -> bool:
        return len(self.rows) == 0

    def to_dict(self, orient: str = "records"):
        if orient != "records":
            raise ValueError("SimpleTable only supports orient='records'")
        return [dict(zip(self.columns, row)) for row in self.rows]

    def to_csv(self, path: str, index: bool = False) -> None:
        del index  # matching pandas signature, but unused
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(self.columns))
            writer.writerows(self.rows)


def _resolve_path(path: str) -> str:
    """Return an existing path for diagnostics, trying csv/parquet variants."""
    if os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    candidates = []
    if ext.lower() != ".csv":
        candidates.append(base + ".csv")
    if ext.lower() != ".parquet":
        candidates.append(base + ".parquet")
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return path


def _iter_edge_rows(path: str) -> Iterator[Dict[str, Union[str, float]]]:
    """Yield rows from the diagnostics file, handling CSV/Parquet."""
    real_path = _resolve_path(path)
    ext = os.path.splitext(real_path)[1].lower()
    if pd is not None:
        try:
            if ext == ".parquet":
                df = pd.read_parquet(real_path)
            else:
                df = pd.read_csv(real_path)
            for record in df.to_dict("records"):
                yield record
            return
        except Exception:
            # Fall through to CSV parsing below.
            pass
    # CSV path (only works if CSV exists)
    csv_path = real_path
    if not csv_path.lower().endswith(".csv"):
        base = os.path.splitext(real_path)[0]
        csv_candidate = base + ".csv"
        if os.path.exists(csv_candidate):
            csv_path = csv_candidate
        else:
            raise FileNotFoundError(
                f"Unable to load diagnostics from {real_path}; pandas not available "
                "and no CSV version present."
            )
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _to_float(val: Union[str, float, int, None]) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val):
            return None
        return float(val)
    if isinstance(val, int):
        return float(val)
    try:
        if val == "" or val == "None":
            return None
        return float(val)
    except Exception:
        return None


def _make_table(rows: List[Tuple], columns: Sequence[str]):
    if pd is not None:
        return pd.DataFrame(rows, columns=columns)
    return SimpleTable(rows, columns)


def _as_records(table) -> List[Dict[str, Union[str, float]]]:
    if table is None:
        return []
    if pd is not None and isinstance(table, pd.DataFrame):
        return table.to_dict("records")
    if isinstance(table, SimpleTable):
        return table.to_dict("records")
    if isinstance(table, list):
        # assume list of tuples matching select_cycle_hotspots output
        cols = ["u", "v", "strength", "dist_km"]
        recs = []
        for row in table:
            if isinstance(row, dict):
                recs.append(row)
            elif isinstance(row, (tuple, list)):
                rec = {}
                for c, val in zip(cols, row):
                    rec[c] = val
                recs.append(rec)
        return recs
    if hasattr(table, "to_dict"):
        try:
            return table.to_dict("records")
        except Exception:
            pass
    return []


def select_cycle_hotspots(
    edges_decomp_path: str,
    k: int = 2000,
    quantile: float = 0.99,
):
    """Return strongest cycle edges (absolute residual or g-pred signal).

    Parameters
    ----------
    edges_decomp_path : str
        Path to `edges_decomp_glm` diagnostics (CSV or Parquet).
    k : int
        Maximum number of edges to return.
    quantile : float
        Quantile threshold (0-1) on |strength| before taking Top-K.
    """
    rows = []
    for row in _iter_edge_rows(edges_decomp_path):
        u = str(row.get("u"))
        v = str(row.get("v"))
        if not u or not v:
            continue
        resid = _to_float(row.get("resid"))
        g_val = _to_float(row.get("g_ij"))
        pred = _to_float(row.get("pred"))
        dist = _to_float(
            row.get("distance_km") or row.get("dist") or row.get("dist_km")
        )
        strength_signal = resid
        if strength_signal is None and g_val is not None and pred is not None:
            strength_signal = g_val - pred
        if strength_signal is None:
            continue
        rows.append((u, v, abs(strength_signal), dist))
    if not rows:
        return _make_table([], ("u", "v", "strength", "dist_km"))
    values = sorted(r[2] for r in rows)
    quantile = min(max(quantile, 0.0), 1.0)
    idx = int(math.floor((len(values) - 1) * quantile))
    idx = max(0, min(len(values) - 1, idx))
    threshold = values[idx]
    filtered = [r for r in rows if r[2] >= threshold]
    filtered.sort(key=lambda x: x[2], reverse=True)
    limit = max(0, int(k)) if k is not None else len(filtered)
    top = filtered[:limit]
    return _make_table(top, ("u", "v", "strength", "dist_km"))


def approx_triangle_cycles(edges_decomp_path: str, topcycles: int = 500):
    """Approximate Top-M triangular cycles from Hodge diagnostics."""
    g_dir: Dict[Tuple[str, str], float] = {}
    neighbors: Dict[str, set] = {}
    for row in _iter_edge_rows(edges_decomp_path):
        u = str(row.get("u"))
        v = str(row.get("v"))
        if not u or not v:
            continue
        g_val = _to_float(row.get("g_ij"))
        if g_val is None:
            # Fall back to pred+resid if available
            pred = _to_float(row.get("pred"))
            resid = _to_float(row.get("resid"))
            if pred is not None and resid is not None:
                g_val = pred + resid
        if g_val is None:
            continue
        g_dir[(u, v)] = g_val
        g_dir[(v, u)] = -g_val
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)
    if not neighbors:
        return _make_table([], ("i", "j", "k", "C_ijk"))
    triangles: List[Tuple[str, str, str, float]] = []
    nodes = sorted(neighbors.keys())
    for i in nodes:
        neigh_i = sorted(n for n in neighbors[i] if n > i)
        if len(neigh_i) < 2:
            continue
        for j_idx, j in enumerate(neigh_i):
            neigh_j = neighbors.get(j, set())
            if not neigh_j:
                continue
            for k in neigh_i[j_idx + 1 :]:
                if k <= j:
                    continue
                if k not in neigh_j:
                    continue
                g_ij = g_dir.get((i, j))
                g_jk = g_dir.get((j, k))
                g_ki = g_dir.get((k, i))
                if g_ij is None or g_jk is None or g_ki is None:
                    continue
                cyc = g_ij + g_jk + g_ki
                triangles.append((i, j, k, cyc))
    triangles.sort(key=lambda t: abs(t[3]), reverse=True)
    limit = max(0, int(topcycles))
    top = triangles[:limit]
    return _make_table(top, ("i", "j", "k", "C_ijk"))


def plot_cycle_hotspots(top_edges_df, nodes_geojson: str, out_png: str) -> bool:
    """Plot cycle hotspot edges as great-circle approximations on centroid plane.

    Parameters
    ----------
    top_edges_df
        DataFrame (or SimpleTable) with columns u,v,strength.
    nodes_geojson : str
        Path to GeoJSON supplying tract centroids (INTPTLAT/INTPTLON).
    out_png : str
        Destination figure path inside results/figures.
    """
    if plt is None:
        print("matplotlib not available; skip cycle hotspot figure.")
        return False
    records = _as_records(top_edges_df)
    if not records:
        print("No cycle hotspots to plot (empty records).")
        return False
    centroids = {}
    if nodes_geojson and os.path.exists(nodes_geojson):
        try:
            centroids = io_mod.load_geojson_centroids(nodes_geojson)
        except Exception as exc:  # pragma: no cover - IO heavy
            print(f"Failed to load centroids from {nodes_geojson}: {exc}")
    if not centroids:
        print("No centroids available; cannot plot hotspots.")
        return False
    xs = []
    ys = []
    lines = []
    strengths = []
    for rec in records:
        u = str(rec.get("u"))
        v = str(rec.get("v"))
        strength = _to_float(rec.get("strength"))
        if not u or not v or strength is None:
            continue
        loc_u = centroids.get(u)
        loc_v = centroids.get(v)
        if not loc_u or not loc_v:
            continue
        lat_u, lon_u = loc_u
        lat_v, lon_v = loc_v
        xs.extend([lon_u, lon_v])
        ys.extend([lat_u, lat_v])
        lines.append(((lon_u, lon_v), (lat_u, lat_v)))
        strengths.append(strength)
    if not lines:
        print("Cycle hotspot plot skipped: no edges with coordinates.")
        return False
    max_strength = max(strengths) if strengths else 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    # Light background nodes
    ax.scatter(xs, ys, s=4, color="#bbbbbb", alpha=0.6, linewidths=0)
    # Hotspot corridors on top
    for (lon_pair, lat_pair), strength in zip(lines, strengths):
        norm = strength / max_strength if max_strength > 0 else 0.0
        lw = 0.4 + 3.6 * (norm ** 0.5)
        alpha = 0.15 + 0.65 * norm
        ax.plot(
            lon_pair,
            lat_pair,
            color="#c02739",
            linewidth=lw,
            alpha=max(0.1, min(1.0, alpha)),
        )
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True
