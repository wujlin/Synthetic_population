import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def _load_records(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        base, ext = os.path.splitext(path)
        csv_path = base + ".csv" if ext != ".csv" else path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"node_signals_geo missing: {path}")
        path = csv_path
    if path.endswith(".parquet"):
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:
            raise RuntimeError("Parquet input requires pandas. Save as CSV instead.") from exc
        df = pd.read_parquet(path)  # type: ignore
        return df.to_dict("records")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def _grid_for_county(
    rows: List[Dict[str, str]],
    grid_res_km: float,
) -> Dict[str, object]:
    cell_deg = grid_res_km / 111.0
    data = []
    for row in rows:
        try:
            lat = float(row.get("lat") or "nan")
            lon = float(row.get("lon") or "nan")
            drhoH = float(row.get("drhoH") or 0.0)
            drhoW = float(row.get("drhoW") or 0.0)
            dIn = float(row.get("dIn") or 0.0)
            rhoH0 = float(row.get("rhoH_2020") or 0.0)
            rhoW0 = float(row.get("rhoW_2020") or 0.0)
            In0 = float(row.get("In_2020") or 0.0)
        except Exception:
            continue
        if math.isnan(lat) or math.isnan(lon):
            continue
        data.append((lat, lon, drhoH, drhoW, dIn, rhoH0, rhoW0, In0))
    if not data:
        return {}
    lat_min = min(d[0] for d in data)
    lat_max = max(d[0] for d in data)
    lon_min = min(d[1] for d in data)
    lon_max = max(d[1] for d in data)
    ny = max(1, int((lat_max - lat_min) / cell_deg) + 1)
    nx = max(1, int((lon_max - lon_min) / cell_deg) + 1)
    X = [[[0.0 for _ in range(nx)] for __ in range(ny)] for __ in range(3)]
    Y = [[[0.0 for _ in range(nx)] for __ in range(ny)] for __ in range(3)]
    counts = [[0 for _ in range(nx)] for __ in range(ny)]
    for lat, lon, drhoH, drhoW, dIn, rhoH0, rhoW0, In0 in data:
        iy = int((lat - lat_min) / cell_deg)
        ix = int((lon - lon_min) / cell_deg)
        iy = max(0, min(ny - 1, iy))
        ix = max(0, min(nx - 1, ix))
        X[0][iy][ix] += rhoH0
        X[1][iy][ix] += rhoW0
        X[2][iy][ix] += In0
        Y[0][iy][ix] += drhoH
        Y[1][iy][ix] += drhoW
        Y[2][iy][ix] += dIn
        counts[iy][ix] += 1
    for iy in range(ny):
        for ix in range(nx):
            if counts[iy][ix] > 0:
                for c in range(3):
                    X[c][iy][ix] /= counts[iy][ix]
                for c in range(len(Y)):
                    Y[c][iy][ix] /= counts[iy][ix]
    mask = [[1 if counts[iy][ix] > 0 else 0 for ix in range(nx)] for iy in range(ny)]
    lat_grid = [lat_min + (i + 0.5) * cell_deg for i in range(ny)]
    lon_grid = [lon_min + (j + 0.5) * cell_deg for j in range(nx)]
    return {
        "grid_res_km": grid_res_km,
        "lat": lat_grid,
        "lon": lon_grid,
        "X": X,
        "Y": Y,
        "mask": mask,
    }


def to_grids(
    node_signals_geo_path: str,
    tracts_geojson: str,
    grid_res_km: float = 1.5,
) -> None:
    """Rasterize node signals to per-county grids."""
    records = _load_records(node_signals_geo_path)
    if not records:
        raise RuntimeError("node_signals_geo has no rows.")
    by_county: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in records:
        county = row.get("county")
        if county:
            by_county[county].append(row)
    out_root = os.path.join("project", "data", "node", "grids")
    os.makedirs(out_root, exist_ok=True)
    for county, rows in sorted(by_county.items()):
        payload = _grid_for_county(rows, grid_res_km)
        if not payload:
            continue
        out_path = os.path.join(out_root, f"{county}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
