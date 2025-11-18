import csv
import os
from typing import Dict, List, Tuple

try:
    import pandas as pd  # type: ignore
    HAVE_PANDAS = True
except Exception:
    pd = None  # type: ignore
    HAVE_PANDAS = False


def _log(msg: str) -> None:
    """Log helper: print to stdout and append to a node-level log file."""
    print(msg)
    try:
        log_dir = os.path.join("project", "results", "node")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "attach_geo.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        # Best-effort logging; never crash on log failure.
        pass


def _read_table_records(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """Read table records from parquet/csv, returning list of dicts and column order."""
    if path.endswith(".parquet") and os.path.exists(path) and HAVE_PANDAS:
        _log(f"[attach_geo] pandas available, reading parquet: {path}")
        df = pd.read_parquet(path)  # type: ignore
        return df.to_dict("records"), list(df.columns)
    # Fallback to CSV
    if path.endswith(".csv") and os.path.exists(path):
        csv_path = path
    else:
        csv_path = path.rsplit(".", 1)[0] + ".csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[attach_geo] Input table not found as parquet or csv: {path} / {csv_path}"
            )
    _log(f"[attach_geo] Reading CSV: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
        cols = reader.fieldnames or []
    return rows, cols


def _write_table_records(records: List[Dict[str, str]], columns: List[str], out_path: str) -> None:
    """Write records to parquet (if possible) or CSV fallback."""
    if out_path.endswith(".parquet") and HAVE_PANDAS:
        try:
            _log(f"[attach_geo] Writing parquet: {out_path}")
            df = pd.DataFrame(records, columns=columns)  # type: ignore
            df.to_parquet(out_path, index=False)
            return
        except Exception as exc:
            _log(f"[attach_geo] Failed to write parquet ({exc}); falling back to CSV.")
    parquet_path = out_path + ".parquet"
    if HAVE_PANDAS and out_path.endswith(".csv"):
        # allow writing both csv and parquet if requested
        _log(f"[attach_geo] Writing parquet alongside CSV: {parquet_path}")
        df = pd.DataFrame(records, columns=columns)  # type: ignore
        df.to_parquet(parquet_path, index=False)
    csv_out = out_path if out_path.endswith(".csv") else out_path.rsplit(".", 1)[0] + ".csv"
    _log(f"[attach_geo] Writing CSV: {csv_out}")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def attach_geo(in_parquet: str, tracts_geojson: str, out_parquet: str) -> None:
    """Attach lat/lon/county5 to a node-level table keyed by geoid.

    Parameters
    ----------
    in_parquet : str
        Input table with a 'geoid' column (parquet or csv).
    tracts_geojson : str
        GeoJSON providing tract centroids / INTPTLAT10/LON10.
    out_parquet : str
        Output path (parquet if pandas available, otherwise csv).
    """
    from .. import io as core_io

    _log(f"[attach_geo] start: in_parquet={in_parquet}, tracts_geojson={tracts_geojson}, out_parquet={out_parquet}")

    # Load centroids (GEOID -> (lat, lon))
    centroids: Dict[str, Tuple[float, float]] = {}
    if os.path.exists(tracts_geojson):
        _log(f"[attach_geo] Loading centroids from {tracts_geojson}...")
        centroids = core_io.load_geojson_centroids(tracts_geojson)
        _log(f"[attach_geo] Loaded {len(centroids)} centroid(s).")
    else:
        _log(f"[attach_geo] WARNING: tracts_geojson not found: {tracts_geojson} (lat/lon will be None).")

    # Load tabular data
    try:
        records, columns = _read_table_records(in_parquet)
        if "geoid" not in columns:
            raise ValueError("[attach_geo] Input table must have a 'geoid' column.")
        _log(f"[attach_geo] Loaded table with {len(records)} rows.")
    except Exception as exc:
        _log(f"[attach_geo] ERROR while loading table: {exc}")
        raise

    def _county5(g: str) -> str:
        g = core_io.geoid11(str(g))
        return g[:5]

    columns = list(columns)
    if "lat" not in columns:
        columns.append("lat")
    if "lon" not in columns:
        columns.append("lon")
    if "county" not in columns:
        columns.append("county")

    for row in records:
        g = str(row.get("geoid", ""))
        latlon = centroids.get(g)
        row["lat"] = latlon[0] if latlon else ""
        row["lon"] = latlon[1] if latlon else ""
        row["county"] = core_io.geoid11(g)[:5]

    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    try:
        _write_table_records(records, columns, out_parquet)
    except Exception as exc:
        _log(f"[attach_geo] ERROR while writing output: {exc}")
        raise
    _log("[attach_geo] done.")
