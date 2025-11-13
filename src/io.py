import csv
import json
import math
import os
from collections import defaultdict
from typing import Dict, Tuple, Iterable, List, Optional


SEMCOG_COUNTIES = {"26163", "26125", "26099", "26161", "26115", "26093", "26147"}


def ensure_dirs():
    base = os.path.join("project", "data")
    for sub in ("processed",):
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    for sub in ("diagnostics", "figures"):
        os.makedirs(os.path.join("project", "results", sub), exist_ok=True)


def read_od_csv(path: str) -> Iterable[Tuple[str, str, int]]:
    """Yield (work, home, S000) from tract→tract OD CSV with header work,home,S000.
    Ignores any index column if present.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [c.strip().lower() for c in header]
        # Try to locate columns
        if cols[:3] == ["", "work", "home"]:
            # Some exports have a leading blank index column
            idx_work, idx_home, idx_s = 1, 2, 3
        else:
            idx_work = cols.index("work")
            idx_home = cols.index("home")
            idx_s = cols.index("s000")
        for row in reader:
            try:
                w = row[idx_work].strip()
                h = row[idx_home].strip()
                s = int(float(row[idx_s]))
                yield (w, h, s)
            except Exception:
                continue


def geoid11(x: str) -> str:
    s = ''.join(ch for ch in str(x) if ch.isdigit())
    if len(s) >= 11:
        return s[:11]
    return s.zfill(11)


def county5(geoid: str) -> str:
    g = geoid11(geoid)
    return g[:5]


def filter_semcog(edges: Iterable[Tuple[str, str, int]]) -> Iterable[Tuple[str, str, int]]:
    for w, h, s in edges:
        wc = w[:5]
        hc = h[:5]
        if wc in SEMCOG_COUNTIES and hc in SEMCOG_COUNTIES:
            yield (w, h, s)


def write_csv(path: str, rows: Iterable[Tuple], header: Optional[List[str]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def try_write_parquet(path: str, rows: List[Tuple], columns: List[str]) -> bool:
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows, columns=columns)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def load_geojson_centroids(path: str) -> Dict[str, Tuple[float, float]]:
    """Return mapping GEOID -> (lat, lon) using INTPTLAT/INTPTLON when available.
    Fallback: compute naive centroid from coordinates if no internal point.
    """
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    mapping: Dict[str, Tuple[float, float]] = {}
    for feat in feats:
        props = feat.get("properties", {})
        geoid = props.get("GEOID") or props.get("GEOID10") or props.get("geoid")
        if not geoid:
            continue
        lat = props.get("INTPTLAT10") or props.get("centroid_lat")
        lon = props.get("INTPTLON10") or props.get("centroid_lon")
        if lat is not None and lon is not None:
            try:
                mapping[str(geoid)] = (float(lat), float(lon))
                continue
            except Exception:
                pass
        # Fallback: naive centroid from geometry coords
        geom = feat.get("geometry", {})
        coords = []
        def collect(g):
            if not g:
                return
            t = g.get("type")
            if t == "Point":
                coords.append(g.get("coordinates"))
            elif t == "MultiPoint":
                coords.extend(g.get("coordinates", []))
            elif t in ("Polygon",):
                for ring in g.get("coordinates", []):
                    coords.extend(ring)
            elif t in ("MultiPolygon",):
                for poly in g.get("coordinates", []):
                    for ring in poly:
                        coords.extend(ring)
            elif t == "GeometryCollection":
                for gg in g.get("geometries", []):
                    collect(gg)
        collect(geom)
        if coords:
            xs = [c[0] for c in coords if c]
            ys = [c[1] for c in coords if c]
            if xs and ys:
                lon_c = (min(xs) + max(xs)) / 2.0
                lat_c = (min(ys) + max(ys)) / 2.0
                mapping[str(geoid)] = (lat_c, lon_c)
    return mapping


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0088  # km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_edge_distances(
    edges: Iterable[Tuple[str, str, int]],
    centroids: Dict[str, Tuple[float, float]],
) -> Iterable[Tuple[str, str, int, Optional[float]]]:
    for w, h, s in edges:
        latlon_w = centroids.get(w)
        latlon_h = centroids.get(h)
        if latlon_w and latlon_h:
            d = haversine_km(latlon_w[0], latlon_w[1], latlon_h[0], latlon_h[1])
        else:
            d = None
        yield (w, h, s, d)


def save_processed(
    edges_path: str,
    geojson_path: str,
    out_nodes: str,
    out_edges: str,
    out_edges_with_dist: str,
    parquet_out: Optional[str] = None,
) -> None:
    ensure_dirs()
    # Read and filter edges
    edges = list(filter_semcog(read_od_csv(edges_path)))
    write_csv(out_edges, edges, header=["work", "home", "S000"])

    # Load centroids if available
    centroids = {}
    if os.path.exists(geojson_path):
        centroids = load_geojson_centroids(geojson_path)
    # Save nodes CSV (only nodes appearing in edges)
    nodes = sorted({n for e in edges for n in (e[0], e[1])})
    node_rows = []
    for n in nodes:
        latlon = centroids.get(n) if centroids else None
        if latlon:
            node_rows.append((n, latlon[0], latlon[1]))
        else:
            node_rows.append((n, None, None))
    write_csv(out_nodes, node_rows, header=["geoid", "lat", "lon"])

    # Distances if we have centroids
    if centroids:
        with_dist = list(compute_edge_distances(edges, centroids))
        write_csv(out_edges_with_dist, with_dist, header=["work", "home", "S000", "distance_km"])
        if parquet_out is not None:
            try_write_parquet(parquet_out, with_dist, ["work", "home", "S000", "distance_km"])
    else:
        # No geo: write edges_with_distance with None distances
        with_dist = [(w, h, s, None) for (w, h, s) in edges]
        write_csv(out_edges_with_dist, with_dist, header=["work", "home", "S000", "distance_km"])


def ingest_od(
    raw_path: str,
    out_parquet: str,
) -> Dict[str, int]:
    """Ingest raw OD CSV, normalize GEOIDs (11-char), add county, filter to SEMCOG both ends,
    and write processed Parquet. Returns simple summary metrics.
    """
    # Read
    rows = list(read_od_csv(raw_path))
    total = sum(s for _w, _h, s in rows)
    # Normalize + filter
    clean: List[Tuple[str, str, str, str, int]] = []
    for w, h, s in rows:
        gw = geoid11(w)
        gh = geoid11(h)
        wc = gw[:5]
        hc = gh[:5]
        if wc in SEMCOG_COUNTIES and hc in SEMCOG_COUNTIES:
            clean.append((gw, gh, wc, hc, int(s)))
    # Write
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(clean, columns=["work", "home", "work_cty", "home_cty", "S000"])
        os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
        df.to_parquet(out_parquet, index=False)
    except Exception:
        # CSV fallback
        csv_out = out_parquet.rsplit('.', 1)[0] + ".csv"
        write_csv(csv_out, clean, header=["work", "home", "work_cty", "home_cty", "S000"])
        out_parquet = csv_out
    tot_clean = sum(r[4] for r in clean)
    # Acceptance helpers
    return {
        "rows_raw": len(rows),
        "rows_clean": len(clean),
        "sum_raw": int(total),
        "sum_clean": int(tot_clean),
    }


def marginals_from_parquet(in_parquet: str, out_nodes_parquet: str) -> Dict[str, int]:
    # Try pandas path
    try:
        import pandas as pd  # type: ignore
        df = pd.read_parquet(in_parquet)
        df["work"] = df["work"].astype(str).map(geoid11)
        df["home"] = df["home"].astype(str).map(geoid11)
        R = df.groupby("work")["S000"].sum().rename("R")
        W = df.groupby("home")["S000"].sum().rename("W")
        Fii = df[df["work"] == df["home"]].groupby("work")["S000"].sum().rename("Fii")
        nodes = pd.DataFrame(index=sorted(set(df["work"]) | set(df["home"])) )
        nodes.index.name = "geoid"
        nodes = nodes.join(R, how="left").join(W, how="left").join(Fii, how="left").fillna(0)
        os.makedirs(os.path.dirname(out_nodes_parquet), exist_ok=True)
        try:
            nodes.reset_index().to_parquet(out_nodes_parquet, index=False)
        except Exception:
            nodes.reset_index().to_csv(out_nodes_parquet.rsplit('.',1)[0]+'.csv', index=False)
        return {
            "nodes": int(nodes.shape[0]),
            "sumR": int(nodes["R"].sum()),
            "sumW": int(nodes["W"].sum()),
            "sumF": int(df["S000"].sum()),
        }
    except Exception:
        # Fallback CSV
        import csv
        R: Dict[str,int] = defaultdict(int)
        W: Dict[str,int] = defaultdict(int)
        Fsum = 0
        nodes_set = set()
        path_csv = in_parquet.rsplit('.',1)[0] + '.csv'
        with open(path_csv, 'r', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                w = geoid11(r['work']); h = geoid11(r['home']); s = int(float(r['S000']))
                R[w] += s; W[h] += s; Fsum += s
                nodes_set.add(w); nodes_set.add(h)
        header = ["geoid","R","W","Fii"]
        rows = []
        for g in sorted(nodes_set):
            fii = R.get(g,0) if g in W and g in R and False else 0  # skip exact Fii without pandas
            rows.append((g, R.get(g,0), W.get(g,0), fii))
        write_csv(out_nodes_parquet.rsplit('.',1)[0]+'.csv', rows, header=header)
        return {
            "nodes": len(nodes_set),
            "sumR": sum(R.values()),
            "sumW": sum(W.values()),
            "sumF": Fsum,
        }


def convert_shp_to_geojson(shp_path: str, out_geojson: str, geoid_fields: Optional[List[str]] = None) -> bool:
    """Convert ESRI Shapefile to GeoJSON with GEOID and centroid properties.
    Tries pyshp; if unavailable, tries ogr2ogr.
    """
    # Support directory-style shapefile bundles
    if os.path.isdir(shp_path):
        # look for a single .dbf inside
        cands = [os.path.join(shp_path, f) for f in os.listdir(shp_path) if f.lower().endswith('.dbf')]
        if cands:
            return convert_dbf_to_geojson_points(cands[0], out_geojson, geoid_fields)
    try:
        import shapefile  # type: ignore
    except Exception:
        # Try ogr2ogr if GDAL present
        rc = os.system(f"ogr2ogr -f GeoJSON {out_geojson} {shp_path} > /dev/null 2>&1")
        if rc == 0:
            return True
        # Try DBF-only centroid export as Points
        dbf_path = shp_path[:-4] + '.dbf' if shp_path.lower().endswith('.shp') else None
        if dbf_path and os.path.exists(dbf_path):
            return convert_dbf_to_geojson_points(dbf_path, out_geojson, geoid_fields)
        return False

    sf = shapefile.Reader(shp_path)
    fields = [f[0] for f in sf.fields[1:]]
    geoid_fields = geoid_fields or ["GEOID", "GEOID10", "geoid", "TRACTCE10", "TRACTCE"]
    # Attempt to identify lon/lat centroid fields if present
    lat_keys = ["INTPTLAT", "INTPTLAT10", "lat", "LAT"]
    lon_keys = ["INTPTLON", "INTPTLON10", "lon", "LON"]
    feats = []
    for sr in sf.shapeRecords():
        rec = {k: sr.record[fields.index(k)] for k in fields if k in fields}
        # GEOID
        geoid = None
        for k in geoid_fields:
            if k in rec and rec[k]:
                v = str(rec[k]).strip()
                if k.upper().startswith("TRACT"):
                    # try build from STATEFP/COUNTYFP
                    st = str(rec.get("STATEFP") or rec.get("STATEFP10") or "").zfill(2)
                    ct = str(rec.get("COUNTYFP") or rec.get("COUNTYFP10") or "").zfill(3)
                    geoid = geoid11(st + ct + v)
                else:
                    geoid = geoid11(v)
                break
        if not geoid:
            # Try STATEFP+COUNTYFP+TRACTCE
            st = str(rec.get("STATEFP") or rec.get("STATEFP10") or "").zfill(2)
            ct = str(rec.get("COUNTYFP") or rec.get("COUNTYFP10") or "").zfill(3)
            tr = str(rec.get("TRACTCE") or rec.get("TRACTCE10") or "").zfill(6)
            if st and ct and tr:
                geoid = geoid11(st + ct + tr)
        if not geoid:
            continue
        # Geometry
        sh = sr.shape
        geom_type = sh.shapeTypeName
        def ring_to_coords(ring):
            return [[float(x), float(y)] for x, y in ring]
        geom = None
        if geom_type in ("POLYGON", "POLYGONZ"):
            geom = {"type": "Polygon", "coordinates": [ring_to_coords(part) for part in sh.parts and [sh.points[i:j] for i, j in zip(sh.parts, list(sh.parts[1:]) + [None])] or [sh.points]]}
        elif geom_type in ("MULTIPOLYGON",):
            # pyshp doesn't directly label MULTIPOLYGON; treat multiple parts as polygon with multiple rings
            geom = {"type": "Polygon", "coordinates": [ring_to_coords(part) for part in sh.parts and [sh.points[i:j] for i, j in zip(sh.parts, list(sh.parts[1:]) + [None])] or [sh.points]]}
        else:
            # fallback: envelope center only
            xs = [p[0] for p in sh.points]
            ys = [p[1] for p in sh.points]
            cx = (min(xs) + max(xs)) / 2 if xs else 0.0
            cy = (min(ys) + max(ys)) / 2 if ys else 0.0
            geom = {"type": "Point", "coordinates": [float(cx), float(cy)]}
        # centroid
        lat = None
        lon = None
        for k in lat_keys:
            if k in rec:
                try:
                    lat = float(rec[k])
                    break
                except Exception:
                    pass
        for k in lon_keys:
            if k in rec:
                try:
                    lon = float(rec[k])
                    break
                except Exception:
                    pass
        if (lat is None or lon is None) and geom and geom.get("type") in ("Polygon",):
            # bbox center
            coords = []
            for ring in geom.get("coordinates", []):
                coords.extend(ring)
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            if xs and ys:
                lon = (min(xs) + max(xs)) / 2
                lat = (min(ys) + max(ys)) / 2
        props = {"GEOID": geoid}
        if lat is not None and lon is not None:
            props["centroid_lat"] = lat
            props["centroid_lon"] = lon
        feats.append({"type": "Feature", "properties": props, "geometry": geom})
    gj = {"type": "FeatureCollection", "features": feats}
    os.makedirs(os.path.dirname(out_geojson), exist_ok=True)
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(gj, f)
    return True


def convert_dbf_to_geojson_points(dbf_path: str, out_geojson: str, geoid_fields: Optional[List[str]] = None) -> bool:
    """Minimal DBF reader to emit centroids-only GeoJSON Points using INTPTLAT/INTPTLON (or LAT/LON) fields.
    This avoids heavy dependencies if geometry is not needed.
    """
    try:
        with open(dbf_path, 'rb') as f:
            header = f.read(32)
            if len(header) < 32:
                return False
            num_records = int.from_bytes(header[4:8], 'little')
            header_len = int.from_bytes(header[8:10], 'little')
            record_len = int.from_bytes(header[10:12], 'little')
            # Read field descriptors
            fields = []
            while True:
                desc = f.read(32)
                if not desc or desc[0] == 0x0D:
                    break
                name = desc[0:11].split(b'\x00',1)[0].decode('latin1').strip().strip('\x00')
                ftype = chr(desc[11])
                length = desc[16]
                fields.append((name, ftype, length))
            # Position at first record
            f.seek(header_len)
            # Build name->(offset,length)
            offsets = {}
            off = 1  # first byte is deletion flag
            for (name, _t, length) in fields:
                offsets[name] = (off, length)
                off += length
            geoid_fields = geoid_fields or ["GEOID","GEOID10","geoid","TRACTCE10","TRACTCE"]
            lat_keys = ["INTPTLAT10","INTPTLAT","lat","LAT"]
            lon_keys = ["INTPTLON10","INTPTLON","lon","LON"]
            feats = []
            for i in range(num_records):
                rec = f.read(record_len)
                if not rec or rec[0:1] == b'*':
                    continue
                def get(name):
                    if name not in offsets:
                        return None
                    off, ln = offsets[name]
                    raw = rec[off:off+ln]
                    try:
                        return raw.decode('latin1').strip()
                    except Exception:
                        return None
                geoid = None
                for k in geoid_fields:
                    val = get(k)
                    if val:
                        if k.upper().startswith('TRACT'):
                            st = get('STATEFP') or get('STATEFP10') or ''
                            ct = get('COUNTYFP') or get('COUNTYFP10') or ''
                            geoid = geoid11((st+ct+val))
                        else:
                            geoid = geoid11(val)
                        break
                if not geoid:
                    st = (get('STATEFP') or get('STATEFP10') or '').zfill(2)
                    ct = (get('COUNTYFP') or get('COUNTYFP10') or '').zfill(3)
                    tr = (get('TRACTCE') or get('TRACTCE10') or '').zfill(6)
                    if st and ct and tr:
                        geoid = geoid11(st+ct+tr)
                if not geoid:
                    continue
                lat = None; lon = None
                for k in lat_keys:
                    v = get(k)
                    if v:
                        try:
                            lat = float(v)
                            break
                        except Exception:
                            pass
                for k in lon_keys:
                    v = get(k)
                    if v:
                        try:
                            lon = float(v)
                            break
                        except Exception:
                            pass
                if lat is None or lon is None:
                    continue
                feats.append({
                    "type":"Feature",
                    "properties": {"GEOID": geoid, "centroid_lat": lat, "centroid_lon": lon},
                    "geometry": {"type":"Point","coordinates":[lon,lat]}
                })
        gj = {"type":"FeatureCollection","features":feats}
        os.makedirs(os.path.dirname(out_geojson), exist_ok=True)
        with open(out_geojson,'w',encoding='utf-8') as fo:
            json.dump(gj, fo)
        return True
    except Exception:
        return False
