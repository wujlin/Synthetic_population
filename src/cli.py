import argparse
import csv
import os
from typing import Optional

from . import io as io_mod
from . import hodge as hodge_mod
from .locality import bin_by_distance, plot_locality_curve, run_locality_curve
from .gravity import residual_from_parquet
import json


def cmd_prepare(args: argparse.Namespace) -> None:
    raw_path = args.raw or os.path.join("project", "data", "raw", "semcog_tract_od_intra_2020.csv")
    geo_path = args.geo or os.path.join("project", "data", "geo", "tracts.geojson")
    out_nodes = os.path.join("project", "data", "processed", "nodes.csv")
    out_edges = os.path.join("project", "data", "processed", "edges.csv")
    out_edges_dist = os.path.join("project", "data", "processed", "edges_with_distance.csv")
    out_parquet = os.path.join("project", "data", "processed", "edges_with_distance.parquet")
    io_mod.save_processed(raw_path, geo_path, out_nodes, out_edges, out_edges_dist, out_parquet)
    print("Prepared processed nodes/edges with distance.")


def cmd_shp2geojson(args: argparse.Namespace) -> None:
    ok = io_mod.convert_shp_to_geojson(args.shp, args.out, geoid_fields=args.geoid_fields)
    if ok:
        print("GeoJSON written:", args.out)
    else:
        print("Failed to convert SHP→GeoJSON; please install pyshp or GDAL (ogr2ogr).")


def load_processed_edges_with_dist(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            w = r["work"].strip()
            h = r["home"].strip()
            s = int(float(r["S000"]))
            d = r.get("distance_km")
            dist = float(d) if d not in (None, "", "None") else None
            rows.append((w, h, s, dist))
    return rows


def cmd_hodge(args: argparse.Namespace) -> None:
    in_path = os.path.join("project", "data", "processed", "edges_with_distance.csv")
    rows = load_processed_edges_with_dist(in_path)
    edges = [(w, h, s) for (w, h, s, _d) in rows]
    nodes, edges_u = hodge_mod.build_pairs(edges, use_independence=args.use_independence, eps=args.eps)
    # Optional cap on number of undirected edges (keep largest weights)
    if args.max_edges and len(edges_u) > args.max_edges:
        edges_u.sort(key=lambda t: t[2], reverse=True)
        edges_u = edges_u[: args.max_edges]
    phi, Et, Eg, Er = hodge_mod.fit_potential(edges_u, nodes, maxiter=args.maxiter, tol=args.tol)
    grad_frac = (Eg / Et) if Et > 0 else 0.0
    resid_frac = (Er / Et) if Et > 0 else 0.0

    # Save node potential
    out_node = os.path.join("project", "results", "diagnostics", "node_potential.csv")
    io_mod.write_csv(out_node, [(n, phi[n]) for n in nodes], header=["geoid", "phi"])

    # Save edge diagnostics (with distance)
    phi_map = phi
    diag_rows = []
    # Map distances for undirected (u,v) to attach to diagnostics
    dist_map = {}
    for (w, h, s, d) in rows:
        u, v = (w, h) if w < h else (h, w)
        # keep min distance if duplicates
        if d is not None:
            if (u, v) not in dist_map:
                dist_map[(u, v)] = d
            else:
                dist_map[(u, v)] = min(dist_map[(u, v)], d)
    for (u, v, w, a, pred, resid) in hodge_mod.edge_diagnostics(edges_u, phi_map):
        d = dist_map.get((u, v))
        diag_rows.append((u, v, w, a, pred, resid, d))
    out_edge = os.path.join("project", "results", "diagnostics", "edge_hodge_metrics.csv")
    io_mod.write_csv(
        out_edge,
        diag_rows,
        header=["u", "v", "weight", "a_ij", "pred", "resid", "distance_km"],
    )

    # Save global metrics
    out_global = os.path.join("project", "results", "diagnostics", "global_metrics.csv")
    io_mod.write_csv(
        out_global,
        [
            (
                len(nodes),
                len(edges_u),
                Et,
                Eg,
                Er,
                grad_frac,
                resid_frac,
                int(args.use_independence),
                args.eps,
            )
        ],
        header=[
            "nodes",
            "edges",
            "energy_total",
            "energy_grad",
            "energy_resid",
            "grad_fraction",
            "resid_fraction",
            "use_independence",
            "eps",
        ],
    )
    print("Hodge diagnostics saved.")


def cmd_locality(args: argparse.Namespace) -> None:
    in_diag = os.path.join("project", "results", "diagnostics", "edge_hodge_metrics.csv")
    rows = []
    with open(in_diag, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            d = r.get("distance_km")
            dist = float(d) if d not in (None, "", "None") else None
            rows.append(
                (
                    r["u"],
                    r["v"],
                    float(r["weight"]),
                    float(r["a_ij"]),
                    float(r["pred"]),
                    float(r["resid"]),
                    dist,
                )
            )
    curve = bin_by_distance(rows, bins=args.bins)
    out_curve = os.path.join("project", "results", "diagnostics", "locality_curve.csv")
    io_mod.write_csv(
        out_curve,
        curve,
        header=[
            "bin_idx",
            "d_lo_km",
            "d_hi_km",
            "edge_count",
            "E_total",
            "E_grad",
            "E_resid",
            "grad_fraction",
            "resid_fraction",
        ],
    )
    fig_path = os.path.join("project", "results", "figures", "locality_curve.png")
    ok = plot_locality_curve(curve, fig_path)
    if ok:
        print("Locality curve saved:", fig_path)
    else:
        print("Locality curve CSV saved; matplotlib not available for PNG.")


def cmd_ingest(args: argparse.Namespace) -> None:
    raw = args.raw or os.path.join("project", "data", "raw", "semcog_tract_od_intra_2020.csv")
    outp = os.path.join("project", "data", "processed", "od_clean.parquet")
    res = io_mod.ingest_od(raw, outp)
    print(json.dumps({"ingest": res}, ensure_ascii=False))


def cmd_marginals(args: argparse.Namespace) -> None:
    inp = os.path.join("project", "data", "processed", "od_clean.parquet")
    outp = os.path.join("project", "data", "processed", "nodes_basic.parquet")
    res = io_mod.marginals_from_parquet(inp, outp)
    print(json.dumps({"marginals": res}, ensure_ascii=False))


def cmd_baseline(args: argparse.Namespace) -> None:
    inp = os.path.join("project", "data", "processed", "od_clean.parquet")
    outp = os.path.join("project", "data", "processed", "od_residual.parquet")
    res = residual_from_parquet(inp, outp, eps=args.eps)
    print(json.dumps({"baseline_gravity": res}, ensure_ascii=False))


def cmd_potential(args: argparse.Namespace) -> None:
    inp = os.path.join("project", "data", "processed", "od_residual.parquet")
    geom = os.path.join("project", "data", "processed", "edges_with_distance.csv")
    # Load residual
    df = None
    try:
        import pandas as pd  # type: ignore
        df = pd.read_parquet(inp)
    except Exception:
        csv_inp = inp.rsplit('.',1)[0]+'.csv'
        import csv as _csv
        rows_tmp=[]
        with open(csv_inp,'r',encoding='utf-8') as f:
            rdr=_csv.DictReader(f)
            for r in rdr:
                rows_tmp.append((r['work'], r['home'], float(r['S000']), float(r['log_resid'])))
        # Directly run without pandas dataframe
        phi, summary, diags = hodge_mod.run_hodge_from_residual(rows_tmp, max_edges=args.max_edges, maxiter=args.maxiter, tol=args.tol)
        # Save outputs
        out_nodes = os.path.join("project", "results", "diagnostics", "nodes_potential.parquet")
        io_mod.write_csv(out_nodes.rsplit('.',1)[0]+'.csv', [(k,v) for k,v in phi.items()], header=["geoid","pi"])
        out_edges = os.path.join("project", "results", "diagnostics", "edges_decomp.parquet")
        io_mod.write_csv(out_edges.rsplit('.',1)[0]+'.csv', diags, header=["u","v","weight","g_ij","pred","resid"])
        out_summary = os.path.join("project", "results", "diagnostics", "summary.json")
        with open(out_summary, 'w', encoding='utf-8') as f:
            json.dump(summary, f)
        print("Potential/Hodge outputs saved.")
        return
    # optional distances join (if edges_with_distance exists)
    dist_map = {}
    if os.path.exists(geom):
        import csv as _csv
        with open(geom, 'r', encoding='utf-8') as f:
            rdr = _csv.DictReader(f)
            for r in rdr:
                dist_map[(r['work'], r['home'])] = r.get('distance_km')
    rows = []
    for w, h, s, lr in df[["work","home","S000","log_resid"]].itertuples(index=False, name=None):
        rows.append((w, h, float(s), float(lr)))
    phi, summary, diags = hodge_mod.run_hodge_from_residual(rows, max_edges=args.max_edges, maxiter=args.maxiter, tol=args.tol)
    # Save outputs
    out_nodes = os.path.join("project", "results", "diagnostics", "nodes_potential.parquet")
    try:
        import pandas as pd  # type: ignore
        pd.DataFrame({"geoid": list(phi.keys()), "pi": list(phi.values())}).to_parquet(out_nodes, index=False)
    except Exception:
        io_mod.write_csv(out_nodes.rsplit('.',1)[0]+'.csv', [(k,v) for k,v in phi.items()], header=["geoid","pi"])
    out_edges = os.path.join("project", "results", "diagnostics", "edges_decomp.parquet")
    try:
        import pandas as pd  # type: ignore
        pd.DataFrame(diags, columns=["u","v","weight","g_ij","pred","resid"]).to_parquet(out_edges, index=False)
    except Exception:
        io_mod.write_csv(out_edges.rsplit('.',1)[0]+'.csv', diags, header=["u","v","weight","g_ij","pred","resid"])
    out_summary = os.path.join("project", "results", "diagnostics", "summary.json")
    with open(out_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f)
    print("Potential/Hodge outputs saved.")


def cmd_locality_curve(args: argparse.Namespace) -> None:
    inp = os.path.join("project", "data", "processed", "od_residual.parquet")
    geom = os.path.join("project", "data", "processed", "edges_with_distance.csv")
    df = None
    try:
        import pandas as pd  # type: ignore
        df = pd.read_parquet(inp)
    except Exception:
        import csv as _csv
        csv_inp = inp.rsplit('.',1)[0] + '.csv'
        # build distance map from edges_with_distance.csv
        dist_map = {}
        with open(geom, 'r', encoding='utf-8') as f:
            rdr = _csv.DictReader(f)
            for r in rdr:
                try:
                    dist = float(r.get('distance_km')) if r.get('distance_km') not in (None, '', 'None') else None
                except Exception:
                    dist = None
                dist_map[(r['work'], r['home'])] = dist
        rows = []
        with open(csv_inp,'r',encoding='utf-8') as f:
            rdr=_csv.DictReader(f)
            for r in rdr:
                dist = dist_map.get((r['work'], r['home']))
                rows.append((r['work'], r['home'], float(r['S000']), float(r['log_resid']), dist))
        radii = args.radii or [5,10,15,20,25,30,40,50]
        runner = lambda residual_rows: hodge_mod.run_hodge_from_residual(residual_rows, max_edges=args.max_edges, maxiter=args.maxiter, tol=args.tol)
        curve = run_locality_curve(rows, radii, runner)
        out_json = os.path.join("project", "results", "diagnostics", "locality_report.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(curve, f)
        print("Locality curve saved.")
        return
    dist_map = {}
    import csv as _csv
    with open(geom, 'r', encoding='utf-8') as f:
        rdr = _csv.DictReader(f)
        for r in rdr:
            try:
                dist = float(r.get('distance_km')) if r.get('distance_km') not in (None, '', 'None') else None
            except Exception:
                dist = None
            dist_map[(r['work'], r['home'])] = dist
    rows = []
    for w, h, s, lr in df[["work","home","S000","log_resid"]].itertuples(index=False, name=None):
        rows.append((w, h, float(s), float(lr), dist_map.get((w,h))))
    radii = args.radii or [5,10,15,20,25,30,40,50]
    runner = lambda residual_rows: hodge_mod.run_hodge_from_residual(residual_rows, max_edges=args.max_edges, maxiter=args.maxiter, tol=args.tol)
    curve = run_locality_curve(rows, radii, runner)
    out_json = os.path.join("project", "results", "diagnostics", "locality_report.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(curve, f)
    # optional figure
    try:
        import matplotlib.pyplot as plt  # type: ignore
        xs = [c['r0'] for c in curve]
        ys = [c['R2'] for c in curve]
        plt.figure(figsize=(6,4))
        plt.plot(xs, ys, marker='o')
        plt.xlabel('distance threshold r0 (km)')
        plt.ylabel('R^2 potential')
        plt.title('Locality curve')
        plt.tight_layout()
        os.makedirs(os.path.join('project','results','figures'), exist_ok=True)
        plt.savefig(os.path.join('project','results','figures','fig_locality_curve.png'), dpi=150)
        plt.close()
    except Exception:
        pass
    print("Locality curve saved.")


def cmd_export_figs(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print('matplotlib not available; skip figure export.')
        return
    # Load potentials (CSV fallback)
    np_base = os.path.join("project", "results", "diagnostics", "nodes_potential")
    np_csv = np_base + '.csv'
    geoid_pi = []
    import csv as _csv
    if os.path.exists(np_csv):
        with open(np_csv,'r',encoding='utf-8') as f:
            rdr=_csv.DictReader(f)
            for r in rdr:
                try:
                    geoid_pi.append((r['geoid'], float(r['pi'])))
                except Exception:
                    pass
    else:
        # try parquet -> csv not present
        try:
            import pandas as pd  # type: ignore
            dfp = pd.read_parquet(np_base+'.parquet')
            geoid_pi = list(dfp[['geoid','pi']].itertuples(index=False, name=None))
        except Exception:
            print('no nodes_potential data; skip figure export.')
            return
    # Boxplot by county
    from collections import defaultdict
    buckets = defaultdict(list)
    for g, pi in geoid_pi:
        buckets[str(g)[:5]].append(pi)
    labels = sorted(buckets.keys())
    data = [buckets[k] for k in labels if buckets[k]]
    plt.figure(figsize=(7,4))
    plt.boxplot(data, labels=[l for l in labels if buckets[l]], showfliers=False)
    plt.xticks(rotation=45)
    plt.title('Potential by county')
    plt.tight_layout()
    os.makedirs(os.path.join('project','results','figures'), exist_ok=True)
    plt.savefig(os.path.join('project','results','figures','pi_box_by_county.png'), dpi=150)
    plt.close()
    # R2/eta bar
    with open(os.path.join('project','results','diagnostics','summary.json'),'r',encoding='utf-8') as f:
        summ = json.load(f)
    plt.figure(figsize=(4,3))
    plt.bar(['R2','eta'], [summ.get('R2',0.0), summ.get('eta',0.0)])
    plt.title('R2 and non-reciprocity')
    plt.tight_layout()
    plt.savefig(os.path.join('project','results','figures','r2_eta.png'), dpi=150)
    plt.close()
    print('Figures exported.')


def main():
    p = argparse.ArgumentParser(description="SEMCOG OD diagnostics pipeline")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("prepare", help="Load OD + GeoJSON, save processed nodes/edges+distance")
    sp.add_argument("--raw", type=str, default=None, help="Path to OD CSV (work,home,S000)")
    sp.add_argument("--geo", type=str, default=None, help="Path to tracts.geojson")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("shp2geojson", help="Convert Shapefile to GeoJSON with GEOID and centroids")
    sp.add_argument("--shp", required=True, help="Path to .shp")
    sp.add_argument("--out", required=True, help="Output GeoJSON path")
    sp.add_argument("--geoid-fields", nargs='*', default=None, help="Candidate GEOID fields")
    sp.set_defaults(func=cmd_shp2geojson)

    sp = sub.add_parser("hodge", help="Potential fit + Hodge indicators")
    sp.add_argument("--use-independence", action="store_true", help="Independence normalization")
    sp.add_argument("--eps", type=float, default=0.5, help="Smoothing epsilon for ratios")
    sp.add_argument("--max-edges", type=int, default=200000, help="Cap undirected edges (by weight)")
    sp.add_argument("--maxiter", type=int, default=500, help="CG max iterations")
    sp.add_argument("--tol", type=float, default=1e-6, help="CG tolerance")
    sp.set_defaults(func=cmd_hodge)

    sp = sub.add_parser("locality", help="Distance-layered R²/energy curves")
    sp.add_argument("--bins", type=int, default=10, help="Number of distance bins")
    sp.set_defaults(func=cmd_locality)

    sp = sub.add_parser("ingest", help="Ingest OD, normalize GEOIDs, filter SEMCOG, write od_clean.parquet")
    sp.add_argument("--raw", type=str, default=None)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("marginals", help="Compute R_i, W_j, F_ii and write nodes_basic.parquet")
    sp.set_defaults(func=cmd_marginals)

    sp = sub.add_parser("baseline_gravity", help="Compute independence baseline and residuals")
    sp.add_argument("--eps", type=float, default=0.5)
    sp.set_defaults(func=cmd_baseline)

    sp = sub.add_parser("potential_hodge", help="Fit potential from residuals and export diagnostics")
    sp.add_argument("--max-edges", type=int, default=200000)
    sp.add_argument("--maxiter", type=int, default=500)
    sp.add_argument("--tol", type=float, default=1e-6)
    sp.set_defaults(func=cmd_potential)

    sp = sub.add_parser("locality_curve", help="R² vs distance thresholds")
    sp.add_argument("--radii", type=int, nargs='*', default=None)
    sp.add_argument("--max-edges", type=int, default=200000)
    sp.add_argument("--maxiter", type=int, default=500)
    sp.add_argument("--tol", type=float, default=1e-6)
    sp.set_defaults(func=cmd_locality_curve)

    sp = sub.add_parser("export_figs", help="Export figures for potential and summary metrics")
    sp.set_defaults(func=cmd_export_figs)

    sp = sub.add_parser("all", help="Run prepare + hodge + locality")
    sp.add_argument("--raw", type=str, default=None)
    sp.add_argument("--geo", type=str, default=None)
    sp.add_argument("--use-independence", action="store_true")
    sp.add_argument("--eps", type=float, default=0.5)
    sp.add_argument("--max-edges", type=int, default=200000)
    sp.add_argument("--maxiter", type=int, default=500)
    sp.add_argument("--tol", type=float, default=1e-6)
    sp.add_argument("--bins", type=int, default=10)
    def run_all(args):
        cmd_prepare(args)
        cmd_hodge(args)
        cmd_locality(args)
    sp.set_defaults(func=run_all)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return
    io_mod.ensure_dirs()
    args.func(args)


if __name__ == "__main__":
    main()
