import argparse
import csv
import os
from typing import Optional

from . import io as io_mod
from . import hodge as hodge_mod
from .locality import bin_by_distance, plot_locality_curve, run_locality_curve
from .gravity import residual_from_parquet, residual_glm_ppml
from .cycles import select_cycle_hotspots, approx_triangle_cycles, plot_cycle_hotspots
from . import pde_fit
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


def cmd_rot_diagnostics(args: argparse.Namespace) -> None:
    """Compute loop diagnostics (eta proxy, triangles) and render hotspot map."""
    diag_dir = os.path.join("project", "results", "diagnostics")
    figs_dir = os.path.join("project", "results", "figures")
    edges_path = os.path.join(diag_dir, "edges_decomp_glm.csv")
    if not os.path.exists(edges_path):
        print("edges_decomp_glm.csv missing. Run potential_hodge_glm first.")
        return
    top_edges = select_cycle_hotspots(edges_path, k=args.topk, quantile=args.quantile)
    os.makedirs(diag_dir, exist_ok=True)
    out_top_edges = os.path.join(diag_dir, "edges_topk_glm.csv")
    if hasattr(top_edges, "to_csv"):
        top_edges.to_csv(out_top_edges, index=False)
    else:
        io_mod.write_csv(out_top_edges, top_edges, header=["u", "v", "strength", "dist_km"])
    top_cycles = approx_triangle_cycles(edges_path, topcycles=args.topcycles)
    out_top_cycles = os.path.join(diag_dir, "top_cycles.csv")
    if hasattr(top_cycles, "to_csv"):
        top_cycles.to_csv(out_top_cycles, index=False)
    else:
        io_mod.write_csv(out_top_cycles, top_cycles, header=["i", "j", "k", "C_ijk"])
    eta_val = None
    for cand in ("summary_robustness.json", "summary.json"):
        cand_path = os.path.join(diag_dir, cand)
        if os.path.exists(cand_path):
            try:
                with open(cand_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    eta_val = meta.get("eta")
                break
            except Exception:
                continue
    summary = {
        "edges_decomp": edges_path,
        "edges_hotspots": out_top_edges,
        "top_cycles": out_top_cycles,
        "topk": args.topk,
        "topcycles": args.topcycles,
        "quantile": args.quantile,
        "eta_global": eta_val,
    }
    out_summary = os.path.join(diag_dir, "rot_summary.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    fig_path = os.path.join(figs_dir, "fig_cycles_hotspots.png")
    ok = plot_cycle_hotspots(top_edges, args.nodes_geo, fig_path)
    if ok:
        print("Cycle hotspot figure saved:", fig_path)
    else:
        print("Cycle hotspot data prepared; figure skipped due to missing deps.")


def _resolve_processed(path_parquet: str):
    if os.path.exists(path_parquet):
        return path_parquet
    if path_parquet.endswith(".parquet"):
        csv_path = path_parquet.rsplit(".", 1)[0] + ".csv"
        if os.path.exists(csv_path):
            return csv_path
    return path_parquet


def cmd_pde_fit_kappa(args: argparse.Namespace) -> None:
    residual_path = _resolve_processed(os.path.join("project", "data", "processed", "od_residual_glm.parquet"))
    nodes_path = os.path.join("project", "results", "diagnostics", "nodes_potential_glm.csv")
    if not os.path.exists(residual_path):
        print("od_residual_glm missing. Run baseline_glm first.")
        return
    if not os.path.exists(nodes_path):
        print("nodes_potential_glm.csv missing. Run potential_hodge_glm first.")
        return
    diag_dir = os.path.join("project", "results", "diagnostics")
    figs_dir = os.path.join("project", "results", "figures")
    out_json = os.path.join(diag_dir, "pde_kappa.json")
    out_png = os.path.join(figs_dir, "fig_kappa_scatter.png")
    try:
        pde_fit.fit_kappa(
            residual_path,
            nodes_path,
            args.rac,
            out_json,
            out_png,
        )
        print("Kappa diagnostics saved:", out_json)
    except Exception as exc:
        print(f"Kappa fit failed: {exc}")


def cmd_pde_fit_diffusion(args: argparse.Namespace) -> None:
    residual_edges = os.path.join("project", "results", "diagnostics", "edges_decomp_glm.csv")
    if not os.path.exists(residual_edges):
        print("edges_decomp_glm.csv missing. Run potential_hodge_glm first.")
        return
    diag_dir = os.path.join("project", "results", "diagnostics")
    figs_dir = os.path.join("project", "results", "figures")
    out_json = os.path.join(diag_dir, "pde_diffusion.json")
    out_png = os.path.join(figs_dir, "fig_diffusion_scatter.png")
    try:
        pde_fit.fit_diffusion(
            residual_edges,
            args.rac,
            out_json,
            out_png,
        )
        print("Diffusion diagnostics saved:", out_json)
    except Exception as exc:
        print(f"Diffusion fit failed: {exc}")


def cmd_pde_fit_interface(args: argparse.Namespace) -> None:
    residual_edges = os.path.join("project", "results", "diagnostics", "edges_decomp_glm.csv")
    if not os.path.exists(residual_edges):
        print("edges_decomp_glm.csv missing. Run potential_hodge_glm first.")
        return
    diag_dir = os.path.join("project", "results", "diagnostics")
    figs_dir = os.path.join("project", "results", "figures")
    out_json = os.path.join(diag_dir, "pde_interface.json")
    out_png = os.path.join(figs_dir, "fig_interface_scatter.png")
    geo_path = args.tracts_geo or os.path.join("project", "data", "geo", "tracts.geojson")
    try:
        pde_fit.fit_interface(
            residual_edges,
            geo_path,
            args.rac,
            out_json,
            out_png,
            knn=args.knn,
        )
        print("Interface diagnostics saved:", out_json)
    except Exception as exc:
        print(f"Interface fit failed: {exc}")


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
    from .export_figs import export_all
    out = export_all()
    made = [k for k,v in out.items() if v]
    skipped = [k for k,v in out.items() if not v]
    if made:
        print('Figures exported:', ', '.join(made))
    if skipped:
        print('Skipped (missing matplotlib or data):', ', '.join(skipped))


def _legacy_main(argv=None):
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

    sp = sub.add_parser(
        "rot_diagnostics",
        help=(
            "Loop diagnostics on edges_decomp_glm: select hotspot edges, top triangle cycles, "
            "and render fig_cycles_hotspots.png."
        ),
    )
    sp.add_argument("--topk", type=int, default=2000, help="Top-K edges ranked by |residual| to export")
    sp.add_argument("--topcycles", type=int, default=500, help="Top triangle cycles (sorted by |C_ijk|)")
    sp.add_argument(
        "--nodes-geo",
        type=str,
        default=os.path.join("project", "data", "geo", "tracts.geojson"),
        help="GeoJSON with GEOID centroids for hotspot plotting",
    )
    sp.add_argument(
        "--quantile",
        type=float,
        default=0.99,
        help="Quantile on |residual| before applying Top-K cap (controls hotspot tail)",
    )
    sp.set_defaults(func=cmd_rot_diagnostics)

    sp = sub.add_parser(
        "pde_fit_kappa",
        help="Estimate kappa via WLS on paired flows: tilde N_ij ~ rho_ij + c_ij + c_ij*Delta pi",
    )
    sp.add_argument("--rac", type=str, default=None, help="Optional RAC density file (parquet/csv)")
    sp.set_defaults(func=cmd_pde_fit_kappa)

    sp = sub.add_parser(
        "pde_fit_diffusion",
        help="Estimate diffusion D by regressing residual skew on -Delta rho (after removing kappa).",
    )
    sp.add_argument("--rac", type=str, default=None, help="Optional RAC density file (parquet/csv)")
    sp.set_defaults(func=cmd_pde_fit_diffusion)

    sp = sub.add_parser(
        "pde_fit_interface",
        help="Estimate interface Gamma via Laplacian gradients: hat N_ij ~ -Gamma * Delta(L rho).",
    )
    sp.add_argument("--rac", type=str, default=None, help="Optional RAC density file (parquet/csv)")
    sp.add_argument("--knn", type=int, default=6, help="k for centroid kNN adjacency when polygons unavailable")
    sp.add_argument(
        "--tracts-geo",
        type=str,
        default=os.path.join("project", "data", "geo", "tracts.geojson"),
        help="GeoJSON with tract geometries/centroids for adjacency",
    )
    sp.set_defaults(func=cmd_pde_fit_interface)

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

    def cmd_baseline_glm(args: argparse.Namespace) -> None:
        inp = os.path.join("project", "data", "processed", "od_clean.parquet")
        if not (os.path.exists(inp)):
            # use CSV version if Parquet missing
            inp = os.path.join("project", "data", "processed", "od_clean.csv")
        edges_dist = os.path.join("project", "data", "processed", "edges_with_distance.csv")
        outp = os.path.join("project", "data", "processed", "od_residual_glm.parquet")
        logp = os.path.join("project", "results", "diagnostics", "baseline_glm_summary.json")
        res = residual_glm_ppml(
            inp, edges_dist, outp, logp,
            use_county_pair_fe=args.county_pair_fe,
            eps=args.eps,
            backend=args.backend,
            sample_n=args.sample_n,
            max_iter=getattr(args, 'max_iter', 3000),
            alpha=getattr(args, 'alpha', 1e-8),
            standardize_dist=getattr(args, 'standardize_dist', False),
        )
        print(json.dumps({"baseline_glm": res}, ensure_ascii=False))
    sp = sub.add_parser("baseline_glm", help="PPML with origin/dest FE + distance [+ county×county FE]")
    sp.add_argument("--eps", type=float, default=0.5)
    sp.add_argument("--county-pair-fe", action="store_true")
    sp.add_argument("--backend", type=str, default='auto', choices=['auto','sklearn','statsmodels'], help="Force backend (auto tries statsmodels then sklearn)")
    sp.add_argument("--sample-n", type=int, default=None, help="Optional downsample N for PPML fit")
    sp.add_argument("--max-iter", type=int, default=3000, help="Max iterations for PPML solver")
    sp.add_argument("--alpha", type=float, default=1e-8, help="L2 regularization for PPML")
    sp.add_argument("--standardize-dist", action="store_true", help="Z-score distance before fitting")
    sp.set_defaults(func=cmd_baseline_glm)

    sp = sub.add_parser("potential_hodge", help="Fit potential from residuals and export diagnostics")
    sp.add_argument("--max-edges", type=int, default=200000)
    sp.add_argument("--maxiter", type=int, default=500)
    sp.add_argument("--tol", type=float, default=1e-6)
    sp.set_defaults(func=cmd_potential)

    def cmd_potential_glm(args: argparse.Namespace) -> None:
        # Load GLM residuals and distances
        inp = os.path.join("project", "data", "processed", "od_residual_glm.parquet")
        if not os.path.exists(inp):
            inp = os.path.join("project", "data", "processed", "od_residual_glm.csv")
        import csv as _csv
        rows_ext=[]
        if inp.endswith('.parquet'):
            try:
                import pandas as pd  # type: ignore
                df = pd.read_parquet(inp)
                for w,h,s,lr,d,mu in df[["work","home","S000","log_resid_glm","dist","mu_hat"]].itertuples(index=False, name=None):
                    rows_ext.append((w,h,float(s),float(lr),float(d) if d==d else None,float(mu)))
            except Exception:
                csv_candidate = inp.rsplit('.', 1)[0] + '.csv'
                if os.path.exists(csv_candidate):
                    inp = csv_candidate
                rows_ext = []
        if not rows_ext:
            with open(inp, 'r', encoding='utf-8', errors='ignore') as f:
                rdr = _csv.DictReader(f)
                for r in rdr:
                    w = r.get('work') or r.get('WORK')
                    h = r.get('home') or r.get('HOME')
                    s = r.get('S000') or r.get('s000')
                    if not w or not h or s in (None, ''):
                        continue
                    d = r.get('dist'); mu = r.get('mu_hat');
                    lr = r.get('log_resid_glm') if 'log_resid_glm' in r else r.get('log_resid')
                    rows_ext.append((w, h, float(s), float(lr), float(d) if d not in (None,'','None') else None, float(mu) if mu not in (None,'','None') else None))
        from .hodge import run_hodge_from_residual_robust
        phi, summary, diags, topk = run_hodge_from_residual_robust(
            rows_ext,
            weight_type=args.weight_type,
            cap_tau=args.cap_tau,
            drop_self=args.drop_self,
            sample_edges=args.sample_edges,
            bins_dist=args.bins_dist,
            bins_weight=args.bins_weight,
            seed=args.seed,
            max_edges=args.max_edges,
            maxiter=args.maxiter,
            tol=args.tol,
        )
        # Save outputs
        out_nodes = os.path.join("project", "results", "diagnostics", "nodes_potential_glm.csv")
        io_mod.write_csv(out_nodes, [(k,v) for k,v in phi.items()], header=["geoid","pi"])
        out_edges = os.path.join("project", "results", "diagnostics", "edges_decomp_glm.csv")
        io_mod.write_csv(out_edges, diags, header=["u","v","weight","g_ij","pred","resid"])
        out_topk = os.path.join("project", "results", "diagnostics", "edges_topk_glm.csv")
        io_mod.write_csv(out_topk, topk, header=["u","v","weight","g_ij","pred","resid"])
        # robust summary
        out_summary = os.path.join("project", "results", "diagnostics", "summary_robustness.json")
        rob = dict(summary)
        rob.update({
            "weight_type": args.weight_type,
            "cap_tau": args.cap_tau,
            "drop_self": bool(args.drop_self),
            "sample_edges": args.sample_edges or 0,
            "bins_dist": args.bins_dist,
            "bins_weight": args.bins_weight,
            "seed": args.seed,
        })
        with open(out_summary,'w',encoding='utf-8') as f:
            json.dump(rob, f)
        print("Potential/Hodge GLM robustness saved.")
    sp = sub.add_parser("potential_hodge_glm", help="Robust Hodge on GLM residuals with sampling/weights")
    sp.add_argument("--weight-type", type=str, default="sum", choices=["sum","cap","mu","eij"])
    sp.add_argument("--cap-tau", type=float, default=100.0)
    sp.add_argument("--drop-self", action="store_true")
    sp.add_argument("--sample-edges", type=int, default=None)
    sp.add_argument("--bins-dist", type=int, default=8)
    sp.add_argument("--bins-weight", type=int, default=8)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--max-edges", type=int, default=200000)
    sp.add_argument("--maxiter", type=int, default=500)
    sp.add_argument("--tol", type=float, default=1e-6)
    sp.set_defaults(func=cmd_potential_glm)

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

    if argv is None:
        import sys as _sys
        argv = _sys.argv[1:]
    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return
    io_mod.ensure_dirs()
    args.func(args)
def main(argv: Optional[list[str]] = None):
    """Router entry point.

    - `python -m project.src.cli edge ...`  → edge-line CLI
    - `python -m project.src.cli node ...`  → node-line CLI
    - `python -m project.src.cli baseline_glm ...` (etc.) keeps working.
    """
    import sys

    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "edge":
        from .edge.cli_edge import main as edge_main

        edge_main(argv[1:])
        return
    if argv and argv[0] == "node":
        from .node.cli_node import main as node_main

        node_main(argv[1:])
        return

    # Fallback to legacy flat CLI
    _legacy_main(argv)


if __name__ == "__main__":
    main()
