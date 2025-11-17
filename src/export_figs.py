import os
import json
from collections import defaultdict

from .cycles import select_cycle_hotspots, plot_cycle_hotspots
from . import pde_fit
from . import io as io_mod


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def _read_nodes_potential(diag_dir):
    """Return list of (geoid, pi). Tries parquet then CSV."""
    path_parq = os.path.join(diag_dir, "nodes_potential.parquet")
    path_csv = os.path.join(diag_dir, "nodes_potential.csv")
    rows = []
    try:
        import pandas as pd  # type: ignore
        if os.path.exists(path_parq):
            df = pd.read_parquet(path_parq)
            rows = list(df[["geoid", "pi"]].itertuples(index=False, name=None))
        elif os.path.exists(path_csv):
            rows = []
            import csv
            with open(path_csv, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    try:
                        rows.append((r["geoid"], float(r["pi"])))
                    except Exception:
                        pass
    except Exception:
        # CSV only
        if os.path.exists(path_csv):
            import csv
            with open(path_csv, "r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    try:
                        rows.append((r["geoid"], float(r["pi"])))
                    except Exception:
                        pass
    return rows


def plot_potential_box(diag_dir: str, figs_dir: str) -> bool:
    plt = _try_import_matplotlib()
    if plt is None:
        return False
    data = _read_nodes_potential(diag_dir)
    if not data:
        return False
    buckets = defaultdict(list)
    for g, pi in data:
        buckets[str(g)[:5]].append(pi)
    labels = sorted(k for k, v in buckets.items() if v)
    series = [buckets[k] for k in labels]
    os.makedirs(figs_dir, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.boxplot(series, labels=labels, showfliers=False)
    plt.xticks(rotation=45)
    plt.title("Potential by county")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "pi_box_by_county.png"), dpi=150)
    plt.close()
    return True


def plot_r2_eta(diag_dir: str, figs_dir: str) -> bool:
    plt = _try_import_matplotlib()
    if plt is None:
        return False
    summ_path = os.path.join(diag_dir, "summary.json")
    if not os.path.exists(summ_path):
        return False
    with open(summ_path, "r", encoding="utf-8") as f:
        summ = json.load(f)
    os.makedirs(figs_dir, exist_ok=True)
    plt.figure(figsize=(4, 3))
    plt.bar(["R2", "eta"], [summ.get("R2", 0.0), summ.get("eta", 0.0)])
    plt.title("R2 and non-reciprocity")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "r2_eta.png"), dpi=150)
    plt.close()
    return True


def plot_locality_curve(diag_dir: str, figs_dir: str) -> bool:
    plt = _try_import_matplotlib()
    if plt is None:
        return False
    path = os.path.join(diag_dir, "locality_report.json")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        curve = json.load(f)
    xs = [c.get("r0") for c in curve]
    ys = [c.get("R2") for c in curve]
    os.makedirs(figs_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("distance threshold r0 (km)")
    plt.ylabel("R^2 potential")
    plt.title("Locality curve")
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "fig_locality_curve.png"), dpi=150)
    plt.close()
    return True


def plot_cycle_hotspots_fig(
    diag_dir: str,
    figs_dir: str,
    nodes_geo: str = os.path.join("project", "data", "geo", "tracts.geojson"),
    topk: int = 2000,
):
    edges_path = os.path.join(diag_dir, "edges_decomp_glm.csv")
    if not os.path.exists(edges_path):
        return False
    try:
        top_edges = select_cycle_hotspots(edges_path, k=topk)
    except Exception as exc:
        print(f"Cycle hotspot selection failed: {exc}")
        return False
    out_topk = os.path.join(diag_dir, "edges_topk_glm.csv")
    try:
        if hasattr(top_edges, "to_csv"):
            top_edges.to_csv(out_topk, index=False)
        else:
            io_mod.write_csv(out_topk, top_edges, header=["u", "v", "strength", "dist_km"])
    except Exception:
        pass
    fig_path = os.path.join(figs_dir, "fig_cycles_hotspots.png")
    return plot_cycle_hotspots(top_edges, nodes_geo, fig_path)


def _try_plot_kappa(diag_dir: str, figs_dir: str):
    json_path = os.path.join(diag_dir, "pde_kappa.json")
    if not os.path.exists(json_path):
        return False
    residual = os.path.join("project", "data", "processed", "od_residual_glm.parquet")
    nodes = os.path.join(diag_dir, "nodes_potential_glm.csv")
    residual_csv = os.path.splitext(residual)[0] + ".csv"
    residual_avail = os.path.exists(residual) or os.path.exists(residual_csv)
    if not os.path.exists(nodes) or not residual_avail:
        return False
    out_png = os.path.join(figs_dir, "fig_kappa_scatter.png")
    try:
        pde_fit.fit_kappa(
            residual,
            nodes,
            None,
            json_path,
            out_png,
        )
        return os.path.exists(out_png)
    except Exception as exc:
        print(f"Kappa scatter skipped: {exc}")
        return False


def _try_plot_diffusion(diag_dir: str, figs_dir: str):
    json_path = os.path.join(diag_dir, "pde_diffusion.json")
    if not os.path.exists(json_path):
        return False
    residual_edges = os.path.join(diag_dir, "edges_decomp_glm.csv")
    if not os.path.exists(residual_edges):
        return False
    out_png = os.path.join(figs_dir, "fig_diffusion_scatter.png")
    try:
        pde_fit.fit_diffusion(
            residual_edges,
            None,
            json_path,
            out_png,
        )
        return os.path.exists(out_png)
    except Exception as exc:
        print(f"Diffusion scatter skipped: {exc}")
        return False


def _try_plot_interface(diag_dir: str, figs_dir: str):
    json_path = os.path.join(diag_dir, "pde_interface.json")
    residual_edges = os.path.join(diag_dir, "edges_decomp_glm.csv")
    geo_path = os.path.join("project", "data", "geo", "tracts.geojson")
    if not os.path.exists(json_path) or not os.path.exists(residual_edges) or not os.path.exists(geo_path):
        return False
    out_png = os.path.join(figs_dir, "fig_interface_scatter.png")
    try:
        pde_fit.fit_interface(
            residual_edges,
            geo_path,
            None,
            json_path,
            out_png,
        )
        return os.path.exists(out_png)
    except Exception as exc:
        print(f"Interface scatter skipped: {exc}")
        return False


def export_all(
    diagnostics_dir: str = os.path.join("project", "results", "diagnostics"),
    figures_dir: str = os.path.join("project", "results", "figures"),
):
    """Export all figures. Returns a dict of figure->bool for success.

    Example (in notebook):
        from project.src.export_figs import export_all
        export_all()
    """
    out = {
        "pi_box_by_county": plot_potential_box(diagnostics_dir, figures_dir),
        "r2_eta": plot_r2_eta(diagnostics_dir, figures_dir),
        "locality_curve": plot_locality_curve(diagnostics_dir, figures_dir),
        "cycle_hotspots": plot_cycle_hotspots_fig(diagnostics_dir, figures_dir),
    }
    out.update(
        {
            "kappa_scatter": _try_plot_kappa(diagnostics_dir, figures_dir),
            "diffusion_scatter": _try_plot_diffusion(diagnostics_dir, figures_dir),
            "interface_scatter": _try_plot_interface(diagnostics_dir, figures_dir),
        }
    )
    return out
