import os
import json
from collections import defaultdict


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
    }
    return out

