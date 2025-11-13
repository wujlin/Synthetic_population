import math
from typing import Dict, Iterable, List, Tuple, Optional


def bin_by_distance(
    edge_rows: Iterable[Tuple[str, str, float, float, float, float, Optional[float]]],
    bins: int = 10,
):
    """Edge rows: (u,v,w,a,pred,resid,distance_km). Returns list of bins with stats.
    Computes energy fractions per bin.
    """
    rows = [r for r in edge_rows if r[6] is not None]
    if not rows:
        return []
    ds = sorted(r[6] for r in rows if r[6] is not None)
    if not ds:
        return []
    # Quantile binning
    qs = [ds[int(len(ds) * k / bins)] for k in range(bins)] + [ds[-1]]
    # Deduplicate boundaries
    bounds = [qs[0]]
    for x in qs[1:]:
        if x > bounds[-1]:
            bounds.append(x)
    # Build bins
    out = []
    for b in range(len(bounds) - 1):
        lo = bounds[b]
        hi = bounds[b + 1]
        Et = Eg = Er = 0.0
        cnt = 0
        for (u, v, w, a, pred, resid, d) in rows:
            if d is None:
                continue
            if (d >= lo and d <= hi) or (b == 0 and d <= hi):
                Et += w * a * a
                Eg += w * pred * pred
                Er += w * resid * resid
                cnt += 1
        frac_g = (Eg / Et) if Et > 0 else 0.0
        frac_r = (Er / Et) if Et > 0 else 0.0
        out.append((b, lo, hi, cnt, Et, Eg, Er, frac_g, frac_r))
    return out


def run_locality_curve(
    residual_rows_with_dist,
    radii_km,
    hodge_runner,
):
    """Run potential fit for subsets with distance <= r0.
    residual_rows_with_dist: iterator of (work,home,S000,log_resid,dist_km)
    hodge_runner: function(residual_rows)->(phi, summary, diags)
    Returns list of dicts with r0 and R2.
    """
    rows = list(residual_rows_with_dist)
    out = []
    for r0 in radii_km:
        sub = [(w,h,s,lr) for (w,h,s,lr,d) in rows if d is not None and d <= r0]
        if not sub:
            out.append({"r0": r0, "R2": 0.0, "edges": 0})
            continue
        _phi, summary, _diags = hodge_runner(sub)
        out.append({"r0": r0, "R2": summary.get("R2", 0.0), "edges": summary.get("edges", 0)})
    return out


def plot_locality_curve(curve_rows: List[Tuple], out_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    xs = [(lo + hi) / 2.0 for (_b, lo, hi, *_rest) in curve_rows]
    frac_g = [r[-2] for r in curve_rows]
    frac_r = [r[-1] for r in curve_rows]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, frac_g, label="gradient fraction")
    plt.plot(xs, frac_r, label="residual fraction")
    plt.xlabel("distance (km)")
    plt.ylabel("energy fraction")
    plt.title("Locality curve (Hodge energy fractions)")
    plt.legend()
    plt.tight_layout()
    try:
        import os
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        return True
    except Exception:
        return False
