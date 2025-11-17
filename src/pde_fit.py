"""
PDE-style regression diagnostics for structural terms (kappa, diffusion, etc.).

This file focuses on estimating the potential-strength coefficient kappa using
GLM residual outputs combined with node potentials and marginal densities.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple, Union

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

from . import io as io_mod
from . import adjacency as adjacency_mod


def _load_dataframe(path: str, columns: Optional[List[str]] = None):
    if pd is None:
        return None
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".parquet"):
            df = pd.read_parquet(path, columns=columns)
        else:
            df = pd.read_csv(path, usecols=columns)
        return df
    except Exception:
        return None


def _load_residual_glm(path: str):
    df = _load_dataframe(path)
    if df is not None:
        return df
    csv_path = path if path.endswith(".csv") else path.rsplit(".", 1)[0] + ".csv"
    rows = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GLM residual file not found: {path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _load_node_potential(path: str):
    df = _load_dataframe(path, columns=["geoid", "pi"])
    if df is not None:
        return {_normalize_geoid(g): float(pi) for g, pi in df.itertuples(index=False, name=None)}
    csv_path = path if path.endswith(".csv") else path.rsplit(".", 1)[0] + ".csv"
    mapping = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    mapping[_normalize_geoid(row["geoid"])] = float(row["pi"])
                except Exception:
                    continue
    return mapping


def _load_density(path: Optional[str]) -> Dict[str, float]:
    if not path:
        # Default to CSV marginals to avoid stale parquet mismatches
        path = os.path.join("project", "data", "processed", "nodes_basic.csv")
    df = _load_dataframe(path)
    if df is not None:
        return {_normalize_geoid(geoid): float(rho) for geoid, rho in zip(df["geoid"], df["R"])}
    csv_path = path if path.endswith(".csv") else path.rsplit(".", 1)[0] + ".csv"
    dens = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    dens[_normalize_geoid(row["geoid"])] = float(row.get("R") or row.get("rho") or 0.0)
                except Exception:
                    continue
    if dens:
        return dens
    raise FileNotFoundError(
        "RAC/marginal file not found; provide --rac or run `project.cli marginals`."
    )


def _ols_wls(X: List[List[float]], y: List[float], weights: List[float], ridge: float = 0.0):
    """Solve beta via normal equations with optional weights and ridge."""
    if not X or not y:
        return None, None
    n = len(y)
    m = len(X[0])
    # Weighted design: multiply rows by sqrt(weight)
    XtW = [[0.0 for _ in range(m)] for __ in range(m)]
    XtWy = [0.0 for _ in range(m)]
    total_w = 0.0
    for i in range(n):
        w = max(weights[i], 1e-12)
        total_w += w
        row = X[i]
        for a in range(m):
            for b in range(m):
                XtW[a][b] += w * row[a] * row[b]
        for a in range(m):
            XtWy[a] += w * row[a] * y[i]

    # Ridge stabilizer
    if ridge > 0.0:
        for i in range(m):
            XtW[i][i] += ridge

    # Invert XtW via Gaussian elimination (m is small: <=3 columns)
    beta = _solve_linear_system(XtW, XtWy)
    if beta is None:
        return None, None
    residuals = []
    rss = 0.0
    tss = 0.0
    y_mean = sum(y) / n if n else 0.0
    for i in range(n):
        pred = sum(beta[j] * X[i][j] for j in range(m))
        resid = y[i] - pred
        residuals.append(resid)
        rss += weights[i] * (resid ** 2)
        tss += weights[i] * ((y[i] - y_mean) ** 2)
    R2 = 1.0 - (rss / tss) if tss > 0 else 0.0
    return beta, {
        "R2": R2,
        "rss": rss,
        "tss": tss,
        "n": n,
        "weights_sum": total_w,
        "residuals": residuals,
    }


def _solve_linear_system(A: List[List[float]], b: List[float]):
    """Basic Gaussian elimination for small dense systems."""
    m = len(b)
    # Form augmented matrix
    aug = [row[:] + [b_val] for row, b_val in zip(A, b)]
    for col in range(m):
        # Pivot
        pivot = col
        while pivot < m and abs(aug[pivot][col]) < 1e-12:
            pivot += 1
        if pivot == m:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        # Normalize pivot row
        pivot_val = aug[col][col]
        scale = pivot_val or 1.0
        aug[col] = [val / scale for val in aug[col]]
        # Eliminate other rows
        for row in range(m):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][i] - factor * aug[col][i] for i in range(m + 1)]
    return [aug[i][-1] for i in range(m)]


def _compute_t_stats(X: List[List[float]], residuals: List[float], weights: List[float], beta: List[float]):
    n = len(residuals)
    m = len(beta)
    if n <= m:
        return [float("nan")] * m
    # Build XtWX inverse
    XtWX = [[0.0 for _ in range(m)] for __ in range(m)]
    for i in range(n):
        w = weights[i]
        row = X[i]
        for a in range(m):
            for b in range(m):
                XtWX[a][b] += w * row[a] * row[b]
    XtWX_inv = _invert_matrix(XtWX)
    if XtWX_inv is None:
        return [float("nan")] * m
    rss = sum(weights[i] * (residuals[i] ** 2) for i in range(n))
    dof = max(n - m, 1)
    sigma2 = rss / dof
    t_stats = []
    for j in range(m):
        var_j = sigma2 * XtWX_inv[j][j]
        if var_j <= 0:
            t_stats.append(float("nan"))
        else:
            t_stats.append(beta[j] / math.sqrt(var_j))
    return t_stats


def _invert_matrix(M: List[List[float]]):
    n = len(M)
    aug = [row[:] + [float(i == j) for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        pivot = col
        while pivot < n and abs(aug[pivot][col]) < 1e-12:
            pivot += 1
        if pivot == n:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        scale = pivot_val or 1.0
        aug[col] = [val / scale for val in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [aug[row][i] - factor * aug[col][i] for i in range(2 * n)]
    inv = [row[n:] for row in aug]
    return inv


def _plot_scatter_generic(
    x_vals: List[float],
    y_vals: List[float],
    preds: List[float],
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
):
    if plt is None:
        print(f"matplotlib unavailable; skip figure {os.path.basename(out_path)}.")
        return False
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(5.5, 4.0))
    plt.scatter(x_vals, y_vals, s=6, alpha=0.3, label="observed", color="#4c72b0")
    plt.scatter(x_vals, preds, s=8, alpha=0.8, label="fitted", color="#dd8452")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def fit_kappa(
    od_residual_glm_path: str,
    nodes_potential_path: str,
    rac_or_marginals_path: Optional[str],
    out_json: str,
    out_png: str,
):
    """Estimate kappa coefficient linking density/potential differential to flow skew."""
    df = _load_residual_glm(od_residual_glm_path)
    node_pi = _load_node_potential(nodes_potential_path)
    rho = _load_density(rac_or_marginals_path)
    if not node_pi:
        raise RuntimeError("Node potential file is empty; run potential_hodge_glm first.")

    pairs: Dict[Tuple[str, str], Dict[str, float]] = {}
    iterable = df.to_dict("records") if hasattr(df, "to_dict") else df
    for row in iterable:
        u = str(row.get("work") or row.get("u") or row.get("geoid_i") or "")
        v = str(row.get("home") or row.get("v") or row.get("geoid_j") or "")
        if not u or not v:
            continue
        F = row.get("S000")
        mu = row.get("mu_hat") or row.get("E_ij")
        if F is None or mu is None:
            continue
        try:
            F = float(F)
            mu = float(mu)
        except Exception:
            continue
        pairs[(u, v)] = {"F": F, "mu": mu}

    total_pairs = 0
    data = []
    zero_density = 0
    zero_c = 0
    zero_dpi = 0
    for (u, v), payload in pairs.items():
        if (v, u) not in pairs:
            continue
        mu_uv = payload["mu"]
        mu_vu = pairs[(v, u)]["mu"]
        F_uv = payload["F"]
        F_vu = pairs[(v, u)]["F"]
        tilde = (F_uv - F_vu) - (mu_uv - mu_vu)
        key_u = _normalize_geoid(u)
        key_v = _normalize_geoid(v)
        rho_u = rho.get(key_u, 0.0)
        rho_v = rho.get(key_v, 0.0)
        rho_avg = 0.5 * (rho_u + rho_v)
        c_ij = mu_uv + mu_vu
        pi_u = node_pi.get(key_u, 0.0)
        pi_v = node_pi.get(key_v, 0.0)
        delta_pi = pi_v - pi_u
        total_pairs += 1
        if (
            abs(delta_pi) <= 1e-8
            or c_ij <= 0.0
            or rho_avg <= 0.0
        ):
            if abs(delta_pi) <= 1e-8:
                zero_dpi += 1
            if c_ij <= 0.0:
                zero_c += 1
            if rho_avg <= 0.0:
                zero_density += 1
            continue
        data.append((tilde, rho_avg, c_ij, delta_pi))

    if not data:
        raise RuntimeError(
            "No usable paired flows found to estimate kappa "
            f"(pairs={total_pairs}, zero_delta_pi={zero_dpi}, nonpositive_c={zero_c}, nonpositive_rho={zero_density})."
        )
    X = []
    y = []
    weights = []
    feature_proxy = []
    for tilde, rho_avg, c_ij, delta_pi in data:
        X.append([rho_avg, c_ij, c_ij * delta_pi])
        y.append(tilde)
        w = max(c_ij, 1e-6)
        weights.append(w)
        feature_proxy.append(rho_avg * c_ij * delta_pi)

    beta, reg = _ols_wls(X, y, weights, ridge=1e-8)
    if beta is None or reg is None:
        raise RuntimeError("Failed to solve kappa regression (singular design).")
    t_stats = _compute_t_stats(X, reg["residuals"], weights, beta)
    preds = [sum(beta[j] * Xrow[j] for j in range(len(beta))) for Xrow in X]
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "kappa": beta[-1],
                "rho_coef": beta[0],
                "c_coef": beta[1],
                "kappa_t": t_stats[-1],
                "rho_t": t_stats[0],
                "c_t": t_stats[1],
                "R2": reg["R2"],
                "n_pairs": len(X),
                "n_pairs_total": total_pairs,
            },
            f,
            indent=2,
        )
    _plot_scatter_generic(
        feature_proxy,
        y,
        preds,
        out_png,
        "Kappa fit diagnostics",
        "rho*c*Delta pi (proxy)",
        "tilde N_ij",
    )


def _load_edge_residuals(source: Union[str, Iterable[Dict], Dict]):
    """Return iterable of dicts containing u,v,residual,weight from various inputs."""
    if source is None:
        raise FileNotFoundError("Edge residual source is required.")
    if isinstance(source, str):
        df = _load_dataframe(source)
        if df is not None:
            return df.to_dict("records")
        csv_path = source if source.endswith(".csv") else source.rsplit(".", 1)[0] + ".csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Residual edge file not found: {source}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    if isinstance(source, dict):
        return [source]
    if isinstance(source, list):
        return source
    if hasattr(source, "to_dict"):
        try:
            return source.to_dict("records")
        except Exception:
            pass
    return list(source)


def _parse_edge_row(row: Dict) -> Optional[Tuple[str, str, float, float]]:
    u = row.get("u") or row.get("work") or row.get("i") or row.get("source")
    v = row.get("v") or row.get("home") or row.get("j") or row.get("target")
    if u is None or v is None:
        return None
    u = str(u)
    v = str(v)
    value_fields = [
        "residual_after",
        "resid_after",
        "hat_N",
        "residual",
        "resid",
        "value",
    ]
    residual = None
    for key in value_fields:
        val = row.get(key)
        if val is not None:
            try:
                residual = float(val)
                break
            except Exception:
                continue
    if residual is None:
        return None
    weight = row.get("weight") or row.get("c_ij") or row.get("w") or row.get("mu_hat")
    try:
        w = float(weight) if weight is not None else 1.0
    except Exception:
        w = 1.0
    return (u, v, residual, max(w, 1e-6))


def _load_residual_rows(source: Union[str, Iterable[Dict], Dict]):
    rows_raw = _load_edge_residuals(source)
    parsed = []
    for row in rows_raw:
        parsed_row = _parse_edge_row(row)
        if parsed_row:
            parsed.append(parsed_row)
    if not parsed:
        raise RuntimeError("No valid residual rows found for diffusion/interface fit.")
    return parsed


def fit_diffusion(
    residual_source: Union[str, Iterable[Dict], Dict],
    rac_or_marginals_path: Optional[str],
    out_json: str,
    out_png: str,
):
    """Estimate diffusion coefficient D using directed residuals and rho gradients."""
    rows = _load_residual_rows(residual_source)
    rho = _load_density(rac_or_marginals_path)
    X = []
    y = []
    wts = []
    feature = []
    for u, v, resid, weight in rows:
        key_u = _normalize_geoid(u)
        key_v = _normalize_geoid(v)
        rho_u = rho.get(key_u, None)
        rho_v = rho.get(key_v, None)
        if rho_u is None or rho_v is None:
            continue
        delta = rho_v - rho_u
        if abs(delta) < 1e-12:
            continue
        X.append([-delta])  # expecting positive D for smoothing
        y.append(resid)
        wts.append(weight)
        feature.append(-delta)
    if not X:
        raise RuntimeError("Insufficient rho differences to estimate diffusion.")
    beta, reg = _ols_wls(X, y, wts)
    if beta is None or reg is None:
        raise RuntimeError("Diffusion regression failed (singular).")
    t_stats = _compute_t_stats(X, reg["residuals"], wts, beta)
    preds = [beta[0] * x[0] for x in X]
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    payload = {
        "diffusion_D": beta[0],
        "diffusion_t": t_stats[0],
        "R2": reg["R2"],
        "delta_R2": reg["R2"],
        "n_edges": len(X),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _plot_scatter_generic(
        feature,
        y,
        preds,
        out_png,
        "Diffusion fit diagnostics",
        "-Delta rho_ij",
        "hat N_ij (after kappa)",
    )


def fit_interface(
    residual_source: Union[str, Iterable[Dict], Dict],
    tracts_geojson: str,
    rac_or_marginals_path: Optional[str],
    out_json: str,
    out_png: str,
    knn: int = 6,
):
    """Estimate interface coefficient Gamma using Laplacian gradients."""
    rows = _load_residual_rows(residual_source)
    rho = _load_density(rac_or_marginals_path)
    edges_adj, laplacian = adjacency_mod.build_adjacency(tracts_geojson, k=knn)
    if not laplacian:
        raise RuntimeError("Adjacency build failed; check tracts GeoJSON.")
    lap_rho = {}
    for node, coeffs in laplacian.items():
        total = 0.0
        for nbr, val in coeffs.items():
            rho_val = rho.get(_normalize_geoid(nbr), 0.0)
            total += val * rho_val
        lap_rho[node] = total
    X = []
    y = []
    wts = []
    feature = []
    for u, v, resid, weight in rows:
        lap_u = lap_rho.get(_normalize_geoid(u))
        lap_v = lap_rho.get(_normalize_geoid(v))
        if lap_u is None or lap_v is None:
            continue
        delta = lap_v - lap_u
        if abs(delta) < 1e-12:
            continue
        X.append([-delta])
        y.append(resid)
        wts.append(weight)
        feature.append(-delta)
    if not X:
        raise RuntimeError("Insufficient Laplacian gradients for interface fit.")
    beta, reg = _ols_wls(X, y, wts)
    if beta is None or reg is None:
        raise RuntimeError("Interface regression failed (singular).")
    t_stats = _compute_t_stats(X, reg["residuals"], wts, beta)
    preds = [beta[0] * x[0] for x in X]
    payload = {
        "gamma": beta[0],
        "gamma_t": t_stats[0],
        "R2": reg["R2"],
        "delta_R2": reg["R2"],
        "n_edges": len(X),
        "adj_edges": len(edges_adj),
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _plot_scatter_generic(
        feature,
        y,
        preds,
        out_png,
        "Interface fit diagnostics",
        "-Delta(L rho)_ij",
        "hat N_ij (after diffusion)",
    )
def _normalize_geoid(val: Union[str, float, int]) -> str:
    s = "".join(ch for ch in str(val) if ch.isdigit())
    if len(s) >= 11:
        return s[:11]
    if s:
        return s.zfill(11)
    return str(val)
