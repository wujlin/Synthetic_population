import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .node import schema as node_schema  # type: ignore

EARTH_RADIUS_KM = 6371.0
SEMCOG_COUNTIES = {"26163", "26125", "26099", "26161", "26115", "26093", "26147"}


def load_node_panel() -> pd.DataFrame:
    """Load node-level panel with geometry and rhoH signals."""
    candidates = [
        "project/data/node/node_signals_geo.parquet",
        "project/data/node/panel_node_geo.parquet",
    ]
    for cand in candidates:
        path = Path(cand)
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        try:
            df = node_schema.normalize_node_signals(df)
        except Exception as exc:
            print(f"Skipping {cand}: {exc}")
            continue
        df = df.dropna(subset=["lat", "lon"])
        if SEMCOG_COUNTIES:
            df = df[df["county"].isin(SEMCOG_COUNTIES)]
        if df.empty:
            continue
        print(f"Loaded node panel from {cand} with {len(df)} nodes.")
        return df.reset_index(drop=True)
    raise FileNotFoundError("Missing node panel with geo columns; run Phase 0/1 to build it.")


def _haversine(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in kilometers."""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def build_graph(nodes_df: pd.DataFrame, r_star_km: float = 6.0, kernel: str = "gaussian"):
    """Return neighbor indices and weights for nodes within r*."""
    if "lat" not in nodes_df.columns or "lon" not in nodes_df.columns:
        raise ValueError("Node panel missing lat/lon columns.")
    coords = nodes_df[["lat", "lon"]].to_numpy(dtype=float)
    n = len(coords)
    neighbors: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    sigma = max(r_star_km / 2.0, 1e-6)
    for i in range(n):
        lat1, lon1 = coords[i]
        dists = _haversine(lat1, lon1, coords[:, 0], coords[:, 1])
        mask = (dists <= r_star_km) & (dists > 0)
        nbr_idx = np.where(mask)[0]
        if nbr_idx.size == 0:
            neighbors.append(np.zeros(0, dtype=int))
            weights.append(np.zeros(0, dtype=float))
            continue
        dist_vals = dists[nbr_idx]
        if kernel == "uniform":
            wij = np.ones_like(dist_vals)
        else:
            wij = np.exp(-(dist_vals ** 2) / (2 * sigma ** 2))
        neighbors.append(nbr_idx.astype(int))
        weights.append(wij.astype(float))
    return neighbors, weights


def gaussian_smooth_from_graph(values: np.ndarray, neighbors, weights, include_self: bool = True) -> np.ndarray:
    """Approximate Gaussian smoothing via weighted neighbor averages."""
    arr = np.asarray(values, dtype=float)
    smoothed = arr.copy()
    for i, nbrs in enumerate(neighbors):
        if nbrs.size == 0:
            continue
        wij = weights[i]
        denom = np.sum(wij)
        if include_self:
            num = arr[i] + np.dot(wij, arr[nbrs])
            denom = 1.0 + denom
        else:
            num = np.dot(wij, arr[nbrs])
        if denom > 0:
            smoothed[i] = num / denom
    return smoothed


def _lap(phi: np.ndarray, neighbors, weights) -> np.ndarray:
    Lphi = np.zeros_like(phi)
    for i, nbrs in enumerate(neighbors):
        if nbrs.size == 0:
            continue
        wij = weights[i]
        diff = phi[nbrs] - phi[i]
        Lphi[i] = np.dot(wij, diff)
    return Lphi


def _bilap(Lphi: np.ndarray, neighbors, weights) -> np.ndarray:
    L2 = np.zeros_like(Lphi)
    for i, nbrs in enumerate(neighbors):
        if nbrs.size == 0:
            continue
        wij = weights[i]
        diff = Lphi[nbrs] - Lphi[i]
        L2[i] = np.dot(wij, diff)
    return L2


def _taxis(phi: np.ndarray, U: np.ndarray, neighbors, weights) -> np.ndarray:
    Phi = np.zeros_like(phi)
    for i, nbrs in enumerate(neighbors):
        if nbrs.size == 0:
            continue
        wij = weights[i]
        ugrad = U[nbrs] - U[i]
        phi_bar = 0.5 * (phi[i] + phi[nbrs])
        Phi[i] = np.dot(wij, phi_bar * ugrad)
    return Phi


def make_columns(
    phi_t0: np.ndarray,
    neighbors,
    weights,
    U_dict: Optional[Dict[str, np.ndarray]] = None,
    include_bilaplacian: bool = False,
) -> Dict[str, np.ndarray]:
    phi = np.asarray(phi_t0, dtype=float)
    cols: Dict[str, np.ndarray] = {"growth": phi}
    Lphi = _lap(phi, neighbors, weights)
    cols["diff"] = Lphi
    if include_bilaplacian:
        cols["bilap"] = _bilap(Lphi, neighbors, weights)
    if U_dict:
        for name, U in U_dict.items():
            cols[f"taxis:{name}"] = _taxis(phi, np.asarray(U, dtype=float), neighbors, weights)
    return cols


def target_delta(phi_t0: np.ndarray, phi_t1: np.ndarray, dt: float = 1.0) -> np.ndarray:
    if dt == 0:
        raise ValueError("dt must be non-zero")
    return (np.asarray(phi_t1, dtype=float) - np.asarray(phi_t0, dtype=float)) / dt


def _stack_columns(X_dict: Dict[str, np.ndarray], names: Sequence[str]) -> Tuple[List[str], np.ndarray]:
    cols = []
    used = []
    for name in names:
        if name not in X_dict:
            raise KeyError(f"Column {name} missing from feature dictionary.")
        arr = np.asarray(X_dict[name], dtype=float)
        cols.append(arr)
        used.append(name)
    if not cols:
        raise ValueError("No columns selected for layer.")
    return used, np.column_stack(cols)


def _run_loco(
    column_names: Sequence[str],
    X_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    county: np.ndarray,
    alpha_grid: Sequence[float],
    fixed_alpha: Optional[float] = None,
    capture_coefs: bool = False,
) -> Dict:
    names, X = _stack_columns(X_dict, column_names)
    y_arr = np.asarray(y, dtype=float)
    county_arr = np.asarray(county)
    unique_counties = np.unique(county_arr)
    if fixed_alpha is not None:
        grid = [float(fixed_alpha)]
    else:
        grid = [float(a) for a in alpha_grid]
    if capture_coefs and len(grid) != 1:
        raise ValueError("capture_coefs=True requires a single alpha.")
    fold_scores = {
        alpha: {"r2": [], "rmse": [], "per_county": {}}
        for alpha in grid
    }
    fold_coefs: List[Dict] = []
    for county_key in unique_counties:
        train_mask = county_arr != county_key
        val_mask = ~train_mask
        if not train_mask.any() or not val_mask.any():
            continue
        X_train = X[train_mask]
        X_val = X[val_mask]
        y_train = y_arr[train_mask]
        y_val = y_arr[val_mask]
        x_mean = X_train.mean(axis=0)
        x_std = X_train.std(axis=0)
        x_std[x_std == 0] = 1.0
        y_mean = y_train.mean()
        y_std = y_train.std()
        if y_std == 0:
            y_std = 1.0
        X_train_z = (X_train - x_mean) / x_std
        X_val_z = (X_val - x_mean) / x_std
        y_train_z = (y_train - y_mean) / y_std
        y_val_z = (y_val - y_mean) / y_std
        for alpha in grid:
            model = Ridge(alpha=alpha, fit_intercept=False)
            model.fit(X_train_z, y_train_z)
            pred_val_z = model.predict(X_val_z)
            y_pred = y_mean + y_std * pred_val_z
            resid = y_val - y_pred
            denom = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = 0.0 if denom <= 0 else 1.0 - np.sum(resid ** 2) / denom
            rmse = math.sqrt(np.mean(resid ** 2))
            fold_scores[alpha]["r2"].append(float(r2))
            fold_scores[alpha]["rmse"].append(float(rmse))
            fold_scores[alpha]["per_county"][str(county_key)] = {"r2": float(r2), "rmse": float(rmse)}
            if capture_coefs:
                coef_raw = (y_std / x_std) * model.coef_
                intercept = y_mean - float(np.dot(coef_raw, x_mean))
                fold_coefs.append(
                    {
                        "county": str(county_key),
                        "coef": coef_raw.tolist(),
                        "intercept": intercept,
                        "names": names,
                    }
                )
    if not any(len(v["r2"]) for v in fold_scores.values()):
        raise RuntimeError("LOCO evaluation produced no folds; check county labels.")
    metrics = {}
    for alpha, vals in fold_scores.items():
        mean_r2 = float(np.mean(vals["r2"])) if vals["r2"] else float("nan")
        se_r2 = float(np.std(vals["r2"], ddof=0) / math.sqrt(len(vals["r2"]))) if len(vals["r2"]) else float("nan")
        mean_rmse = float(np.mean(vals["rmse"])) if vals["rmse"] else float("nan")
        se_rmse = float(np.std(vals["rmse"], ddof=0) / math.sqrt(len(vals["rmse"]))) if len(vals["rmse"]) else float("nan")
        metrics[alpha] = {
            "mean_r2": mean_r2,
            "se_r2": se_r2,
            "mean_rmse": mean_rmse,
            "se_rmse": se_rmse,
            "per_county": vals["per_county"],
        }
    best_alpha = max(
        metrics.keys(),
        key=lambda a: (metrics[a]["mean_r2"], -a),
    )
    result = {
        "columns": names,
        "best_alpha": float(best_alpha),
        "metrics": metrics,
        "fold_coefs": fold_coefs if capture_coefs else None,
    }
    return result


def _summarize_layer(layer_name: str, run_result: Dict) -> Dict:
    alpha = run_result["best_alpha"]
    metric = run_result["metrics"][alpha]
    return {
        "name": layer_name,
        "columns": run_result["columns"],
        "alpha": alpha,
        "r2": {"mean": metric["mean_r2"], "se": metric["se_r2"]},
        "rmse": {"mean": metric["mean_rmse"], "se": metric["se_rmse"]},
        "per_county": metric["per_county"],
    }


def _aggregate_coef_stats(fold_coefs: List[Dict]) -> List[Dict]:
    if not fold_coefs:
        return []
    names = fold_coefs[0]["names"]
    coef_matrix = np.array([entry["coef"] for entry in fold_coefs], dtype=float)
    intercepts = np.array([entry["intercept"] for entry in fold_coefs], dtype=float)
    stats = []
    for idx, name in enumerate(names):
        vals = coef_matrix[:, idx]
        mean = float(np.mean(vals))
        se = float(np.std(vals, ddof=0) / math.sqrt(len(vals)))
        sign_ref = np.sign(mean)
        if sign_ref == 0:
            stability = float(np.mean(np.sign(vals) == 0))
        else:
            stability = float(np.mean(np.sign(vals) == sign_ref))
        stats.append(
            {
                "name": name,
                "mean": mean,
                "se": se,
                "sign_stability": stability,
            }
        )
    # Append intercept stats
    mean_int = float(np.mean(intercepts))
    se_int = float(np.std(intercepts, ddof=0) / math.sqrt(len(intercepts)))
    if mean_int == 0:
        stability_int = float(np.mean(np.sign(intercepts) == 0))
    else:
        stability_int = float(np.mean(np.sign(intercepts) == np.sign(mean_int)))
    stats.append(
        {
            "name": "intercept",
            "mean": mean_int,
            "se": se_int,
            "sign_stability": stability_int,
        }
    )
    return stats


def fit_with_loco(
    X_dict: Dict[str, np.ndarray],
    y: np.ndarray,
    county: np.ndarray,
    model_layers: Sequence[str] = ("M0", "M1", "M2", "M3"),
    alpha_grid: Optional[Sequence[float]] = None,
    one_se: bool = True,
    keep_top_k_taxis: int = 2,
):
    """Fit Ridge with LOCO CV and one-SE layer selection."""
    if alpha_grid is None:
        alpha_grid = np.logspace(-6, -1, 10)
    available_layers: List[str] = []
    layer_details: Dict[str, Dict] = {}
    y_arr = np.asarray(y, dtype=float)
    county_arr = np.asarray(county)
    taxis_cols = sorted([k for k in X_dict.keys() if k.startswith("taxis:")])
    layer_order = [layer for layer in model_layers if layer in {"M0", "M1", "M2", "M3"}]

    # M0
    if "growth" not in X_dict:
        raise KeyError("Feature 'growth' missing; ensure make_columns ran.")
    if "M0" in layer_order:
        res = _run_loco(["growth"], X_dict, y_arr, county_arr, alpha_grid)
        layer_details["M0"] = res
        available_layers.append("M0")

    # M1
    base_cols = ["growth"]
    if "diff" in X_dict:
        base_cols.append("diff")
    if "M1" in layer_order:
        res = _run_loco(base_cols, X_dict, y_arr, county_arr, alpha_grid)
        layer_details["M1"] = res
        available_layers.append("M1")

    # Select taxis
    selected_taxis: List[str] = []
    if "M2" in layer_order and taxis_cols:
        base_res = layer_details.get("M1") or layer_details.get("M0")
        if base_res is None:
            raise RuntimeError("M2 requires base layer results; run M0/M1 first.")
        base_mean = base_res["metrics"][base_res["best_alpha"]]["mean_r2"]
        taxi_scores = []
        for col in taxis_cols:
            cols = base_cols + [col]
            res = _run_loco(cols, X_dict, y_arr, county_arr, alpha_grid)
            mean_r2 = res["metrics"][res["best_alpha"]]["mean_r2"]
            taxi_scores.append((mean_r2, col, res))
        taxi_scores.sort(reverse=True, key=lambda t: t[0])
        for mean_r2, col, res in taxi_scores:
            if len(selected_taxis) >= keep_top_k_taxis:
                break
            if mean_r2 >= base_mean:
                selected_taxis.append(col)
        if selected_taxis:
            cols = base_cols + selected_taxis
            res = _run_loco(cols, X_dict, y_arr, county_arr, alpha_grid)
            layer_details["M2"] = res
            available_layers.append("M2")

    # M3
    if "M3" in layer_order and "bilap" in X_dict:
        base_for_m3 = layer_details.get("M2") or layer_details.get("M1") or layer_details.get("M0")
        if base_for_m3:
            prev_cols = base_for_m3["columns"]
            cols = list(prev_cols) + ["bilap"]
            # ensure uniqueness order
            seen = set()
            cols_unique = []
            for c in cols:
                if c not in seen:
                    cols_unique.append(c)
                    seen.add(c)
            res = _run_loco(cols_unique, X_dict, y_arr, county_arr, alpha_grid)
            layer_details["M3"] = res
            available_layers.append("M3")

    if not available_layers:
        raise RuntimeError("No layers evaluated; check inputs.")

    summaries = [(_summarize_layer(layer, layer_details[layer]), layer) for layer in available_layers]
    layer_order_summary = [s[0] for s in summaries]

    # Determine best layer and one-SE selection
    best_perf_layer = max(
        available_layers,
        key=lambda name: layer_details[name]["metrics"][layer_details[name]["best_alpha"]]["mean_r2"],
    )
    best_metric = layer_details[best_perf_layer]["metrics"][layer_details[best_perf_layer]["best_alpha"]]
    threshold = best_metric["mean_r2"]
    if one_se and not math.isnan(best_metric["se_r2"]):
        threshold = best_metric["mean_r2"] - best_metric["se_r2"]
    chosen_layer = None
    for layer in available_layers:
        metric = layer_details[layer]["metrics"][layer_details[layer]["best_alpha"]]
        if metric["mean_r2"] >= threshold:
            chosen_layer = layer
            break
    if chosen_layer is None:
        chosen_layer = best_perf_layer
    best_alpha = layer_details[chosen_layer]["best_alpha"]
    capture = _run_loco(
        layer_details[chosen_layer]["columns"],
        X_dict,
        y_arr,
        county_arr,
        alpha_grid,
        fixed_alpha=best_alpha,
        capture_coefs=True,
    )
    coef_stats = _aggregate_coef_stats(capture["fold_coefs"])
    cv_report = {
        "layers": layer_order_summary,
        "selection": {
            "best_layer": best_perf_layer,
            "chosen_layer": chosen_layer,
            "threshold_r2": threshold,
            "one_se": one_se,
        },
        "taxis_selected": selected_taxis,
    }
    return chosen_layer, {
        "layer": chosen_layer,
        "alpha": best_alpha,
        "coefficients": coef_stats,
    }, cv_report


def save_results(out_dir: str, best_layer: str, coefs: Dict, cv_report: Dict, meta: Dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    node_path = os.path.join(out_dir, "pde_node.json")
    report_path = os.path.join(out_dir, "cv_report.json")
    payload = {"best_layer": best_layer, "detail": coefs, "meta": meta}
    with open(node_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(cv_report, f, indent=2, ensure_ascii=False)

    try:
        import matplotlib.pyplot as plt  # type: ignore

        # Figure 1: Ablation
        layers = cv_report["layers"]
        names = [layer["name"] for layer in layers]
        means = [layer["r2"]["mean"] for layer in layers]
        ses = [layer["r2"]["se"] for layer in layers]
        plt.figure(figsize=(6, 4))
        x = np.arange(len(names))
        plt.bar(x, means, yerr=ses, capsize=4)
        plt.xticks(x, names)
        plt.ylabel("OOS R²")
        plt.title(f"PDE ablation (one-SE choice: {cv_report['selection']['chosen_layer']})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fig_pde_ablation.png"), dpi=150)
        plt.close()

        # Figure 2: Coefficients
        coef_rows = [c for c in coefs["coefficients"] if c["name"] != "intercept"]
        if coef_rows:
            coef_rows_sorted = sorted(coef_rows, key=lambda r: r["name"])
            labels = [r["name"] for r in coef_rows_sorted]
            vals = [r["mean"] for r in coef_rows_sorted]
            errs = [r["se"] for r in coef_rows_sorted]
            plt.figure(figsize=(6, max(3, 0.3 * len(labels))))
            y_pos = np.arange(len(labels))
            plt.barh(y_pos, vals, xerr=errs, capsize=4)
            plt.yticks(y_pos, labels)
            plt.xlabel("Coefficient (per raw unit)")
            plt.title("PDE coefficients (mean ± SE)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_pde_coeffs.png"), dpi=150)
            plt.close()
        else:
            plt.figure(figsize=(4, 3))
            plt.text(0.5, 0.5, "No coefficients retained", ha="center", va="center")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "fig_pde_coeffs.png"), dpi=150)
            plt.close()
    except Exception:
        print("Matplotlib not available; skipped figure export.")
