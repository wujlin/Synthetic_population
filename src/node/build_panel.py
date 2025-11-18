import os
from typing import Dict, Iterable, List, Sequence


def _ensure_node_dirs() -> None:
    """Ensure node data/results directories exist."""
    os.makedirs(os.path.join("project", "data", "node"), exist_ok=True)
    os.makedirs(os.path.join("project", "results", "node"), exist_ok=True)


def _load_lodes_table(path: str, key_col: str, value_col: str) -> Dict[str, float]:
    """Aggregate a LODES CSV.gz by truncated GEOID.

    Parameters
    ----------
    path : str
        Path to CSV.gz file.
    key_col : str
        Column containing block-level geocode (e.g. h_geocode / w_geocode).
    value_col : str
        Count column to aggregate (e.g. C000).
    """
    import gzip
    import csv

    agg: Dict[str, float] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            geocode = row.get(key_col)
            if not geocode:
                continue
            tract = str(geocode)[:11]
            try:
                val = float(row.get(value_col, "0") or 0.0)
            except Exception:
                continue
            agg[tract] = agg.get(tract, 0.0) + val
    return agg


def _load_od_year(root: str, year: int) -> Iterable[tuple[str, str, float]]:
    """Yield (home_tract, work_tract, S000) for a given year from main+aux."""
    import gzip
    import csv

    paths: List[str] = []
    for basename in (f"mi_od_main_JT01_{year}.csv.gz", f"mi_od_aux_JT01_{year}.csv.gz"):
        cand = _resolve_lodes_path(root, "od", year, basename)
        if os.path.exists(cand):
            paths.append(cand)
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = row.get("h_geocode")
                w = row.get("w_geocode")
                if not h or not w:
                    continue
                home = str(h)[:11]
                work = str(w)[:11]
                try:
                    s = float(row.get("S000", "0") or 0.0)
                except Exception:
                    continue
                if s <= 0:
                    continue
                yield (home, work, s)


def _resolve_lodes_path(root: str, subdir: str, year: int, basename: str) -> str:
    """Resolve LODES path, preferring by_year/<subdir>/year but falling back to root/<subdir>.

    This handles setups where by_year entries are broken symlinks but flat files exist
    under e.g. `rac/mi_rac_S000_JT01_2020.csv.gz`.
    """
    by_year_path = os.path.join(root, "by_year", subdir, str(year), basename)
    flat_path = os.path.join(root, subdir, basename)
    if os.path.exists(by_year_path):
        return by_year_path
    return flat_path


def build_panel(lodes_root: str, years: Sequence[int] = (2020, 2021)) -> None:
    """Build node-level panel for rho_H, rho_W, In, Out across years.

    Outputs
    -------
    project/data/node/panel_node.parquet
    project/data/node/node_signals.parquet
    project/results/node/panel_checks.json
    """
    import json

    _ensure_node_dirs()
    years = list(years)

    # Containers: year -> geoid -> values
    rho_H_by_year: Dict[int, Dict[str, float]] = {}
    rho_W_by_year: Dict[int, Dict[str, float]] = {}
    In_by_year: Dict[int, Dict[str, float]] = {}
    Out_by_year: Dict[int, Dict[str, float]] = {}

    for year in years:
        # H: RAC
        rac_path = _resolve_lodes_path(
            lodes_root, "rac", year, f"mi_rac_S000_JT01_{year}.csv.gz"
        )
        rho_H_by_year[year] = (
            _load_lodes_table(rac_path, key_col="h_geocode", value_col="C000")
            if os.path.exists(rac_path)
            else {}
        )

        # W: WAC
        wac_path = _resolve_lodes_path(
            lodes_root, "wac", year, f"mi_wac_S000_JT01_{year}.csv.gz"
        )
        rho_W_by_year[year] = (
            _load_lodes_table(wac_path, key_col="w_geocode", value_col="C000")
            if os.path.exists(wac_path)
            else {}
        )

        # OD → In/Out
        In: Dict[str, float] = {}
        Out: Dict[str, float] = {}
        for home, work, s in _load_od_year(lodes_root, year):
            Out[home] = Out.get(home, 0.0) + s
            In[work] = In.get(work, 0.0) + s
        In_by_year[year] = In
        Out_by_year[year] = Out

    # Long panel
    panel_rows: List[tuple[str, int, float, float, float, float]] = []
    for year in years:
        rho_H = rho_H_by_year.get(year, {})
        rho_W = rho_W_by_year.get(year, {})
        In = In_by_year.get(year, {})
        Out = Out_by_year.get(year, {})
        all_geoids = set(rho_H) | set(rho_W) | set(In) | set(Out)
        for g in sorted(all_geoids):
            panel_rows.append(
                (
                    g,
                    int(year),
                    float(rho_H.get(g, 0.0)),
                    float(rho_W.get(g, 0.0)),
                    float(In.get(g, 0.0)),
                    float(Out.get(g, 0.0)),
                )
            )

    panel_path = os.path.join("project", "data", "node", "panel_node.parquet")
    try:
        import pandas as pd  # type: ignore

        df_panel = pd.DataFrame(
            panel_rows,
            columns=["geoid", "year", "rho_H", "rho_W", "In", "Out"],
        )
        df_panel.to_parquet(panel_path, index=False)
    except Exception:
        import csv

        csv_path = panel_path.rsplit(".", 1)[0] + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["geoid", "year", "rho_H", "rho_W", "In", "Out"])
            writer.writerows(panel_rows)
        panel_path = csv_path

    # Wide table for 2020/2021
    if len(years) >= 2 and 2020 in years and 2021 in years:
        y0, y1 = 2020, 2021
    else:
        years_sorted = sorted(years)
        if len(years_sorted) < 2:
            return
        y0, y1 = years_sorted[0], years_sorted[-1]

    rho_H0 = rho_H_by_year.get(y0, {})
    rho_H1 = rho_H_by_year.get(y1, {})
    rho_W0 = rho_W_by_year.get(y0, {})
    rho_W1 = rho_W_by_year.get(y1, {})
    In0 = In_by_year.get(y0, {})
    In1 = In_by_year.get(y1, {})
    Out0 = Out_by_year.get(y0, {})
    all_geoids = set(rho_H0) | set(rho_H1) | set(rho_W0) | set(rho_W1) | set(In0) | set(In1) | set(Out0)

    wide_rows: List[tuple] = []
    for g in sorted(all_geoids):
        h0 = float(rho_H0.get(g, 0.0))
        h1 = float(rho_H1.get(g, 0.0))
        w0 = float(rho_W0.get(g, 0.0))
        w1 = float(rho_W1.get(g, 0.0))
        In_val0 = float(In0.get(g, 0.0))
        In_val1 = float(In1.get(g, 0.0))
        Out_val = float(Out0.get(g, 0.0))
        drhoH = h1 - h0
        drhoW = w1 - w0
        dIn = In_val1 - In_val0
        wide_rows.append(
            (
                g,
                h0,
                h1,
                drhoH,
                w0,
                w1,
                drhoW,
                In_val0,
                In_val1,
                dIn,
                Out_val,
            )
        )

    wide_path = os.path.join("project", "data", "node", "node_signals.parquet")
    try:
        import pandas as pd  # type: ignore

        df_wide = pd.DataFrame(
            wide_rows,
            columns=[
                "geoid",
                f"rhoH_{y0}",
                f"rhoH_{y1}",
                "drhoH",
                f"rhoW_{y0}",
                f"rhoW_{y1}",
                "drhoW",
                f"In_{y0}",
                f"In_{y1}",
                "dIn",
                f"Out_{y0}",
            ],
        )
        df_wide.to_parquet(wide_path, index=False)
    except Exception:
        import csv

        csv_path = wide_path.rsplit(".", 1)[0] + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "geoid",
                    f"rhoH_{y0}",
                    f"rhoH_{y1}",
                    "drhoH",
                    f"rhoW_{y0}",
                    f"rhoW_{y1}",
                    "drhoW",
                    f"In_{y0}",
                    f"In_{y1}",
                    "dIn",
                    f"Out_{y0}",
                ]
            )
            writer.writerows(wide_rows)
        wide_path = csv_path

    # Panel checks
    checks = {}
    for year in years:
        H = rho_H_by_year.get(year, {})
        W = rho_W_by_year.get(year, {})
        In = In_by_year.get(year, {})
        Out = Out_by_year.get(year, {})
        sum_H = sum(H.values())
        sum_W = sum(W.values())
        sum_In = sum(In.values())
        sum_Out = sum(Out.values())
        checks[str(year)] = {
            "sum_rho_H": sum_H,
            "sum_rho_W": sum_W,
            "sum_In": sum_In,
            "sum_Out": sum_Out,
            "rel_err_In_vs_W": (sum_In - sum_W) / sum_W if sum_W else None,
            "rel_err_Out_vs_H": (sum_Out - sum_H) / sum_H if sum_H else None,
            "n_tracts": len(set(H) | set(W) | set(In) | set(Out)),
        }

    # Overlap between years
    if len(years) >= 2:
        y0, y1 = years[0], years[-1]
        g0 = set(rho_H_by_year.get(y0, {})) | set(rho_W_by_year.get(y0, {}))
        g1 = set(rho_H_by_year.get(y1, {})) | set(rho_W_by_year.get(y1, {}))
        common = g0 & g1
        union = g0 | g1
        checks["overlap"] = {
            "y0": int(y0),
            "y1": int(y1),
            "n_common": len(common),
            "n_union": len(union),
            "match_rate": len(common) / len(union) if union else None,
        }

    checks_path = os.path.join("project", "results", "node", "panel_checks.json")
    with open(checks_path, "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)
