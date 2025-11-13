import math
from collections import defaultdict
from typing import Dict, Tuple, Iterable, List, Optional


def independence_expected(edges: Iterable[Tuple[str, str, int]]):
    """Compute independence baseline E_ij = O_i * D_j / T.
    Returns dicts: O, D, total T, and expected for present (i,j).
    """
    O: Dict[str, int] = defaultdict(int)
    D: Dict[str, int] = defaultdict(int)
    flows: Dict[Tuple[str, str], int] = {}
    T = 0
    for w, h, s in edges:
        flows[(w, h)] = flows.get((w, h), 0) + int(s)
        O[w] += int(s)
        D[h] += int(s)
        T += int(s)
    expected: Dict[Tuple[str, str], float] = {}
    if T == 0:
        return O, D, T, expected, flows
    for (w, h), s in flows.items():
        expected[(w, h)] = (O[w] * D[h]) / float(T)
    return O, D, T, expected, flows


def adjusted_flow_ratio(
    w: str,
    h: str,
    flows: Dict[Tuple[str, str], int],
    expected: Dict[Tuple[str, str], float],
    eps: float = 0.5,
) -> float:
    """Return (F_ij/E_ij + eps) / (F_ji/E_ji + eps)."""
    f_ij = float(flows.get((w, h), 0))
    f_ji = float(flows.get((h, w), 0))
    e_ij = float(expected.get((w, h), 0.0))
    e_ji = float(expected.get((h, w), 0.0))
    a = (f_ij / e_ij) if e_ij > 0 else 0.0
    b = (f_ji / e_ji) if e_ji > 0 else 0.0
    return (a + eps) / (b + eps)


def residual_from_parquet(
    in_parquet: str,
    out_parquet: str,
    eps: float = 0.5,
) -> Dict[str, float]:
    """Compute independence baseline E_ij on present pairs, rescale so sum(E)=sum(F), and log_resid.
    Writes od_residual.parquet with columns: work,home,S000,E_ij,log_resid.
    """
    # Try pandas first
    edges: List[Tuple[str,str,int]] = []
    T = 0
    try:
        import pandas as pd  # type: ignore
        df = pd.read_parquet(in_parquet)
        edges = [(w, h, int(s)) for w, h, s in df[["work", "home", "S000"]].itertuples(index=False, name=None)]
        T = int(df["S000"].sum())
    except Exception:
        # CSV fallback
        import csv
        path_csv = in_parquet.rsplit('.',1)[0]+'.csv'
        with open(path_csv,'r',encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                w = r['work']; h=r['home']; s=int(float(r['S000']))
                edges.append((w,h,s)); T += s
    O, D, T, E_map, F_map = independence_expected(edges)
    # Rescale E to match T on observed support
    sum_E = sum(E_map.values())
    scale = (T / sum_E) if sum_E > 0 else 1.0
    Es = []
    logs = []
    for w, h, s in edges:
        e = E_map.get((w, h), 0.0) * scale
        Es.append(e)
        logs.append((math.log((s + eps) / (e + eps))) if e >= 0 else 0.0)
    # Write output
    try:
        import pandas as pd  # type: ignore
        df_out = pd.DataFrame(edges, columns=["work","home","S000"])
        df_out["E_ij"] = Es
        df_out["log_resid"] = logs
        df_out.to_parquet(out_parquet, index=False)
        valid_ratio = float((df_out["log_resid"].notna()).mean())
    except Exception:
        import csv
        out_csv = out_parquet.rsplit('.',1)[0]+'.csv'
        with open(out_csv,'w',newline='',encoding='utf-8') as f:
            wtr = csv.writer(f)
            wtr.writerow(["work","home","S000","E_ij","log_resid"])
            for (w,h,s),e,lr in zip(edges, Es, logs):
                wtr.writerow([w,h,s,e,lr])
        valid_ratio = 1.0
    return {"sumF": float(T), "sumE": float(sum(Es)), "valid_ratio": valid_ratio}
