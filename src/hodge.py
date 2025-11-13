import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Optional

from .gravity import independence_expected, adjusted_flow_ratio


def build_pairs(
    edges: Iterable[Tuple[str, str, int]], use_independence: bool = False, eps: float = 0.5
):
    """Construct undirected edge set with weights and antisymmetric log-ratio a_e.

    Returns:
      nodes: sorted list of GEOIDs
      edges_u: list of tuples (u,v,w,a) with u<v, weight w = F_uv + F_vu, a = log(ratio)
    """
    # Gather flows
    E: Dict[Tuple[str, str], int] = {}
    nodes_set = set()
    for w, h, s in edges:
        E[(w, h)] = E.get((w, h), 0) + int(s)
        nodes_set.add(w)
        nodes_set.add(h)

    if use_independence:
        O, D, T, expected, flows = independence_expected([(w, h, s) for (w, h), s in E.items()])

    seen = set()
    edges_u: List[Tuple[str, str, float, float]] = []
    for (i, j), s_ij in E.items():
        if (j, i) in seen:
            continue
        seen.add((i, j))
        seen.add((j, i))
        s_ji = E.get((j, i), 0)
        w = float(s_ij + s_ji)
        if w <= 0:
            continue
        # Use consistent orientation u<v lexicographically
        u, v = (i, j) if i < j else (j, i)
        # ratio oriented u->v
        if use_independence:
            ratio = adjusted_flow_ratio(u, v, flows, expected, eps=eps)
        else:
            f_uv = float(E.get((u, v), 0))
            f_vu = float(E.get((v, u), 0))
            ratio = (f_uv + eps) / (f_vu + eps)
        a = math.log(ratio)
        edges_u.append((u, v, w, a))
    nodes = sorted(nodes_set)
    return nodes, edges_u


def laplacian_matvec(
    x: List[float],
    edges_u: List[Tuple[int, int, float, float]],
    pivot: int,
) -> List[float]:
    """Compute y = L x for the reduced system (pivot removed).
    edges_u uses integer indices; w is weight.
    """
    n = len(x) + 1  # original dimension
    y = [0.0 for _ in range(n - 1)]
    def idx(k: int) -> Optional[int]:
        if k == pivot:
            return None
        return k if k < pivot else k - 1
    for (u, v, w, _a) in edges_u:
        iu = idx(u)
        iv = idx(v)
        if iu is None and iv is None:
            continue
        if iu is not None and iv is not None:
            du = x[iu]
            dv = x[iv]
            y[iu] += w * (du - dv)
            y[iv] += w * (dv - du)
        elif iu is not None:
            # v is pivot (phi_v = 0)
            du = x[iu]
            y[iu] += w * (du - 0.0)
        elif iv is not None:
            dv = x[iv]
            y[iv] += w * (dv - 0.0)
    return y


def cg_solve(
    matvec,
    b: List[float],
    tol: float = 1e-6,
    maxiter: int = 500,
) -> List[float]:
    n = len(b)
    x = [0.0] * n
    r = [bi for bi in b]
    p = [ri for ri in r]
    rsold = sum(ri * ri for ri in r)
    if rsold == 0:
        return x
    for _ in range(maxiter):
        Ap = matvec(p)
        pAp = sum(pi * api for pi, api in zip(p, Ap))
        if pAp == 0:
            break
        alpha = rsold / pAp
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * api for ri, api in zip(r, Ap)]
        rsnew = sum(ri * ri for ri in r)
        if rsnew < tol * tol * rsold:
            break
        beta = rsnew / rsold
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rsold = rsnew
    return x


def fit_potential(
    edges_u: List[Tuple[str, str, float, float]],
    nodes: List[str],
    maxiter: int = 500,
    tol: float = 1e-6,
) -> Tuple[Dict[str, float], float, float, float]:
    """Solve min_phi sum_e w (a_e - (phi_v - phi_u))^2.

    Returns: (phi dict), energy_total, energy_grad, energy_resid.
    """
    # Map node to index
    idx = {n: i for i, n in enumerate(nodes)}
    # Convert edges to indexed tuples
    E_idx = [(idx[u], idx[v], w, a) for (u, v, w, a) in edges_u]
    n = len(nodes)
    pivot = 0  # fix gauge phi[pivot]=0
    # Build b = B^T W a
    b_full = [0.0] * n
    for (u, v, w, a) in E_idx:
        b_full[u] += -w * a
        b_full[v] += +w * a
    # Reduced b
    b = [b_full[i] for i in range(n) if i != pivot]
    # CG solve
    matvec = lambda x: laplacian_matvec(x, E_idx, pivot)
    x = cg_solve(matvec, b, tol=tol, maxiter=maxiter)
    # Assemble phi with pivot=0
    phi = [0.0] * n
    it = iter(x)
    for i in range(n):
        if i == pivot:
            phi[i] = 0.0
        else:
            phi[i] = next(it)
    phi_map = {nodes[i]: phi[i] for i in range(n)}

    # Energies
    E_total = 0.0
    E_grad = 0.0
    E_resid = 0.0
    for (u, v, w, a) in E_idx:
        pred = phi[v] - phi[u]
        E_total += w * (a * a)
        E_grad += w * (pred * pred)
        E_resid += w * ((a - pred) ** 2)
    return phi_map, E_total, E_grad, E_resid


def fit_potential_components(
    edges_u: List[Tuple[str, str, float, float]],
    nodes: List[str],
    maxiter: int = 500,
    tol: float = 1e-6,
) -> Tuple[Dict[str, float], float, float, float]:
    """Solve potential per connected component (each with its own anchor)."""
    # Build adjacency on indices
    idx = {n: i for i, n in enumerate(nodes)}
    E_idx = [(idx[u], idx[v], w, a) for (u, v, w, a) in edges_u]
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for (u, v, w, a) in E_idx:
        adj[u].append(v)
        adj[v].append(u)
    # DFS to get components
    comp = [-1] * n
    comps: List[List[int]] = []
    for i in range(n):
        if comp[i] != -1:
            continue
        stack = [i]
        comp[i] = len(comps)
        cur = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = comp[i]
                    stack.append(v)
                    cur.append(v)
        comps.append(cur)
    # Separate edges per comp and solve
    phi_map: Dict[str, float] = {}
    Et = Eg = Er = 0.0
    for cid, nodes_c in enumerate(comps):
        if len(nodes_c) == 1:
            phi_map[nodes[nodes_c[0]]] = 0.0
            continue
        # map old index -> local
        loc = {i: k for k, i in enumerate(nodes_c)}
        edges_c = []
        for (u, v, w, a) in E_idx:
            if comp[u] == cid and comp[v] == cid:
                edges_c.append((u, v, w, a))
        # reuse solver with pivot=0 in local indices
        nodes_local = [nodes[i] for i in nodes_c]
        # Remap indices to local
        edges_local = [(loc[u], loc[v], w, a) for (u, v, w, a) in edges_c]
        # Build b and matvec like fit_potential
        nloc = len(nodes_local)
        pivot = 0
        b_full = [0.0] * nloc
        for (u, v, w, a) in edges_local:
            b_full[u] += -w * a
            b_full[v] += +w * a
        b = [b_full[i] for i in range(nloc) if i != pivot]
        matvec = lambda x: laplacian_matvec(x, edges_local, pivot)
        x = cg_solve(matvec, b, tol=tol, maxiter=maxiter)
        phi = [0.0] * nloc
        it = iter(x)
        for i in range(nloc):
            phi[i] = 0.0 if i == pivot else next(it)
        for k, i in enumerate(nodes_c):
            phi_map[nodes[i]] = phi[k]
        # energies on component
        for (u, v, w, a) in edges_local:
            pred = phi[v] - phi[u]
            Et += w * (a * a)
            Eg += w * (pred * pred)
            Er += w * ((a - pred) ** 2)
    return phi_map, Et, Eg, Er


def run_hodge_from_residual(
    residual_rows: Iterable[Tuple[str, str, float, float]],
    use_weights_from_F: bool = True,
    max_edges: Optional[int] = None,
    maxiter: int = 500,
    tol: float = 1e-6,
):
    """
    residual_rows: iterator of (work, home, S000, log_resid)
    Returns: phi_map, summary dict, edge diagnostics (u,v,w,a,pred,resid)
    """
    # Build directed dicts
    F: Dict[Tuple[str, str], float] = {}
    LR: Dict[Tuple[str, str], float] = {}
    nodes = set()
    for w, h, s, lr in residual_rows:
        F[(w, h)] = F.get((w, h), 0.0) + float(s)
        LR[(w, h)] = float(lr)
        nodes.add(w); nodes.add(h)
    # Build undirected pairs with g_ij and weight
    seen = set()
    edges_u: List[Tuple[str, str, float, float]] = []
    for (i, j), _ in F.items():
        if (j, i) in seen:
            continue
        seen.add((i, j)); seen.add((j, i))
        w = F.get((i, j), 0.0) + F.get((j, i), 0.0)
        if w <= 0: continue
        u, v = (i, j) if i < j else (j, i)
        g = LR.get((u, v), 0.0) - LR.get((v, u), 0.0)
        edges_u.append((u, v, w, g))
    if max_edges and len(edges_u) > max_edges:
        edges_u.sort(key=lambda t: t[2], reverse=True)
        edges_u = edges_u[:max_edges]
    nodes = sorted(nodes)
    phi, Et, Eg, Er = fit_potential_components(edges_u, nodes, maxiter=maxiter, tol=tol)
    R2 = 1.0 - (Er / Et) if Et > 0 else 0.0
    # Non-reciprocity eta
    num = den = 0.0
    for (u, v, w, a) in edges_u:
        pred = phi[v] - phi[u]
        num += w * abs(a - pred)
        den += w * abs(a)
    eta = (num / den) if den > 0 else 0.0
    # Edge diagnostics
    diags = []
    for (u, v, w, a) in edges_u:
        pred = phi[v] - phi[u]
        diags.append((u, v, w, a, pred, a - pred))
    summary = {"energy_total": Et, "energy_grad": Eg, "energy_resid": Er, "R2": R2, "eta": eta, "edges": len(edges_u), "nodes": len(nodes)}
    return phi, summary, diags


def top_triangle_cycles(diags: List[Tuple[str, str, float, float, float, float]], K: int = 100):
    """Heuristic top-K 3-cycles by weight: restrict to top-neighborhood per node.
    Returns list of (i,j,k,cycle_sum, min_w).
    """
    # Build adjacency and store a_ij
    nbr = defaultdict(set)
    aij = {}
    wmap = {}
    for (u, v, w, a, pred, resid) in diags:
        nbr[u].add(v); nbr[v].add(u)
        aij[(u, v)] = a; aij[(v, u)] = -a
        wmap[(u, v)] = w; wmap[(v, u)] = w
    # Limit neighbors per node by top weights
    max_deg = 50
    for n in list(nbr.keys()):
        neighbors = list(nbr[n])
        neighbors.sort(key=lambda m: wmap.get(tuple(sorted((n, m))), 0.0), reverse=True)
        nbr[n] = set(neighbors[:max_deg])
    tri_list = []
    nodes = list(nbr.keys())
    nodes.sort()
    for i_idx, i in enumerate(nodes):
        Ni = sorted(nbr[i])
        for j in Ni:
            if j <= i: continue
            Nj = nbr[j]
            common = [k for k in Ni if k in Nj and k > j]
            for k in common:
                cik = aij.get((i, k), 0.0)
                cij = aij.get((i, j), 0.0)
                cjk = aij.get((j, k), 0.0)
                cyc = cij + cjk + aij.get((k, i), 0.0)
                mw = min(wmap.get((i, j), 0.0), wmap.get((j, k), 0.0), wmap.get((k, i), 0.0))
                tri_list.append((i, j, k, cyc, mw))
        if len(tri_list) > 5 * K:
            break
    tri_list.sort(key=lambda t: (abs(t[3]), t[4]), reverse=True)
    return tri_list[:K]


def edge_diagnostics(
    edges_u: List[Tuple[str, str, float, float]], phi: Dict[str, float]
) -> Iterable[Tuple[str, str, float, float, float, float]]:
    for (u, v, w, a) in edges_u:
        pred = phi[v] - phi[u]
        resid = a - pred
        yield (u, v, w, a, pred, resid)
