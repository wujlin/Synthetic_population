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


def residual_glm_ppml(
    od_clean_path: str,
    edges_with_dist_path: str,
    out_path: str,
    log_path: str,
    use_county_pair_fe: bool = False,
    eps: float = 0.5,
    backend: str = 'auto',  # 'auto' | 'sklearn' | 'statsmodels'
    sample_n: Optional[int] = None,
    max_iter: int = 1000,
    alpha: float = 1e-8,
    standardize_dist: bool = False,
) -> Dict[str, float]:
    """Fit PPML with origin/dest FE and distance (optionally county×county FE).
    Writes od_residual_glm with columns: work,home,S000,mu_hat,log_resid_glm,dist.
    Also writes a JSON log with lambda (distance coef), stderr/pvalue, and deviance.
    If statsmodels is not available, falls back to independence baseline and notes fallback.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        # Fallback to independence baseline
        base = residual_from_parquet(od_clean_path, out_path, eps=eps)
        with open(log_path, 'w', encoding='utf-8') as f:
            import json
            json.dump({"fallback": "independence", **base}, f)
        return {"fallback": 1.0}
    # Load data
    if od_clean_path.endswith('.parquet'):
        df = pd.read_parquet(od_clean_path)
    else:
        df = pd.read_csv(od_clean_path)
    # Join distance
    dist_map = {}
    import csv as _csv
    with open(edges_with_dist_path, 'r', encoding='utf-8') as f:
        rdr = _csv.DictReader(f)
        for r in rdr:
            try:
                d = float(r.get('distance_km')) if r.get('distance_km') not in (None, '', 'None') else None
            except Exception:
                d = None
            dist_map[(r['work'], r['home'])] = d
    df['dist'] = [dist_map.get((w,h), 0.0) for w,h in df[['work','home']].itertuples(index=False, name=None)]
    df['work_cty'] = df['work'].astype(str).str[:5]
    df['home_cty'] = df['home'].astype(str).str[:5]
    # Optional downsample for sanity/diagnostics
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    # Optional distance standardization (z-score)
    dist_mean = None
    dist_std = None
    if standardize_dist:
        try:
            import numpy as _np
            dist_mean = float(df['dist'].mean())
            dist_std = float(df['dist'].std()) or 1.0
            df['dist'] = (df['dist'] - dist_mean) / dist_std
        except Exception:
            pass
    # Build design with FEs
    # Helper to write fallback log
    def _write_log(payload: Dict[str, object]):
        import json
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f)

    # If user forces sklearn backend, skip statsmodels entirely
    if backend == 'sklearn':
        try:
            import numpy as np  # type: ignore
            from sklearn.preprocessing import OneHotEncoder  # type: ignore
            from sklearn.metrics import mean_poisson_deviance  # type: ignore
            from sklearn.linear_model import PoissonRegressor  # type: ignore
            import scipy.sparse as sp  # type: ignore
            # Handle sklearn API change (sparse -> sparse_output)
            def make_ohe():
                try:
                    return OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=True)
                except TypeError:
                    return OneHotEncoder(handle_unknown='ignore', drop='first', sparse=True)
            # Encoders (origin/dest)
            ohe_w = make_ohe()
            ohe_h = make_ohe()
            Xw = ohe_w.fit_transform(df[['work']])
            Xh = ohe_h.fit_transform(df[['home']])
            blocks = [Xw, Xh]
            if use_county_pair_fe:
                cc = (df['work_cty'].astype(str) + '_' + df['home_cty'].astype(str)).values.reshape(-1,1)
                ohe_cc = make_ohe()
                Xcc = ohe_cc.fit_transform(cc)
                blocks.append(Xcc)
            dist_col = df['dist'].fillna(0.0).values.astype(float).reshape(-1,1)
            Xdist = sp.csr_matrix(dist_col)
            blocks.append(Xdist)
            X = sp.hstack(blocks).tocsr()
            y = df['S000'].values.astype(float)
            model = PoissonRegressor(alpha=alpha, max_iter=max_iter, tol=1e-8, fit_intercept=True, warm_start=False)
            model.fit(X, y)
            mu = model.predict(X)
            df_out = df.copy()
            df_out['mu_hat'] = mu
            df_out['log_resid_glm'] = np.log((df_out['S000'] + eps) / (df_out['mu_hat'] + eps))
            try:
                df_out[['work','home','S000','dist','mu_hat','log_resid_glm']].to_parquet(out_path, index=False)
            except Exception:
                df_out[['work','home','S000','dist','mu_hat','log_resid_glm']].to_csv(out_path.rsplit('.',1)[0]+'.csv', index=False)
            dev_mean = mean_poisson_deviance(y, mu)
            payload = {
                'n_obs': int(df.shape[0]),
                'use_county_pair_fe': bool(use_county_pair_fe),
                'lambda_dist': float(model.coef_[-1]) if hasattr(model, 'coef_') else None,
                'deviance': float(dev_mean * len(y)),
                'backend': 'sklearn',
                'max_iter': int(max_iter),
                'alpha': float(alpha),
                'standardize_dist': bool(standardize_dist),
            }
            if standardize_dist:
                payload['dist_mean'] = dist_mean
                payload['dist_std'] = dist_std
            _write_log(payload)
            return {'n_obs': float(df.shape[0]), 'lambda': float(model.coef_[-1]) if hasattr(model, 'coef_') else float('nan')}
        except Exception as e:
            # Record reason and then base metrics
            base = residual_from_parquet(od_clean_path, out_path, eps=eps)
            base['backend'] = 'independence'
            base['reason'] = f'sklearn_error: {type(e).__name__}: {str(e)}'
            _write_log(base)
            return {"fallback": 1.0}

    # Try statsmodels first; if memory error, fallback to sklearn sparse
    try:
        import statsmodels.api as sm  # type: ignore
        import patsy  # type: ignore
        if use_county_pair_fe:
            formula = 'S000 ~ 0 + C(work) + C(home) + dist + C(work_cty):C(home_cty)'
        else:
            formula = 'S000 ~ 0 + C(work) + C(home) + dist'
        y, X = patsy.dmatrices(formula, df, return_type='dataframe')
        model = sm.GLM(y, X, family=sm.families.Poisson())
        res = model.fit()
        mu = res.mu
        df_out = df.copy()
        df_out['mu_hat'] = mu
        import numpy as np  # type: ignore
        df_out['log_resid_glm'] = np.log((df_out['S000'] + eps) / (df_out['mu_hat'] + eps))
        try:
            df_out[['work','home','S000','dist','mu_hat','log_resid_glm']].to_parquet(out_path, index=False)
        except Exception:
            df_out[['work','home','S000','dist','mu_hat','log_resid_glm']].to_csv(out_path.rsplit('.',1)[0]+'.csv', index=False)
        coef = res.params.get('dist', float('nan'))
        bse = res.bse.get('dist', float('nan')) if hasattr(res, 'bse') else float('nan')
        pval = res.pvalues.get('dist', float('nan')) if hasattr(res, 'pvalues') else float('nan')
        dev = getattr(res, 'deviance', float('nan'))
        import json
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                'n_obs': int(df.shape[0]),
                'use_county_pair_fe': bool(use_county_pair_fe),
                'lambda_dist': float(coef) if coef==coef else None,
                'lambda_stderr': float(bse) if bse==bse else None,
                'lambda_pvalue': float(pval) if pval==pval else None,
                'deviance': float(dev) if dev==dev else None,
                'backend': 'statsmodels'
            }, f)
        return {'n_obs': float(df.shape[0]), 'lambda': float(coef) if coef==coef else float('nan')}
    except Exception as e:
        # Fallback to sklearn sparse PPML (PoissonRegressor)
        try:
            import numpy as np  # type: ignore
            from sklearn.preprocessing import OneHotEncoder  # type: ignore
            from sklearn.metrics import mean_poisson_deviance  # type: ignore
            from sklearn.linear_model import PoissonRegressor  # type: ignore
            import scipy.sparse as sp  # type: ignore
            def make_ohe():
                try:
                    return OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=True)
                except TypeError:
                    return OneHotEncoder(handle_unknown='ignore', drop='first', sparse=True)
            # Encoders
            ohe_w = make_ohe()
            ohe_h = make_ohe()
            Xw = ohe_w.fit_transform(df[['work']])
            Xh = ohe_h.fit_transform(df[['home']])
            blocks = [Xw, Xh]
            if use_county_pair_fe:
                cc = (df['work_cty'].astype(str) + '_' + df['home_cty'].astype(str)).values.reshape(-1,1)
                ohe_cc = make_ohe()
                Xcc = ohe_cc.fit_transform(cc)
                blocks.append(Xcc)
            # distance column; fill NaN with 0
            dist_col = df['dist'].fillna(0.0).values.astype(float).reshape(-1,1)
            Xdist = sp.csr_matrix(dist_col)
            blocks.append(Xdist)
            X = sp.hstack(blocks).tocsr()
            y = df['S000'].values.astype(float)
            # PoissonRegressor
            model = PoissonRegressor(alpha=alpha, max_iter=max_iter, tol=1e-8, fit_intercept=True, warm_start=False)
            model.fit(X, y)
            mu = model.predict(X)
            df_out = df.copy()
            df_out['mu_hat'] = mu
            df_out['log_resid_glm'] = np.log((df_out['S000'] + eps) / (df_out['mu_hat'] + eps))
            try:
                df_out[['work','home','S000','dist','mu_hat','log_resid_glm']].to_parquet(out_path, index=False)
            except Exception:
                df_out[['work','home','S000','dist','mu_hat','log_resid_glm']].to_csv(out_path.rsplit('.',1)[0]+'.csv', index=False)
            # dist coef is the last column weight
            coef_dist = float(model.coef_[-1]) if hasattr(model, 'coef_') else float('nan')
            dev_mean = mean_poisson_deviance(y, mu)
            dev_total = float(dev_mean * len(y))
            payload = {
                'n_obs': int(df.shape[0]),
                'use_county_pair_fe': bool(use_county_pair_fe),
                'lambda_dist': coef_dist,
                'lambda_stderr': None,
                'lambda_pvalue': None,
                'deviance': dev_total,
                'backend': 'sklearn',
                'statsmodels_error': f'{type(e).__name__}: {str(e)}'
            }
            if standardize_dist:
                payload['dist_mean'] = dist_mean
                payload['dist_std'] = dist_std
            payload['max_iter'] = int(max_iter)
            payload['alpha'] = float(alpha)
            payload['standardize_dist'] = bool(standardize_dist)
            _write_log(payload)
            return {'n_obs': float(df.shape[0]), 'lambda': coef_dist}
        except Exception:
            # Final fallback
            base = residual_from_parquet(od_clean_path, out_path, eps=eps)
            base['backend'] = 'independence'
            base['reason'] = f'final_fallback: {type(e).__name__}: {str(e)}'
            _write_log(base)
            return {"fallback": 1.0}
