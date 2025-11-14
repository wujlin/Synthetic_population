Project: SEMCOG 2020 OD Diagnostics

## Overview (1‑liner + 3 commands + key figures)

- One‑liner: A minimal, reproducible pipeline on SEMCOG 2020 tract→tract OD to diagnose "potential + non‑reciprocity" structure and evaluate locality (CSV/Parquet + figures).

### Quickstart (Phase 1–2)
```bash
# 1) Baseline (PPML + distance / admin FE)
python -m project.src.cli baseline_glm --eps 0.5 --county-pair-fe --backend sklearn --max-iter 5000 --standardize-dist

# 2) Robust Hodge (on GLM residuals)
python -m project.src.cli potential_hodge_glm \
  --weight-type cap --cap-tau 200 --drop-self \
  --sample-edges 120000 --bins-dist 8 --bins-weight 8 \
  --seed 42 --max-edges 150000 --maxiter 300 --tol 1e-5

# 3) Export figures
python -m project.src.cli export_figs
```

<p align="center">
  <img src="results/figures/r2_eta.png" width="520" alt="R² and η bars">
  <br><em>R² (potential fit) higher → more "potential‑like"; η (non‑reciprocity) higher → stronger loops.</em>
</p>

<p align="center">
  <img src="results/figures/pi_box_by_county.png" width="520" alt="County‑level potential (π) boxplots">
  <br><em>Low‑π → High‑π is the residual flow preference; county‑level π reveals net attraction gradients.</em>
</p>

Goal
- Based on 2020 SEMCOG 7-county tract→tract OD snapshot (columns: `work,home,S000`), build a minimal, executable pipeline to diagnose "potential + non-reciprocity" structure and evaluate locality, and export reusable results (CSV/Parquet + figures).

Why
- Remove pure scale effects first; test if a single potential is sufficient (reciprocal) or if cycles/non-reciprocity dominate.
- Use geometry to produce distance-layered locality curves.
- Provide a foundation for subsequent "single/multi-potential + external features U(x)" fitting.

Layout
- data/
  - raw/semcog_tract_od_intra_2020.csv          # tract→tract OD, columns: work,home,S000 (you can symlink or copy from dataset/od_flow/)
  - geo/tracts.geojson                           # your local tract boundaries or centroids (see below)
  - processed/                                   # generated
- results/
  - diagnostics/                                 # generated
  - figures/                                     # generated
- src/
  - io.py            # read/write, normalization, geometry/centroids, distance computation
  - gravity.py       # baseline independence normalization (optional)
  - hodge.py         # least-squares potential fit + Hodge indicators (gradient/curl fractions)
  - locality.py      # distance-layered R²/energy curves
  - features.py      # placeholder for node/edge features
  - cli.py           # command-line entry

## Workflow (Mermaid)

```mermaid
flowchart LR
  A[Phase 0 数据就绪] --> B[Phase 1 PPML基线\nlog μ=α+β-λ·dist (+FE)]
  B --> C[Phase 2 Hodge诊断\n拟合 π → R², η]
  C --> D{Phase 3+}
  D --> D1[局域性曲线 r*]
  D --> D2[多势/反对称项]
  D --> D3[外势 U(x) 回归]
```

## Requirements
- Python 3.8+.
- No hard external deps. If `numpy`, `pandas`, `pyarrow`, or `matplotlib` are available, the pipeline will leverage them; otherwise it will degrade gracefully to CSV-only and simple plotting is skipped.

## Geo centroids
- `data/geo/tracts.geojson` should include tract GEOIDs and either internal points or centroids:
  - GEOID keys accepted: `GEOID`, `GEOID10`, or `geoid`.
  - Lat/Lon keys accepted (preferred): `INTPTLAT10`, `INTPTLON10`, `centroid_lat`, `centroid_lon`.
  - If none found, a naive centroid (bbox mid) is computed as a fallback.

## Quickstart
- Place your OD CSV at `data/raw/semcog_tract_od_intra_2020.csv` (columns: `work,home,S000`).
- Place your tract GeoJSON at `data/geo/tracts.geojson`.
- Run end-to-end:
  - `python -m project.src.cli all`
- Or step-by-step:
  - `python -m project.src.cli prepare`
  - `python -m project.src.cli hodge --use-independence`
  - `python -m project.src.cli locality --bins 12`

## Data & artifacts

| 路径 | 阶段 | 含义/关键字段 |
|---|---|---|
| `data/raw/semcog_tract_od_intra_2020.csv` | Phase 0 | 原始 OD（`work, home, S000`） |
| `data/geo/tracts.geojson` | Phase 0 | 质心（`GEOID, INTPTLAT/LON`） |
| `data/processed/od_clean.parquet|csv` | Phase 0 | 过滤与标准化后的 OD |
| `data/processed/edges_with_distance.csv` | Phase 0 | OD + `distance_km` |
| `data/processed/od_residual_glm.parquet|csv` | Phase 1 | `mu_hat, log_resid_glm` |
| `results/diagnostics/baseline_glm_summary.json` | Phase 1 | PPML 收敛、距离系数 λ、deviance、backend |
| `results/diagnostics/nodes_potential_glm.csv` | Phase 2 | 每 tract 的 π |
| `results/diagnostics/edges_decomp_glm.csv` | Phase 2 | `g_ij, pred=π_j-π_i, resid, weight` |
| `results/diagnostics/locality_report.json` | Phase 2 | 局域性曲线数据（r0 vs R²） |
| `results/figures/*.png` | Phase 2 | `r2_eta.png, pi_box_by_county.png, fig_locality_curve.png` |

## Metrics & acceptance

### Metrics
- R² (potential fit): fraction of directional signal explained by the potential (higher is better)
- η (non‑reciprocity): fraction of one‑way/loop signal (lower is better)

### Acceptance checklist
- [ ] PPML converged; λ (distance) reasonable (often negative); `Σμ≈ΣF`
- [ ] `nodes_potential_glm.csv` and `edges_decomp_glm.csv` generated
- [ ] vs independent baseline: **R² ↑, η ↓** under the same sampling/weight settings

## Glossary

| Symbol/Name | Meaning | Notes |
|---|---|---|
| `F_ij` | Raw OD (work=i, home=j job count) | CSV column `S000` |
| `R_i=Σ_j F_ij` | Total outflow at work tract i | Marginal |
| `W_j=Σ_i F_ij` | Total inflow at home tract j | Marginal |
| `E_ij` | Independence expected = `R_i·W_j/ΣF` | Scale‑effect baseline |
| `μ̂_ij` | PPML expectation | Phase 1 output |
| `log_resid` | `log((F_ij+ε)/(E_ij+ε))` | Independent residual |
| `log_resid_glm` | `log((F_ij+ε)/(μ̂_ij+ε))` | GLM residual (recommended) |
| `g_ij` | Directional signal = `log_resid_ij − log_resid_ji` | Anti‑symmetric |
| `w_ij` | Weight, default `F_ij+F_ji` (can cap `min(·,τ)`) | Hodge weighting |
| `π` | Potential (node value) | `g_ij ≈ π(j) − π(i)` |
| `R²` | Potential fit | Higher is better |
| `η` | Non‑reciprocity | Lower is better |
- data/processed/
  - nodes.csv: tract GEOID, lat, lon
  - edges.csv: work, home, S000 (filtered to SEMCOG 7 counties)
  - edges_with_distance.csv: edges + distance_km
  - edges.parquet (if pyarrow available)
- results/diagnostics/
  - global_metrics.csv: N nodes, E edges, gradient/curl energy fractions, totals
  - node_potential.csv: GEOID, phi
  - edge_hodge_metrics.csv: work,home,weight,a_ij,pred,resid
  - locality_curve.csv: bin stats and energy/R² fractions vs distance
- results/figures/
  - locality_curve.png (if matplotlib available)

## Counties (SEMCOG 7)
- Wayne(26163), Oakland(26125), Macomb(26099), Washtenaw(26161), Monroe(26115), Livingston(26093), St. Clair(26147).

## Pipeline
- 详细工作流与验收标准：`project/PIPELINE.md`
- 阶段性结果总结与图表：`project/REPORT.md`

## Troubleshooting
<details><summary>⚠️ No figures?</summary>

- `matplotlib` missing: `pip install matplotlib`, then rerun `export_figs`.
- Diagnostics missing: run `baseline_glm` and `potential_hodge_glm` first.
</details>

<details><summary>⚠️ PPML λ = 0?</summary>

- Increase iterations: `--max-iter 5000` (or higher); standardize distance: `--standardize-dist`.
- Start without county×county FE; confirm nonzero λ with only origin/dest FE + distance, then add complexity.
- Check `edges_with_distance.csv` coverage: share of zero distances should not be high.
</details>

## Appendix: Independent baseline (for teaching/benchmark)

```bash
# Hodge on independence residuals (g_ij from log((F_ij+ε)/(E_ij+ε)))
python -m project.src.cli hodge --use-independence --eps 0.5 --max-edges 150000 --maxiter 300 --tol 1e-5

# Locality curve (train+evaluate) under independence residuals
python -m project.src.cli locality --bins 10
```

## Directory layout
```
project/
  data/
    raw/
    geo/
    processed/
  results/
    diagnostics/
    figures/
  src/
    io.py  gravity.py  hodge.py  locality.py  features.py  export_figs.py  cli.py
  PIPELINE.md  REPORT.md  README.md
```
