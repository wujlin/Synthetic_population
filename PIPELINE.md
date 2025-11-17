Pipeline: SEMCOG 2020 OD Diagnostics

## Three‑figure logic (at a glance)

- R²/η bars
  - Observation: After independence baseline, R²≈0.194, η≈0.898.
  - Meaning: A single potential explains ~19% of directions; ~80% comes from loops/non‑reciprocity.
  - Implication: Move toward multi‑potential or potential + anti‑symmetric edge terms.

- Potential (π) boxplots
  - Observation: County‑level π differs systematically (e.g., Livingston/Washtenaw/St. Clair higher; Oakland/Macomb/Monroe/Wayne lower).
  - Reading: g_ij≈π_j−π_i; low‑π→high‑π aligns with net attraction (after scale effect removal).
  - Implication: π is a target for interpretable external‑feature regression U(x) (access, POIs, housing/income, etc.).

- Locality curve
  - Observation: R²≈0.58→0.38 for 5–15 km; adding longer links reduces R² ≈ 0.22.
  - Meaning: Short distances are more potential‑like; medium/long distances show stronger non‑reciprocity.
  - Implication: Baseline must remove distance/time friction; multi‑potential or anti‑symmetric terms should focus on medium/long distances.
- Loop hotspots (new 4th visual)
  - Observation: Top |resid| corridors link SEMCOG sub-centers (tri-county ↔ Washtenaw/St. Clair).
  - Meaning: η mass is spatially clustered; supports “retain anti-symmetric term” decision.
  - Implication: Use `rot_diagnostics` before adding PDE regressions; treat highlighted corridors separately.

Overall: Short‑distance ≈ potential‑like; medium/long‑distance ≈ loop‑dominated. Actions: PPML baseline → robust Hodge → multi‑potential/anti‑symmetric modeling → U(x) regression.

## Workflow

```mermaid
flowchart LR
  A[Phase 0 Data readiness] --> B[Phase 1 PPML baseline<br/>log mu = alpha + beta - lambda * dist (+FE)]
  B --> C[Phase 2 Hodge diagnostics<br/>fit pi -> R2, eta]
  C --> D{Phase 3+}
  D --> D1[Locality curves (r*)]
  D --> D2[Multi-potential / Anti-symmetric]
  D --> D3[External potential U(x)]
```

## End‑to‑end pipeline (purpose → steps → outputs/acceptance)

- Environment
  - conda create -n semcog-od python=3.10 -y && conda activate semcog-od
  - conda install -c conda-forge numpy pandas scipy matplotlib statsmodels patsy pyarrow pyshp scikit-learn -y

### Phase 0 — Data readiness (completed)
**Purpose**: Standardize OD and produce centroids/distances.

**Inputs**: `data/raw/semcog_tract_od_intra_2020.csv`, `data/geo/mi-tracts-demo-work.shp`

**Process**:
- Read OD: `python -m project.src.cli ingest --raw dataset/od_flow/semcog_tract_od_intra_2020.csv`
- Marginals: `python -m project.src.cli marginals`
- Centroids/distances: `python -m project.src.cli shp2geojson --shp dataset/geo/mi-tracts-demo-work.shp --out project/data/geo/tracts.geojson && python -m project.src.cli prepare`

**Outputs**: `data/processed/od_clean.csv`, `nodes_basic.csv`, `edges_with_distance.csv`

**Acceptance**: ΣR=ΣW=ΣF; 11‑digit GEOID; reasonable `distance_km`.

### Phase 1 — PPML + distance/admin FE (priority)
**Purpose**: Remove distance/admin effects to obtain cleaner residuals.

**Process**:
```bash
python -m project.src.cli baseline_glm --backend sklearn --eps 0.5 \
  --max-iter 5000 --alpha 1e-8 --standardize-dist [--county-pair-fe]
```
Definition: $\log \mu_{ij} = \alpha_i + \beta_j - \lambda\cdot \mathrm{dist}_{ij}$ (optionally with county‑by‑county FE); $\;\log\_\mathrm{resid\_glm} = \log\tfrac{F\_{ij}+\epsilon}{\hat\mu\_{ij}+\epsilon}$.

**Outputs**: `data/processed/od_residual_glm.parquet|csv` (`mu_hat, log_resid_glm`), `results/diagnostics/baseline_glm_summary.json` (λ, deviance, backend).

**Latest run**: λ_dist ≈ **−0.94** (`backend=sklearn`, `standardize_dist=True`, no county×county FE), `dist_mean ≈ 28.88 km`, `dist_std ≈ 19.92 km`.

**Acceptance**: vs independent baseline, downstream Hodge shows **R² ↑, η ↓**.

### Phase 2 — Robust Hodge: potential/non-reciprocity
**Purpose**: Make conclusions robust to sampling/numerics/weights.

**Process**:
```bash
python -m project.src.cli potential_hodge_glm --weight-type cap --cap-tau 200 --drop-self \
  --sample-edges 120000 --bins-dist 8 --bins-weight 8 --seed 42 \
  --max-edges 150000 --maxiter 300 --tol 1e-5
```
Notes: stratified sampling (`dist×(F_ij+F_ji)`), component‑wise anchoring (sparse Laplacian `Lπ=div(g)`), weight baselines, ε sweeps, self‑loop toggle.

**Outputs**: `results/diagnostics/nodes_potential_glm.csv`, `edges_decomp_glm.csv`, `edges_topk_glm.csv`, `summary_robustness.json`.

**Latest run**: `R² ≈ 0.108`, `η ≈ 0.935`, 1475 nodes / 113,011 edges（`summary_robustness.json`）。

**Acceptance**: conclusions stable under settings; reasonable variance in R²/η; Top‑K residual edges for inspection.

### From diagnostics to PDE (4 visuals + 4 analyses)

Once Phase 2 artifacts exist, run the structure-first suite to quantify each PDE term and add the fourth visual (cycle hotspots).

| Structure item | Goal / definition | CLI command | Diagnostics / figures | Acceptance |
|---|---|---|---|---|
| Loops / η(r) | Map non-reciprocity corridors; `C_{ijk}=g_{ij}+g_{jk}+g_{ki}` | `python3 -m project.src.cli rot_diagnostics --topk 2000 --topcycles 500` | `rot_summary.json`, `top_cycles.csv`, `fig_cycles_hotspots.png` | η level reported; corridors match geography |
| Potential term κ | $\tilde N_{ij}=(F_{ij}-F_{ji})-(\mu_{ij}-\mu_{ji})\sim \rho_{ij},c_{ij},c_{ij}\Delta\pi_{ij}$ | `python3 -m project.src.cli pde_fit_kappa` | `pde_kappa.json`, `fig_kappa_scatter.png` | κ significant (t-value) or documented as negligible |
| Diffusion term D | Residual gradient $\hat N_{ij}\sim -D(\rho_j-\rho_i)$ | `python3 -m project.src.cli pde_fit_diffusion` | `pde_diffusion.json`, `fig_diffusion_scatter.png` | ΔR² > 0 or term dropped |
| Interface term Γ | Boundary curvature $\hat N^{(after D)}_{ij}\sim -\Gamma(\nabla^2\rho_j-\nabla^2\rho_i)$ | `python3 -m project.src.cli pde_fit_interface --knn 6` | `pde_interface.json`, `fig_interface_scatter.png` | Γ significant or justified as 0 |

**Latest metrics (SEMCOG 2020 run)**  
κ ≈ **−0.80** (t ≈ −105, `R² ≈ 0.069`), diffusion D ≈ `3.2e-12` (ΔR² `~6e-4`), interface Γ ≈ `3.9e-12` (ΔR² `~6e-4`) — retain κ, document D/Γ as negligible.

Flow (Phase 2 → structure terms):

```mermaid
flowchart LR
  H[PPML residuals] --> R[Robust Hodge<br/>(π, g, η)]
  R --> C[Cycle diagnostics<br/>(rot_diagnostics)]
  C --> Kappa[pde_fit_kappa]
  Kappa --> Diff[pde_fit_diffusion]
  Diff --> Interface[pde_fit_interface]
```

- Phase 3 — Locality (two curves, unified criteria)
  - Purpose: Separate “short‑distance advantage” from “subset training effect”.
  - Actions:
    - Curve A (implemented): train+evaluate on `dist ≤ r0`: `python -m project.src.cli locality_curve --radii 5 10 15 20 25 30 40 50`
    - Curve B (suggested): evaluate only with a globally trained π (ensure monotone non‑decreasing).
    - Report half‑energy radius r* (minimum r0 with R²≥0.5) with CI (repeated sampling).
  - Outputs: `results/diagnostics/locality_report.json` + `results/figures/fig_locality_curve.png`
  - Acceptance: Curve B monotone non‑decreasing; Curve A above B; report r* and CI.

- Phase 4 — Loop structure (make η spatial)
  - Purpose: Turn non‑reciprocity into interpretable cycles/corridors.
  - Actions: approximate triangle strength $C\_{ijk}=g\_{ij}+g\_{jk}+g\_{ki}$ on `edges_decomp_glm`; export Top‑K cycles and hotspots.
  - Outputs: `results/diagnostics/top_cycles.csv`, `results/figures/fig_cycles_hotspots.png`.
  - Acceptance: Main cycles align with inter‑county / medium‑long corridors.

- Phase 5 — Multi‑potential / potential + anti‑symmetric (choose one to start)
  - Route 5A (multi‑potential): cluster (NMF/spectral) into K subnetworks; per subnet: PPML→Hodge→$\pi^{(k)}$; combine $\log F\_{ij}=\alpha\_i+\beta\_j-\lambda d\_{ij}+\log\sum\_k w\_k e^{\pi^{(k)}(j)-\pi^{(k)}(i)}$; validate on held‑out edges.
  - Route 5B (potential + anti‑symmetric edge term): $\log F\_{ij}=\alpha\_i+\beta\_j-\lambda d\_{ij}+[\pi(j)-\pi(i)]+A\_{ij}$ with $A\_{ij}=-A\_{ji}$; design anti‑symmetric features; fit $A\_{ij}$; validate on held‑out edges.
  - Outputs: `multi_potential_nmf/*` or `antisym_edge_model/*`.
  - Acceptance: Significant uplift vs "single potential + distance" on held‑out; directions match spatial intuition.

- Phase 6 — External potential $U(x)$ (node‑level, interpretable)
  - Purpose: explain $\pi$ (or $\pi^{(k)}$) by actionable factors.
  - Actions: linear/ridge/GAM (interpretable; small MLP as baseline); cross‑validation; coefficients/SHAP + potential maps.
  - Outputs: `u_regression/*`, `results/figures/fig_U_maps.png`.
  - Acceptance: adequate fit; spatial patterns and signs reasonable; scenario analysis feasible.

- Phase 7 — Report & visualization
  - Purpose: unify outputs and reproducibility.
  - Actions: `python -m project.src.cli export_figs` (needs matplotlib); `REPORT.md` collects parameters and acceptance.
  - Outputs: `results/figures/*`, `REPORT.md`.
  - Acceptance: one‑click reruns reproduce; figures match JSON metrics.

**Acceptance focus**

- Baseline: vs independence, R² up and η down under identical sampling/weights.
- Locality: Curve B monotone; report half‑energy radius r* with CI.
- Modeling (5A/5B): held‑out uplift over "single potential + distance"; interpretable signs.
