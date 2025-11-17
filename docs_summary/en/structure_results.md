# Structure Diagnostics (English)

## Phase 1 – PPML baseline
- Command: `python -m project.src.cli baseline_glm --backend sklearn --max-iter 10000 --alpha 1e-10 --standardize-dist`
- Observations: **496,096**
- Distance coefficient: **λ_dist ≈ −0.94** (sklearn PoissonRegressor, standardized distance, no county×county FE)
- Log file: `results/diagnostics/baseline_glm_summary.json`

## Phase 2 – Robust Hodge + locality
- Settings: weight-type=`cap`, τ=200, drop-self, stratified sample 120k edges, bins_dist=8, bins_weight=8, seed=42
- Outputs: `nodes_potential_glm.csv`, `edges_decomp_glm.csv`, `edges_topk_glm.csv`, `summary_robustness.json`
- Latest metrics: **R² ≈ 0.108**, **η ≈ 0.935**, nodes=1475, edges used=113,011
- Figures: `results/figures/r2_eta.png`, `pi_box_by_county.png`, `fig_locality_curve.png`

## Loop diagnostics (rot_diagnostics)
- Command: `python -m project.src.cli rot_diagnostics --topk 2000 --topcycles 500 --nodes-geo project/data/geo/tracts.geojson`
- Summary: `results/diagnostics/rot_summary.json` (η_global ≈ 0.935, 2k hotspot edges, 500 top triangle cycles)
- Figures/CSVs: `fig_cycles_hotspots.png`, `results/diagnostics/top_cycles.csv`, `results/diagnostics/edges_topk_glm.csv`

## Phase 3 – PDE structure terms
- κ (`pde_kappa.json`):
  - κ ≈ **−0.80** (t ≈ −105, `n_pairs ≈ 150k`, `R² ≈ 0.069`)
  - Figure: `results/figures/fig_kappa_scatter.png`
- Diffusion D (`pde_diffusion.json`):
  - D ≈ `3.2×10⁻¹²`, ΔR² ≈ `6.1×10⁻⁴` (effectively zero)
  - Figure: `results/figures/fig_diffusion_scatter.png`
- Interface Γ (`pde_interface.json`):
  - Γ ≈ `3.9×10⁻¹²`, ΔR² ≈ `6.1×10⁻⁴` (effectively zero)
  - Figure: `results/figures/fig_interface_scatter.png`

## Minimal PDE conclusion
- Keep the **potential term (κ)** and the **loop term (η)** as the only persistent structure terms.
- Document D/Γ as negligible on the current SEMCOG 2020 dataset (ΔR² < 0.001).
- All seven figures for the “4 visuals + 4 analyses” suite exist under `results/figures/`.
