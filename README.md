Project: SEMCOG 2020 OD Diagnostics

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

Requirements
- Python 3.8+.
- No hard external deps. If `numpy`, `pandas`, `pyarrow`, or `matplotlib` are available, the pipeline will leverage them; otherwise it will degrade gracefully to CSV-only and simple plotting is skipped.

Geo centroids
- `data/geo/tracts.geojson` should include tract GEOIDs and either internal points or centroids:
  - GEOID keys accepted: `GEOID`, `GEOID10`, or `geoid`.
  - Lat/Lon keys accepted (preferred): `INTPTLAT10`, `INTPTLON10`, `centroid_lat`, `centroid_lon`.
  - If none found, a naive centroid (bbox mid) is computed as a fallback.

Quickstart
- Place your OD CSV at `data/raw/semcog_tract_od_intra_2020.csv` (columns: `work,home,S000`).
- Place your tract GeoJSON at `data/geo/tracts.geojson`.
- Run end-to-end:
  - `python -m project.src.cli all`
- Or step-by-step:
  - `python -m project.src.cli prepare`
  - `python -m project.src.cli hodge --use-independence`
  - `python -m project.src.cli locality --bins 12`

Outputs
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

Counties (SEMCOG 7)
- Wayne(26163), Oakland(26125), Macomb(26099), Washtenaw(26161), Monroe(26115), Livingston(26093), St. Clair(26147).
