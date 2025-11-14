Pipeline: SEMCOG 2020 OD Diagnostics

## Three‑figure logic (at a glance)

- R²/η 柱图
  - Observation: After independence baseline, R²≈0.194, η≈0.898.
  - Meaning: A single potential explains ~19% of directions; ~80% comes from loops/non‑reciprocity.
  - Implication: Move toward multi‑potential or potential + anti‑symmetric edge terms.

- 位势箱线图
  - Observation: County‑level π differs systematically (e.g., Livingston/Washtenaw/St. Clair higher; Oakland/Macomb/Monroe/Wayne lower).
  - Reading: g_ij≈π_j−π_i; low‑π→high‑π aligns with net attraction (after scale effect removal).
  - Implication: π is a target for interpretable external‑feature regression U(x) (access, POIs, housing/income, etc.).

- 局域性曲线
  - Observation: R²≈0.58→0.38 for 5–15 km; adding longer links reduces R² ≈ 0.22.
  - Meaning: Short distances are more potential‑like; medium/long distances show stronger non‑reciprocity.
  - Implication: Baseline must remove distance/time friction; multi‑potential or anti‑symmetric terms should focus on medium/long distances.

Overall: Short‑distance ≈ potential‑like; medium/long‑distance ≈ loop‑dominated. Actions: PPML baseline → robust Hodge → multi‑potential/anti‑symmetric modeling → U(x) regression.

## 流程图

```mermaid
flowchart LR
  A[Phase 0 数据就绪] --> B[Phase 1 PPML基线\nlog μ=α+β-λ·dist (+FE)]
  B --> C[Phase 2 Hodge诊断\n拟合 π → R², η]
  C --> D{Phase 3+}
  D --> D1[局域性曲线 r*]
  D --> D2[多势/反对称项]
  D --> D3[外势 U(x) 回归]
```

## 完整可执行 Pipeline（目的→要做→产物/验收）

- 环境建议
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
Definition: `log μ_ij = α_i + β_j − λ·dist_ij [+ county×county FE]`; `log_resid_glm = log((F_ij+ε)/(μ_ij+ε))`.

**Outputs**: `data/processed/od_residual_glm.parquet|csv` (`mu_hat, log_resid_glm`), `results/diagnostics/baseline_glm_summary.json` (λ, deviance, backend).

**Acceptance**: vs independent baseline, downstream Hodge shows **R² ↑, η ↓**.

### Phase 2 — Robust Hodge: potential/non‑reciprocity
**Purpose**: Make conclusions robust to sampling/numerics/weights.

**Process**:
```bash
python -m project.src.cli potential_hodge_glm --weight-type cap --cap-tau 200 --drop-self \
  --sample-edges 120000 --bins-dist 8 --bins-weight 8 --seed 42 \
  --max-edges 150000 --maxiter 300 --tol 1e-5
```
Notes: stratified sampling (`dist×(F_ij+F_ji)`), component‑wise anchoring (sparse Laplacian `Lπ=div(g)`), weight baselines, ε sweeps, self‑loop toggle.

**Outputs**: `results/diagnostics/nodes_potential_glm.csv`, `edges_decomp_glm.csv`, `edges_topk_glm.csv`, `summary_robustness.json`.

**Acceptance**: conclusions stable under settings; reasonable variance in R²/η; Top‑K residual edges for inspection.

- Phase 3｜局域性两条曲线（口径统一）
  - 目的：区分“短距更好解释”与“训练子集效应”。
  - 动作：
    - 曲线 A（已实现）：在 dist ≤ r0 子集上训练并评估：python -m project.src.cli locality_curve --radii 5 10 15 20 25 30 40 50
    - 曲线 B（建议新增）：基于全量训练的 π，仅在 dist ≤ r0 上评估（保证单调不降）。
    - 计算半能量半径 r*（R²≥0.5 的最小 r0）与 CI（重复抽样）。
  - 产物：results/diagnostics/locality_report.json + results/figures/fig_locality_curve.png。
  - 验收：曲线 B 单调不降；曲线 A 高于 B；给出 r* 与 CI。

- Phase 4｜闭环结构刻画（让 η 有地理形状）
  - 目的：把“非互易”结构化成可解释的回路/走廊。
  - 动作：在 edges_decomp_glm 上近似三元环强度 C_ijk=g_ij+g_jk+g_ki；输出 Top-K 回路清单与热区图（后续命令可加）。
  - 产物：results/diagnostics/top_cycles.csv、results/figures/fig_cycles_hotspots.png。
  - 验收：主回路与跨县/中长距走廊直觉一致。

- Phase 5｜多势场 / 势+反对称（两条路线二选一）
  - 路线 5A（多势）：NMF/谱聚类得到 K 个子网络；对子网络做 PPML→Hodge→π^(k)；组合 log F_ij = α_i + β_j − λ d_ij + log Σ_k w_k exp(π^(k)(j) − π^(k)(i))；留出集验证 ΔR²/ΔLL。
  - 路线 5B（势+反对称边项）：log F_ij = α_i + β_j − λ d_ij + [π(j)−π(i)] + A_ij（A_ij=-A_ji）；设计反对称特征；拟合 A_ij；留出集验证。
  - 产物：multi_potential_nmf/* 或 antisym_edge_model/*。
  - 验收：留出集相对“单势+距离”显著提升；特征方向与地理直觉一致。

- Phase 6｜外势 U(x) 可解释回归（节点级）
  - 目的：把 π（或 π^(k)）解释为可操作外部因子。
  - 动作：线性/岭/GAM（优先可解释），必要时轻量 MLP 作对照；交叉验证；输出系数/SHAP 与位势地图。
  - 产物：u_regression/*、results/figures/fig_U_maps.png。
  - 验收：解释度达标；空间分布与系数方向合理；可做情景推演。

- Phase 7｜报告与可视化
  - 目的：统一输出与复现。
  - 动作：python -m project.src.cli export_figs（需 matplotlib）；补 REPORT.md 记录参数与验收表。
  - 产物：results/figures/*、REPORT.md。
  - 验收：一键重跑可复现；图表与 JSON 指标一致。

**关键验收口径**

- 基线：独立→PPML 后，R² 上升、η 下降（同抽样/权重设置）。
- 局域性：曲线 B 单调不降；给出半能量半径 r* 与 CI。
- 建模（5A/5B）：留出边相对“单势+距离”有显著提升（ΔR² 或 ΔLL），且解释项方向合理。
