Pipeline: SEMCOG 2020 OD Diagnostics

**三图逻辑总结**

- R²/η 柱图
  - 现象：独立模型基线后，R²≈0.194，η≈0.898。
  - 含义：单势场仅解释约19%方向性；约80%能量来自闭环/非互易。
  - 启发：需要“多势”或“势+反对称边项”来吸纳闭环并提升解释力。

- 位势箱线图
  - 现象：π 县际分布差异明显：Livingston/Washtenaw/St. Clair 较高，Oakland/Macomb/Monroe/Wayne 较低。
  - 读法：g_ij≈π_j−π_i，低π→高π对应净吸引方向（已剥离规模效应后）。
  - 启发：π 可作为外势 U(x) 的监督目标，用可解释特征做回归（可达性、POI、房价/收入等）。

- 局域性曲线
  - 现象：5–15km 内 R²≈0.58→0.38；引入更远边后降至≈0.22。
  - 含义：短距更“势场化”；中长距更非互易/路径依赖。
  - 启发：基线需显式扣除距离/时间摩擦；多势或反对称项重点解释中长距。

整体结论：短距≈势场化；中长距≈闭环主导。行动顺序：升级基线（PPML+距离/行政FE）→ 稳健 Hodge → 多势/反对称建模 → 外势回归 U(x)。

**完整可执行 Pipeline（目的→要做→产物/验收）**

- 环境建议
  - conda create -n semcog-od python=3.10 -y && conda activate semcog-od
  - conda install -c conda-forge numpy pandas scipy matplotlib statsmodels patsy pyarrow pyshp scikit-learn -y

- Phase 0｜数据就绪（已完成）
  - 目的：标准化 OD 与生成质心/距离。
  - 动作：
    - 读 OD：python -m project.src.cli ingest --raw dataset/od_flow/semcog_tract_od_intra_2020.csv
    - 边际：python -m project.src.cli marginals
    - 质心/距离：python -m project.src.cli shp2geojson --shp dataset/geo/mi-tracts-demo-work.shp --out project/data/geo/tracts.geojson && python -m project.src.cli prepare
  - 产物：data/processed/od_clean.csv、nodes_basic.csv、edges_with_distance.csv
  - 验收：ΣR=ΣW=ΣF；GEOID=11位；distance_km 合理。

- Phase 1｜基线升级：PPML + 距离/行政 FE（优先）
  - 目的：扣除可由距离/行政边界解释的部分，得到更纯的残差。
  - 动作：
    - PPML：python -m project.src.cli baseline_glm --eps 0.5 --county-pair-fe
      - 稀疏回退（scikit-learn OneHot+PoissonRegressor）已内置，避免内存爆。
    - 定义：log μ_ij = α_i + β_j − λ·dist_ij [+ county×county FE]；log_resid_glm = log((F_ij+ε)/(μ_ij+ε))。
  - 产物：data/processed/od_residual_glm.parquet|csv（含 mu_hat、log_resid_glm）、results/diagnostics/baseline_glm_summary.json（λ、deviance、backend）。
  - 验收：对比独立模型，Hodge 复算时 R²↑、η↓。

- Phase 2｜稳健的势/非互易诊断（Hodge）
  - 目的：结论对采样/数值/权重不敏感。
  - 动作：
    - 稳健 Hodge（GLM 残差）：
      - python -m project.src.cli potential_hodge_glm --weight-type cap --cap-tau 200 --drop-self --sample-edges 120000 --bins-dist 8 --bins-weight 8 --seed 42 --max-edges 150000 --maxiter 300 --tol 1e-5
      - 分层抽样：按 dist×(F_ij+F_ji) 均衡抽样；seed 固定。
      - 多分量锚定：各连通分量各自锚点，稀疏拉普拉斯解 Lπ=div(g)。
      - 权重对照：sum（F_ij+F_ji）、cap（截断τ）、mu（μ_ij+μ_ji）、eij（E_ij+E_ji）。
      - ε 巡检：ε∈{0.1,0.5,1.0）；自流开关：--drop-self。
  - 产物：results/diagnostics/nodes_potential_glm.csv、edges_decomp_glm.csv、edges_topk_glm.csv、summary_robustness.json。
  - 验收：主口径结论稳健；R²/η 方差合理；Top-K 残差边便于巡查异常。

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

