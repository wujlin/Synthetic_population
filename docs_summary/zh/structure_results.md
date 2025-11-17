# 结构诊断摘要（中文）

## Phase 1：PPML 基线
- 命令：`python -m project.src.cli baseline_glm --backend sklearn --max-iter 10000 --alpha 1e-10 --standardize-dist`
- 样本量：**496,096**
- 距离系数：**λ_dist ≈ −0.94**（sklearn PoissonRegressor、距离标准化、无县×县双向固定效应）
- 日志：`results/diagnostics/baseline_glm_summary.json`

## Phase 2：Robust Hodge + Locality
- 参数：weight-type=`cap`，τ=200，去自环，距离×流量分层抽样 120k，bins_dist=8，bins_weight=8，seed=42
- 产物：`nodes_potential_glm.csv`、`edges_decomp_glm.csv`、`edges_topk_glm.csv`、`summary_robustness.json`
- 最新指标：**R² ≈ 0.108**，**η ≈ 0.935**，节点 1475，使用边 113,011
- 图像：`results/figures/r2_eta.png`、`pi_box_by_county.png`、`fig_locality_curve.png`

## 旋度与循环热点
- 命令：`python -m project.src.cli rot_diagnostics --topk 2000 --topcycles 500 --nodes-geo project/data/geo/tracts.geojson`
- 概览：`results/diagnostics/rot_summary.json`（全局 η≈0.935，热点边 2000 条，三角环 500 条）
- 相关文件：`fig_cycles_hotspots.png`、`results/diagnostics/top_cycles.csv`、`results/diagnostics/edges_topk_glm.csv`

## Phase 3：PDE 结构项
- κ（`pde_kappa.json`）：
  - κ ≈ **−0.80**，t ≈ −105，`n_pairs ≈ 150k`，`R² ≈ 0.069`
  - 图像：`results/figures/fig_kappa_scatter.png`
- 扩散 D（`pde_diffusion.json`）：
  - D ≈ `3.2×10⁻¹²`，ΔR² ≈ `6.1×10⁻⁴`，贡献可忽略
  - 图像：`results/figures/fig_diffusion_scatter.png`
- 界面 Γ（`pde_interface.json`）：
  - Γ ≈ `3.9×10⁻¹²`，ΔR² ≈ `6.1×10⁻⁴`，贡献可忽略
  - 图像：`results/figures/fig_interface_scatter.png`

## 最小 PDE 结论
- 保留 **势项 κ** 与 **循环/η**，它们是唯一在本数据上显著的结构项。
- 扩散 D、界面 Γ 的增益 <0.001，可在报告中标注为“不保留”。
- “4 张图 + 4 分析” 套件已全部生成，位于 `results/figures/` 与 `results/diagnostics/`。
