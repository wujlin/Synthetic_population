# 项目简介（中文）

本项目基于 SEMCOG 2020 tract→tract OD 数据，提供「Phase 0–2 清洗 + Phase 3 结构项」的可重复管线，核心目标是：
1. 通过 PPML 去除规模与距离摩擦，得到残差 $\log\frac{F_{ij}+\varepsilon}{\hat\mu_{ij}+\varepsilon}$；
2. 用 Robust Hodge 拆解残差，测量 $R^2$ 与非互易性 $\eta$，并输出循环热点；
3. 依次拟合势项 $\kappa$、扩散 $D$、界面 $\Gamma$，形成 “4 图 + 4 analyses” 套件。

## 快速开始（Phase 1–2）
```bash
# PPML baseline（sklearn PoissonRegressor，需 pandas / scikit-learn / statsmodels）
python -m project.src.cli baseline_glm --backend sklearn --max-iter 10000 --alpha 1e-10 --standardize-dist

# Robust Hodge（GLM 残差上）
python -m project.src.cli potential_hodge_glm \
  --weight-type cap --cap-tau 200 --drop-self \
  --sample-edges 120000 --bins-dist 8 --bins-weight 8 \
  --seed 42 --max-edges 150000 --maxiter 300 --tol 1e-5

# 结构项命令
python -m project.src.cli rot_diagnostics --topk 2000 --topcycles 500 --nodes-geo project/data/geo/tracts.geojson
python -m project.src.cli pde_fit_kappa
python -m project.src.cli pde_fit_diffusion
python -m project.src.cli pde_fit_interface --knn 6

# 集中导出 7 张图
python -m project.src.cli export_figs
```

## 关键产物
| 阶段 | 输出 | 说明 |
| --- | --- | --- |
| Phase 0 | `data/processed/{od_clean.csv, nodes_basic.csv, edges_with_distance.csv}` | 清洗后的 OD、节点与距离 |
| Phase 1 | `data/processed/od_residual_glm.{parquet,csv}`，`results/diagnostics/baseline_glm_summary.json` | PPML 残差 + 距离系数 $\lambda_{dist}\approx -0.94$ |
| Phase 2 | `nodes_potential_glm.csv`、`edges_decomp_glm.csv`、`summary_robustness.json` | Robust Hodge：$R^2 \approx 0.108$，$\eta \approx 0.935$ |
| Phase 2+ | `rot_summary.json`、`top_cycles.csv`、`fig_cycles_hotspots.png` | 循环热点图与三角环 |
| Phase 3 | `pde_kappa.json`、`pde_diffusion.json`、`pde_interface.json` | $\kappa \approx -0.80$ 显著；$D,\Gamma$ 均 $\mathcal{O}(10^{-12})$ |
| 图像 | `results/figures/` | 三张基础图 + `fig_cycles_hotspots` + 三张散点图 |

## 依赖
- Phase 1（baseline_glm）必须安装：`pandas`, `scikit-learn`, `statsmodels`, `patsy`。缺任何库命令会直接报错，不再 fallback。
- 其它阶段若缺 `matplotlib` / `pyarrow` 会退化为 CSV-only，并在控制台提示需要安装。

## 结构项简释
- 势项 $\kappa$：基于 $\tilde N_{ij}=(F_{ij}-F_{ji})-(\mu_{ij}-\mu_{ji})$，使用特征 $[\rho_{ij}, c_{ij}, c_{ij}\Delta\pi_{ij}]$ 进行 WLS；当前结果 $\kappa \approx -0.80$（t≈−105）。
- 扩散 $D$：拟合 $\hat N_{ij} \sim -D(\rho_j-\rho_i)$；当前 $\Delta R^2 \approx 6\times 10^{-4}$，可忽略。
- 界面 $\Gamma$：利用图拉普拉斯 $L\rho$ 构造 $-\Gamma(\nabla^2\rho_j-\nabla^2\rho_i)$；当前贡献同样 <0.001。
- 旋度/循环：`rot_diagnostics` 输出 η 层级与 top cycles，并生成循环热点图 `fig_cycles_hotspots.png`。

## 可视化套件
现有 7 张图位于 `results/figures/`：
1. `r2_eta.png`：独立基线 vs Robust Hodge；
2. `pi_box_by_county.png`：县级 π 分布；
3. `fig_locality_curve.png`：R² vs 距离阈值；
4. `fig_cycles_hotspots.png`：循环热点走廊；
5. `fig_kappa_scatter.png`：势项 κ 拟合；
6. `fig_diffusion_scatter.png`：扩散 D 拟合；
7. `fig_interface_scatter.png`：界面 Γ 拟合。

更多细节可参考 `project/docs_summary/zh/structure_results.md`（中文）与 `project/docs_summary/en/structure_results.md`（英文）。
