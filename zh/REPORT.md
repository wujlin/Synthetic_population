# 结果报告（中文）

## 1. Phase 0–2 概览
- 数据：SEMCOG 2020 tract→tract OD（7 县，约 50 万对）。
- PPML（Phase 1）：`baseline_glm`（sklearn PoissonRegressor，距离标准化），得到 $\lambda_{dist} \approx -0.94$。
- Robust Hodge（Phase 2）：cap 权重 + 分层抽样（120k 边），$R^2 \approx 0.108$，$\eta \approx 0.935$。

### 关键图像
1. `r2_eta.png`（独立基线 vs Robust Hodge）
2. `pi_box_by_county.png`（县级 π）
3. `fig_locality_curve.png`（R² vs 距离阈值）

## 2. 循环诊断
- `rot_diagnostics`: $\eta_{\text{global}} \approx 0.935$，热点边 2000 条，top cycles 500 条。
- 图像：`fig_cycles_hotspots.png` 展示 Detroit–Ann Arbor / 湖岸走廊的非互易走廊。

## 3. PDE 结构项
| 项 | 结果 | 说明 |
| --- | --- | --- |
| 势项 $\kappa$ | **−0.80**（t≈−105，`R² ≈ 0.069`） | 明显保留 |
| 扩散 $D$ | $3.2\times 10^{-12}$，$\Delta R^2 \approx 6.1\times10^{-4}$ | 影响极小，可记为 0 |
| 界面 $\Gamma$ | $3.9\times 10^{-12}$，$\Delta R^2 \approx 6.1\times10^{-4}$ | 同样可忽略 |

对应图像：`fig_kappa_scatter.png`、`fig_diffusion_scatter.png`、`fig_interface_scatter.png`。

## 4. 最小 PDE 结论
- 结构项主线为：**势项 κ + 循环/η**；D 与 Γ 在当前数据上不足以提升解释力（ΔR² < 0.001）。
- 建议在报告/展示时明确写出：“保留 κ 与 η；D、Γ 记录为 0”。

## 5. 参考文件
- JSON：`baseline_glm_summary.json`、`summary_robustness.json`、`rot_summary.json`、`pde_*.json`
- 图像：所有 PNG 位于 `results/figures/`
- 双语概览：`project/docs_summary/{en,zh}/structure_results.md`
