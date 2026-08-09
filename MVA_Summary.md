# MVA Summary: Multivariate Joint-Effect Analysis / MVA汇总：多属性联合作用分析

> Generated dynamically by `scripts/mva_analysis.py` / 由 MVA 分析脚本动态生成
> Dataset / 数据源: 244 valid MOFs; X = 51 full descriptors; Y = 19 metrics (grouped by |r| >= 0.99)

## 1. Y-Target Redundancy Grouping / Y目标冗余分组（|r| ≥ 0.99 只罗列）

| Group / 组 | Members / 成员 | N | Representative / 代表 |
| :--- | :--- | :---: | :--- |
| 1 | CO2_ads_0.15bar; CO2_VSA_capacity; CO2_TSA_capacity; CO2_TSA_regen_heat; log10_CO2_TSA_regen_heat | 5 | **log10_CO2_TSA_regen_heat** |
| 2 | CO2_Qst_CC_mean; CO2_Qst_CC_0.1bar; CO2_Qst_CC_0.5bar; CO2_Qst_GCMC_flue_mean; CO2_Qst_Widom | 5 | **CO2_Qst_CC_mean** |
| 3 | CO2_Qst_CC_1bar | 1 | **CO2_Qst_CC_1bar** |
| 4 | CO2N2_actual_selectivity; log10_CO2N2_actual_selectivity | 2 | **log10_CO2N2_actual_selectivity** |
| 5 | CO2N2_Henry_selectivity; log10_CO2N2_Henry_selectivity | 2 | **log10_CO2N2_Henry_selectivity** |
| 6 | PE_VSA_parasitic_energy; log10_PE_VSA_parasitic_energy | 2 | **log10_PE_VSA_parasitic_energy** |
| 7 | N2_Qst_Widom | 1 | **N2_Qst_Widom** |
| 8 | N2_Qst_GCMC_flue_mean; N2_Qst_1bar | 2 | **N2_Qst_GCMC_flue_mean** |
| 9 | N2_ads_0.75bar; N2_ads_1bar | 2 | **N2_ads_0.75bar** |
| 10 | Qst_diff_CO2_N2 | 1 | **Qst_diff_CO2_N2** |

共 10 组，实际建模 **10** 个代表 target（其余只罗列不重复建模）。

## 2. Cross-Validation Model Performance per Target / 各代表target的CV模型表现

| Target / 目标 | Best Model / 最佳模型 | R² (mean ± std) | MAE | RMSE |
| :--- | :--- | :---: | :---: | :---: |
| **CO2_Qst_CC_1bar** | ExtraTrees | **0.710 ± 0.136** | 1.250 | 1.927 |
| **CO2_Qst_CC_mean** | ExtraTrees | **0.693 ± 0.142** | 1.437 | 2.167 |
| **N2_Qst_GCMC_flue_mean** | ExtraTrees | **0.758 ± 0.120** | 0.865 | 1.335 |
| **N2_Qst_Widom** | ExtraTrees | **0.539 ± 0.235** | 1.119 | 2.001 |
| **N2_ads_0.75bar** | RandomForest | **0.482 ± 0.148** | 0.057 | 0.082 |
| **Qst_diff_CO2_N2** | RandomForest | **0.348 ± 0.185** | 1.246 | 2.046 |
| **log10_CO2N2_Henry_selectivity** | XGBoost | **0.533 ± 0.141** | 0.114 | 0.189 |
| **log10_CO2N2_actual_selectivity** | ExtraTrees | **0.716 ± 0.131** | 0.070 | 0.106 |
| **log10_CO2_TSA_regen_heat** | ExtraTrees | **0.528 ± 0.157** | 0.137 | 0.220 |
| **log10_PE_VSA_parasitic_energy** | RandomForest | **0.675 ± 0.185** | 0.020 | 0.032 |

## 3. Key Findings per Target / 各target核心发现

### 3.x log10_CO2_TSA_regen_heat

**Dominance（子集回归平均贡献）Top 5:**

| Feature / 特征 | Dominance | Unique | Total R² (single) |
| :--- | :---: | :---: | :---: |
| void_fraction | 0.1026 | 0.0895 | 0.1676 |
| LCD_A | 0.0744 | 0.0231 | 0.1867 |
| PLD_A | 0.0730 | 0.0100 | 0.1991 |
| ligand_n_aromatic_rings | 0.0333 | 0.0341 | 0.0218 |
| ASA_m2_cm3 | 0.0331 | 0.0702 | 0.0004 |

**ML Top-5 Features（RF/XGB/permutation 综合排名）:**

- `PLD_A` (rank 1.3); `void_fraction` (rank 1.7); `LFPD_A` (rank 4.0); `LCD_A` (rank 4.3); `C_mass_frac` (rank 4.3)

**显著交互（FDR < 0.05）:**

- `PLD_A × ASA_m2_g`: β = -0.163（拮抗/抵消），p_fdr = 0.0083
- `PLD_A × density_g_cm3`: β = 0.251（协同/放大），p_fdr = 0.0083
- `PLD_A × LCD_A`: β = -0.085（拮抗/抵消），p_fdr = 0.0441
- `PLD_A × pore_vol_cm3_g`: β = 0.251（协同/放大），p_fdr = 0.0083
- `LCD_A × void_fraction`: β = -0.274（拮抗/抵消），p_fdr = 0.0083
- `density_g_cm3 × void_fraction`: β = 0.382（协同/放大），p_fdr = 0.0167


### 3.x CO2_Qst_CC_mean

**Dominance（子集回归平均贡献）Top 5:**

| Feature / 特征 | Dominance | Unique | Total R² (single) |
| :--- | :---: | :---: | :---: |
| void_fraction | 0.1278 | 0.0600 | 0.5048 |
| ASA_m2_g | 0.0896 | 0.0080 | 0.4578 |
| LFPD_A | 0.0555 | 0.0034 | 0.3510 |
| LCD_A | 0.0548 | 0.0112 | 0.3554 |
| ASA_m2_cm3 | 0.0344 | 0.0368 | 0.0993 |

**ML Top-5 Features（RF/XGB/permutation 综合排名）:**

- `void_fraction` (rank 1.0); `LCD_A` (rank 2.7); `PLD_A` (rank 3.0); `ASA_m2_g` (rank 5.0); `LFPD_A` (rank 5.3)


### 3.x CO2_Qst_CC_1bar

**Dominance（子集回归平均贡献）Top 5:**

| Feature / 特征 | Dominance | Unique | Total R² (single) |
| :--- | :---: | :---: | :---: |
| void_fraction | 0.2193 | 0.0944 | 0.5609 |
| LFPD_A | 0.1170 | 0.0056 | 0.4258 |
| LCD_A | 0.1144 | 0.0100 | 0.4283 |
| ASA_m2_cm3 | 0.0478 | 0.0286 | 0.0922 |
| topo_pcu | 0.0211 | 0.0169 | 0.0357 |

**ML Top-5 Features（RF/XGB/permutation 综合排名）:**

- `void_fraction` (rank 1.0); `LCD_A` (rank 2.3); `PLD_A` (rank 5.0); `LFPD_A` (rank 5.0); `ASA_m2_g` (rank 6.0)

**显著交互（FDR < 0.05）:**

- `density_g_cm3 × void_fraction`: β = -0.315（拮抗/抵消），p_fdr = 0.0265


### 3.x log10_CO2N2_actual_selectivity

**Dominance（子集回归平均贡献）Top 5:**

| Feature / 特征 | Dominance | Unique | Total R² (single) |
| :--- | :---: | :---: | :---: |
| void_fraction | 0.1672 | 0.0921 | 0.5299 |
| LFPD_A | 0.0734 | 0.0014 | 0.4220 |
| LCD_A | 0.0731 | 0.0067 | 0.4253 |
| PLD_A | 0.0499 | 0.0023 | 0.3329 |
| ASA_m2_cm3 | 0.0425 | 0.0310 | 0.0733 |

**ML Top-5 Features（RF/XGB/permutation 综合排名）:**

- `void_fraction` (rank 1.0); `LCD_A` (rank 2.3); `PLD_A` (rank 3.0); `LFPD_A` (rank 4.0); `ASA_m2_g` (rank 7.3)

**显著交互（FDR < 0.05）:**

- `PLD_A × density_g_cm3`: β = -0.170（拮抗/抵消），p_fdr = 0.0210
- `PLD_A × pore_vol_cm3_g`: β = -0.170（拮抗/抵消），p_fdr = 0.0210
- `LCD_A × void_fraction`: β = 0.178（协同/放大），p_fdr = 0.0210
- `density_g_cm3 × void_fraction`: β = -0.295（拮抗/抵消），p_fdr = 0.0210


### 3.x log10_CO2N2_Henry_selectivity

**Dominance（子集回归平均贡献）Top 5:**

| Feature / 特征 | Dominance | Unique | Total R² (single) |
| :--- | :---: | :---: | :---: |
| void_fraction | 0.2122 | 0.0692 | 0.3725 |
| ASA_m2_g | 0.1480 | 0.0015 | 0.3155 |
| topo_dia | 0.0393 | 0.0339 | 0.0528 |
| ligand_n_aromatic_rings | 0.0278 | 0.0462 | 0.0003 |
| ASA_m2_cm3 | 0.0266 | 0.0173 | 0.0714 |

**ML Top-5 Features（RF/XGB/permutation 综合排名）:**

- `void_fraction` (rank 1.0); `PLD_A` (rank 2.3); `LCD_A` (rank 3.0); `uc_volume_A3` (rank 6.0); `ASA_m2_g` (rank 6.3)


### 3.x log10_PE_VSA_parasitic_energy

**Dominance（子集回归平均贡献）Top 5:**

| Feature / 特征 | Dominance | Unique | Total R² (single) |
| :--- | :---: | :---: | :---: |
| void_fraction | 0.0883 | 0.0516 | 0.5152 |
| ASA_m2_g | 0.0763 | 0.0071 | 0.4679 |
| LFPD_A | 0.0627 | 0.0048 | 0.5209 |
| LCD_A | 0.0571 | 0.0098 | 0.5145 |
| ASA_m2_cm3 | 0.0458 | 0.0618 | 0.0245 |

**ML Top-5 Features（RF/XGB/permutation 综合排名）:**

- `void_fraction` (rank 1.0); `LFPD_A` (rank 2.3); `PLD_A` (rank 4.3); `LCD_A` (rank 5.0); `pore_vol_cm3_g` (rank 5.0)


### 3.x N2_Qst_Widom

**Dominance（子集回归平均贡献）Top 5:**

| Feature / 特征 | Dominance | Unique | Total R² (single) |
| :--- | :---: | :---: | :---: |
| void_fraction | 0.2694 | 0.1061 | 0.4575 |
| LFPD_A | 0.1587 | 0.0245 | 0.3233 |
| topo_pcu | 0.0431 | 0.0339 | 0.0560 |
| LCD_PLD_ratio | 0.0167 | 0.0111 | 0.0452 |
| NASA_m2_cm3 | 0.0089 | 0.0019 | 0.0098 |

**ML Top-5 Features（RF/XGB/permutation 综合排名）:**

- `void_fraction` (rank 1.0); `PLD_A` (rank 2.0); `LCD_A` (rank 3.7); `pore_vol_cm3_g` (rank 3.7); `LFPD_A` (rank 4.7)


### 3.x N2_Qst_GCMC_flue_mean

**Dominance（子集回归平均贡献）Top 5:**

| Feature / 特征 | Dominance | Unique | Total R² (single) |
| :--- | :---: | :---: | :---: |
| void_fraction | 0.2640 | 0.0886 | 0.6173 |
| LCD_A | 0.1431 | 0.0049 | 0.4496 |
| PLD_A | 0.0925 | 0.0053 | 0.3326 |
| ASA_m2_cm3 | 0.0620 | 0.0178 | 0.1219 |
| topo_pcu | 0.0143 | 0.0107 | 0.0267 |

**ML Top-5 Features（RF/XGB/permutation 综合排名）:**

- `void_fraction` (rank 1.0); `LCD_A` (rank 2.7); `PLD_A` (rank 3.7); `pore_vol_cm3_g` (rank 3.7); `LFPD_A` (rank 6.7)

**显著交互（FDR < 0.05）:**

- `PLD_A × density_g_cm3`: β = -0.160（拮抗/抵消），p_fdr = 0.0307
- `PLD_A × pore_vol_cm3_g`: β = -0.160（拮抗/抵消），p_fdr = 0.0307
- `LCD_A × void_fraction`: β = 0.159（协同/放大），p_fdr = 0.0307


### 3.x N2_ads_0.75bar

**Dominance（子集回归平均贡献）Top 5:**

| Feature / 特征 | Dominance | Unique | Total R² (single) |
| :--- | :---: | :---: | :---: |
| ligand_n_aromatic_rings | 0.0485 | 0.0194 | 0.0718 |
| PLD_A | 0.0327 | 0.0219 | 0.0536 |
| ASA_m2_cm3 | 0.0266 | 0.0466 | 0.0210 |
| LCD_A | 0.0216 | 0.0247 | 0.0241 |
| void_fraction | 0.0204 | 0.0386 | 0.0056 |

**ML Top-5 Features（RF/XGB/permutation 综合排名）:**

- `LCD_A` (rank 1.0); `C_mass_frac` (rank 2.3); `ligand_n_aromatic_rings` (rank 2.7); `PLD_A` (rank 4.7); `LFPD_A` (rank 6.3)

**显著交互（FDR < 0.05）:**

- `PLD_A × ASA_m2_g`: β = 0.261（协同/放大），p_fdr = 0.0001
- `PLD_A × density_g_cm3`: β = -0.349（拮抗/抵消），p_fdr = 0.0003
- `PLD_A × LCD_A`: β = 0.169（协同/放大），p_fdr = 0.0001
- `PLD_A × pore_vol_cm3_g`: β = -0.349（拮抗/抵消），p_fdr = 0.0003
- `LCD_A × void_fraction`: β = 0.402（协同/放大），p_fdr = 0.0000


### 3.x Qst_diff_CO2_N2

**Dominance（子集回归平均贡献）Top 5:**

| Feature / 特征 | Dominance | Unique | Total R² (single) |
| :--- | :---: | :---: | :---: |
| catenation | 0.0566 | 0.0410 | 0.0577 |
| ASA_m2_g | 0.0490 | 0.0172 | 0.0775 |
| ligand_n_aromatic_rings | 0.0485 | 0.0456 | 0.0319 |
| topo_dia | 0.0414 | 0.0356 | 0.0502 |
| void_fraction | 0.0407 | 0.0133 | 0.0658 |

**ML Top-5 Features（RF/XGB/permutation 综合排名）:**

- `ASA_m2_g` (rank 1.7); `catenation` (rank 3.3); `LFPD_A` (rank 4.7); `uc_volume_A3` (rank 5.3); `LCD_A` (rank 5.7)



## 4. Cross-Target Stable Interactions / 跨target稳健交互

| Interaction Pair / 交互对 | # Targets (FDR<0.05) |
| :--- | :---: |
| `LCD_A × void_fraction` | 4 |
| `PLD_A × density_g_cm3` | 4 |
| `PLD_A × pore_vol_cm3_g` | 4 |
| `density_g_cm3 × void_fraction` | 3 |
| `PLD_A × ASA_m2_g` | 2 |
| `PLD_A × LCD_A` | 2 |

在多个 target 上同时显著的交互对，是物理上更可信的联合作用信号。


## 5. Suppression Effects（单变量 r 与净效应 β 方向反转）

| Target / 目标 | Feature / 特征 | Spearman r | β (净效应) |
| :--- | :--- | :---: | :---: |
| log10_CO2_TSA_regen_heat | `LCD_A` | 0.520 | -0.722 |
| log10_CO2_TSA_regen_heat | `pore_vol_cm3_g` | -0.325 | 0.174 |
| log10_CO2_TSA_regen_heat | `ASA_m2_g` | 0.282 | -0.195 |
| log10_CO2_TSA_regen_heat | `NASA_m2_cm3` | -0.140 | 0.500 |
| log10_CO2_TSA_regen_heat | `density_g_cm3` | -0.325 | 0.174 |
| log10_CO2_TSA_regen_heat | `N_mass_frac` | 0.149 | -0.144 |
| log10_CO2_TSA_regen_heat | `ligand_n_heavy` | -0.113 | 1.224 |
| CO2_Qst_CC_mean | `LCD_A` | -0.548 | 0.335 |
| CO2_Qst_CC_mean | `LCD_PLD_ratio` | -0.132 | 0.154 |
| CO2_Qst_CC_mean | `ASA_m2_cm3` | -0.374 | 0.280 |
| CO2_Qst_CC_mean | `NASA_m2_cm3` | 0.203 | -0.466 |
| CO2_Qst_CC_mean | `O_coord_ratio` | -0.137 | 0.215 |
| CO2_Qst_CC_1bar | `LCD_A` | -0.609 | 0.475 |
| CO2_Qst_CC_1bar | `ASA_m2_cm3` | -0.387 | 0.276 |
| CO2_Qst_CC_1bar | `NASA_m2_cm3` | 0.171 | -0.496 |
| CO2_Qst_CC_1bar | `O_coord_ratio` | -0.133 | 0.182 |
| log10_CO2N2_actual_selectivity | `LCD_A` | -0.644 | 0.471 |
| log10_CO2N2_actual_selectivity | `ASA_m2_cm3` | -0.336 | 0.309 |
| log10_CO2N2_actual_selectivity | `NASA_m2_cm3` | 0.155 | -0.354 |
| log10_CO2N2_Henry_selectivity | `LCD_A` | -0.513 | 0.140 |
| log10_CO2N2_Henry_selectivity | `LCD_PLD_ratio` | -0.101 | 0.169 |
| log10_CO2N2_Henry_selectivity | `ASA_m2_cm3` | -0.323 | 0.290 |
| log10_CO2N2_Henry_selectivity | `NASA_m2_cm3` | 0.213 | -0.403 |
| log10_CO2N2_Henry_selectivity | `O_coord_ratio` | -0.130 | 0.257 |
| log10_PE_VSA_parasitic_energy | `LCD_A` | 0.711 | -0.713 |
| log10_PE_VSA_parasitic_energy | `ASA_m2_cm3` | 0.293 | -0.524 |
| log10_PE_VSA_parasitic_energy | `NASA_m2_cm3` | -0.108 | 0.300 |
| log10_PE_VSA_parasitic_energy | `ligand_n_amino` | 0.248 | -0.150 |
| N2_Qst_Widom | `LCD_PLD_ratio` | -0.155 | 0.199 |
| N2_Qst_Widom | `ASA_m2_cm3` | -0.368 | 0.136 |
| N2_Qst_Widom | `NASA_m2_cm3` | 0.132 | -0.289 |
| N2_Qst_Widom | `ligand_n_heavy` | -0.108 | 1.013 |
| N2_Qst_GCMC_flue_mean | `LCD_A` | -0.658 | 0.222 |
| N2_Qst_GCMC_flue_mean | `ASA_m2_cm3` | -0.394 | 0.199 |
| N2_Qst_GCMC_flue_mean | `NASA_m2_cm3` | 0.114 | -0.357 |
| N2_ads_0.75bar | `LCD_A` | -0.229 | 0.839 |
| N2_ads_0.75bar | `LFPD_A` | -0.232 | 0.244 |
| N2_ads_0.75bar | `LCD_PLD_ratio` | 0.101 | -0.339 |
| N2_ads_0.75bar | `ligand_MW` | 0.320 | -0.625 |
| N2_ads_0.75bar | `ligand_n_heavy` | 0.322 | -0.315 |
| Qst_diff_CO2_N2 | `LCD_A` | -0.104 | 0.159 |
| Qst_diff_CO2_N2 | `PLD_A` | -0.225 | 0.274 |
| Qst_diff_CO2_N2 | `pore_vol_cm3_g` | 0.129 | -0.117 |
| Qst_diff_CO2_N2 | `ASA_m2_cm3` | -0.186 | 0.305 |
| Qst_diff_CO2_N2 | `NASA_m2_cm3` | 0.302 | -0.606 |
| Qst_diff_CO2_N2 | `density_g_cm3` | 0.129 | -0.117 |
| Qst_diff_CO2_N2 | `O_coord_ratio` | -0.155 | 0.240 |
| Qst_diff_CO2_N2 | `ligand_MW` | 0.184 | -0.151 |

这些特征单变量相关性与控制其他变量后的净效应方向相反，提示存在共线性掩盖或间接效应——这是单变量分析无法揭示的。

## 6. Output Files / 输出文件清单

| File / 文件 | Content / 内容 |
| :--- | :--- |
| `y_redundant_groups.csv` | Y目标冗余分组（|r|≥0.99，只罗列） |
| `y_redundant_pairs.csv` | 冗余对明细 |
| `vif_diagnostic.csv` | 全特征VIF（诊断，未剔除） |
| `mva_ols_beta.csv` | 全特征标准化β / 半偏相关 / p值 |
| `mva_ols_details.csv` | OLS与Lasso的R²摘要 |
| `mva_lasso_features.csv` | Lasso稀疏特征选择 |
| `mva_univariate_spearman.csv` | 单变量Spearman相关（对比基准） |
| `mva_suppression_findings.csv` | 抑制效应（方向反转） |
| `mva_dominance.csv` | Dominance分析（top-12特征） |
| `mva_commonality.csv` | Commonality分解（unique/两两共同） |
| `mva_interactions.csv` | 物理候选对交互项回归（BH-FDR） |
| `mva_shap_interaction_pairs.csv` | SHAP全对扫描的top交互对（无预设假设） |
| `mva_shap_decomposition.csv` | SHAP主效应 vs 交互效应分解 |
| `mva_model_metrics.csv` | 5模型重复5折CV指标 |
| `mva_top_features.csv` | 每target综合Top-15特征 |
| `mva_beta_heatmap.png` | 标准化β热图 |
| `mva_dominance_heatmap.png` | Dominance热图 |
| `mva_model_cv.png` | 模型CV R²对比 |
| `pdp2d_*.png` | 2D部分依赖图（top SHAP交互对） |
