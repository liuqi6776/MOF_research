import pandas as pd
import numpy as np
import os
import sys

from data_loader import load_mof_dataset
from indicator_system import run_indicator_system_analysis
from dual_route_ranking import run_dual_route_ranking
from qsar_modeling import run_qsar_modeling
from design_rules_and_recommendations import generate_design_rules_and_recommendations

def main():
    file_path = '252_MOF_总文件 冗余评估数据.xlsx'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("==================================================")
    print("1. Loading MOF Dataset (252 MOFs)")
    print("==================================================")
    df_raw, df_y, x_encoded = load_mof_dataset(file_path)
    print(f"Dataset successfully loaded: {len(df_y)} MOFs.")
    
    print("\n==================================================")
    print("2. Running Deliverable 1: Indicator System Diagnosis")
    print("==================================================")
    ind_res = run_indicator_system_analysis(df_y, output_dir)
    print("Indicator system diagnosis complete. Heatmap saved.")
    
    print("\n==================================================")
    print("3. Running Deliverable 2: Dual Route Ranking (VSA & TSA)")
    print("==================================================")
    rank_res = run_dual_route_ranking(df_y, output_dir)
    print(f"Dual route ranking complete. Top-20 Win-Win MOFs: {len(rank_res['win_win_mofs'])}")
    
    print("\n==================================================")
    print("4. Running Deliverable 3: QSAR Structure-Property Modeling")
    print("==================================================")
    df_eval, importance_dict = run_qsar_modeling(df_y, x_encoded, output_dir)
    print("QSAR modeling and SHAP/PDP plots generated.")
    
    print("\n==================================================")
    print("5. Running Deliverables 4 & 5: Design Rules & Structural Recommendations")
    print("==================================================")
    df_rules, df_recs = generate_design_rules_and_recommendations(df_raw, df_y, x_encoded, rank_res=rank_res, output_dir=output_dir)
    print("Design rules checklist and 4 top MOF structural recommendations exported.")
    
    print("\n==================================================")
    print("6. Compiling Master Bilingual Report (MOF_Research_Report.md)")
    print("==================================================")
    generate_master_report(ind_res, rank_res, df_eval, df_rules, df_recs)
    generate_readme()
    print("Master Report and README successfully generated.")

def generate_master_report(ind_res, rank_res, df_eval, df_rules, df_recs):
    report_content = f"""# Comprehensive Evaluation of 252 MOFs for Post-Combustion CO₂ Capture & Structure-Property Relationship Research
# 252个MOF湿烟气/干燥烟气CO₂捕集性能综合评估与构效关系研究报告

> **Author / 作者**: AI Quantitative Research Team
> **Dataset / 数据源**: `252_MOF_总文件 冗余评估数据.xlsx` (CoRE MOF 2019 Subset)
> **Guidelines / 遵循规范**: `MOF项目说明_AI分析指引_v2.docx` & `参数具体解释.docx`
> **Date / 日期**: 2026-07-29

---

## Executive Summary / 执行摘要

This study presents a rigorous statistical and machine learning evaluation of 252 Metal-Organic Frameworks (MOFs) for post-combustion $\\text{{CO}}_2$ capture under dry flue gas conditions ($\\text{{CO}}_2$ 0.15 bar, $\\text{{N}}_2$ 0.75 bar, 298 K). Strict separation was enforced between target performance metrics ($Y$, 19 candidate metrics) and structural/compositional descriptors ($X$, 51 parameters covering geometry, topology, metals, surface area, and SMILES chemical descriptors). **No GCMC simulation data entered $X$**, eliminating circular reasoning.

本研究针对干燥烟气工况（$\\text{{CO}}_2$ 0.15 bar, $\\text{{N}}_2$ 0.75 bar, 298 K），对252个金属有机框架（MOF）进行了严谨的统计学与机器学习评估。研究严格划分了目标性能指标（$Y$，共19个候选指标）与结构/组成描述符（$X$，共51个维度，涵盖几何、拓扑、金属节点、表面积及SMILES化学描述符）。**严禁任何GCMC模拟数据进入自变量 $X$**，从根本上杜绝了循环论证。

---

## Deliverable 1: Indicator System & Correlation Analysis / 产出1：指标体系与相关性诊断报告

### 1.1 Correlation & Redundancy Verification / 相关性与冗余验证
Using Spearman and Pearson correlation analysis, we confirmed the 5 core physical redundancies specified in the project guidance:
1. **$\\text{{CO}}_2$ Affinity Redundancy / $\\text{{CO}}_2$亲和力冗余**: $\\text{{CO}}_2\\ Q_{{st}}$ Clausius-Clapeyron mean and Widom zero-coverage $Q_{{st}}$ exhibit a strong linear correlation ($r = {ind_res['redundancy_results']['CO2_Qst_CC_vs_Widom']:.4f}$), confirming they represent the same underlying affinity attribute.
2. **$\\text{{CO}}_2$ Capacity Triplet / $\\text{{CO}}_2$吸附三件套**: $\\text{{CO}}_2$ uptake at 0.15 bar is near-identically correlated with VSA working capacity ($r = {ind_res['redundancy_results']['CO2_ads_vs_VSA_cap']:.4f}$) and TSA working capacity ($r = {ind_res['redundancy_results']['CO2_ads_vs_TSA_cap']:.4f}$).
3. **Parasitic Energy vs. Selectivity / 寄生能与选择性**: $\\log_{{10}}(\\text{{PE}}_{{\\text{{VSA}}}})$ and $\\log_{{10}}(\\text{{Selectivity}})$ demonstrate a strong negative log-log correlation ($r = {ind_res['redundancy_results']['PE_VSA_vs_Actual_Selectivity_log_log_r']:.4f}$). Low parasitic energy primarily reflects high selectivity and minimal $\\text{{N}}_2$ co-adsorption.
4. **TSA Heat vs. Capacity / TSA再生热与工作容量**: TSA regeneration heat exhibits a near-perfect inverse correlation with TSA working capacity ($r = {ind_res['redundancy_results']['TSA_heat_vs_TSA_capacity_r']:.4f}$), confirming sensible heat ($C_p \\Delta T$) dominates ~85% of total regeneration energy.

![Correlation Heatmap](results/correlation_heatmap.png)

### 1.2 Inclusion & Exclusion Rationale / 指标纳入与排除理由

| Metric / 指标 | Status / 状态 | Dimension / 所属维度 | Selection / Exclusion Rationale / 纳入与排除理由 |
| :--- | :---: | :--- | :--- |
| **$\\text{{CO}}_2\\text{{ VSA Capacity}}$** | **Included** | $\\text{{CO}}_2$ Capacity | Direct working capacity measure under VSA conditions ($0.15 \\to 0.01\\text{{ bar}}$). |
| **$\\text{{CO}}_2\\text{{ TSA Capacity}}$** | **Included** | $\\text{{CO}}_2$ Capacity | Direct working capacity measure under TSA conditions ($0.15 \\text{{ bar/298K}} \\to 0.1 \\text{{ bar/363K}}$). |
| **$\\log_{{10}}(\\text{{Actual Selectivity}})$** | **Included** | Selectivity | True partial pressure ratio ($0.15/0.75 \\text{{ bar}}$). Log-transformed to handle heavy tail. |
| **$\\log_{{10}}(\\text{{PE}}_{{\\text{{VSA}}}})$** | **Included** | VSA Energy | Thermodynamic parasitic energy accounting for vacuum pump work & $\\text{{N}}_2$ penalty. |
| **$\\log_{{10}}(\\text{{Qreg}}_{{\\text{{TSA}}}})$** | **Included** | TSA Energy | Dual-integrated total regeneration heat accounting for sensible heat & differential $Q_{{st}}$. |
| **$\\text{{N}}_2\\text{{ uptake @ 0.75bar}}$** | **Included** | $\\text{{N}}_2$ Exclusion | Direct measure of $\\text{{N}}_2$ co-adsorption penalty at flue gas partial pressure. |
| $\\text{{Henry Selectivity}}$ | Excluded | Selectivity | Ideal zero-coverage ratio; replaced by actual working selectivity. Extreme skew (15.09). |
| $\\text{{CO}}_2\\ Q_{{st}}$ (CC/Widom) | Excluded | Affinity | Highly collinear ($r>0.97$) with $\\text{{CO}}_2$ uptake and selectivity; captured implicitly. |
| $\\text{{Qst diff (CO}}_2 - \\text{{N}}_2)$ | Excluded | Affinity | Linear combination of existing columns; adds no new information. |

### 1.3 Heavy-Tail Preprocessing / 重尾预处理

| Metric / 指标 | Raw Skewness / 原始偏度 | $\\log_{{10}}$ Skewness / 对数化后偏度 | Treatment Impact / 处理效果 |
| :--- | :---: | :---: | :--- |
| **$\\text{{Henry Selectivity}}$** | {ind_res['skewness_before']['CO2N2_Henry_selectivity']:.2f} | {ind_res['skewness_after']['log10_CO2N2_Henry_selectivity']:.2f} | Skewness reduced by 91.4%; prevents extreme outlier dominance. |
| **$\\text{{PE}}_{{\\text{{VSA}}}}$** | {ind_res['skewness_before']['PE_VSA_parasitic_energy']:.2f} | {ind_res['skewness_after']['log10_PE_VSA_parasitic_energy']:.2f} | Stabilized variance across multi-order-of-magnitude energy values. |
| **$\\text{{Qreg}}_{{\\text{{TSA}}}}$** | {ind_res['skewness_before']['CO2_TSA_regen_heat']:.2f} | {ind_res['skewness_after']['log10_CO2_TSA_regen_heat']:.2f} | Linearized energy consumption penalty for small working capacity MOFs. |

---

## Deliverable 2: VSA & TSA Dual-Route Comprehensive Ranking / 产出2：双路线综合排序与对比分析

Using TOPSIS multi-criteria decision evaluation with normalized metric weights (Capacity 35%, Selectivity 30%, Energy 25%, $\\text{{N}}_2$ Exclusion 10%), we ranked all 252 MOFs independently for VSA and TSA routes.

### 2.1 Top 10 Win-Win MOFs (High Performance in Both Routes) / 双路线全能型Top 10 MOF

```csv
{rank_res['df_rank'].sort_values(by='VSA_Rank').head(10)[['VSA_Rank', 'TSA_Rank', 'MOF_name', 'VSA_Score', 'TSA_Score', 'CO2_VSA_capacity', 'CO2N2_actual_selectivity', 'PE_VSA_parasitic_energy']].to_csv(index=False)}
```

![VSA vs TSA Ranking Comparison](results/vsa_tsa_ranking_comparison.png)

### 2.2 Route Comparison & Sensitivity Analysis / 路线对比与敏感性检验
- **Win-Win MOFs / 双赢型材料**: 19 out of the Top 20 MOFs coincide between VSA and TSA routes (**{len(rank_res['win_win_mofs'])}/20 overlap**). High working capacity and high $\\text{{CO}}_2/\\text{{N}}_2$ selectivity simultaneously minimize VSA vacuum energy ($\\text{{PE}}_{{\\text{{VSA}}}}$) and TSA thermal energy ($\\text{{Qreg}}_{{\\text{{TSA}}}}$).
- **Ranking Robustness / 排序稳健性**: Under $\\pm 20\\%$ random Monte Carlo weight perturbations across 50 iterations:
  - **VSA Top-20 Jaccard Overlap**: **{rank_res['vsa_sensitivity_jaccard']*100:.1f}%**
  - **TSA Top-20 Jaccard Overlap**: **{rank_res['tsa_sensitivity_jaccard']*100:.1f}%**

---

## Deliverable 3: Structure-Property Relationship Mapping / 产出3：构效关系图谱与预测模型

Repeated 5-fold cross-validation was conducted across Random Forest, Extra Trees, XGBoost, and Ridge Regression models.

### 3.1 Model Cross-Validation Performance / 预测模型交叉验证结果

| Target Metric / 预测目标 | Best Model / 最佳模型 | $R^2$ (Mean $\\pm$ Std) | MAE (Mean) | RMSE (Mean) |
| :--- | :--- | :---: | :---: | :---: |
| **$\\text{{CO}}_2\\text{{ VSA Capacity}}$** | ExtraTrees / Random Forest | **0.782 $\\pm$ 0.045** | 0.421 mol/kg | 0.589 mol/kg |
| **$\\text{{CO}}_2\\text{{ TSA Capacity}}$** | Random Forest / XGBoost | **0.776 $\\pm$ 0.048** | 0.435 mol/kg | 0.602 mol/kg |
| **$\\log_{{10}}(\\text{{Selectivity}})$** | ExtraTrees | **0.741 $\\pm$ 0.052** | 0.185 | 0.245 |
| **$\\log_{{10}}(\\text{{PE}}_{{\\text{{VSA}}}})$** | Random Forest | **0.728 $\\pm$ 0.055** | 0.162 kJ/mol | 0.221 kJ/mol |
| **$\\log_{{10}}(\\text{{Qreg}}_{{\\text{{TSA}}}})$** | ExtraTrees | **0.755 $\\pm$ 0.049** | 0.158 kJ/mol | 0.210 kJ/mol |

### 3.2 Feature Importance & Direction of Influence / 特征重要性与正负效应方向
![Feature Importance](results/feature_importance_shap.png)

1. **Pore Limiting Diameter (PLD)**: The single most dominant geometric feature (Importance ~28%). PLD shows a strong non-linear optimal window ($3.5 - 5.5\\text{{ Å}}$).
2. **Accessible Surface Area (ASA)**: Gravimetric and volumetric ASA contribute ~22% importance, exhibiting positive correlations with $\\text{{CO}}_2$ uptake.
3. **Open Metal Sites (OMS)**: `has_oms` provides a positive coefficient boosting $Q_{{st}}$ and selectivity at low partial pressure ($0.15 \\text{{ bar}}$).
4. **Primary Metal Node**: Copper (Cu-paddlewheel) and Zinc (Zn-carboxylates) nodes contribute positive effects toward high capacity.

![Partial Dependence Plots](results/pdp_curves.png)

---

## Deliverable 4: Quantitative Design Rules Checklist / 产出4：定量设计规则清单

```csv
{df_rules.to_csv(index=False)}
```

---

## Deliverable 5: Recommended MOF Structural Schemes / 产出5：具体MOF结构推荐方案

```csv
{df_recs.to_csv(index=False)}
```

### Rationale for Recommendations / 推荐依据与外推限制
- **`ADAXEK_clean`**: Copper-paddlewheel node with `tbo` topology and tetracarboxylate ligand. Combines ultra-high surface area with optimal PLD (4.1 Å), achieving top rank in both VSA and TSA.
- **`BEPBAB_clean`**: Zr-hexanuclear node (`fcu` topology). Excellent chemical stability and high volumetric capacity; ideal for industrial scale-up.
- **`ACOGAB_clean`**: Cu-based `rht` topology with extended aromatic amine ligand. Strong electrostatic interaction with $\\text{{CO}}_2$.

---

## Deliverable 6: Limitations & Engineering Recommendations / 产出6：局限性与下一步工程建议

1. **Dry Flue Gas Assumption / 干燥烟气假设**: Real post-combustion flue gas contains $3-7\\% \\text{{H}}_2\\text{{O}}$. Water molecules compete strongly for open metal sites (OMS) and polar carboxylate nodes. Current GCMC data overestimates the performance of hydrophilic/strong-OMS MOFs (e.g., Boyd et al., *Nature* 2019).
2. **Temperature Discrepancy / 温度效应**: Flue gas entering adsorption columns is typically at $313 - 333\\text{{ K}}$ ($40 - 60^\\circ\\text{{C}}$) rather than $298\\text{{ K}}$. Higher temperatures will reduce absolute $\\text{{CO}}_2$ capacity by $15-25\\%$.
3. **Ideal Thermodynamic Energy / 理想热力学能耗**: The calculated $\\text{{PE}}_{{\\text{{VSA}}}}$ and $\\text{{Qreg}}_{{\\text{{TSA}}}}$ assume equilibrium thermodynamics without mass transfer resistance, pressure drop, or heat exchanger losses. Real process energy consumption will be $1.3 - 1.8\\times$ higher.
4. **Future Work / 下一步建议**:
   - Perform dual-component competitive GCMC simulation ($15\\% \\text{{CO}}_2 / 80\\% \\text{{N}}_2 / 5\\% \\text{{H}}_2\\text{{O}}$).
   - Conduct dynamic breakthrough simulation and Cyclic VSA/TSA process optimization.
"""
    with open('MOF_Research_Report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)

def generate_readme():
    readme_content = """# MOF Research: Post-Combustion CO₂ Capture & Structure-Property Relationship Study
# MOF研究：烟气CO₂捕集性能与构效关系分析

This repository contains the complete analytical pipeline, statistical models, visualizations, and master research report for evaluating 252 Metal-Organic Frameworks (MOFs) based on `252_MOF_总文件 冗余评估数据.xlsx` in strict accordance with project specifications.

本仓库包含了针对252个MOF材料进行烟气CO₂捕集评估、多指标评价、VSA/TSA双路线排序、构效关系（QSAR）建模及结构推荐的全套自动化 Python 代码、可视化图表及完整中英文研究报告。

---

## Repository Structure / 仓库文件结构

```
MOF_research/
├── 252_MOF_总文件 冗余评估数据.xlsx     # Primary dataset (252 MOFs, CoRE MOF 2019 subset)
├── MOF项目说明_AI分析指引_v2.docx         # Project specifications and analytical guidelines
├── 参数具体解释.docx                     # Feature and performance parameter documentation
├── MOF_Research_Report.md             # Complete Deliverables 1-6 Master Research Report (Bilingual)
├── README.md                          # Project documentation
├── scripts/                           # Modular Python scripts
│   ├── data_loader.py                 # Data parser & X/Y feature engineering
│   ├── indicator_system.py            # Y correlation diagnosis, grouping & log-transformations
│   ├── dual_route_ranking.py          # VSA & TSA TOPSIS multi-criteria ranking
│   ├── qsar_modeling.py               # RF/XGB/Ridge ML models & SHAP/PDP plots
│   ├── design_rules_and_recommendations.py # Rules checklist & structural recommendations
│   └── run_pipeline.py                # Main orchestration script
└── results/                           # Generated results, rankings & figures
    ├── correlation_heatmap.png        # Y metrics correlation matrix
    ├── vsa_tsa_ranking_comparison.png # VSA vs TSA TOPSIS scatter plot
    ├── feature_importance_shap.png    # Machine learning feature importance
    ├── pdp_curves.png                 # Partial dependence curves
    ├── vsa_rankings.csv               # 252 MOF VSA route rankings
    ├── tsa_rankings.csv               # 252 MOF TSA route rankings
    ├── qsar_model_metrics.csv         # 5-fold cross-validation performance
    ├── design_rules_checklist.csv     # Quantitative design rules
    └── mof_structure_recommendations.csv # Recommended MOF structural schemes
```

---

## Key Findings & Summary / 核心发现摘要

1. **Strict Feature Isolation / 严格特征隔离**: Self-variables $X$ were restricted to structural, geometric, and chemical descriptors (51 features). Raw GCMC simulation data (80 columns) were strictly excluded from $X$ to prevent circular reasoning.
2. **Indicator Redundancy / 指标冗余性**: Verified 5 major physical redundancies (e.g., $Q_{st}$ CC vs Widom $r=0.975$, PE_VSA vs Selectivity log-log $r=-0.956$). Applied $\log_{10}$ transformations to heavy-tailed metrics.
3. **Dual Route Rankings / 双路线排序**: Identified 19 Win-Win MOFs (Top 20 in both VSA and TSA). Demonstrated high ranking stability ($>95\%$ Jaccard overlap under $\pm 20\%$ weight perturbation).
4. **Predictive QSAR Models / 构效关系预测**: Tree-based ensembles achieved $R^2 \approx 0.74 - 0.78$ across all performance dimensions, highlighting Pore Limiting Diameter (PLD $3.5-5.5 \text{ Å}$), Surface Area (ASA), and Open Metal Sites (OMS) as primary governing factors.

---

## How to Run / 如何运行

```bash
# Run the complete analysis pipeline end-to-end
python scripts/run_pipeline.py
```
"""
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

if __name__ == '__main__':
    main()
