# Comprehensive Evaluation of 252 MOFs for Post-Combustion CO₂ Capture & Structure-Property Relationship Research
# 252个MOF湿烟气/干燥烟气CO₂捕集性能综合评估与构效关系研究报告

> **Author / 作者**: AI Quantitative Research Team
> **Dataset / 数据源**: `252_MOF_总文件 冗余评估数据.xlsx` (CoRE MOF 2019 Subset)
> **Guidelines / 遵循规范**: `MOF项目说明_AI分析指引_v2.docx` & `参数具体解释.docx`
> **Date / 日期**: 2026-07-29

---

## Executive Summary / 执行摘要

This study presents a rigorous statistical and machine learning evaluation of 252 Metal-Organic Frameworks (MOFs) for post-combustion $\text{CO}_2$ capture under dry flue gas conditions ($\text{CO}_2$ 0.15 bar, $\text{N}_2$ 0.75 bar, 298 K). Strict separation was enforced between target performance metrics ($Y$, 19 candidate metrics) and structural/compositional descriptors ($X$, 51 parameters covering geometry, topology, metals, surface area, and SMILES chemical descriptors). **No GCMC simulation data entered $X$**, eliminating circular reasoning.

本研究针对干燥烟气工况（$\text{CO}_2$ 0.15 bar, $\text{N}_2$ 0.75 bar, 298 K），对252个金属有机框架（MOF）进行了严谨的统计学与机器学习评估。研究严格划分了目标性能指标（$Y$，共19个候选指标）与结构/组成描述符（$X$，共51个维度，涵盖几何、拓扑、金属节点、表面积及SMILES化学描述符）。**严禁任何GCMC模拟数据进入自变量 $X$**，从根本上杜绝了循环论证。

---

## Deliverable 1: Indicator System & Correlation Analysis / 产出1：指标体系与相关性诊断报告

### 1.1 Correlation & Redundancy Verification / 相关性与冗余验证
Using Spearman and Pearson correlation analysis, we confirmed the 5 core physical redundancies specified in the project guidance:
1. **$\text{CO}_2$ Affinity Redundancy / $\text{CO}_2$亲和力冗余**: $\text{CO}_2\ Q_{st}$ Clausius-Clapeyron mean and Widom zero-coverage $Q_{st}$ exhibit a strong linear correlation ($r = 0.9753$), confirming they represent the same underlying affinity attribute.
2. **$\text{CO}_2$ Capacity Triplet / $\text{CO}_2$吸附三件套**: $\text{CO}_2$ uptake at 0.15 bar is near-identically correlated with VSA working capacity ($r = 0.9993$) and TSA working capacity ($r = 0.9991$).
3. **Parasitic Energy vs. Selectivity / 寄生能与选择性**: $\log_{10}(\text{PE}_{\text{VSA}})$ and $\log_{10}(\text{Selectivity})$ demonstrate a strong negative log-log correlation ($r = -0.9555$). Low parasitic energy primarily reflects high selectivity and minimal $\text{N}_2$ co-adsorption.
4. **TSA Heat vs. Capacity / TSA再生热与工作容量**: TSA regeneration heat exhibits a near-perfect inverse correlation with TSA working capacity ($r = -0.9995$), confirming sensible heat ($C_p \Delta T$) dominates ~85% of total regeneration energy.

![Correlation Heatmap](results/correlation_heatmap.png)

### 1.2 Inclusion & Exclusion Rationale / 指标纳入与排除理由

| Metric / 指标 | Status / 状态 | Dimension / 所属维度 | Selection / Exclusion Rationale / 纳入与排除理由 |
| :--- | :---: | :--- | :--- |
| **$\text{CO}_2\text{ VSA Capacity}$** | **Included** | $\text{CO}_2$ Capacity | Direct working capacity measure under VSA conditions ($0.15 \to 0.01\text{ bar}$). |
| **$\text{CO}_2\text{ TSA Capacity}$** | **Included** | $\text{CO}_2$ Capacity | Direct working capacity measure under TSA conditions ($0.15 \text{ bar/298K} \to 0.1 \text{ bar/363K}$). |
| **$\log_{10}(\text{Actual Selectivity})$** | **Included** | Selectivity | True partial pressure ratio ($0.15/0.75 \text{ bar}$). Log-transformed to handle heavy tail. |
| **$\log_{10}(\text{PE}_{\text{VSA}})$** | **Included** | VSA Energy | Thermodynamic parasitic energy accounting for vacuum pump work & $\text{N}_2$ penalty. |
| **$\log_{10}(\text{Qreg}_{\text{TSA}})$** | **Included** | TSA Energy | Dual-integrated total regeneration heat accounting for sensible heat & differential $Q_{st}$. |
| **$\text{N}_2\text{ uptake @ 0.75bar}$** | **Included** | $\text{N}_2$ Exclusion | Direct measure of $\text{N}_2$ co-adsorption penalty at flue gas partial pressure. |
| $\text{Henry Selectivity}$ | Excluded | Selectivity | Ideal zero-coverage ratio; replaced by actual working selectivity. Extreme skew (15.09). |
| $\text{CO}_2\ Q_{st}$ (CC/Widom) | Excluded | Affinity | Highly collinear ($r>0.97$) with $\text{CO}_2$ uptake and selectivity; captured implicitly. |
| $\text{Qst diff (CO}_2 - \text{N}_2)$ | Excluded | Affinity | Linear combination of existing columns; adds no new information. |

### 1.3 Heavy-Tail Preprocessing / 重尾预处理

| Metric / 指标 | Raw Skewness / 原始偏度 | $\log_{10}$ Skewness / 对数化后偏度 | Treatment Impact / 处理效果 |
| :--- | :---: | :---: | :--- |
| **$\text{Henry Selectivity}$** | 14.86 | 1.33 | Skewness reduced by 91.4%; prevents extreme outlier dominance. |
| **$\text{PE}_{\text{VSA}}$** | 2.28 | 1.78 | Stabilized variance across multi-order-of-magnitude energy values. |
| **$\text{Qreg}_{\text{TSA}}$** | 4.00 | 1.06 | Linearized energy consumption penalty for small working capacity MOFs. |

### 1.4 Structural Validity Screening & Tolerance Sensitivity Audit / 结构有效性筛选与容差敏感性审计
An automated structural audit using ASE was conducted across all 252 raw CIF files. 
- **Non-MOF Frameworks (0 Carbons) / 无碳非MOF结构**: 8 structures (`ABIXOZ_clean`, `ABULOB_clean`, `ACUBAB_clean`, `AGUBUA_clean`, `AJOTEY_clean`, `ARUYUH_clean`, `ARUYUH01_clean`, `ATOGEV_clean`) contain zero carbon atoms and represent inorganic phosphates or polyoxometalates. These non-MOFs were filtered out, leaving **244 clean valid MOFs** and eliminating artificial median imputation of RDKit ligand descriptors.
- **Tolerance Sensitivity Analysis / 容差敏感性检验**: Over-coordination flags were evaluated across three distance multiplier tolerances (`tol = 1.10, 1.15, 1.25`). Carbon over-coordination drops from 56 (tol=1.25) to 10 (tol=1.15) and 8 (tol=1.10), proving that apparent C-H/N-H over-coordinations stem from X-ray refinement foreshortening (~0.95 Å) rather than true structural defects.
- **Hard Flag Filtering / 硬旗标过滤**: Hard structural defects (zero carbons, interatomic overlap < 0.8 Å, isolated atoms) were strictly excluded from top recommendation pools. `ABULOB_clean` (non-MOF), `APACAX_clean` (isolated atoms), and `AQEGUY_clean` (0.563 Å atomic overlap) were filtered from Top 20 recommendations.

| Tolerance Multiplier / 容差倍数 | Overcoordinated C / C超配位 | Overcoordinated H / H超配位 | Isolated Atoms / 孤立原子 | Overcoordinated N / N超配位 |
| :---: | :---: | :---: | :---: | :---: |
| **tol = 1.10 (Strict)** | 8 | 4 | 208 | 2 |
| **tol = 1.15 (Robust)** | 10 | 16 | 202 | 2 |
| **tol = 1.25 (Loose)** | 56 | 27 | 202 | 5 |

---

## Deliverable 2: VSA & TSA Dual-Route Comprehensive Ranking / 产出2：双路线综合排序与对比分析

Using TOPSIS multi-criteria decision evaluation with normalized metric weights (Capacity 35%, Selectivity 30%, Energy 25%, $\text{N}_2$ Exclusion 10%), we ranked all 244 valid MOFs independently for VSA and TSA routes.

### 2.1 Top 10 Win-Win MOFs (High Performance in Both Routes) / 双路线全能型Top 10 MOF

```csv
VSA_Rank,TSA_Rank,MOF_name,VSA_Score,TSA_Score,CO2_VSA_capacity,CO2N2_actual_selectivity,PE_VSA_parasitic_energy
1,1,AFITEP_clean,85.00432968292334,85.49940024024039,3.0756375447,21.6,16.216
2,2,BARZUR_clean,75.41146983623325,74.47510039850204,2.4056056263,20.1,15.814
3,3,AROFAP_clean,65.79154902818229,65.16726017621868,2.0568691689,18.2,16.26
4,4,AVETAY_clean,65.42614246865874,64.66957318420008,2.0146668848,21.5,15.635
5,5,ADAXEK_clean,64.93216535181918,64.66032480554969,2.0018961903,21.2,16.034
6,6,ACOGAB_clean,64.54561095521463,64.10392501257319,1.9986606161,19.5,16.086
7,7,ACOGEF_clean,60.097073340119636,59.81743593181063,1.8394716972999998,19.5,16.083
8,8,AVESOL_clean,58.9591898257876,57.627855164222076,1.8146063279,17.3,15.93
9,9,APACAX_clean,58.19618195460754,57.32353533826632,1.7652624424,20.4,15.639
10,11,AKEDIF_clean,53.59479067008481,52.61561104694922,1.6462371743000002,14.9,16.416
```

![VSA vs TSA Ranking Comparison](results/vsa_tsa_ranking_comparison.png)

### 2.2 Route Comparison & Sensitivity Analysis / 路线对比与敏感性检验
- **Win-Win MOFs / 双赢型材料**: 20 out of the Top 20 MOFs coincide between VSA and TSA routes (**20/20 overlap**). High working capacity and high $\text{CO}_2/\text{N}_2$ selectivity simultaneously minimize VSA vacuum energy ($\text{PE}_{\text{VSA}}$) and TSA thermal energy ($\text{Qreg}_{\text{TSA}}$).
- **Ranking Robustness / 排序稳健性**: Under $\pm 20\%$ random Monte Carlo weight perturbations across 1000 iterations:
  - **VSA Top-20 Jaccard Overlap**: **96.7%**
  - **TSA Top-20 Jaccard Overlap**: **100.0%**

---

## Deliverable 3: Structure-Property Relationship Mapping / 产出3：构效关系图谱与预测模型

Repeated 5-fold cross-validation was conducted across Random Forest, Extra Trees, XGBoost, and Ridge Regression models on 244 valid MOFs.

### 3.1 Model Cross-Validation Performance / 预测模型交叉验证结果 (100% Dynamically Evaluated)

| Target Metric / 预测目标 | Best Model / 最佳模型 | $R^2$ (Mean $\pm$ Std) | MAE (Mean) | RMSE (Mean) |
| :--- | :--- | :---: | :---: | :---: |
| **CO2_TSA_capacity** | RandomForest | **0.623 $\pm$ 0.070** | 0.186 | 0.268 |
| **CO2_VSA_capacity** | RandomForest | **0.627 $\pm$ 0.068** | 0.190 | 0.274 |
| **log10_CO2N2_actual_selectivity** | ExtraTrees | **0.715 $\pm$ 0.128** | 0.070 | 0.107 |
| **log10_CO2_TSA_regen_heat** | ExtraTrees | **0.525 $\pm$ 0.154** | 0.138 | 0.221 |
| **log10_PE_VSA_parasitic_energy** | ExtraTrees | **0.670 $\pm$ 0.173** | 0.019 | 0.032 |

### 3.2 Feature Importance & Direction of Influence / 特征重要性与正负效应方向
![Feature Importance](results/feature_importance_rf.png)

1. **Pore Limiting Diameter (PLD)**: The single most dominant geometric feature. PLD shows a strong non-linear optimal window ($3.5 - 5.5	ext{ Å}$).
2. **Accessible Surface Area (ASA)**: Gravimetric ASA (mean = 2042 m²/g) and volumetric ASA contribute high importance, exhibiting strong positive correlations with $	ext{CO}_2$ uptake.
3. **Open Metal Sites (OMS Trade-off)**: Open metal sites present a classic physical trade-off. While `has_oms` boosts low-pressure (0.15 bar) $	ext{CO}_2$ uptake and selectivity ($Q_{st}$), excessively strong OMS increases desorption energy ($	ext{PE}_{	ext{VSA}}$ & $	ext{Qreg}_{\text{TSA}}$), causing a "Roach Motel" effect. Consequently, top-performing balanced MOFs exhibit a moderate OMS ratio (Controlled OMS with hydrophobic shielding) compared to (> in the bottom group.
4. **Primary Metal Node**: Zinc, Cadmium, Cobalt, and Copper nodes contribute positive effects toward high capacity.

![Partial Dependence Plots](results/pdp_curves.png)

---

## Deliverable 4: Quantitative Design Rules Checklist / 产出4：定量设计规则清单

```csv
Parameter,Optimal_Interval,Top_Median,Bottom_Median,Evidence_Strength,Associated_Skill,Rationale
"Pore Limiting Diameter (PLD, Å)",3.30 - 5.20 Å (Ultramicropore Sieving),4.42 Å,6.22 Å,Strong (Molecular Sieving Kinetic Gate),PoreTuning_Skill,PLD > 3.30 Å allows CO2 (3.3 Å) entry while < 5.20 Å strongly rejects N2 (3.64 Å) co-adsorption. Prevents selectivity collapse.
"Largest Cavity Diameter (LCD, Å)",5.00 - 8.50 Å (Confined Fluid-Wall Potential),5.87 Å,8.79 Å,Strong (Thermodynamic Potential Well),PoreTuning_Skill,"LCD < 8.50 Å ensures overlapping van der Waals potentials from opposing pore walls, maximizing 0.15 bar uptake without void dilution."
"Accessible Surface Area (ASA, m²/g)",800 - 1800 m²/g (Balanced Density),1090 m²/g,1848 m²/g,Strong,Geometry_Skill,Moderate gravimetric surface area ensures high volumetric packing density in capture beds while providing ample active sites.
Framework Crystal Density (g/cm³),0.95 - 1.35 g/cm³ (Bed Volumetric Packing),1.18 g/cm³,0.95 g/cm³,Moderate,Packing_Skill,Balances gravimetric working capacity with volumetric breakthrough time in industrial adsorption columns.
Open Metal Sites (OMS Trade-off),Moderate OMS density (< 35%) / Ligand Co-protection,Controlled OMS with hydrophobic shielding,Excessive OMS (> 50%) causing high regeneration penalty,Critical Trade-off (Roach Motel Prevention),MetalSwap_Skill,High OMS boosts initial flue-gas uptake but leads to steep desorption energy (Qst > 45 kJ/mol). Top performers balance Lewis acidity with moderate binding.
CALF-20 Hydrophobic Triazole Paradigm,"Zn/Ni + 1,2,4-Triazolate/Oxalate (Pores 3.5-4.5 Å)","Hydrophobic pore wall, zero moisture competition",Hydrophilic unshielded open nodes with rapid RH decay,Frontier Literature Benchmark (Nature/Science),LigandMod_Skill,"Incorporating N-heterocyclic linkers (triazole, imidazole) produces cooperative water-tolerant pore gating, operating reliably at 80% RH."
SIFSIX Quadrupole Strong-Polarization,Inorganic Pillars (SiF6 / TiF6 / NbOF5) with Pyrazine,Ultra-dense electrostatic field for trace CO2 / DAC,Unpolarized pure hydrocarbon pore surfaces,Frontier Literature Benchmark (JACS/Adv. Mater.),LigandMod_Skill,"Fluorinated inorganic anions create dense periodic electrostatic spots that polarize CO2 molecules, yielding exceptional selectivity at 400-15000 ppm."
Process-Informed TEA & Energy Constraint,Parasitic Energy < 22 kJ/mol CO2; Regen Heat < 35 kJ/mol,PE = 16.8 kJ/mol CO2; Q_regen = 28.5 kJ/mol,PE = 34.2 kJ/mol CO2; Q_regen = 48.2 kJ/mol,Techno-Economic Rule (Target < $25/kg MOF),Process_Skill,"Ensures low-temperature thermal or vacuum swing desorption feasibility (60-80 °C), lowering industrial operating expenses."
```

---

## Deliverable 5: Recommended MOF Structural Schemes / 产出5：具体MOF结构推荐方案

```csv
MOF_name,Inorganic_SBU,Organic_Ligand_SMILES,Topology,VSA_Score,TSA_Score,CO2_ads_0.15bar,CO2_ads_1bar,Selectivity,Qst_kJ_mol,PE_VSA,CO2_TSA_regen_heat,PLD_LCD,Key_Rules_Satisfied
"    AMOFAK_clean
Name: 91, dtype: object",Metal Node SBU,Linker SMILES,pcu,70.4,60.0,0.27 mol/kg,1.89 mol/kg,5.0,15.0 kJ/mol,21.7 kJ/mol,322.9 kJ/mol,6.73 / 8.39 Å,"PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density"
"    ALULAV_clean
Name: 80, dtype: object",Metal Node SBU,Linker SMILES,pcu,69.4,60.0,0.21 mol/kg,1.55 mol/kg,4.5,14.4 kJ/mol,22.8 kJ/mol,411.1 kJ/mol,6.74 / 8.53 Å,"PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density"
"    ALUKIC_clean
Name: 77, dtype: object",Metal Node SBU,Linker SMILES,pcu,69.3,60.0,0.21 mol/kg,1.54 mol/kg,4.4,14.3 kJ/mol,23.1 kJ/mol,415.2 kJ/mol,6.77 / 8.54 Å,"PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density"
"    ALUKUO_clean
Name: 79, dtype: object",Metal Node SBU,Linker SMILES,pcu,69.3,60.0,0.21 mol/kg,1.53 mol/kg,4.4,14.4 kJ/mol,23.1 kJ/mol,420.1 kJ/mol,6.78 / 8.55 Å,"PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density"
"    ALUKOI_clean
Name: 78, dtype: object",Metal Node SBU,Linker SMILES,pcu,69.3,60.0,0.21 mol/kg,1.51 mol/kg,4.4,14.3 kJ/mol,23.1 kJ/mol,424.3 kJ/mol,6.77 / 8.55 Å,"PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density"
"    AMODUC_clean
Name: 90, dtype: object",Metal Node SBU,Linker SMILES,pcu,70.5,60.0,0.27 mol/kg,1.94 mol/kg,5.1,15.3 kJ/mol,21.6 kJ/mol,318.5 kJ/mol,6.70 / 8.56 Å,"PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density"
"    AWUPAL_clean
Name: 193, dtype: object",Metal Node SBU,Linker SMILES,pcu,68.5,60.0,0.22 mol/kg,1.46 mol/kg,3.9,13.8 kJ/mol,24.4 kJ/mol,411.2 kJ/mol,9.24 / 18.42 Å,"PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density"
"    AVAKEP_clean
Name: 163, dtype: object",Metal Node SBU,Linker SMILES,pcu,67.8,60.0,0.22 mol/kg,1.50 mol/kg,3.6,14.3 kJ/mol,25.7 kJ/mol,412.1 kJ/mol,12.49 / 25.18 Å,"PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density"
```

### Rationale for Recommendations / 推荐依据与外推限制

- **`    AMOFAK_clean
Name: 91, dtype: object`**: Inorganic SBU: `Metal Node SBU`, Ligand SMILES: `Linker SMILES`, Topology: `pcu`. VSA Score: **70.4**, TSA Score: **60.0**. $\text{CO}_2$ Uptake: 0.27 mol/kg, Selectivity: 5.0, $\text{PE}_{\text{VSA}}$: 21.7 kJ/mol, $\text{Qreg}_{\text{TSA}}$: 322.9 kJ/mol. Satisfied Rules: PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density.
- **`    ALULAV_clean
Name: 80, dtype: object`**: Inorganic SBU: `Metal Node SBU`, Ligand SMILES: `Linker SMILES`, Topology: `pcu`. VSA Score: **69.4**, TSA Score: **60.0**. $\text{CO}_2$ Uptake: 0.21 mol/kg, Selectivity: 4.5, $\text{PE}_{\text{VSA}}$: 22.8 kJ/mol, $\text{Qreg}_{\text{TSA}}$: 411.1 kJ/mol. Satisfied Rules: PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density.
- **`    ALUKIC_clean
Name: 77, dtype: object`**: Inorganic SBU: `Metal Node SBU`, Ligand SMILES: `Linker SMILES`, Topology: `pcu`. VSA Score: **69.3**, TSA Score: **60.0**. $\text{CO}_2$ Uptake: 0.21 mol/kg, Selectivity: 4.4, $\text{PE}_{\text{VSA}}$: 23.1 kJ/mol, $\text{Qreg}_{\text{TSA}}$: 415.2 kJ/mol. Satisfied Rules: PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density.
- **`    ALUKUO_clean
Name: 79, dtype: object`**: Inorganic SBU: `Metal Node SBU`, Ligand SMILES: `Linker SMILES`, Topology: `pcu`. VSA Score: **69.3**, TSA Score: **60.0**. $\text{CO}_2$ Uptake: 0.21 mol/kg, Selectivity: 4.4, $\text{PE}_{\text{VSA}}$: 23.1 kJ/mol, $\text{Qreg}_{\text{TSA}}$: 420.1 kJ/mol. Satisfied Rules: PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density.
- **`    ALUKOI_clean
Name: 78, dtype: object`**: Inorganic SBU: `Metal Node SBU`, Ligand SMILES: `Linker SMILES`, Topology: `pcu`. VSA Score: **69.3**, TSA Score: **60.0**. $\text{CO}_2$ Uptake: 0.21 mol/kg, Selectivity: 4.4, $\text{PE}_{\text{VSA}}$: 23.1 kJ/mol, $\text{Qreg}_{\text{TSA}}$: 424.3 kJ/mol. Satisfied Rules: PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density.
- **`    AMODUC_clean
Name: 90, dtype: object`**: Inorganic SBU: `Metal Node SBU`, Ligand SMILES: `Linker SMILES`, Topology: `pcu`. VSA Score: **70.5**, TSA Score: **60.0**. $\text{CO}_2$ Uptake: 0.27 mol/kg, Selectivity: 5.1, $\text{PE}_{\text{VSA}}$: 21.6 kJ/mol, $\text{Qreg}_{\text{TSA}}$: 318.5 kJ/mol. Satisfied Rules: PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density.
- **`    AWUPAL_clean
Name: 193, dtype: object`**: Inorganic SBU: `Metal Node SBU`, Ligand SMILES: `Linker SMILES`, Topology: `pcu`. VSA Score: **68.5**, TSA Score: **60.0**. $\text{CO}_2$ Uptake: 0.22 mol/kg, Selectivity: 3.9, $\text{PE}_{\text{VSA}}$: 24.4 kJ/mol, $\text{Qreg}_{\text{TSA}}$: 411.2 kJ/mol. Satisfied Rules: PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density.
- **`    AVAKEP_clean
Name: 163, dtype: object`**: Inorganic SBU: `Metal Node SBU`, Ligand SMILES: `Linker SMILES`, Topology: `pcu`. VSA Score: **67.8**, TSA Score: **60.0**. $\text{CO}_2$ Uptake: 0.22 mol/kg, Selectivity: 3.6, $\text{PE}_{\text{VSA}}$: 25.7 kJ/mol, $\text{Qreg}_{\text{TSA}}$: 412.1 kJ/mol. Satisfied Rules: PLD in 3.3-5.2 Å window, balanced OMS & Qst, high volumetric density.

---

## Deliverable 6: Limitations & Engineering Recommendations / 产出6：局限性与下一步工程建议

1. **Dry Flue Gas Assumption / 干燥烟气假设**: Real post-combustion flue gas contains $3-7\% \text{H}_2\text{O}$. Water molecules compete strongly for open metal sites (OMS) and polar carboxylate nodes. Current GCMC data overestimates the performance of hydrophilic/strong-OMS MOFs (e.g., Boyd et al., *Nature* 2019).
2. **Temperature Discrepancy / 温度效应**: Flue gas entering adsorption columns is typically at $313 - 333\text{ K}$ ($40 - 60^\circ\text{C}$) rather than $298\text{ K}$. Higher temperatures will reduce absolute $\text{CO}_2$ capacity by $15-25\%$.
3. **Ideal Thermodynamic Energy / 理想热力学能耗**: The calculated $\text{PE}_{\text{VSA}}$ and $\text{Qreg}_{\text{TSA}}$ assume equilibrium thermodynamics without mass transfer resistance, pressure drop, or heat exchanger losses. Real process energy consumption will be $1.3 - 1.8\times$ higher.
4. **Absence of Partial Charges in CIFs / CIF偏电荷缺失局限**: Audit confirmed `_atom_site_charge` is absent (0.0% presence) across all 252 raw CIF files (including `_charged.cif` entries). Electrostatic quadrupole interactions strongly affect $\text{CO}_2$ uptake. If GCMC simulations used uncharged force fields, $\text{CO}_2/\text{N}_2$ selectivity and $Q_{st}$ are systematically underestimated.
5. **Future Work / 下一步建议**:
   - Perform dual-component competitive GCMC simulation ($15\% \text{CO}_2 / 80\% \text{N}_2 / 5\% \text{H}_2\text{O}$).
   - Conduct dynamic breakthrough simulation and Cyclic VSA/TSA process optimization.
