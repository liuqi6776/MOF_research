# MOF Research: Post-Combustion CO₂ Capture & Structure-Property Relationship Study
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
├── MOF_Research_Report.md             # Complete Deliverables 1-6 Master Research Report (Bilingual, 100% Dynamic)
├── README.md                          # Project documentation
├── legacy/                            # Legacy scratch and audit documents
├── scripts/                           # Modular Python scripts
│   ├── data_loader.py                 # Data parser & X/Y feature engineering (ASA bug fixed)
│   ├── indicator_system.py            # Y correlation diagnosis, grouping & log-transformations
│   ├── dual_route_ranking.py          # VSA & TSA TOPSIS multi-criteria ranking (1000 MC iterations)
│   ├── qsar_modeling.py               # RF/XGB/Ridge ML models & feature importance/PDP plots
│   ├── design_rules_and_recommendations.py # Dynamic rules checklist & structural recommendations
│   └── run_pipeline.py                # Main orchestration script
└── results/                           # Generated results, rankings & figures
    ├── correlation_heatmap.png        # Y metrics correlation matrix
    ├── vsa_tsa_ranking_comparison.png # VSA vs TSA TOPSIS scatter plot
    ├── feature_importance_rf.png      # Machine learning feature importance
    ├── pdp_curves.png                 # Partial dependence curves
    ├── vsa_rankings.csv               # 252 MOF VSA route rankings
    ├── tsa_rankings.csv               # 252 MOF TSA route rankings
    ├── qsar_model_metrics.csv         # 5-fold cross-validation performance
    ├── design_rules_checklist.csv     # Quantitative design rules (Dynamic)
    └── mof_structure_recommendations.csv # Recommended MOF structural schemes (Dynamic)
```

---

## Key Findings & Summary / 核心发现摘要

1. **Strict Feature Isolation / 严格特征隔离**: Self-variables $X$ were restricted to structural, geometric, and chemical descriptors (51 features). Raw GCMC simulation data (80 columns) were strictly excluded from $X$ to prevent circular reasoning.
2. **Corrected Surface Area Feature / 修复表面积特征**: Fixed string matching in `data_loader.py`. Restored Accessible Surface Area (`ASA_m2_g`, mean = 2042 m²/g), eliminating the 0 m²/g anomaly.
3. **100% Dynamic Evaluation / 零硬编码**: All $R^2$, MAE, RMSE, metal node proportions, and structural recommendations in `MOF_Research_Report.md` are dynamically compiled from empirical pipeline evaluation.
4. **Dual Route Rankings / 双路线排序**: Identified 19 Win-Win MOFs (Top 20 in both VSA and TSA). Demonstrated high ranking stability ($>95\%$ Jaccard overlap under 1000 Monte Carlo perturbations).
5. **Predictive QSAR Models / 构效关系预测**: Tree-based ensembles achieved strong cross-validation metrics across all performance dimensions, highlighting Pore Limiting Diameter (PLD $3.5-5.5 	ext{ Å}$), Surface Area (ASA), and Open Metal Sites (OMS) as primary governing factors.

---

## How to Run / 如何运行

```bash
# Run the complete analysis pipeline end-to-end
python scripts/run_pipeline.py
```
