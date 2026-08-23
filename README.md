# MOF Research: Multi-Modal Graph RAG, 3D Crystal Embeddings & QSAR Inverse Design
# MOF 材料研究：多模态图检索增强（Graph RAG）、3D 晶体表征与 QSAR 逆向设计

This repository contains the complete analytical pipeline, 3D crystallographic feature extraction, heterogeneous materials knowledge graph, multi-modal fused QSAR machine learning models, and an interactive Web AI Assistant (**MOF Chatbot**) for evaluating Metal-Organic Frameworks (MOFs) in post-combustion $CO_2$ capture.

本仓库包含了针对 252 个 MOF 晶体材料进行 3D 晶体特征向量提取（768维表征）、异构材料知识图谱构建、多模态融合 QSAR 机器学习性能预测、全新 CIF 零样本预测、逆向结构优化改性建议以及交互式 Web AI 助手（**MOF Chatbot**）的完整代码与科研成果。

---

## 🌟 Key Capabilities & New Deliverables / 核心功能与最新研究成果

### 1. 3D Crystallographic Embedding Pipeline (768-dim) / 3D 晶体特征表征
- Extracted 768-dimensional normalized structural vectors for 252 crystallographic CIF files (`scripts/extract_mof_embeddings.py`).
- Captures 3D lattice parameters, cell volume, density, atomic number distribution, radial distribution functions (RDF), and local coordination topology.
- Saved in `results/mof_structural_embeddings.npy` ($252 \times 768$) and indexed via `results/mof_embedding_index.csv`.

### 2. Fused Multi-Modal QSAR ML Modeling / 多模态融合 QSAR 机器学习
- **Representation Benchmark**: Compared Traditional Descriptors ($9\text{d}$), Pure 768d Embeddings ($768\text{d}$), and Fused Multi-Modal Representation ($X_{\text{fused}} = [X_{\text{trad}}, \text{PCA}_{16}(X_{\text{emb}})]$, $25\text{d}$).
- **Performance Discovery**: Fused Multi-Modal feature matrix achieved the highest 5-fold cross-validation accuracy ($R^2 = 0.675$ on $CO_2/N_2$ selectivity, $R^2 = 0.658$ on VSA parasitic energy), combining physical geometry bounds with latent coordination symmetry.
- Trained production model bundle saved in `results/models/mof_property_predictors.joblib`.

### 3. Zero-Shot Out-of-Distribution Property Prediction / 未知 CIF 零样本即时预测
- When users upload any **new, unseen CIF** outside the 252 database, the engine dynamically extracts its 3D crystal parameters, projects structural embeddings, and runs ML ensemble models to predict:
  - $CO_2$ Uptake @ 0.15 bar (flue gas) & 1.0 bar ($\text{mol/kg}$)
  - $CO_2/N_2$ Actual Selectivity (0.15/0.85)
  - $CO_2$ Adsorption Heat $Q_{st}$ ($\text{kJ/mol}$)
  - VSA/TSA Working Capacity & Parasitic Energy ($\text{kJ/mol } CO_2$)

### 4. Target-Driven Inverse Modification Recommender / 目标导向逆向结构优化
- Automated feature sensitivity and QSAR partial dependence diagnostic engine (`scripts/mof_property_predictor.py`).
- Generates 4-step actionable modification strategies:
  1. **Pore Sieving Window Tuning**: Ligand functionalization ($-\text{NH}_2, -\text{OH}, -\text{CF}_3$) / catenation to gate PLD into the golden window ($3.8-4.8\text{ \AA}$).
  2. **Active Site Engineering**: Transmetalation ($\text{Zn} \to \text{Cu/Mg}$ open metal sites) to optimize $Q_{st} \in [28-35\text{ kJ/mol}]$.
  3. **Capacity & Surface Area Optimization**: Ligand extension ($\text{BDC} \to \text{BPDC}$).
  4. **Framework Stability**: Transition to high-connectivity topologies ($\text{tbo, nbo, dia}$).

### 5. MOF Chatbot Web UI (`app.py`) / 材料智能科研助手
- Modern clean white & blue scientific interface with 3D crystal WebGL viewer (3Dmol.js).
- Default English with one-click English $\leftrightarrow$ Chinese bilingual toggle.
- Integrated with DeepSeek multi-modal materials science reasoning.
- Supports one-click public HTTPS access tunnel (`scripts/start_public_service.py`).

---

## 📁 Repository Structure / 仓库文件结构

```
MOF_research/
├── 252_MOF_总文件 冗余评估数据.xlsx     # Primary dataset (156 columns, CoRE MOF 2019 subset)
├── 252_MOF_CIFs/                         # 252 Crystallographic Information Files (*.cif)
├── app.py                                # MOF Chatbot Web Application (Gradio, Port 7860)
├── MOF_Research_Report.md                # Bilingual Master Research Report
├── README.md                             # Project documentation
├── scripts/
│   ├── extract_mof_embeddings.py         # 768-dim 3D crystal embedding extractor
│   ├── train_mof_predictor_models.py     # Fused QSAR ML model trainer & 5-fold CV
│   ├── mof_property_predictor.py         # Zero-shot property predictor & inverse design engine
│   ├── mof_graph_rag_engine.py           # Heterogeneous materials knowledge graph & DeepSeek RAG
│   ├── start_public_service.py           # Production public server & HTTPS tunnel launcher
│   ├── start_public_tunnel.py            # Standalone public tunnel launcher
│   ├── data_loader.py                    # Classical X/Y feature engineering
│   ├── indicator_system.py               # Indicator correlation diagnosis
│   ├── dual_route_ranking.py             # VSA & TSA TOPSIS multi-criteria ranking
│   └── qsar_modeling.py                  # Classical QSAR modeling
└── results/
    ├── models/
    │   └── mof_property_predictors.joblib # Serialized multi-target ML models
    ├── mof_structural_embeddings.npy     # (252, 768) structural embedding matrix
    ├── mof_embedding_index.csv           # MOF index mapping table
    ├── qsar_fused_model_metrics.csv      # Fused representation 5-fold CV metrics
    ├── vsa_rankings.csv                  # VSA route rankings
    └── tsa_rankings.csv                  # TSA route rankings
```

---

## 🚀 How to Run / 如何运行

### 1. Launch MOF Chatbot Web Interface / 启动 Web 界面
```bash
# Start local service on port 7860
python app.py

# Or start with public HTTPS tunnel for external collaborators
python scripts/start_public_service.py
```

### 2. Retrain Fused QSAR ML Models / 重新训练多模态预测模型
```bash
python scripts/train_mof_predictor_models.py
```

### 3. Extract 3D Crystal Embeddings / 批量提取 CIF 结构向量
```bash
python scripts/extract_mof_embeddings.py
```
