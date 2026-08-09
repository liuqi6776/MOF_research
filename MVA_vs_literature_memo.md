# MVA 发现 vs 主流文献对比 Memo / MVA Findings vs Mainstream Literature

> 伴生文档 / Companion to `MVA_Summary.md`
> 生成日期 / Date: 2026-08-09
> 文献来源 / Sources: Google Scholar 检索（8 组查询，约 60 篇），检索记录保存于 `scholar_*.csv`
>
> **分级规则 / Reliability rule**：仅当对应 target 的 CV R² ≥ 0.6 时才将结果表述为"发现"；
> R² < 0.6 的 target 一律降级为"可能（需后续数据验证）"。
>
> | 可靠（R² ≥ 0.6） | R² | 弱（R² < 0.6，仅写"可能"） | R² |
> | :--- | :---: | :--- | :---: |
> | N2_Qst_GCMC_flue_mean | 0.758 | N2_Qst_Widom | 0.539 |
> | log10_CO2N2_actual_selectivity | 0.716 | log10_CO2N2_Henry_selectivity | 0.533 |
> | CO2_Qst_CC_1bar | 0.710 | log10_CO2_TSA_regen_heat | 0.528 |
> | CO2_Qst_CC_mean | 0.693 | N2_ads_0.75bar | 0.482 |
> | log10_PE_VSA_parasitic_energy | 0.675 | Qst_diff_CO2_N2 | 0.348 |

---

## 1. 与主流一致的发现（验证方法可靠性）

| MVA 发现 | 主流依据 |
| :--- | :--- |
| void_fraction / LCD / PLD 是几何描述符核心变量，主导 Qst、选择性、N2 吸附 | Fernandez & Barnard (2016, *ACS Comb. Sci.*) 证明几何性质可预测低压 CO₂/N₂ 吸附；Sumer & Keskin (2016, *I&EC Research*) 以 PLD/LCD/密度作为 MOF 排序的关键描述符 |
| LCD_A 与 Qst 单变量负相关（r ≈ -0.55 ~ -0.66），即小孔 → 高吸附热 | Hu et al. (2017, *AIChE J.*) 最优孔口 ~3.93 Å；Auti et al. (2023, *J. Mater. Chem. A*) 孔径-吸附放热关系 |
| 几何 + 拓扑描述符共同决定 CO₂ 捕集性能 | Anderson et al. (2018, *Chem. Mater.*, 268 引) 孔道化学与拓扑的作用 |

---

## 2. 超越主流的发现（仅列可靠模型支持的）

### 2.1 VSA 寄生能耗的纯结构描述符模型 ✅（R² = 0.675）

- **MVA**：log₁₀_PE_VSA_parasitic_energy 可由 51 个结构描述符预测（CV R² = 0.675），dominance 排序为 void_fraction > ASA_m2_g > LFPD_A > LCD_A > ASA_m2_cm3。
- **主流现状**：工艺界普遍认为平衡态筛选指标**无法**预测工艺性能——Rajagopalan et al. (2016, *Int. J. Greenh. Gas Control*, 231 引) 明确提出此问题；Burns et al. (2020, *Environ. Sci. Technol.*, 228 引)、Khurana & Farooq (2016–2019)、Danaci et al. (2020, *MSDE*) 均强调必须做全流程过程优化才能得到寄生能耗。
- **增量**：descriptor → 工艺能耗的定量构效关系在文献中基本空白。**这是本数据集最有发表潜力的方向之一**。

### 2.2 带符号方向的跨 target 交互图谱 ✅（交互对均由 ≥2 个可靠 target 支持）

- **MVA**：以下交互在多个 R² ≥ 0.6 的 target 上 FDR < 0.05 显著：
  - `density_g_cm3 × void_fraction`：CO2_Qst_CC_1bar（β = -0.315）、actual_selectivity（β = -0.295）——拮抗
  - `PLD_A × density_g_cm3` / `PLD_A × pore_vol_cm3_g`：actual_selectivity（β = -0.170）、N2_Qst_GCMC（β = -0.160）——拮抗
  - `LCD_A × void_fraction`：actual_selectivity（β = +0.178）、N2_Qst_GCMC（β = +0.159）——协同
- **主流现状**：SHAP 交互分析仅见于非 CO₂ 体系（甲苯：Zhang et al., 2026, *Carbon Neutralization*；NH₃：Ti et al., 2025），且无统计显著性检验与符号方向分析；几何描述符之间的物理交互（如"密度放大/抵消孔隙率效应"）从未被系统报道。
- **增量**：**单变量筛选与单 target 研究天然看不到此类联合作用**——这是 MVA 框架的独有产出。

### 2.3 交互符号随 target 反转所揭示的权衡结构（部分可靠）

- `LCD_A × void_fraction` 在选择性/N2 吸附上协同，但在 TSA 再生热上拮抗（β = -0.274）。再生热模型 R² = 0.528 < 0.6，故该反转目前只能表述为**可能的权衡信号，需后续数据验证**（见 §4.2）。
- 若验证成立，意味着"同时优化选择性与再生能耗"存在结构性冲突——主流文献中无此类定量表述。

---

## 3. 与主流相悖或张力较大的结论

### 3.1 控制其他变量后，更大 LCD 反而提高 Qst 与选择性（可靠 target 支持，但机制存疑）⚠️

- **MVA**：LCD_A 在 CO2_Qst_CC_mean（β = +0.335）、CO2_Qst_CC_1bar（β = +0.475）、actual_selectivity（β = +0.471）上净效应为正，而单变量 r 为 -0.55 ~ -0.64。三个 target R² 均 ≥ 0.69，统计上可靠。
- **相悖点**：与"小孔限域增强吸附"的主流共识直接冲突（Hu et al., 2017；Auti et al., 2023）。
- **可能解释**：LCD 与 void_fraction / LFPD 高度共线导致的 Simpson 悖论（统计假象），而非真实物理。**写作时必须作为共线性伪影的候选解释明确讨论，不宜直接宣称物理结论。**

### 3.2 化学描述符被系统性低估 ⚠️（更可能是描述符集合偏倚）

- **MVA**：N_mass_frac（r = +0.149 → β = -0.144）、ligand_n_amino（r = +0.248 → β = -0.150，于寄生能耗）净效应为负；化学特征普遍排在几何特征之后。
- **相悖点**：主流强调孔道化学/官能团主导低压 CO₂ 亲和——Anderson et al. (2018) 得出"化学 > 拓扑"；Torrisi et al. (2010, *JCP*)、Xiang & Cao (2012, *JPCC*) 证明 N 基/芳香官能团增强 CO₂ 结合；Orhan et al. (2023, *Commun. Chem.*) 发现 Henry 系数相对重要性达 0.83（本质是化学亲和项）。
- **可能解释**：本数据集 51 个描述符几乎全为几何量，化学维度覆盖不全 → 化学效应被几何代理变量吸收。**不能作为"化学不重要"的证据。**

### 3.3 拓扑哑变量进入 Dominance Top-5 ⚠️

- **MVA**：topo_pcu 出现在 CO2_Qst_CC_1bar（可靠，R² = 0.710）与 N2_Qst_GCMC（可靠，R² = 0.758）的 dominance Top-5。
- **相悖点**：Anderson et al. (2018) 明确得出拓扑重要性低于化学。
- **可能解释**：同 §3.2，化学描述符缺失导致拓扑相对排名膨胀。拓扑主效应本身与文献中 pcu 拓扑 MOF 高 CO₂/N₂ 选择性的报道（Shabangu et al., 2025, *J. Mater. Chem. A*）方向一致。

---

## 4. 可能的新发现（弱模型 R² < 0.6，需后续数据验证，不作为结论）

### 4.1 catenation 是 Qst_diff 的 Top-2 dominance 特征（R² = 0.348）

- 文献中穿插/互穿仅为定性策略（Liu et al., 2012, *Chem. Soc. Rev.*, 1521 引；Yang et al., 2012, *Nat. Mater.*；双重互穿提升选择性 Liu et al., 2024, *ACS AMI*）。
- 本结果**可能**是"穿插程度定量预测 CO₂–N₂ 差分吸附热"的首次信号，但模型过弱（R² = 0.348），需扩充数据或改进模型后验证。

### 4.2 TSA 再生热的描述符模型及交互（R² = 0.528）

- void_fraction / PLD_A 主导再生热、`PLD_A × density_g_cm3` 协同（β = +0.251）、`LCD_A × void_fraction` 拮抗（β = -0.274）、ligand_n_aromatic_rings 进 Top-5 等结果均**可能**成立，但 R² < 0.6，需后续验证。再生热的 descriptor 级构效关系文献空白，若验证将有较高价值。

### 4.3 Henry 选择性中 topo_dia 的 dominance 地位（R² = 0.533）

- **可能**提示 dia 拓扑对无限稀释选择性有特异作用，需验证。

### 4.4 N2 吸附的强交互（R² = 0.482）

- `LCD_A × void_fraction` 协同 β = +0.402、`PLD_A × density_g_cm3` 拮抗 β = -0.349（p_fdr ≈ 0.0001–0.0003，统计显著性很强），但模型 R² 不足，**可能**为真实强交互，需后续数据确认。

### 4.5 N2_Qst_Widom 上 ligand_n_heavy 的抑制效应（R² = 0.539）

- β = +1.01~1.22 的大幅度方向反转**可能**反映配体尺寸的真实间接效应，需验证。

---

## 5. 总体判断 / Summary

| 类别 | 内容 | 可发表性 |
| :--- | :--- | :--- |
| 最强增量 | §2.1 寄生能耗描述符模型；§2.2 带符号的多 target 交互图谱 | 高（文献空白 + 模型可靠） |
| 需谨慎表述 | §3.1 LCD 净效应反号；§3.2 化学被低估 | 应作为共线性/描述符覆盖问题讨论，不宜作为物理结论 |
| 待验证线索 | §4.1–4.5（catenation、再生热、N2 强交互等） | 有潜力，但当前证据不足 |

**方法论注意事项**：n = 244 个 MOF；特征间共线性强（|r| ≥ 0.99 分组后仍有大量高 VIF）；抑制效应中相当比例应为共线性产物而非物理间接效应。建议后续：扩充化学描述符（电荷、偶极、OMS 标记）、对 §4 各线索做留出集/外部验证。

---

## 6. 参考文献（本轮回合实际检索到的条目）

1. Orhan IB, Le TC, Babarao R, Thornton AW. *Accelerating the prediction of CO₂ capture at low partial pressures in MOFs using new ML descriptors.* Commun. Chem., 2023.
2. Fernandez M, Barnard AS. *Geometrical Properties Can Predict CO₂ and N₂ Adsorption Performance of MOFs at Low Pressure.* ACS Comb. Sci., 2016.
3. Sumer Z, Keskin S. *Ranking of MOF Adsorbents for CO₂ Separations: A Molecular Simulation Study.* I&EC Research, 2016.
4. Anderson R, Rodgers J, Argueta E, et al. *Role of Pore Chemistry and Topology in the CO₂ Capture Capabilities of MOFs.* Chem. Mater., 2018.
5. Hu Z, Wang Y, Farooq S, Zhao D. *A highly stable MOF with optimum aperture size for CO₂ capture.* AIChE J., 2017.
6. Auti G, Kametani Y, Kimura H, et al. *Effect of pore size on heat release from CO₂ adsorption in MIL-101, MOF-177, and UiO-66.* J. Mater. Chem. A, 2023.
7. Rajagopalan AK, Avila AM, Rajendran A. *Do adsorbent screening metrics predict process performance?* Int. J. Greenh. Gas Control, 2016.
8. Burns TD, Pai KN, Subraveti SG, et al. *Prediction of MOF Performance in VSA Systems for Postcombustion CO₂ Capture.* Environ. Sci. Technol., 2020.
9. Khurana M, Farooq S. *Adsorbent Screening for Postcombustion CO₂ Capture / Integrated adsorbent-process optimization.* I&EC Research 2016; AIChE J. 2017, 2019.
10. Danaci D, Bui M, Mac Dowell N, Petit C. *Exploring the limits of adsorption-based CO₂ capture using MOFs with PVSA.* Mol. Syst. Des. Eng., 2020.
11. Liu J, Thallapally PK, McGrail BP, et al. *Progress in adsorption-based CO₂ capture by MOFs.* Chem. Soc. Rev., 2012.
12. Yang S, Lin X, Lewis W, et al. *A partially interpenetrated MOF for selective hysteretic sorption of CO₂.* Nat. Mater., 2012.
13. Liu S, Wang L, Zhang H, et al. *Efficient CO₂ Capture and Separation in MOFs: Effect from Isoreticular Double Interpenetration.* ACS Appl. Mater. Interfaces, 2024.
14. Zhang J, He C, Ji Y, et al. *An Explainable Stacked ML Approach for Toluene Capture in MOFs (SHAP interactions).* Carbon Neutralization, 2026.
15. Ti H, Yang L, Yan W, et al. *Structure-based ML for screening MOFs with high ammonia capture capacity (SHAP interactions).* Process Saf. Environ. Prot., 2025.
16. Torrisi A, Mellot-Draznieks C, Bell RG. *Impact of ligands on CO₂ adsorption in MOFs: interaction of CO₂ with functionalized benzenes.* J. Chem. Phys., 2010.
17. Xiang Z, Leng S, Cao D. *Functional Group Modification of MOFs for CO₂ Capture.* J. Phys. Chem. C, 2012.
18. Shabangu SM, Eaby AC, Javan Nikkhah S, et al. *A pcu topology MOF with high CO₂/N₂ selectivity and low water vapour affinity.* J. Mater. Chem. A, 2025.
19. Zhang Z, Palakkal AS, Wu X, Jiang J, et al. *Discovering ultra-stable MOFs for CO₂ capture from wet flue gas: integrating ML and molecular simulation.* Environ. Sci. Technol., 2025.
20. Burner J, Schwiedrzik L, Krykunov M, et al. *High-Performing Deep Learning Regression Models for Low-Pressure CO₂ Adsorption Properties of MOFs.* J. Phys. Chem. C, 2020.
