# PMTransformer 数据包 — 695 个已完成 GCMC 的 MOF（全覆盖）

CoRE MOF 2019 上乌镇超算已跑完全部 24 个 GCMC/Widom 条件的 **695 个 MOF**，全部完成 PMTransformer 转换。

## 内容

| 文件/目录 | 说明 |
|---|---|
| `embeddings.csv` | 695 行 × 769 列。第 1 列 `MOF_ID`，其后 `emb_0 … emb_767` = PMTransformer（预训练权重 `pmtransformer.ckpt`）输出的 **768 维 CLS 嵌入**。直接作为特征喂下游回归器预测 CO₂ / N₂ 吸附量。 |
| `mof_list.txt` | 695 个 MOF 名单。 |
| `moftransformer_inputs/` | 每 MOF 4 个文件（共 2780），MOFTransformer/PMTransformer 已转换输入，可直接用于微调，无需重跑 GRIDAY： |
| &nbsp;&nbsp;`<MOF>.griddata16` | GRIDAY 能量网格，30×30×30，UFF 力场 + **CH₄ 单点探针**，float16 pickle。与气体无关。**微调必需。** |
| &nbsp;&nbsp;`<MOF>.graphdata` | 原子图数据（近邻列表 + 原子特征）。**微调必需。** |
| &nbsp;&nbsp;`<MOF>.grid` | GRIDAY 网格头（文本）。 |
| &nbsp;&nbsp;`<MOF>.cif` | 转换所用结构（`_make_supercell` 处理后）。 |

## 来源与口径

- 685 个来自 2026-08 的批处理（`batch_prepare_5529.py`，默认参数）。
- 9 个大结构（ALAHUS/ASINEX/ASINIB/AVAJUE/BIHBAX/HUVNAP/IZIHOQ/LAFQUG/LETQAE _clean）当时因内存上限与
  `max_num_unique_atoms=300` / `max_length=60` 被拒，本次**放开这两个限制**单独补算（2026-08-31）。
  其能量网格仍是标准 30³；图的拓扑唯一原子数 >300，属预训练管线默认会拒绝的结构，能用但口径略有出入。
- `JUTCUW_SL`：原胞仅 10 原子（8.94 Å），近邻数 <12。用 **3×3×3 预扩胞**（270 原子）后正常转换。
- CO₂/N₂ 的 GCMC 标签见 `CoRE MOF 2019 数据包/CoRE_MOF_2019_GCMC_695_总文件.xlsx`。

## 关键说明

**能量网格与气体无关**：每个 MOF 只有一张网格（CH₄/UFF）。CO₂ 与 N₂ 的区别只在微调时用的标签。

## 两种下游用法

1. **嵌入 + 经典回归**（小数据首选，几分钟）：读 `embeddings.csv`，与 xlsx 吸附量对齐，训 GBM/Ridge。
2. **微调 PMTransformer**：`moftransformer_inputs/` 按 8:1:1 切分，写 `{split}.json`（`{MOF_ID: 标签}`），
   从 `pmtransformer.ckpt` 微调。**注意：本机 RTX 5060 (sm_120) 比 moftransformer 锁定的 torch 1.13
   (≤sm_86) 新，跑 GPU 会挂起——嵌入抽取和微调都只能在 CPU 上，或需自行升级 torch 并适配。**

moftransformer 2.2.0，导出 2026-08-31。
