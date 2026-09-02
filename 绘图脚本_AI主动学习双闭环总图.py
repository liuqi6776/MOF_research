# -*- coding: utf-8 -*-
"""
CoRE MOF 2025 · AI 主动学习双闭环总图  —— 绘图脚本
===================================================
生成日期: 2026-08-23

用法:
    python 绘图脚本_AI主动学习双闭环总图.py

依赖:
    pip install matplotlib numpy

说明:
  * 纯 matplotlib 绘制，无外部图片素材，所有元素均由坐标定义，可任意修改。
  * 画布坐标系为 200 x 144（左下角原点），与英寸尺寸解耦；
    改 figsize 只改输出分辨率，不影响布局。
  * 三个可配置项见下方 CONFIG 区。
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge, Circle, Rectangle
from matplotlib import font_manager as fm
import numpy as np

# ============================ CONFIG ============================
# 1) 中文字体：按顺序尝试，用第一个存在的
FONT_CANDIDATES = [
    '/mnt/c/Windows/Fonts/msyh.ttc',            # WSL 下的微软雅黑
    'C:/Windows/Fonts/msyh.ttc',                # Windows
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',   # Linux 文泉驿
    '/System/Library/Fonts/PingFang.ttc',       # macOS
]
# 2) 输出文件
OUTPUT = 'AI主动学习双闭环总图_CoREMOF2025.png'
# 3) 输出分辨率（画布逻辑坐标固定 200x144，改这里只改清晰度）
FIGSIZE, DPI = (20, 14.4), 150
# ===============================================================

import os
FP = next((p for p in FONT_CANDIDATES if os.path.isfile(p)), None)
if FP:
    fm.fontManager.addfont(FP)
    ZH = fm.FontProperties(fname=FP)
else:
    print('[警告] 未找到中文字体，中文将显示为方块；请在 FONT_CANDIDATES 中补充路径')
    ZH = fm.FontProperties()
plt.rcParams['axes.unicode_minus'] = False

PETROL='#16564F'; EMBER='#C2571A'; BLUE='#2E5E8E'; GREEN='#3D8B5F'
PURPLE='#6B4E9B'; ORANGE='#C97B2E'; GREY='#6B7F7C'; INK='#12211F'
LBLUE='#DCE7F2'; LGREEN='#DFEDE4'; LPURPLE='#E8E1F2'; LORANGE='#F7E7D5'
LGREY='#ECEFEE'; LTEAL='#E1EDEA'; LRED='#F7DFD5'

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
ax.set_xlim(0, 200); ax.set_ylim(0, 144); ax.axis('off')
ax.set_facecolor('white')

def box(x, y, w, h, fc, ec, lw=1.4, r=1.2, alpha=1.0, ls='-'):
    b = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, linestyle=ls, zorder=2)
    ax.add_patch(b); return b

def txt(x, y, s, size=10, color=INK, weight='normal', ha='center', va='center', zorder=5):
    ax.text(x, y, s, fontproperties=ZH, fontsize=size, color=color,
            fontweight=weight, ha=ha, va=va, zorder=zorder, linespacing=1.55)

def arrow(p1, p2, color=GREY, lw=2.0, style='-|>', ls='-', rad=0.0, ms=18, zorder=4):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, lw=lw,
                        color=color, linestyle=ls, zorder=zorder,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2)
    ax.add_patch(a); return a

# ==================== 标题 ====================
txt(100, 139.5, 'CoRE MOF 2025 · AI 主动学习双闭环：库内迭代 + 生成设计验证回流', 21, PETROL, 'bold')
txt(100, 134.8, '4,100 个 CR/ASR 结构  │  种子 SEED-400 → 每轮 400 个 GCMC → HV-300 冻结盲测 → 收敛后全量补完 → 生成设计 → DFT/MLIP 优化 → GCMC 验证 → 回流再训练',
    11, GREY)

# ================= 左：输入 =================
box(2.5, 72, 31, 57, 'white', GREY, 1.2, 1.4, ls=(0,(4,3)))
txt(18, 126, '输 入  I N P U T', 12.5, GREY, 'bold')

box(4, 113, 27.5, 10, LGREEN, GREEN, 1.5)
txt(17.75, 120.2, 'R0 种子集 SEED-400', 11.5, GREEN, 'bold')
txt(17.75, 116.6, '词典序优化：代表性最优 → 复用最大化\nCSD 6 字母词干级去重，零泄漏', 9.2, INK)

box(4, 100.5, 27.5, 10, LBLUE, BLUE, 1.5)
txt(17.75, 107.7, '四模态特征矩阵 X', 11.5, BLUE, 'bold')
txt(17.75, 103.8, '几何 13 + RACs 176 + 能量格点 ~50\n+ PMTransformer 768   ≈ 1,007 维', 9.2, INK)

box(4, 93.6, 27.5, 5.6, LRED, EMBER, 1.2)
txt(17.75, 96.4, '★ 严守红线\nCp 与 Widom/GEMC 不得入特征（标签泄漏）', 8.2, EMBER, 'bold')

box(4, 83.5, 27.5, 9, LGREY, GREY, 1.5)
txt(17.75, 90.3, '候选池 POOL-3,400', 11.5, GREY, 'bold')
txt(17.75, 86.8, '每轮从中选 400 个\n（含遗留集，选中即免费）', 9.2, INK)

box(4, 75, 27.5, 6.6, LORANGE, ORANGE, 1.4)
txt(17.75, 79.3, '预注册假设 H1–H10', 10.5, ORANGE, 'bold')
txt(17.75, 76.9, '受限窗 / OMS 力场盲区 / 分子筛分窗口', 8.8, INK)

# ================= 中：主动学习闭环 =================
CX, CY, R = 89, 99, 19.0
segs = [(58, 122, BLUE, 'R1'), (328, 58, PURPLE, 'R2'),
        (238, 328, ORANGE, 'R3'), (122, 238, GREEN, 'R4')]
for (t1, t2, c, lab), la in zip(segs, [45, 315, 225, 135]):
    ax.add_patch(Wedge((CX, CY), R+3.0, t1, t2, width=3.0, fc=c, alpha=0.88, zorder=3))
    ar = np.radians(la)
    txt(CX + (R+6.4)*np.cos(ar), CY + (R+6.4)*np.sin(ar), lab, 14, c, 'bold')

ax.add_patch(Circle((CX, CY), 11.2, fc='white', ec=PETROL, lw=2.2, zorder=4))
txt(CX, CY+6.0, '闭环迭代', 12.5, PETROL, 'bold', zorder=6)
txt(CX, CY+2.0, 'R1 › R2 › R3 › R4', 10.5, INK, 'bold', zorder=6)
txt(CX, CY-2.2, '每轮 +400 个 GCMC', 9.4, INK, zorder=6)
txt(CX, CY-6.0, '训练集累计\n400→800→1200→1600→2000', 8.4, GREY, zorder=6)

box(CX-17, CY+20.5, 34, 10.5, LBLUE, BLUE, 1.6)
txt(CX, CY+28.2, '① 代理模型训练', 11.2, BLUE, 'bold')
txt(CX, CY+24.4, 'XGBoost / CatBoost（主力）\n对照：CGCNN · PMTransformer 微调\n分组 5 折 CV（按 CSD 词干）│ 树间方差 = 不确定性', 8.9, INK)

box(CX+23, CY-5.5, 33, 10.5, LPURPLE, PURPLE, 1.6)
txt(CX+39.5, CY+1.8, '② 全库预测', 11.5, PURPLE, 'bold')
txt(CX+39.5, CY-2.0, '对候选池 3,400 个逐一打分\n输出 E_regen 预测值 + 不确定性', 8.9, INK)

box(CX-17, CY-31.5, 34, 10.5, LORANGE, ORANGE, 1.6)
txt(CX, CY-24.2, '③ 采集函数选点  n=400', 11.5, ORANGE, 'bold')
txt(CX, CY-28.2, '200 开发（预测 Top，k-means 去冗余）\n100 不确定性  │  100 分层随机', 8.9, INK)

box(CX-56, CY-5.5, 33, 10.5, LGREEN, GREEN, 1.6)
txt(CX-39.5, CY+1.8, '④ GCMC 计算', 11.5, GREEN, 'bold')
txt(CX-39.5, CY-2.0, '烟气工况 4 条件/结构 → 新增 400 真值\nQC 核验 → 并入训练集', 8.9, INK)

for a0, a1, c in [(150, 105, BLUE), (60, 15, PURPLE), (330, 285, ORANGE), (240, 195, GREEN)]:
    t0, t1 = np.radians(a0), np.radians(a1)
    arrow((CX+(R-2.4)*np.cos(t0), CY+(R-2.4)*np.sin(t0)),
          (CX+(R-2.4)*np.cos(t1), CY+(R-2.4)*np.sin(t1)), c, 2.4, rad=-0.32, ms=20)

arrow((31.5, 114), (CX-16, CY+24), GREEN, 2.2, rad=-0.18)
txt(48, 119, '训练', 10.5, GREEN, 'bold')
arrow((31.5, 87), (CX-18, CY-26), GREY, 1.8, ls=(0,(5,3)), rad=0.18)
txt(50, 76, '候选', 10.5, GREY, 'bold')

# ================= 右：盲验证 =================
box(140, 74, 57, 55, 'white', EMBER, 1.4, 1.4, ls=(0,(4,3)))
txt(168.5, 126, '独 立 盲 验 证   V A L I D A T I O N', 12.5, EMBER, 'bold')

box(142.5, 114.5, 52, 9.5, LRED, EMBER, 1.6)
txt(168.5, 121.3, 'HV-300 冻结盲验证集', 12, EMBER, 'bold')
txt(168.5, 117.4, '零已算 · 与种子 CSD 词干严格互斥 · 永不入训练\n全程同一集合，跨轮可比', 9.2, INK)

box(142.5, 96, 30, 16.5, 'white', GREY, 1.3)
txt(157.5, 110.3, '每轮六项检验', 10.8, INK, 'bold')
txt(157.5, 102.8, '① 预测 R²        ④ 规则前瞻确认\n② 排序 Spearman  ⑤ 不确定性有效性\n③ Top30 富集     ⑥ 校准斜率', 8.8, INK)

ax_in = fig.add_axes([0.876, 0.678, 0.098, 0.088])
r = np.array([0.742, 0.790, 0.822, 0.836])
ax_in.plot([1,2,3,4], r, '-o', color=EMBER, lw=2.0, ms=5)
ax_in.fill_between([1,2,3,4], r-0.028, r+0.028, color=EMBER, alpha=0.18)
ax_in.set_xticks([1,2,3,4]); ax_in.set_xticklabels(['R1','R2','R3','R4'], fontsize=7)
ax_in.tick_params(labelsize=7); ax_in.set_ylim(0.70, 0.87)
ax_in.set_title('HV 上的验证曲线 R²', fontproperties=ZH, fontsize=8.4)
for s in ax_in.spines.values(): s.set_linewidth(0.8)

box(142.5, 86, 52, 8.5, LORANGE, ORANGE, 1.4)
txt(168.5, 91.9, '第二重：本轮随机片 100（前瞻确认）', 10.2, ORANGE, 'bold')
txt(168.5, 88.3, '计算前模型已有预测 → 确认上一轮规则的外推性\n（验证后并入训练；HV 永不并入）', 8.6, INK)

box(142.5, 76.5, 52, 7.5, LGREY, GREY, 1.4)
txt(168.5, 81.8, '第三重：CSD 词干级去重交叉验证', 10.2, GREY, 'bold')
txt(168.5, 78.5, '同一化合物的多次测定不得跨训练/验证侧（防伪重复虚高）', 8.6, INK)

arrow((CX+40, CY+7), (142, 116), EMBER, 2.0, ls=(0,(5,3)), rad=-0.2)
txt(128, 119.5, '每轮模型\n送 HV 盲测', 9.4, EMBER, 'bold')
arrow((142, 99), (CX+40, CY-3), EMBER, 1.8, ls=(0,(5,3)), rad=-0.2)
txt(129, 91, '验证反馈\n(不达标→暂停诊断)', 9.2, EMBER)

# ================= 下：生成设计闭环（新增） =================
box(2.5, 4, 195, 61, 'white', PURPLE, 1.5, 1.4, ls=(0,(4,3)))
txt(100, 61.5, '生 成 设 计 与 验 证 回 流   G E N E R A T I V E   L O O P', 13, PURPLE, 'bold')

# 收敛/全量
box(5.5, 49, 40, 9, LGREEN, GREEN, 1.6)
txt(25.5, 55.4, 'R5 全量补完 ≈ 2,700', 11.5, GREEN, 'bold')
txt(25.5, 51.6, '4,100 全真值 → 终模型 + 设计规则\nSHAP · PDP/ICE · 符号回归', 9.0, INK)
arrow((CX-14, CY-32), (25.5, 58.5), GREEN, 2.2, rad=0.22)
txt(52, 63.5, '收敛后沉淀', 10, GREEN, 'bold')

GY = 28.5
def gbox(x, w, title, body, c, lc, h=15):
    box(x, GY, w, h, lc, c, 1.6)
    txt(x+w/2, GY+h-3.6, title, 11.2, c, 'bold')
    txt(x+w/2, GY+h/2-2.9, body, 8.8, INK)

gbox(6, 34, 'Ⅰ  改造式设计（主线）', '以 Top-N 实测最优 MOF 为母体\n换金属位点 · 换连接体 · 官能团修饰\n构建单元来自 mofid：1,886 连接体 + 163 节点\n→ 可合成性最高', PURPLE, LPURPLE)
gbox(44, 30, 'Ⅱ  理想模型（探索）', '不受可合成性约束\n由设计规则反解最优结构参数\n给出性能上界与"理想靶心"\n→ 指明改造方向', BLUE, LBLUE)
gbox(78, 30, 'Ⅲ  生成模型（先进性）', 'MOFDiff 粗粒化扩散 / MOFGPT\n必须报告有效性·新颖性·可合成性\n用 Ⅰ 的构建单元库交叉检验', ORANGE, LORANGE)

gbox(112, 38, 'Ⅳ  几何优化（关键补强）', '① 全部候选：MLIP 优化（MACE / CHGNet）\n   —— 成本比 DFT 低 3–4 个数量级\n② Top 10–20：周期性 DFT 精修（CP2K/QE）\n不优化直接算 GCMC = 骨架本身就是错的', EMBER, LRED)

gbox(154, 40, 'Ⅴ  GCMC 验证', '与主库完全相同的工况与参数\n（313 K，CO₂ 0.15 + N₂ 0.85 bar）\n否则预测-验证不可比\n→ 得到假想 MOF 的真值', GREEN, LGREEN)

for x0, x1 in [(40, 44), (74, 78), (108, 112), (150, 154)]:
    arrow((x0, GY+7.5), (x1, GY+7.5), GREY, 2.0)

# 回流箭头
arrow((174, GY), (100, 16.5), EMBER, 2.6, rad=0.22)
box(58, 9.5, 84, 8, LRED, EMBER, 1.8)
txt(100, 15.4, 'Ⅵ  验证结果回流 → 再训练 → 第二轮生成', 11.8, EMBER, 'bold')
txt(100, 11.8, '预测 vs 实测配对比较：第二轮候选是否优于第一轮 · 预测-验证一致性是否提升 · 样本须落在训练分布之外方有信息量', 8.8, INK)
arrow((58, 13.5), (23, GY-1), EMBER, 2.4, rad=0.22)
txt(34, 21, '回流再训练', 10.2, EMBER, 'bold')

fig.savefig(OUTPUT, dpi=DPI, bbox_inches='tight', facecolor='white')
print('已生成:', os.path.abspath(OUTPUT))
