# -*- coding: utf-8 -*-
"""
MVA: Multivariate Analysis of MOF Structure-Property Relationships
多属性联合作用分析（Multivariate Analysis）

目标：将研究从"单变量相关性"升级为"多个属性对 target 的共同作用"：
  L1  线性净效应   —— 全特征 OLS（标准化 beta / 半偏相关 / p 值）+ Ridge/Lasso 互补
  L2  方差分解     —— top-12 特征精确 Dominance / Commonality 分解 + 全特征 SHAP main/interaction 分解
  L3  交互效应     —— 物理候选对交互项回归（BH-FDR 校正）+ SHAP interaction 全对扫描 + 2D PDP
  L4  ML 建模      —— SVR/RF/ExtraTrees/XGBoost/Ridge CV 对比 + permutation + SHAP top-features 综合排名

设计约定（用户指定）：
  1) Y 目标冗余分组：|r| >= 0.99 的目标对只罗列，不重复建模，每组保留 1 个代表 target
  2) X 使用全特征（51 结构描述符，不含 GCMC 原始数据），不做 VIF 剔除，VIF 仅作诊断
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as sp_stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import xgboost as xgb
import shap

try:
    from scripts.data_loader import load_mof_dataset
    from scripts.indicator_system import run_indicator_system_analysis
except ImportError:
    from data_loader import load_mof_dataset
    from indicator_system import run_indicator_system_analysis

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
RNG = 42

# ---------------------------------------------------------------
# 常数定义
# ---------------------------------------------------------------
NON_X_COLS = ['MOF_id', 'MOF_name', 'primary_metal', 'topology',
              'primary_metal_grouped', 'topology_grouped']
NON_Y_COLS = ['MOF_id', 'MOF_name']

# 重尾指标：建模用 log10 版本（与 QSAR 模型一致）
LOG10_COLS = ['CO2N2_Henry_selectivity', 'CO2N2_actual_selectivity',
              'PE_VSA_parasitic_energy', 'CO2_TSA_regen_heat']

# 代表 target 选择优先级（命中即作为该冗余组的代表）
REP_PRIORITY = [
    'log10_CO2N2_actual_selectivity', 'log10_PE_VSA_parasitic_energy',
    'log10_CO2_TSA_regen_heat', 'log10_CO2N2_Henry_selectivity',
    'CO2_VSA_capacity', 'CO2_TSA_capacity', 'CO2_ads_0.15bar',
    'CO2_Qst_CC_mean', 'CO2_Qst_CC_0.1bar', 'CO2_Qst_CC_0.5bar',
    'CO2_Qst_CC_1bar', 'CO2_Qst_Widom', 'CO2_Qst_GCMC_flue_mean',
    'N2_ads_0.75bar', 'N2_ads_1bar', 'N2_Qst_Widom',
    'N2_Qst_GCMC_flue_mean', 'N2_Qst_1bar', 'Qst_diff_CO2_N2'
]

# 物理上有依据的交互候选对（L3 回归检验）
PHYSICAL_INTERACTION_PAIRS = [
    ('PLD_A', 'ASA_m2_g'), ('PLD_A', 'has_oms'), ('PLD_A', 'density_g_cm3'),
    ('PLD_A', 'LCD_A'), ('PLD_A', 'pore_vol_cm3_g'), ('ASA_m2_g', 'density_g_cm3'),
    ('ASA_m2_g', 'pore_vol_cm3_g'), ('ASA_m2_g', 'void_fraction'),
    ('LCD_A', 'void_fraction'), ('density_g_cm3', 'void_fraction'),
    ('has_oms', 'density_g_cm3'), ('PLD_A', 'ligand_MW'),
    ('has_oms', 'ASA_m2_g'), ('PLD_A', 'C_mass_frac'),
]

REDUNDANCY_THRESHOLD = 0.99

MODELS = {
    'SVR': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=10.0, gamma='scale'))
    ]),
    'RandomForest': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('rf', RandomForestRegressor(n_estimators=200, random_state=RNG))
    ]),
    'ExtraTrees': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('et', ExtraTreesRegressor(n_estimators=200, random_state=RNG))
    ]),
    'XGBoost': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('xgb', xgb.XGBRegressor(n_estimators=150, learning_rate=0.05,
                                  max_depth=4, subsample=0.9, random_state=RNG))
    ]),
    'Ridge': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc', StandardScaler()),
        ('ridge', Ridge(alpha=10.0))
    ]),
}


# ---------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------
def log10_transform(df_metrics):
    """对重尾指标做 log10 变换，返回含 log10 列的新表。"""
    df_t = df_metrics.copy()
    for col in LOG10_COLS:
        df_t[f'log10_{col}'] = np.log10(df_t[col].clip(lower=1e-4))
    return df_t


def union_find_groups(pairs, nodes):
    """根据高相关边 pairs 对 nodes 做连通分量分组。"""
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for a, b in pairs:
        union(a, b)
    groups = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


def calculate_vif(X):
    """计算 VIF（多重共线性诊断，仅作报告不用于剔除）。"""
    vifs = {}
    for col in X.columns:
        others = X.drop(columns=[col])
        y = X[col]
        try:
            reg = LinearRegression().fit(others, y)
            r2 = reg.score(others, y)
            vifs[col] = np.inf if r2 >= 0.9999 else 1.0 / (1.0 - r2)
        except Exception:
            vifs[col] = np.inf
    return pd.Series(vifs)


def standardized_ols(X, y):
    """全特征标准化 OLS：返回 beta/se/t/p/semipartial/R2。NaN/inf 用中位数插补。"""
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    # 中位数插补（拟合后再标准化，避免插补值影响统计推断的合理性有限但稳健）
    X = X.fillna(X.median())
    Xs = (X - X.mean()) / (X.std(ddof=0) + 1e-12)
    ys = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
    Xc = sm.add_constant(Xs)
    res = sm.OLS(ys, Xc).fit()
    n, k = Xc.shape
    df = n - k
    t = res.tvalues[1:]
    r2 = res.rsquared
    semipartial = t * np.sqrt((1.0 - r2) / (df + t ** 2))
    out = pd.DataFrame({
        'beta': res.params[1:],
        'se': res.bse[1:],
        't': t,
        'p_value': res.pvalues[1:],
        'semipartial': semipartial,
    })
    return res, out, r2


def _fast_r2(Xmat, y):
    """OLS R² 快速计算（带截距）。Xmat: 标准化设计矩阵（不含截距列）。"""
    Xc = np.column_stack([np.ones(len(y)), Xmat])
    coef, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def dominance_analysis(X, y, feats):
    """General dominance analysis：枚举全部子集，返回每特征的平均增量贡献。"""
    k = len(feats)
    Xs = (X[feats] - X[feats].mean()) / (X[feats].std(ddof=0) + 1e-12)
    ys = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
    yv = ys.values.astype(float)
    r2_map = {0: 0.0}
    for mask in range(1, 1 << k):
        cols = [i for i in range(k) if mask & (1 << i)]
        r2_map[mask] = _fast_r2(Xs.values[:, cols], yv)
    dom = {}
    for i in range(k):
        incs = []
        for mask in range(1 << k):
            if mask & (1 << i):
                continue
            incs.append(r2_map[mask | (1 << i)] - r2_map[mask])
        dom[feats[i]] = np.mean(incs)
    return pd.Series(dom), r2_map


def commonality_analysis(X, y, feats, r2_map, full_mask):
    """unique / pairwise-common / total 贡献分解（基于已枚举的子集 R²）。"""
    k = len(feats)
    Xs = (X[feats] - X[feats].mean()) / (X[feats].std(ddof=0) + 1e-12)
    ys = (y - y.mean()) / (y.std(ddof=0) + 1e-12)
    yv = ys.values.astype(float)
    r2_full = r2_map[full_mask]
    # 单变量 R² (total contribution)
    total = {}
    for i, f in enumerate(feats):
        total[f] = _fast_r2(Xs.values[:, [i]], yv)
    # unique: R2_full - R2_without_i
    unique = {}
    for i, f in enumerate(feats):
        mask_wo = full_mask & ~(1 << i)
        unique[f] = r2_full - r2_map[mask_wo]
    # pairwise common: R2_without_j - R2_without_ij - unique_i
    common = {}
    for i, j in itertools.combinations(range(k), 2):
        fi, fj = feats[i], feats[j]
        m_woj = full_mask & ~(1 << j)
        m_woij = full_mask & ~(1 << i) & ~(1 << j)
        common[(fi, fj)] = r2_map[m_woj] - r2_map[m_woij] - unique[fi]
    # 高阶共同（未分解的剩余部分）
    high_order = r2_full - sum(unique.values()) - sum(common.values())
    return pd.Series(total), pd.Series(unique), common, high_order


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR 校正。"""
    pvals = np.asarray(pvals, dtype=float)
    if len(pvals) == 0:
        return pvals
    rejected, pcorr, _, _ = multipletests(pvals, method='fdr_bh')
    return pcorr


# ---------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------
def run_mva_analysis(df_y, x_encoded, output_dir='results/mva'):
    os.makedirs(output_dir, exist_ok=True)
    print("[MVA] ================================================")
    print("[MVA] 数据准备：Y 指标 + log10 变换 + X 全特征")

    # ---------- 1. Y 指标表（含 log10）----------
    metric_cols = [c for c in df_y.columns if c not in NON_Y_COLS]
    df_metrics = df_y[metric_cols].copy()
    df_metrics = log10_transform(df_metrics)
    y_metric_cols = [c for c in df_metrics.columns if c not in NON_Y_COLS]

    # ---------- 2. Y 冗余分组（|r| >= 0.99 只罗列）----------
    spearman = df_metrics[y_metric_cols].corr(method='spearman')
    pairs = []
    for a, b in itertools.combinations(y_metric_cols, 2):
        r = spearman.loc[a, b]
        if abs(r) >= REDUNDANCY_THRESHOLD:
            pairs.append((a, b, r))
    groups = union_find_groups([(a, b) for a, b, _ in pairs], y_metric_cols)

    # 选择代表 target：优先级命中优先，否则取组内与其他成员平均 |r| 最高者
    group_records = []
    representative_targets = []
    for gi, grp in enumerate(groups):
        rep = None
        for p in REP_PRIORITY:
            if p in grp:
                rep = p
                break
        if rep is None:
            avg_r = {g: np.nanmean([abs(spearman.loc[g, o]) for o in grp if o != g])
                     for g in grp}
            rep = max(avg_r, key=avg_r.get)
        group_records.append({'group_id': gi + 1, 'group_members': '; '.join(grp),
                              'n_members': len(grp), 'representative': rep})
        representative_targets.append(rep)

    df_groups = pd.DataFrame(group_records)
    df_groups.to_csv(os.path.join(output_dir, 'y_redundant_groups.csv'),
                     index=False, encoding='utf-8-sig')
    df_pairs = pd.DataFrame(pairs, columns=['var_a', 'var_b', 'spearman_r'])
    df_pairs.to_csv(os.path.join(output_dir, 'y_redundant_pairs.csv'),
                    index=False, encoding='utf-8-sig')

    n_red = len(pairs)
    print(f"[MVA] Y 冗余分组完成：{len(y_metric_cols)} 个指标 -> "
          f"{len(groups)} 组（{n_red} 对 |r|>=0.99 冗余对）")
    print(f"[MVA] 实际建模 target：{len(representative_targets)} 个 -> "
          f"{representative_targets}")

    # ---------- 3. X 全特征 ----------
    x_feats = [c for c in x_encoded.columns if c not in NON_X_COLS]
    X = x_encoded[x_feats].apply(pd.to_numeric, errors='coerce')
    # one-hot 每组去掉参照类，避免完美共线
    metal_cols = sorted([c for c in x_feats if c.startswith('metal_')])
    topo_cols = sorted([c for c in x_feats if c.startswith('topo_')])
    drop_ref = set()
    if metal_cols:
        drop_ref.add(metal_cols[0])
    if topo_cols:
        drop_ref.add(topo_cols[0])
    ols_feats = [c for c in x_feats if c not in drop_ref]

    # VIF 诊断（全特征）
    X_vif = X.fillna(X.median())
    vif = calculate_vif(X_vif)
    vif.to_csv(os.path.join(output_dir, 'vif_diagnostic.csv'), encoding='utf-8-sig')
    print(f"[MVA] X 全特征 {len(x_feats)} 列（OLS 用 {len(ols_feats)} 列，"
          f"去除参照类 {sorted(drop_ref)}）；VIF 诊断已输出")

    # 全局中位数插补（清理 NaN/inf），供所有层统一使用
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    X_clean = X.copy()

    # ---------- 4. 逐 target 建模 ----------
    beta_records, detail_records = [], []
    univariate_records, suppress_records = [], []
    lasso_records = []
    dom_records, com_records = [], []
    inter_records = []
    model_cv_records = []
    top_feat_records = []
    shap_int_pairs_records = []
    shap_decomp_records = []
    best_pdp_pairs = []  # 用于 2D PDP 绘制的 top 交互对

    for t_name in representative_targets:
        y_vec = pd.to_numeric(df_metrics[t_name], errors='coerce')
        print(f"\n[MVA] === Target: {t_name} (N={y_vec.notna().sum()}) ===")

        # ---- L1: 线性净效应（全特征 OLS + Ridge/Lasso）----
        res, beta_df, r2 = standardized_ols(X[ols_feats], y_vec)
        for f in ols_feats:
            beta_records.append({'Target': t_name, 'Feature': f,
                                 'beta_std': beta_df.loc[f, 'beta'],
                                 'semipartial': beta_df.loc[f, 'semipartial'],
                                 'p_value': beta_df.loc[f, 'p_value']})
        detail_records.append({'Target': t_name, 'model': 'OLS_full', 'R2': r2,
                               'adj_R2': res.rsquared_adj,
                               'n_features': len(ols_feats)})

        # RidgeCV / LassoCV 互补（共线稳健性）
        X_imp = SimpleImputer(strategy='median').fit_transform(X)
        y_imp = y_vec.fillna(y_vec.median()).values
        sc = StandardScaler()
        Xs = sc.fit_transform(X_imp)
        lasso = LassoCV(cv=5, random_state=RNG, max_iter=50000).fit(Xs, y_imp)
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit(Xs, y_imp)
        lasso_sel = [x_feats[i] for i, c in enumerate(lasso.coef_) if abs(c) > 1e-6]
        detail_records.append({'Target': t_name, 'model': 'LassoCV',
                               'R2': lasso.score(Xs, y_imp), 'adj_R2': np.nan,
                               'n_features': len(lasso_sel)})
        lasso_records.append({'Target': t_name, 'lasso_features': '; '.join(lasso_sel)})

        # ---- 单变量 Spearman 与 β 对比（抑制效应识别）----
        for f in ols_feats:
            r_sp, p_sp = sp_stats.spearmanr(X[f], y_vec, nan_policy='omit')
            univariate_records.append({'Target': t_name, 'Feature': f,
                                       'spearman_r': r_sp, 'spearman_p': p_sp})
        suppress = []
        for f in ols_feats:
            b = beta_df.loc[f, 'beta']
            r_row = next((r for r in univariate_records
                          if r['Target'] == t_name and r['Feature'] == f), None)
            if r_row is None:
                continue
            rv = r_row['spearman_r']
            if abs(rv) > 0.1 and abs(b) > 0.1 and np.sign(rv) != np.sign(b):
                suppress.append({'Target': t_name, 'Feature': f,
                                 'spearman_r': rv, 'beta_std': b,
                                 'note': '方向反转(单变量r vs 净效应β)'})
        suppress_records.extend(suppress)
        if suppress:
            print(f"[MVA]   抑制效应特征: {[s['Feature'] for s in suppress]}")

        # ---- L2: top-12 Dominance / Commonality ----
        top12 = beta_df['beta'].abs().sort_values(ascending=False).head(12).index.tolist()
        top12 = [f for f in top12 if f in X.columns]
        dom_ser, r2_map = dominance_analysis(X, y_vec, top12)
        full_mask = (1 << len(top12)) - 1
        total_ser, unique_ser, common_dict, high_order = commonality_analysis(
            X, y_vec, top12, r2_map, full_mask)
        for f in top12:
            dom_records.append({'Target': t_name, 'Feature': f,
                                'dominance': dom_ser[f], 'unique': unique_ser[f],
                                'total_R2_single': total_ser[f]})
        for (fi, fj), v in sorted(common_dict.items(),
                                  key=lambda x: -abs(x[1]))[:8]:
            com_records.append({'Target': t_name, 'Feature_i': fi, 'Feature_j': fj,
                                'common_contrib': v})
        if high_order != 0:
            com_records.append({'Target': t_name, 'Feature_i': '(higher-order)',
                                'Feature_j': '', 'common_contrib': high_order})
        print(f"[MVA]   L2 dominance top5: "
              f"{dom_ser.sort_values(ascending=False).head(5).round(3).to_dict()}")

        # ---- L3a: 物理候选对交互项回归 + BH-FDR ----
        Xs_ols = (X - X.mean()) / (X.std(ddof=0) + 1e-12)
        t_inter = []
        for f1, f2 in PHYSICAL_INTERACTION_PAIRS:
            if f1 not in X.columns or f2 not in X.columns:
                continue
            base_cols = [c for c in ols_feats if c not in (f1, f2)]
            inter = Xs_ols[f1] * Xs_ols[f2]
            inter.name = f'{f1}__x__{f2}'
            Xint = sm.add_constant(pd.concat([Xs_ols[base_cols], inter], axis=1))
            try:
                r_int = sm.OLS((y_vec - y_vec.mean()) / (y_vec.std(ddof=0) + 1e-12),
                               Xint).fit()
                p_int = r_int.pvalues.iloc[-1]
                b_int = r_int.params.iloc[-1]
            except Exception:
                p_int, b_int = np.nan, np.nan
            t_inter.append({'Target': t_name, 'feat_i': f1, 'feat_j': f2,
                            'interaction_beta': b_int, 'p_raw': p_int})
        if t_inter:
            pcorr = bh_fdr([r['p_raw'] for r in t_inter])
            for r, pc in zip(t_inter, pcorr):
                r['p_fdr'] = pc
            inter_records.extend(t_inter)
            sig = [r for r in t_inter if r['p_fdr'] < 0.05]
            if sig:
                print(f"[MVA]   L3 显著交互(FDR<0.05): "
                      f"{[(r['feat_i'], r['feat_j'], round(r['p_fdr'],4)) for r in sig]}")

        # ---- L4: ML 模型 CV + SHAP ----
        cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RNG)
        for m_name, model in MODELS.items():
            scores = cross_validate(model, X, y_vec, cv=cv,
                                    scoring=['r2', 'neg_mean_absolute_error',
                                             'neg_root_mean_squared_error'])
            model_cv_records.append({
                'Target': t_name, 'Model': m_name,
                'R2_mean': np.mean(scores['test_r2']),
                'R2_std': np.std(scores['test_r2']),
                'MAE_mean': -np.mean(scores['test_neg_mean_absolute_error']),
                'RMSE_mean': -np.mean(scores['test_neg_root_mean_squared_error'])
            })
        best_m = max(model_cv_records[-len(MODELS):], key=lambda r: r['R2_mean'])
        print(f"[MVA]   L4 best model: {best_m['Model']} R2={best_m['R2_mean']:.3f}")

        # XGBoost + SHAP（importance 与 interaction）
        xgb_pipe = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('xgb', xgb.XGBRegressor(n_estimators=150, learning_rate=0.05,
                                      max_depth=4, subsample=0.9, random_state=RNG))
        ])
        xgb_pipe.fit(X, y_vec)
        X_imp2 = SimpleImputer(strategy='median').fit_transform(X)
        xgb_model = xgb_pipe.named_steps['xgb']
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(X_imp2)
        shap_imp = np.mean(np.abs(shap_vals), axis=0)
        # SHAP interaction（全对扫描，用于"新发现"）
        shap_int = None
        try:
            shap_int = explainer.shap_interaction_values(X_imp2)
            abs_int = np.mean(np.abs(shap_int), axis=0)
            np.fill_diagonal(abs_int, 0.0)
            # main vs interaction 分解（L2 全特征层面）
            main_abs = np.mean(np.abs(shap_vals), axis=0)
            inter_sum = np.mean(np.abs(shap_int), axis=0).sum(axis=1)
            for idx, f in enumerate(x_feats):
                shap_decomp_records.append({
                    'Target': t_name, 'Feature': f,
                    'shap_main': main_abs[idx],
                    'shap_interaction_total': inter_sum[idx],
                    'shap_interaction_share': inter_sum[idx] / (main_abs[idx] + inter_sum[idx] + 1e-12)
                })
            pairs_sorted = sorted(
                [(i, j, abs_int[i, j]) for i in range(len(x_feats))
                 for j in range(i + 1, len(x_feats))],
                key=lambda x: -x[2])[:10]
            for i, j, v in pairs_sorted:
                if v > 0:
                    shap_int_pairs_records.append({
                        'Target': t_name, 'feat_i': x_feats[i], 'feat_j': x_feats[j],
                        'shap_interaction_magnitude': v})
            # 记录 top1 交互对用于 2D PDP
            if pairs_sorted and pairs_sorted[0][2] > 0:
                best_pdp_pairs.append((t_name, x_feats[pairs_sorted[0][0]],
                                       x_feats[pairs_sorted[0][1]]))
        except Exception as e:
            print(f"[MVA]   SHAP interaction 计算失败: {e}")

        # permutation importance（RF）
        rf_pipe = Pipeline([
            ('imp', SimpleImputer(strategy='median')),
            ('rf', RandomForestRegressor(n_estimators=200, random_state=RNG))
        ])
        rf_pipe.fit(X, y_vec)
        perm = permutation_importance(rf_pipe, X, y_vec, n_repeats=10,
                                      random_state=RNG, scoring='r2')
        perm_imp = pd.Series(perm.importances_mean, index=x_feats)

        # 综合 top-N：RF importance + XGB SHAP + RF permutation 三路 rank 平均
        rf_imp = pd.Series(rf_pipe.named_steps['rf'].feature_importances_, index=x_feats)
        rank_df = pd.DataFrame({
            'rf_importance': rf_imp,
            'xgb_shap': pd.Series(shap_imp, index=x_feats),
            'rf_permutation': perm_imp,
        })
        rank_sum = rank_df.rank(ascending=False, method='first').mean(axis=1)
        for f, rk in rank_sum.sort_values().head(15).items():
            top_feat_records.append({'Target': t_name, 'Feature': f,
                                     'rank_avg': rk,
                                     'rf_importance': rf_imp[f],
                                     'shap_mean_abs': rank_df.loc[f, 'xgb_shap'],
                                     'rf_permutation': perm_imp[f]})

    # ---------- 5. 汇总输出 ----------
    pd.DataFrame(beta_records).to_csv(
        os.path.join(output_dir, 'mva_ols_beta.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(detail_records).to_csv(
        os.path.join(output_dir, 'mva_ols_details.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(lasso_records).to_csv(
        os.path.join(output_dir, 'mva_lasso_features.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(univariate_records).to_csv(
        os.path.join(output_dir, 'mva_univariate_spearman.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(suppress_records).to_csv(
        os.path.join(output_dir, 'mva_suppression_findings.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(dom_records).to_csv(
        os.path.join(output_dir, 'mva_dominance.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(com_records).to_csv(
        os.path.join(output_dir, 'mva_commonality.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(inter_records).to_csv(
        os.path.join(output_dir, 'mva_interactions.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(model_cv_records).to_csv(
        os.path.join(output_dir, 'mva_model_metrics.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(top_feat_records).to_csv(
        os.path.join(output_dir, 'mva_top_features.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(shap_int_pairs_records).to_csv(
        os.path.join(output_dir, 'mva_shap_interaction_pairs.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(shap_decomp_records).to_csv(
        os.path.join(output_dir, 'mva_shap_decomposition.csv'), index=False, encoding='utf-8-sig')

    # 图：OLS beta 热图
    df_beta_pivot = pd.DataFrame(beta_records).pivot_table(
        index='Target', columns='Feature', values='beta_std')
    df_beta_pivot = df_beta_pivot.reindex(columns=
        df_beta_pivot.abs().mean().sort_values(ascending=False).index[:25])
    fig, ax = plt.subplots(figsize=(14, max(4, len(df_beta_pivot) * 0.6)))
    sns.heatmap(df_beta_pivot, cmap='vlag', center=0, vmin=-0.6, vmax=0.6,
                annot=True, fmt='.2f', linewidths=.3, ax=ax)
    ax.set_title('Standardized OLS Beta (Net Effects) per Target', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mva_beta_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()

    # 图：dominance 堆叠对比
    df_dom = pd.DataFrame(dom_records)
    if not df_dom.empty:
        dom_pivot = df_dom.pivot_table(index='Target', columns='Feature',
                                       values='dominance').reindex(
            columns=df_dom.groupby('Feature')['dominance'].mean().sort_values(
                ascending=False).index[:12])
        fig, ax = plt.subplots(figsize=(12, max(4, len(dom_pivot) * 0.6)))
        sns.heatmap(dom_pivot, cmap='YlGnBu', annot=True, fmt='.3f',
                    linewidths=.3, ax=ax)
        ax.set_title('Dominance Analysis (General Dominance) per Target', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mva_dominance_heatmap.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    # 图：模型 CV R² 对比
    df_cv = pd.DataFrame(model_cv_records)
    if not df_cv.empty:
        fig, ax = plt.subplots(figsize=(12, max(4, df_cv['Target'].nunique() * 0.5)))
        sns.barplot(data=df_cv, x='R2_mean', y='Target', hue='Model',
                    palette='deep', ax=ax)
        ax.set_title('Cross-Validation R² by Model per Target', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'mva_model_cv.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()

    # 图：2D PDP（每个 target 的 top1 SHAP 交互对）
    for t_name, f1, f2 in best_pdp_pairs:
        try:
            y_vec = pd.to_numeric(df_metrics[t_name], errors='coerce')
            xgb_pipe = Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('xgb', xgb.XGBRegressor(n_estimators=150, learning_rate=0.05,
                                          max_depth=4, subsample=0.9, random_state=RNG))
            ])
            xgb_pipe.fit(X, y_vec)
            fig, ax = plt.subplots(figsize=(7, 6))
            PartialDependenceDisplay.from_estimator(
                xgb_pipe, X, [(f1, f2)], kind='average', ax=ax,
                grid_resolution=25, contour_kw={'cmap': 'viridis'})
            ax.set_title(f'2D PDP: {f1} × {f2} -> {t_name}', fontsize=11)
            plt.tight_layout()
            safe = f"{t_name}_{f1}_x_{f2}".replace('/', '_')
            plt.savefig(os.path.join(output_dir, f'pdp2d_{safe}.png'),
                        dpi=200, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"[MVA]   2D PDP 绘制失败 ({t_name}, {f1}, {f2}): {e}")

    print(f"\n[MVA] 完成。全部输出写入 {output_dir}/")
    return {
        'representative_targets': representative_targets,
        'redundant_groups': df_groups,
        'beta': df_beta_pivot,
        'dominance': df_dom,
        'interactions': pd.DataFrame(inter_records),
        'top_features': pd.DataFrame(top_feat_records),
        'model_metrics': df_cv,
    }


if __name__ == '__main__':
    file_path = '252_MOF_总文件 冗余评估数据.xlsx'
    _, df_y, x_encoded = load_mof_dataset(file_path)
    res = run_mva_analysis(df_y, x_encoded)
