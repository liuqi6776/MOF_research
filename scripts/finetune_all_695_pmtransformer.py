# -*- coding: utf-8 -*-
import os, sys, io
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

emb_path = r'C:/Users/liuqi/MOF_research/PMtransformer/PMTransformer_695GCMC_695(1)/PMTransformer_695GCMC_695/embeddings.csv'
excel_path = r'C:/Users/liuqi/MOF_research/695_MOF/CoRE_MOF_2019_GCMC_695_总文件.xlsx'
os.makedirs('results/models', exist_ok=True)

print('[*] Loading PMTransformer CIF embeddings (695 x 768)...')
emb_df = pd.read_csv(emb_path)
df = pd.read_excel(excel_path, header=1)
mof_col = [c for c in df.columns if 'MOF' in str(c) or '名称' in str(c)][0]
df['MOF_ID'] = df[mof_col].astype(str).str.strip()

merged = pd.merge(emb_df, df, on='MOF_ID', how='inner')
print(f'[*] Total valid MOFs merged: {len(merged)}')

emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
X_emb = merged[emb_cols].values

# 提取物理化学描述符
num_cols = [c for c in df.columns if any(k in str(c) for k in ['LCD', 'PLD', 'LFPD', '孔体积', '表面积', '体积分数', '密度', '原子', '最短M-M', 'OMS', '占比', '质量分数', '晶胞'])]
X_phys = merged[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
X_fused = np.hstack([X_emb, X_phys])

print(f'[*] Representation matrices prepared:')
print(f'    - Pure Transformer Embedding: {X_emb.shape}')
print(f'    - Fused Multi-Modal Matrix:   {X_fused.shape} (768 Trans + {X_phys.shape[1]} Phys/Chem)')

targets = [
    # 结构描述符 X
    (15, 'LCD (A) Largest Cavity Diameter', 'Structure X'),
    (16, 'PLD (A) Pore Limiting Diameter', 'Structure X'),
    (18, 'Pore Volume (cm3/g)', 'Structure X'),
    (20, 'Gravimetric Surface Area (m2/g)', 'Structure X'),
    (23, 'Accessible Void Fraction', 'Structure X'),
    # 吸附与热力学性质 y
    (63, 'CO2 Uptake @0.001bar (mol/kg)', 'Low-P Uptake y'),
    (66, 'CO2 Uptake @0.01bar (mol/kg)', 'Low-P Uptake y'),
    (72, 'CO2 Uptake @0.15bar FlueGas (mol/kg)', 'FlueGas Uptake y'),
    (57, 'CO2 Uptake @1.0bar 1atm (mol/kg)', '1atm Uptake y'),
    (143, 'CO2 Qst Widom Heat (kJ/mol)', 'Heat Qst y'),
    (144, 'CO2 VSA Working Capacity (mol/kg)', 'Separation Cap y'),
    (146, 'CO2/N2 Actual Selectivity', 'Selectivity y'),
    (134, 'N2 Uptake @1.0bar (mol/kg)', 'N2 Uptake y')
]

cv = KFold(n_splits=5, shuffle=True, random_state=42)
all_metrics = []
pred_dict = {'MOF_ID': merged['MOF_ID'].values}
trained_models = {}

print('\n' + '='*130)
print('{:<16} | {:<36} | {:<8} | {:<8} | {:<10} | {:<10} | {:<10}'.format('Category', 'Target Property', 'Raw R2', 'Finetuned R2', 'MAE', 'Pearson r', 'Spearman rho'))
print('='*130)

for col_idx, name, t_type in targets:
    col_name = merged.columns[769 + col_idx]
    y_raw = pd.to_numeric(merged[col_name], errors='coerce').values
    valid_mask = ~np.isnan(y_raw)
    
    # 1. 原始未经物理微调的基线 (纯 Transformer Embedding ExtraTrees)
    X_sub_emb = X_emb[valid_mask]
    y_sub = y_raw[valid_mask]
    
    m_base = ExtraTreesRegressor(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1)
    y_pred_base = cross_val_predict(m_base, X_sub_emb, y_sub, cv=cv, n_jobs=-1)
    r2_raw = r2_score(y_sub, y_pred_base)
    
    # 2. 全量微调的多模态模型 (Fused Multi-Modal ExtraTrees)
    if t_type == 'Structure X':
        X_sub_ft = X_sub_emb
        X_full_ft = X_emb
    else:
        X_sub_ft = X_fused[valid_mask]
        X_full_ft = X_fused
        
    m_ft = ExtraTreesRegressor(n_estimators=220, max_depth=18, min_samples_split=2, random_state=42, n_jobs=-1)
    y_pred_ft = cross_val_predict(m_ft, X_sub_ft, y_sub, cv=cv, n_jobs=-1)
    
    r2_ft = r2_score(y_sub, y_pred_ft)
    mae_ft = mean_absolute_error(y_sub, y_pred_ft)
    rmse_ft = np.sqrt(mean_squared_error(y_sub, y_pred_ft))
    pr_ft, _ = pearsonr(y_sub, y_pred_ft)
    sr_ft, _ = spearmanr(y_sub, y_pred_ft)
    
    # 全量拟合并保存生产模型
    m_ft.fit(X_sub_ft, y_sub)
    trained_models[name] = m_ft
    
    all_metrics.append({
        'Category': t_type,
        'Target': name,
        'Raw_ZeroShot_R2': round(r2_raw, 4),
        'Finetuned_All695_R2': round(r2_ft, 4),
        'R2_Improvement': round(r2_ft - r2_raw, 4),
        'MAE': round(mae_ft, 4),
        'RMSE': round(rmse_ft, 4),
        'Pearson_r': round(pr_ft, 4),
        'Spearman_rho': round(sr_ft, 4)
    })
    
    pred_dict['True_' + name] = y_raw
    full_pred = np.full(len(y_raw), np.nan)
    full_pred[valid_mask] = y_pred_ft
    pred_dict['Finetuned_Pred_' + name] = full_pred
    
    print('{:<16} | {:<36} | {:8.4f} | {:12.4f} | {:10.4f} | {:10.4f} | {:10.4f}'.format(t_type, name, r2_raw, r2_ft, mae_ft, pr_ft, sr_ft))

print('='*130)

# 保存模型包与预测文件
bundle_path = 'results/models/mof_finetuned_pmtransformer_bundle.joblib'
joblib.dump({'models': trained_models, 'num_cols': num_cols, 'metrics': all_metrics}, bundle_path)

df_metrics = pd.DataFrame(all_metrics)
df_metrics.to_csv('results/pmtransformer_finetuned_all695_metrics.csv', index=False, encoding='utf-8-sig')

df_preds = pd.DataFrame(pred_dict)
df_preds.to_csv('results/pmtransformer_finetuned_all695_predictions_vs_excel.csv', index=False, encoding='utf-8-sig')

print(f'\n[DONE] Full 695 MOF fine-tuning complete!')
print(f'  - Saved Production Model Bundle: {bundle_path}')
print(f'  - Saved Metrics: results/pmtransformer_finetuned_all695_metrics.csv')
print(f'  - Saved Full Predictions: results/pmtransformer_finetuned_all695_predictions_vs_excel.csv')



