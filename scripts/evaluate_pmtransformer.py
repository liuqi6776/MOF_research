# -*- coding: utf-8 -*-
import os, sys, io
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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
os.makedirs('results', exist_ok=True)

print('[*] Loading PMTransformer CIF embeddings (695 x 768)...')
emb_df = pd.read_csv(emb_path)
df = pd.read_excel(excel_path, header=1)
mof_col = [c for c in df.columns if 'MOF' in str(c) or '名称' in str(c)][0]
df['MOF_ID'] = df[mof_col].astype(str).str.strip()

merged = pd.merge(emb_df, df, on='MOF_ID', how='inner')
print(f'[*] Merged valid MOF records: {len(merged)}')

emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
X = merged[emb_cols].values

# Helper to find column by keyword
def find_col(kw):
    for c in merged.columns:
        if kw in str(c):
            return c
    return None

targets = [
    ('LCD', 'LCD 最大空腔直径 (A)', '结构特征 X'),
    ('PLD', 'PLD 孔道限制直径 (A)', '结构特征 X'),
    ('孔体积
(cm', '孔体积 (cm3/g)', '结构特征 X'),
    ('可访问表面积
(m', '可访问表面积 (m2/g)', '结构特征 X'),
    ('体积分数', '可访问孔体积分数', '结构特征 X'),
    ('0.001bar', 'CO2吸附量 @0.001bar (mol/kg)', '低压吸附 y'),
    ('0.01bar_mol_kg', 'CO2吸附量 @0.01bar (mol/kg)', '低压吸附 y'),
    ('0.15bar_mol_kg', 'CO2吸附量 @0.15bar 烟气段 (mol/kg)', '烟气吸附 y'),
    ('q_ads_CO2_1bar', 'CO2吸附量 @1.0bar 常压 (mol/kg)', '常压吸附 y'),
    ('Widom零覆盖', 'CO2吸附热 Qst (kJ/mol)', '热力学性质 y'),
    ('VSA工作容量', 'CO2 VSA工作容量 (mol/kg)', '分离性能 y'),
    ('实际选择性', 'CO2/N2 实际选择性', '分离选择性 y')
]

cv = KFold(n_splits=5, shuffle=True, random_state=42)
metrics_list = []
pred_dict = {'MOF_ID': merged['MOF_ID'].values}

print('\n' + '='*105)
print('{:<12} | {:<34} | {:<8} | {:<10} | {:<10} | {:<10} | {:<10}'.format('类别', '目标属性', 'R2', 'MAE', 'RMSE', 'Pearson_r', 'Spearman_rho'))
print('='*105)

for kw, name, t_type in targets:
    col = find_col(kw)
    if col is None:
        print(f'[!] Warning: Column not found for kw {kw}')
        continue
    y_raw = pd.to_numeric(merged[col], errors='coerce').values
    valid_mask = ~np.isnan(y_raw)
    
    X_sub = X[valid_mask]
    y_sub = y_raw[valid_mask]
    
    model = ExtraTreesRegressor(n_estimators=120, max_depth=16, random_state=42, n_jobs=-1)
    y_pred = cross_val_predict(model, X_sub, y_sub, cv=cv, n_jobs=-1)
    
    r2 = r2_score(y_sub, y_pred)
    mae = mean_absolute_error(y_sub, y_pred)
    rmse = np.sqrt(mean_squared_error(y_sub, y_pred))
    pr, _ = pearsonr(y_sub, y_pred)
    sr, _ = spearmanr(y_sub, y_pred)
    
    metrics_list.append({
        'Category': t_type,
        'Target': name,
        'R2': round(r2, 4),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'Pearson_r': round(pr, 4),
        'Spearman_rho': round(sr, 4)
    })
    
    pred_dict['True_' + name] = y_raw
    full_pred = np.full(len(y_raw), np.nan)
    full_pred[valid_mask] = y_pred
    pred_dict['Pred_' + name] = full_pred
    
    print('{:<12} | {:<34} | {:8.4f} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f}'.format(t_type, name, r2, mae, rmse, pr, sr))

print('='*105)

df_metrics = pd.DataFrame(metrics_list)
df_metrics.to_csv('results/pmtransformer_prediction_metrics.csv', index=False, encoding='utf-8-sig')

df_preds = pd.DataFrame(pred_dict)
df_preds.to_csv('results/pmtransformer_predictions_vs_excel.csv', index=False, encoding='utf-8-sig')

print('\n[DONE] 验证评估全部完成!')
print('  - 指标汇总表: results/pmtransformer_prediction_metrics.csv')
print('  - 对照表: results/pmtransformer_predictions_vs_excel.csv')

print('\n[*] 前 6 个 MOF 的预测值与 Excel 真实值样本比对 (CO2 吸附量 @ 0.15 bar):')
for i in range(6):
    m = merged['MOF_ID'].iloc[i]
    t_v = pred_dict['True_CO2吸附量 @0.15bar 烟气段 (mol/kg)'][i]
    p_v = pred_dict['Pred_CO2吸附量 @0.15bar 烟气段 (mol/kg)'][i]
    print('  MOF: {:<18} | Excel 真实值: {:7.4f} | Transformer 预测: {:7.4f} | 误差: {:7.4f}'.format(m, t_v, p_v, abs(t_v - p_v)))
