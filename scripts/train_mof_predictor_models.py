"""
MOF Multi-Target Property Predictor Trainer (Fused Multi-Modal Architecture)
训练基于 3D 结构嵌入向量 + 物理描述符的多目标性能预测机器学习模型
"""
import sys
import io
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def train_and_save_mof_models(
    excel_path: str = "252_MOF_总文件 冗余评估数据.xlsx",
    struct_emb_path: str = "results/mof_structural_embeddings.npy",
    index_csv_path: str = "results/mof_embedding_index.csv",
    output_model_path: str = "results/models/mof_property_predictors.joblib",
    n_pca_components: int = 16
):
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    print("[*] Loading data and building Fused Multi-Modal Representation Matrix...")
    
    df = pd.read_excel(excel_path, header=1)
    idx_df = pd.read_csv(index_csv_path)
    X_emb_all = np.load(struct_emb_path)
    
    df['mof_name'] = df['MOF名称'].astype(str).str.strip()
    merged = pd.merge(idx_df, df, on='mof_name', how='inner')
    indices = merged['index'].values
    X_emb = X_emb_all[indices]
    
    trad_cols = [
        'LCD (Å)\n最大空腔直径', 'PLD (Å)\n孔道限制直径', 'LFPD (Å)\n最大自由路径',
        '孔体积\n(cm³/g)', '可访问表面积\n(m²/g)', '可访问表面积\n(m²/cm³)',
        '不可访问表面积\n(m²/g)', '可访问孔\n体积分数', '可访问孔体积\n(cm³/g)'
    ]
    X_trad = merged[trad_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    
    pca_scaler = StandardScaler()
    X_emb_scaled = pca_scaler.fit_transform(X_emb)
    pca = PCA(n_components=n_pca_components, random_state=42)
    X_emb_pca = pca.fit_transform(X_emb_scaled)
    
    X_fused = np.hstack([X_trad, X_emb_pca])
    print(f"  [OK] Fused matrix created: {X_fused.shape} (9d geometric + {n_pca_components}d PCA structure components)")
    
    targets = {
        'co2_015bar': {
            'label': 'CO2 Uptake @ 0.15 bar (mol/kg)',
            'data': pd.to_numeric(merged['q_CO2_298K_0.15bar_mol_kg'], errors='coerce').fillna(0).values,
            'log_transform': False
        },
        'co2_1bar': {
            'label': 'CO2 Uptake @ 1.0 bar (mol/kg)',
            'data': pd.to_numeric(merged['q_ads_CO2_1bar_298K_mol_kg'], errors='coerce').fillna(0).values,
            'log_transform': False
        },
        'selectivity_real': {
            'label': 'CO2/N2 Actual Selectivity (0.15/0.85)',
            'data': np.log10(np.clip(pd.to_numeric(merged['CO2N2实际选择性\n越高越好'], errors='coerce').fillna(1.0).values, 1.0, 1000.0)),
            'log_transform': True
        },
        'qst_widom': {
            'label': 'CO2 Adsorption Heat Qst (kJ/mol)',
            'data': pd.to_numeric(merged['CO2_Qst_Widom零覆盖(kJ/mol)\n越高越好'], errors='coerce').fillna(0).values,
            'log_transform': False
        },
        'vsa_working_capacity': {
            'label': 'VSA Working Capacity (mol/kg)',
            'data': pd.to_numeric(merged['CO2_VSA工作容量(mol/kg)\n越高越好'], errors='coerce').fillna(0).values,
            'log_transform': False
        },
        'pe_vsa': {
            'label': 'VSA Parasitic Energy (kJ/mol CO2)',
            'data': np.log10(np.clip(pd.to_numeric(merged['PE_VSA寄生能(kJ/mol CO2)\n越低越好'], errors='coerce').fillna(100.0).values, 1.0, 500.0)),
            'log_transform': True
        }
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    trained_models = {}
    metrics_summary = []
    
    print("\n[*] Starting 5-Fold Cross Validation & Training Production Predictors:")
    for key, target_info in targets.items():
        y = target_info['data']
        valid_mask = ~np.isnan(y) & (y != 0)
        X_train = X_fused[valid_mask]
        y_train = y[valid_mask]
        
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', ExtraTreesRegressor(n_estimators=120, max_depth=12, random_state=42))
        ])
        
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=['r2', 'neg_mean_absolute_error'])
        r2_m = float(np.mean(cv_scores['test_r2']))
        mae_m = float(-np.mean(cv_scores['test_neg_mean_absolute_error']))
        
        pipe.fit(X_train, y_train)
        trained_models[key] = {
            'pipeline': pipe,
            'label': target_info['label'],
            'log_transform': target_info['log_transform'],
            'r2_cv': r2_m,
            'mae_cv': mae_m
        }
        
        metrics_summary.append({
            'Target_Key': key,
            'Target_Label': target_info['label'],
            'R2_5Fold_CV': round(r2_m, 4),
            'MAE_5Fold_CV': round(mae_m, 4)
        })
        print(f"  [OK] {target_info['label']:<40} -> 5-Fold R2 = {r2_m:.4f}, MAE = {mae_m:.4f}")
        
    bundle = {
        'models': trained_models,
        'pca_scaler': pca_scaler,
        'pca': pca,
        'trad_cols': trad_cols,
        'n_pca_components': n_pca_components,
        'metrics_summary': pd.DataFrame(metrics_summary)
    }
    
    joblib.dump(bundle, output_model_path)
    bundle['metrics_summary'].to_csv("results/qsar_fused_model_metrics.csv", index=False)
    print(f"\n[DONE] All models trained and saved to: {output_model_path}")
    return bundle

if __name__ == "__main__":
    train_and_save_mof_models()
