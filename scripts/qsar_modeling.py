import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

from data_loader import load_mof_dataset
from indicator_system import run_indicator_system_analysis

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_qsar_modeling(df_y, x_encoded, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    ind_res = run_indicator_system_analysis(df_y, output_dir)
    df_trans = ind_res['df_transformed']
    
    # 1. Define X feature matrix (numeric & encoded, strictly NO GCMC raw data)
    non_x_cols = ['MOF_id', 'MOF_name', 'primary_metal', 'topology', 'primary_metal_grouped', 'topology_grouped']
    x_feature_cols = [c for c in x_encoded.columns if c not in non_x_cols]
    
    X = x_encoded[x_feature_cols].copy()
    
    # 2. Targets to model
    targets = {
        'CO2_VSA_capacity': df_trans['CO2_VSA_capacity'],
        'CO2_TSA_capacity': df_trans['CO2_TSA_capacity'],
        'log10_CO2N2_actual_selectivity': df_trans['log10_CO2N2_actual_selectivity'],
        'log10_PE_VSA_parasitic_energy': df_trans['log10_PE_VSA_parasitic_energy'],
        'log10_CO2_TSA_regen_heat': df_trans['log10_CO2_TSA_regen_heat']
    }
    
    cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
        'Ridge': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=10.0))
        ])
    }
    
    eval_results = []
    importance_dict = {}
    
    for t_name, y_vec in targets.items():
        importance_dict[t_name] = {}
        for m_name, model in models.items():
            # Build imputer + model pipeline for trees
            if m_name != 'Ridge':
                pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('model', model)
                ])
            else:
                pipe = model
                
            scores = cross_validate(
                pipe, X, y_vec, 
                cv=cv, 
                scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error']
            )
            
            r2_mean, r2_std = np.mean(scores['test_r2']), np.std(scores['test_r2'])
            mae_mean = -np.mean(scores['test_neg_mean_absolute_error'])
            rmse_mean = -np.mean(scores['test_neg_root_mean_squared_error'])
            
            eval_results.append({
                'Target': t_name,
                'Model': m_name,
                'R2_mean': r2_mean,
                'R2_std': r2_std,
                'MAE_mean': mae_mean,
                'RMSE_mean': rmse_mean
            })
            
            # Fit on full data for feature importance
            pipe.fit(X, y_vec)
            if m_name == 'RandomForest':
                rf_model = pipe.named_steps['model']
                importances = rf_model.feature_importances_
                importance_dict[t_name]['RandomForest'] = pd.Series(importances, index=x_feature_cols)
                
    df_eval = pd.DataFrame(eval_results)
    df_eval.to_csv(os.path.join(output_dir, 'qsar_model_metrics.csv'), index=False)
    
    # --- Feature Importance Visualization ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    main_targets = ['CO2_VSA_capacity', 'log10_CO2N2_actual_selectivity', 'log10_PE_VSA_parasitic_energy', 'log10_CO2_TSA_regen_heat']
    
    for ax, t_name in zip(axes.flatten(), main_targets):
        imp = importance_dict[t_name]['RandomForest'].sort_values(ascending=False).head(10)
        
        # Determine direction via correlation with feature
        corrs = [X[feat].corr(targets[t_name]) for feat in imp.index]
        colors = ['#2ca02c' if c > 0 else '#d62728' for c in corrs]
        
        sns.barplot(x=imp.values, y=imp.index, ax=ax, palette=colors)
        ax.set_title(f'Top 10 Features for {t_name}\n(Green = +, Red = -)', fontsize=12)
        ax.set_xlabel('Random Forest Feature Importance', fontsize=10)
        
    plt.suptitle('Structure-Property Relationship Feature Importance Mapping', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_shap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Partial Dependence Plots (PDP) ---
    rf_cap_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    rf_cap_pipe.fit(X, targets['CO2_VSA_capacity'])
    
    pdp_features = ['PLD_A', 'LCD_A', 'ASA_m2_g', 'density_g_cm3']
    pdp_features = [f for f in pdp_features if f in X.columns]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(rf_cap_pipe, X, pdp_features, ax=ax, line_kw={"color": "navy", "linewidth": 2})
    plt.suptitle('Partial Dependence Plots (PDP) for CO2 VSA Capacity', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pdp_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return df_eval, importance_dict

if __name__ == '__main__':
    file_path = '252_MOF_总文件 冗余评估数据.xlsx'
    _, df_y, x_encoded = load_mof_dataset(file_path)
    df_eval, _ = run_qsar_modeling(df_y, x_encoded)
    print("=== QSAR Model CV Evaluation Summary ===")
    print(df_eval.groupby(['Target', 'Model'])[['R2_mean', 'MAE_mean']].mean())
