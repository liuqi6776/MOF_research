import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

try:
    from scripts.data_loader import load_mof_dataset
except ImportError:
    from data_loader import load_mof_dataset

# Set matplotlib parameters for publication quality figures
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_indicator_system_analysis(df_y, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract numerical Y columns
    metric_cols = [c for c in df_y.columns if c not in ['MOF_id', 'MOF_name']]
    df_metrics = df_y[metric_cols].copy()
    
    # 1. Calculate Correlation Matrices (Spearman & Pearson)
    spearman_corr = df_metrics.corr(method='spearman')
    pearson_corr = df_metrics.corr(method='pearson')
    
    spearman_corr.to_csv(os.path.join(output_dir, 'spearman_corr_matrix.csv'))
    pearson_corr.to_csv(os.path.join(output_dir, 'pearson_corr_matrix.csv'))
    
    # 2. Plot Correlation Heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
    sns.heatmap(
        spearman_corr, 
        mask=mask, 
        cmap='vlag', 
        vmin=-1, vmax=1, 
        center=0, 
        annot=True, 
        fmt='.2f', 
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .8}
    )
    plt.title('Spearman Correlation Matrix of 19 MOF Candidate Performance Metrics ($Y$)', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()
    
    # 3. Verify Specific Known Redundancy Relationships
    redundancy_results = {}
    
    # Redundancy 1: CO2 Qst variants (CC vs Widom)
    r_cc_widom = spearman_corr.loc['CO2_Qst_CC_mean', 'CO2_Qst_Widom']
    redundancy_results['CO2_Qst_CC_vs_Widom'] = r_cc_widom
    
    # Redundancy 2: CO2 capacity triplet
    r_ads_vsa = spearman_corr.loc['CO2_ads_0.15bar', 'CO2_VSA_capacity']
    r_ads_tsa = spearman_corr.loc['CO2_ads_0.15bar', 'CO2_TSA_capacity']
    redundancy_results['CO2_ads_vs_VSA_cap'] = r_ads_vsa
    redundancy_results['CO2_ads_vs_TSA_cap'] = r_ads_tsa
    
    # Redundancy 3: PE_VSA vs Actual Selectivity (log-log)
    log_pe = np.log10(df_metrics['PE_VSA_parasitic_energy'].clip(lower=1e-3))
    log_sel = np.log10(df_metrics['CO2N2_actual_selectivity'].clip(lower=1e-3))
    log_log_r = log_pe.corr(log_sel, method='pearson')
    redundancy_results['PE_VSA_vs_Actual_Selectivity_log_log_r'] = log_log_r
    
    # Redundancy 4: TSA Heat vs TSA Capacity
    r_tsa_heat_cap = spearman_corr.loc['CO2_TSA_regen_heat', 'CO2_TSA_capacity']
    redundancy_results['TSA_heat_vs_TSA_capacity_r'] = r_tsa_heat_cap
    
    # 4. Heavy-tail Preprocessing Diagnostics & Transformation
    df_transformed = df_metrics.copy()
    
    # Heavy-tailed / Log-scaling columns: Henry Selectivity, Actual Selectivity, PE_VSA, TSA Regeneration Heat
    heavy_tail_cols = ['CO2N2_Henry_selectivity', 'CO2N2_actual_selectivity', 'PE_VSA_parasitic_energy', 'CO2_TSA_regen_heat']
    skewness_before = {}
    skewness_after = {}
    
    for col in heavy_tail_cols:
        skewness_before[col] = df_metrics[col].skew()
        log_col = f"log10_{col}"
        df_transformed[log_col] = np.log10(df_metrics[col].clip(lower=1e-4))
        skewness_after[log_col] = df_transformed[log_col].skew()
        
    # 5. Core Metric Selection & Weighting Scheme Definition
    # Non-redundant Core Dimension Grouping:
    # Dimension 1: CO2 Capacity (Weight: 0.35) -> VSA Capacity (VSA) / TSA Capacity (TSA)
    # Dimension 2: CO2/N2 Selectivity & Affinity (Weight: 0.30) -> log10(CO2N2_actual_selectivity)
    # Dimension 3: Regeneration Energy (Weight: 0.25) -> log10(PE_VSA) for VSA / log10(CO2_TSA_regen_heat) for TSA
    # Dimension 4: N2 Exclusion (Weight: 0.10) -> N2_ads_0.75bar
    
    scheme = {
        'VSA_route': {
            'CO2_VSA_capacity': {'direction': 'max', 'weight': 0.35, 'transform': 'none'},
            'CO2N2_actual_selectivity': {'direction': 'max', 'weight': 0.30, 'transform': 'log10'},
            'PE_VSA_parasitic_energy': {'direction': 'min', 'weight': 0.25, 'transform': 'log10'},
            'N2_ads_0.75bar': {'direction': 'min', 'weight': 0.10, 'transform': 'none'}
        },
        'TSA_route': {
            'CO2_TSA_capacity': {'direction': 'max', 'weight': 0.35, 'transform': 'none'},
            'CO2N2_actual_selectivity': {'direction': 'max', 'weight': 0.30, 'transform': 'log10'},
            'CO2_TSA_regen_heat': {'direction': 'min', 'weight': 0.25, 'transform': 'log10'},
            'N2_ads_0.75bar': {'direction': 'min', 'weight': 0.10, 'transform': 'none'}
        }
    }
    
    return {
        'spearman_corr': spearman_corr,
        'pearson_corr': pearson_corr,
        'redundancy_results': redundancy_results,
        'skewness_before': skewness_before,
        'skewness_after': skewness_after,
        'df_transformed': df_transformed,
        'scheme': scheme
    }

if __name__ == '__main__':
    file_path = '252_MOF_总文件 冗余评估数据.xlsx'
    _, df_y, _ = load_mof_dataset(file_path)
    res = run_indicator_system_analysis(df_y)
    print("=== Redundancy Verification ===")
    for k, v in res['redundancy_results'].items():
        print(f"  {k}: {v:.4f}")
    print("\n=== Heavy-Tail Skewness Diagnostic ===")
    for col in res['skewness_before']:
        print(f"  {col}: skew before = {res['skewness_before'][col]:.2f} -> log10 skew = {res['skewness_after']['log10_'+col]:.2f}")
