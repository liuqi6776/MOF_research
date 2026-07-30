import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

try:
    from scripts.data_loader import load_mof_dataset
    from scripts.indicator_system import run_indicator_system_analysis
except ImportError:
    from data_loader import load_mof_dataset
    from indicator_system import run_indicator_system_analysis

def topsis_score(matrix, weights, directions):
    """
    Computes TOPSIS scores for a feature matrix.
    matrix: np.ndarray (n_samples, n_criteria)
    weights: np.ndarray (n_criteria,)
    directions: list of 'max' or 'min'
    """
    # 1. Vector normalization
    norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    
    # 2. Weighted normalized matrix
    weighted_matrix = norm_matrix * weights
    
    # 3. Ideal best and ideal worst
    ideal_best = np.zeros(matrix.shape[1])
    ideal_worst = np.zeros(matrix.shape[1])
    
    for i, d in enumerate(directions):
        if d == 'max':
            ideal_best[i] = np.max(weighted_matrix[:, i])
            ideal_worst[i] = np.min(weighted_matrix[:, i])
        else:
            ideal_best[i] = np.min(weighted_matrix[:, i])
            ideal_worst[i] = np.max(weighted_matrix[:, i])
            
    # 4. Euclidean distance to ideal best and worst
    dist_best = np.sqrt(((weighted_matrix - ideal_best) ** 2).sum(axis=1))
    dist_worst = np.sqrt(((weighted_matrix - ideal_worst) ** 2).sum(axis=1))
    
    # 5. Relative closeness to ideal solution
    scores = dist_worst / (dist_best + dist_worst + 1e-12)
    return scores

def run_dual_route_ranking(df_y, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    ind_res = run_indicator_system_analysis(df_y, output_dir)
    df_trans = ind_res['df_transformed']
    
    # --- VSA Route Ranking ---
    vsa_cols = ['CO2_VSA_capacity', 'log10_CO2N2_actual_selectivity', 'log10_PE_VSA_parasitic_energy', 'N2_ads_0.75bar']
    vsa_weights = np.array([0.35, 0.30, 0.25, 0.10])
    vsa_dirs = ['max', 'max', 'min', 'min']
    
    X_vsa = df_trans[vsa_cols].values
    vsa_scores = topsis_score(X_vsa, vsa_weights, vsa_dirs)
    
    # --- TSA Route Ranking ---
    tsa_cols = ['CO2_TSA_capacity', 'log10_CO2N2_actual_selectivity', 'log10_CO2_TSA_regen_heat', 'N2_ads_0.75bar']
    tsa_weights = np.array([0.35, 0.30, 0.25, 0.10])
    tsa_dirs = ['max', 'max', 'min', 'min']
    
    X_tsa = df_trans[tsa_cols].values
    tsa_scores = topsis_score(X_tsa, tsa_weights, tsa_dirs)
    
    # Build Ranking DataFrames
    df_rank = pd.DataFrame({
        'MOF_id': df_y['MOF_id'],
        'MOF_name': df_y['MOF_name'],
        'VSA_Score': vsa_scores * 100,
        'TSA_Score': tsa_scores * 100,
        'CO2_VSA_capacity': df_y['CO2_VSA_capacity'],
        'CO2_TSA_capacity': df_y['CO2_TSA_capacity'],
        'CO2N2_actual_selectivity': df_y['CO2N2_actual_selectivity'],
        'PE_VSA_parasitic_energy': df_y['PE_VSA_parasitic_energy'],
        'CO2_TSA_regen_heat': df_y['CO2_TSA_regen_heat'],
        'N2_ads_0.75bar': df_y['N2_ads_0.75bar']
    })
    
    df_rank['VSA_Rank'] = df_rank['VSA_Score'].rank(ascending=False, method='min').astype(int)
    df_rank['TSA_Rank'] = df_rank['TSA_Score'].rank(ascending=False, method='min').astype(int)
    
    df_vsa_sorted = df_rank.sort_values(by='VSA_Rank').reset_index(drop=True)
    df_tsa_sorted = df_rank.sort_values(by='TSA_Rank').reset_index(drop=True)
    
    df_vsa_sorted.to_csv(os.path.join(output_dir, 'vsa_rankings.csv'), index=False)
    df_tsa_sorted.to_csv(os.path.join(output_dir, 'tsa_rankings.csv'), index=False)
    
    # Identify Win-Win MOFs (Top 20 in both)
    top20_vsa = set(df_vsa_sorted.head(20)['MOF_name'])
    top20_tsa = set(df_tsa_sorted.head(20)['MOF_name'])
    win_win_mofs = top20_vsa.intersection(top20_tsa)
    vsa_only_mofs = top20_vsa - top20_tsa
    tsa_only_mofs = top20_tsa - top20_vsa
    
    # --- Sensitivity Analysis ---
    # Perturb weights by +/- 20% across 1000 Monte Carlo simulations
    np.random.seed(42)
    vsa_top20_overlaps = []
    tsa_top20_overlaps = []
    
    for _ in range(1000):
        # Random perturbation
        vsa_w_rand = vsa_weights * (1 + np.random.uniform(-0.2, 0.2, size=len(vsa_weights)))
        vsa_w_rand /= vsa_w_rand.sum()
        vsa_s_rand = topsis_score(X_vsa, vsa_w_rand, vsa_dirs)
        top20_rand_vsa = set(df_rank.iloc[np.argsort(-vsa_s_rand)[:20]]['MOF_name'])
        
        vsa_jaccard = len(top20_vsa.intersection(top20_rand_vsa)) / len(top20_vsa.union(top20_rand_vsa))
        vsa_top20_overlaps.append(vsa_jaccard)
        
        tsa_w_rand = tsa_weights * (1 + np.random.uniform(-0.2, 0.2, size=len(tsa_weights)))
        tsa_w_rand /= tsa_w_rand.sum()
        tsa_s_rand = topsis_score(X_tsa, tsa_w_rand, tsa_dirs)
        top20_rand_tsa = set(df_rank.iloc[np.argsort(-tsa_s_rand)[:20]]['MOF_name'])
        
        tsa_jaccard = len(top20_tsa.intersection(top20_rand_tsa)) / len(top20_tsa.union(top20_rand_tsa))
        tsa_top20_overlaps.append(tsa_jaccard)
        
    vsa_sensitivity_jaccard = np.mean(vsa_top20_overlaps)
    tsa_sensitivity_jaccard = np.mean(tsa_top20_overlaps)
    
    # --- Plot VSA vs TSA Comparison Scatter ---
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_rank, 
        x='VSA_Score', 
        y='TSA_Score', 
        hue='CO2N2_actual_selectivity', 
        size='CO2_VSA_capacity',
        sizes=(30, 200),
        palette='viridis', 
        alpha=0.85
    )
    
    # Annotate Top 5 Win-Win MOFs
    for idx, row in df_rank.iterrows():
        if row['MOF_name'] in win_win_mofs and (row['VSA_Rank'] <= 10 or row['TSA_Rank'] <= 10):
            plt.text(row['VSA_Score'] + 0.5, row['TSA_Score'] + 0.5, row['MOF_name'], fontsize=9, weight='bold')
            
    plt.axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5, label='Equal Performance Line')
    plt.title('VSA vs TSA Route Performance Ranking Comparison (n=252 MOFs)', fontsize=14, pad=15)
    plt.xlabel('VSA Route Comprehensive TOPSIS Score (0-100)', fontsize=12)
    plt.ylabel('TSA Route Comprehensive TOPSIS Score (0-100)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vsa_tsa_ranking_comparison.png'), dpi=300)
    plt.close()
    
    return {
        'df_rank': df_rank,
        'win_win_mofs': sorted(list(win_win_mofs)),
        'vsa_only_mofs': sorted(list(vsa_only_mofs)),
        'tsa_only_mofs': sorted(list(tsa_only_mofs)),
        'vsa_sensitivity_jaccard': vsa_sensitivity_jaccard,
        'tsa_sensitivity_jaccard': tsa_sensitivity_jaccard
    }

if __name__ == '__main__':
    file_path = '252_MOF_总文件 冗余评估数据.xlsx'
    _, df_y, _ = load_mof_dataset(file_path)
    res = run_dual_route_ranking(df_y)
    print("=== Win-Win MOFs (Top 20 in both VSA & TSA) ===")
    print(res['win_win_mofs'])
    print(f"\nVSA Top-20 Sensitivity Jaccard Overlap: {res['vsa_sensitivity_jaccard']:.4f}")
    print(f"TSA Top-20 Sensitivity Jaccard Overlap: {res['tsa_sensitivity_jaccard']:.4f}")
