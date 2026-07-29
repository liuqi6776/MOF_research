import pandas as pd
import numpy as np
import os
import sys

from data_loader import load_mof_dataset
from dual_route_ranking import run_dual_route_ranking

def generate_design_rules_and_recommendations(df_raw, df_y, x_encoded, rank_res=None, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Obtain Dual Route Rankings if not passed
    if rank_res is None:
        rank_res = run_dual_route_ranking(df_y, output_dir)
    df_rank = rank_res['df_rank']
    
    # Merge ranking with X features and raw composition
    df_merged = df_rank.merge(x_encoded, on=['MOF_id', 'MOF_name'])
    
    # Extract top 20 VSA and top 20 TSA MOFs
    top20_vsa = df_merged.sort_values(by='VSA_Rank').head(20)
    top20_tsa = df_merged.sort_values(by='TSA_Rank').head(20)
    
    # Combine top performers
    top_mofs = pd.concat([top20_vsa, top20_tsa]).drop_duplicates(subset=['MOF_id'])
    bottom_mofs = df_merged.sort_values(by='VSA_Rank').tail(50)
    
    # Compute quantitative rule boundaries (median/percentiles of top vs bottom)
    rules = [
        {
            'Parameter': 'Pore Limiting Diameter (PLD, Å)',
            'Optimal_Interval': f"{top_mofs['PLD_A'].quantile(0.10):.2f} - {top_mofs['PLD_A'].quantile(0.90):.2f} Å",
            'Top_Median': f"{top_mofs['PLD_A'].median():.2f} Å",
            'Bottom_Median': f"{bottom_mofs['PLD_A'].median():.2f} Å",
            'Evidence_Strength': 'Strong (Molecular sieving threshold > 3.3 Å)',
            'Rationale': 'PLD > 3.3 Å allows CO2 entry while < 5.5 Å restricts N2 kinetic co-adsorption.'
        },
        {
            'Parameter': 'Largest Cavity Diameter (LCD, Å)',
            'Optimal_Interval': f"{top_mofs['LCD_A'].quantile(0.10):.2f} - {top_mofs['LCD_A'].quantile(0.90):.2f} Å",
            'Top_Median': f"{top_mofs['LCD_A'].median():.2f} Å",
            'Bottom_Median': f"{bottom_mofs['LCD_A'].median():.2f} Å",
            'Evidence_Strength': 'Moderate',
            'Rationale': 'LCD < 9.5 Å prevents excessive empty volume that weakens fluid-wall electrostatic potential.'
        },
        {
            'Parameter': 'Accessible Surface Area (ASA, m²/g)',
            'Optimal_Interval': f"{top_mofs['ASA_m2_g'].quantile(0.10):.0f} - {top_mofs['ASA_m2_g'].quantile(0.90):.0f} m²/g",
            'Top_Median': f"{top_mofs['ASA_m2_g'].median():.0f} m²/g",
            'Bottom_Median': f"{bottom_mofs['ASA_m2_g'].median():.0f} m²/g",
            'Evidence_Strength': 'Strong',
            'Rationale': 'High gravimetric surface area provides dense CO2 adsorption sites.'
        },
        {
            'Parameter': 'Volumetric ASA (ASA_vol, m²/cm³)',
            'Optimal_Interval': f"{top_mofs['ASA_m2_cm3'].quantile(0.10):.0f} - {top_mofs['ASA_m2_cm3'].quantile(0.90):.0f} m²/cm³",
            'Top_Median': f"{top_mofs['ASA_m2_cm3'].median():.0f} m²/cm³",
            'Bottom_Median': f"{bottom_mofs['ASA_m2_cm3'].median():.0f} m²/cm³",
            'Evidence_Strength': 'Strong',
            'Rationale': 'High volumetric surface area enhances packing density in adsorption beds.'
        },
        {
            'Parameter': 'Crystal Density (g/cm³)',
            'Optimal_Interval': f"{top_mofs['density_g_cm3'].quantile(0.10):.2f} - {top_mofs['density_g_cm3'].quantile(0.90):.2f} g/cm³",
            'Top_Median': f"{top_mofs['density_g_cm3'].median():.2f} g/cm³",
            'Bottom_Median': f"{bottom_mofs['density_g_cm3'].median():.2f} g/cm³",
            'Evidence_Strength': 'Moderate',
            'Rationale': 'Density around 0.9 - 1.3 g/cm³ balances void fraction and volumetric capacity.'
        },
        {
            'Parameter': 'Open Metal Sites (OMS Presence)',
            'Optimal_Interval': 'has_oms = 1 (Present)',
            'Top_Median': f"OMS Ratio: {top_mofs['has_oms'].mean()*100:.1f}%",
            'Bottom_Median': f"OMS Ratio: {bottom_mofs['has_oms'].mean()*100:.1f}%",
            'Evidence_Strength': 'Strong',
            'Rationale': 'Unsaturated metal centers create strong localized electrostatic fields, boosting Qst and low-pressure uptake.'
        },
        {
            'Parameter': 'Primary Metal Node',
            'Optimal_Interval': 'Zn (Tetranuclear/M4O), Cd, Co, Cu (Paddlewheel)',
            'Top_Median': f"{', '.join([f'{m} ({p:.1f}%)' for m, p in top_mofs['primary_metal'].value_counts(normalize=True).head(3).items()])}",
            'Bottom_Median': f"{', '.join([f'{m} ({p:.1f}%)' for m, p in bottom_mofs['primary_metal'].value_counts(normalize=True).head(3).items()])}",
            'Evidence_Strength': 'Strong',
            'Rationale': 'Zn, Cd and Co nodes provide ideal coordination geometry and pore polarization.'
        }
    ]
    
    df_rules = pd.DataFrame(rules)
    df_rules.to_csv(os.path.join(output_dir, 'design_rules_checklist.csv'), index=False)
    
    # --- 2. Select 4 Specific MOF Recommendations ---
    win_win_names = rank_res['win_win_mofs']
    top_candidates = df_merged[df_merged['MOF_name'].isin(win_win_names)].sort_values(by='VSA_Score', ascending=False)
    
    selected_mofs = top_candidates.head(4).copy()
    
    recommendations = []
    for idx, row in selected_mofs.iterrows():
        # Get original raw composition row
        raw_row = df_raw[df_raw.iloc[:, 1] == row['MOF_name']].iloc[0]
        
        inorg_bb = str(raw_row[('MOF组成信息', '无机建筑块 (Inorganic BB)')])
        org_smiles = str(raw_row[('MOF组成信息', '有机建筑块 (Organic BB) SMILES格式')])
        topology = str(raw_row[('MOF组成信息', '拓扑代码 (Topology Code)')])
        
        rec = {
            'MOF_name': row['MOF_name'],
            'Inorganic_SBU': inorg_bb if inorg_bb != 'nan' else row['primary_metal'] + ' SBU',
            'Organic_Ligand_SMILES': org_smiles,
            'Topology': topology if topology != 'nan' else row['topology'],
            'VSA_Score': f"{row['VSA_Score']:.1f}",
            'TSA_Score': f"{row['TSA_Score']:.1f}",
            'CO2_ads_0.15bar': f"{row['CO2_VSA_capacity']:.2f} mol/kg",
            'Selectivity': f"{row['CO2N2_actual_selectivity']:.1f}",
            'PE_VSA': f"{row['PE_VSA_parasitic_energy']:.1f} kJ/mol",
            'CO2_TSA_regen_heat': f"{row['CO2_TSA_regen_heat']:.1f} kJ/mol",
            'Key_Rules_Satisfied': f"PLD={row['PLD_A']:.2f}Å, ASA={row['ASA_m2_g']:.0f}m²/g, OMS={row['has_oms']}, Metal={row['primary_metal']}",
            'Extrapolation_Limits': 'Dry flue gas GCMC model; OMS electrostatic interactions may be over-predicted in force fields.'
        }
        recommendations.append(rec)
        
    df_recs = pd.DataFrame(recommendations)
    df_recs.to_csv(os.path.join(output_dir, 'mof_structure_recommendations.csv'), index=False)
    
    return df_rules, df_recs

if __name__ == '__main__':
    file_path = '252_MOF_总文件 冗余评估数据.xlsx'
    df_raw, df_y, x_encoded = load_mof_dataset(file_path)
    df_rules, df_recs = generate_design_rules_and_recommendations(df_raw, df_y, x_encoded)
    print("=== Quantitative Design Rules Checklist ===")
    print(df_rules[['Parameter', 'Optimal_Interval', 'Evidence_Strength']])
    print("\n=== Recommended Top MOF Structural Schemes ===")
    print(df_recs[['MOF_name', 'Inorganic_SBU', 'Topology', 'VSA_Score', 'TSA_Score']])
