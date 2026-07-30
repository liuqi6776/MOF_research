import pandas as pd
import numpy as np
import re
import os
import sys

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def load_mof_dataset(file_path):
    """
    Loads and structures the 252 MOF dataset from Excel.
    Ensures strict separation between X (structural descriptors) and Y (performance metrics).
    """
    df_raw = pd.read_excel(file_path, header=[0, 1])
    
    # Filter out header/note rows (keep only valid numeric 序号 1..252)
    valid_idx = pd.to_numeric(df_raw.iloc[:, 0], errors='coerce').notnull()
    df_raw = df_raw[valid_idx].reset_index(drop=True)
    
    # 1. Extract MOF metadata & identification
    mof_names = df_raw.iloc[:, 1].values # MOF名称
    mof_ids = df_raw.iloc[:, 0].values   # 序号
    
    # 2. Extract Y performance metrics (19 columns)
    y_cat = 'MOF性能指标'
    y_cols = df_raw[y_cat].columns.tolist()
    df_y = df_raw[y_cat].copy()
    
    # Rename Y columns for clarity
    y_rename_map = {
        'CO₂吸附@0.15bar(mol/kg)\n越高越好': 'CO2_ads_0.15bar',
        'CO2_Qst_CC均值(kJ/mol)\n越高越好': 'CO2_Qst_CC_mean',
        'CO2_Qst_CC@0.1bar(kJ/mol)\n越高越好': 'CO2_Qst_CC_0.1bar',
        'CO2_Qst_CC@0.5bar(kJ/mol)\n越高越好': 'CO2_Qst_CC_0.5bar',
        'CO2_Qst_CC@1bar(kJ/mol)\n越高越好': 'CO2_Qst_CC_1bar',
        'CO2_Qst_GCMC烟气段均值(kJ/mol)\n越高越好': 'CO2_Qst_GCMC_flue_mean',
        'CO2_Qst_Widom零覆盖(kJ/mol)\n越高越好': 'CO2_Qst_Widom',
        'CO2_VSA工作容量(mol/kg)\n越高越好': 'CO2_VSA_capacity',
        'CO2_TSA工作容量(mol/kg)\n越高越好': 'CO2_TSA_capacity',
        'CO2N2实际选择性\n越高越好': 'CO2N2_actual_selectivity',
        'CO2N2_Henry选择性\n越高越好': 'CO2N2_Henry_selectivity',
        'PE_VSA寄生能(kJ/mol CO2)\n越低越好': 'PE_VSA_parasitic_energy',
        'CO2_TSA双积分法计算再生热(kJ/mol)\n越低越好': 'CO2_TSA_regen_heat',
        'N2_Qst_Widom零覆盖(kJ/mol)\n越低越好': 'N2_Qst_Widom',
        'N2_Qst_GCMC烟气段均值(kJ/mol)\n越低越好': 'N2_Qst_GCMC_flue_mean',
        'N2_Qst_@1bar(kJ/mol)\n越低越好': 'N2_Qst_1bar',
        'N2吸附@0.75bar(mol/kg)\n越低越好': 'N2_ads_0.75bar',
        'N2吸附@1bar(mol/kg)\n越低越好': 'N2_ads_1bar',
        'Qst差(CO2-N2)(kJ/mol)\n越高越好': 'Qst_diff_CO2_N2'
    }
    
    df_y = df_y.rename(columns=y_rename_map)
    df_y.insert(0, 'MOF_name', mof_names)
    df_y.insert(0, 'MOF_id', mof_ids)
    
    # 3. Extract X feature categories (Strictly NO raw GCMC data)
    # Categories: MOF组成信息, 孔道几何参数, 孔体积与表面积, 金属与活性位, 晶体信息
    x_features = pd.DataFrame({'MOF_id': mof_ids, 'MOF_name': mof_names})
    
    # --- 3a. Pore Geometry ---
    if '孔道几何参数' in df_raw.columns.levels[0]:
        geom = df_raw['孔道几何参数']
        for col in geom.columns:
            clean_name = col.split('\n')[0].strip()
            if 'LCD' in clean_name:
                x_features['LCD_A'] = pd.to_numeric(geom[col], errors='coerce')
            elif 'PLD' in clean_name:
                x_features['PLD_A'] = pd.to_numeric(geom[col], errors='coerce')
            elif 'LFPD' in clean_name:
                x_features['LFPD_A'] = pd.to_numeric(geom[col], errors='coerce')
    
    # Derived geometry ratio
    x_features['LCD_PLD_ratio'] = x_features['LCD_A'] / (x_features['PLD_A'] + 1e-6)
    
    # --- 3b. Pore Volume & Surface Area ---
    if '孔体积与表面积' in df_raw.columns.levels[0]:
        pv_sa = df_raw['孔体积与表面积']
        for col in pv_sa.columns:
            clean_name = col.replace('\n', ' ').strip()
            if '孔体积' in clean_name and 'cm³/g' in clean_name and '可访问' not in clean_name and '不可访问' not in clean_name:
                x_features['pore_vol_cm3_g'] = pd.to_numeric(pv_sa[col], errors='coerce')
            elif '不可访问表面积' in clean_name and 'm²/g' in clean_name:
                x_features['NASA_m2_g'] = pd.to_numeric(pv_sa[col], errors='coerce')
            elif '不可访问表面积' in clean_name and 'm²/cm³' in clean_name:
                x_features['NASA_m2_cm3'] = pd.to_numeric(pv_sa[col], errors='coerce')
            elif '可访问表面积' in clean_name and 'm²/cm³' in clean_name:
                x_features['ASA_m2_cm3'] = pd.to_numeric(pv_sa[col], errors='coerce')
            elif '可访问表面积' in clean_name and 'm²/g' in clean_name:
                x_features['ASA_m2_g'] = pd.to_numeric(pv_sa[col], errors='coerce')
            elif '可访问孔' in clean_name and '体积分数' in clean_name:
                x_features['void_fraction'] = pd.to_numeric(pv_sa[col], errors='coerce')
            elif '可访问孔体积' in clean_name and 'cm³/g' in clean_name:
                x_features['accessible_pore_vol'] = pd.to_numeric(pv_sa[col], errors='coerce')
    
    # --- 3c. Metal & Active Sites ---
    if '金属与活性位' in df_raw.columns.levels[0]:
        metal_cat = df_raw['金属与活性位']
        if '主要金属元素' in metal_cat.columns:
            x_features['primary_metal'] = metal_cat['主要金属元素'].fillna('Other').astype(str)
        elif '金属元素' in metal_cat.columns:
            x_features['primary_metal'] = metal_cat['金属元素'].fillna('Other').astype(str)
        
        if '含OMS?' in metal_cat.columns:
            oms_val = metal_cat['含OMS?']
            x_features['has_oms'] = oms_val.apply(lambda v: 1 if str(v).strip().lower() in ['true', '1', 'yes', '是'] else 0)
    
    # --- 3d. Crystal Information ---
    if '晶体信息' in df_raw.columns.levels[0]:
        cryst = df_raw['晶体信息']
        for col in cryst.columns:
            if '晶体密度' in col:
                x_features['density_g_cm3'] = pd.to_numeric(cryst[col], errors='coerce')
            elif '晶胞体积' in col and 'A3' in col:
                x_features['uc_volume_A3'] = pd.to_numeric(cryst[col], errors='coerce')
            elif 'C原子的质量分数' in col:
                x_features['C_mass_frac'] = pd.to_numeric(cryst[col], errors='coerce')
            elif 'N原子的质量分数' in col:
                x_features['N_mass_frac'] = pd.to_numeric(cryst[col], errors='coerce')
            elif 'O原子的质量分数' in col:
                x_features['O_mass_frac'] = pd.to_numeric(cryst[col], errors='coerce')
    
    # --- 3e. MOF Composition & SMILES Descriptors ---
    if 'MOF组成信息' in df_raw.columns.levels[0]:
        comp = df_raw['MOF组成信息']
        if '拓扑代码 (Topology Code)' in comp.columns:
            x_features['topology'] = comp['拓扑代码 (Topology Code)'].fillna('UNKNOWN').astype(str).str.strip().str.lower()
        if '穿插度 (Catenation)' in comp.columns:
            x_features['catenation'] = pd.to_numeric(comp['穿插度 (Catenation)'], errors='coerce').fillna(0)
        if 'O配位占比' in comp.columns:
            x_features['O_coord_ratio'] = pd.to_numeric(comp['O配位占比'], errors='coerce')
        if '最短M-M距离(Å)' in comp.columns:
            x_features['shortest_MM_dist'] = pd.to_numeric(comp['最短M-M距离(Å)'], errors='coerce')
        
        # SMILES extraction
        smiles_col = [c for c in comp.columns if 'SMILES' in c][0]
        smiles_series = comp[smiles_col].fillna('').astype(str)
        
        smiles_feats = extract_smiles_features(smiles_series)
        x_features = pd.concat([x_features, smiles_feats], axis=1)

    # Clean and encode primary metal and topology categories
    top_metals = ['Cu', 'Zn', 'Ni', 'Co', 'Mn', 'Fe', 'Zr']
    x_features['primary_metal_grouped'] = x_features['primary_metal'].apply(lambda m: m if m in top_metals else 'Other')
    
    # Frequency encoding for topology
    top_counts = x_features['topology'].value_counts()
    x_features['topology_grouped'] = x_features['topology'].apply(lambda t: t if top_counts.get(t, 0) >= 4 and t != 'unknown' else 'Other')
    
    # One-hot encoding for primary metal and topology
    metal_dummies = pd.get_dummies(x_features['primary_metal_grouped'], prefix='metal')
    topo_dummies = pd.get_dummies(x_features['topology_grouped'], prefix='topo')
    
    x_encoded = pd.concat([x_features, metal_dummies, topo_dummies], axis=1)
    
    # Filter out 8 non-MOF zero-carbon structures (inorganic phosphates / POMs)
    non_mof_names = ['ABIXOZ_clean', 'ABULOB_clean', 'ACUBAB_clean', 'AGUBUA_clean', 
                     'AJOTEY_clean', 'ARUYUH01_clean', 'ARUYUH_clean', 'ATOGEV_clean']
    valid_mof_mask = ~x_encoded['MOF_name'].isin(non_mof_names)
    
    df_raw = df_raw[valid_mof_mask].reset_index(drop=True)
    df_y = df_y[valid_mof_mask].reset_index(drop=True)
    x_encoded = x_encoded[valid_mof_mask].reset_index(drop=True)
    
    return df_raw, df_y, x_encoded

def extract_smiles_features(smiles_series):
    """
    Extracts chemical descriptors from organic ligand SMILES strings.
    """
    records = []
    for s in smiles_series:
        feat = {
            'ligand_MW': np.nan,
            'ligand_LogP': np.nan,
            'ligand_n_heavy': np.nan,
            'ligand_n_aromatic_rings': 0,
            'ligand_n_carboxylate': 0,
            'ligand_n_N_hetero': 0,
            'ligand_n_halogen': 0,
            'ligand_n_amino': 0,
            'ligand_TPSA': np.nan
        }
        
        s_clean = s.strip()
        if not s_clean or s_clean == 'nan':
            records.append(feat)
            continue
            
        # Regex based counts (works on all SMILES strings)
        feat['ligand_n_carboxylate'] = len(re.findall(r'C\(=O\)\[O-\]|C\(=O\)O', s_clean))
        feat['ligand_n_halogen'] = len(re.findall(r'F|Cl|Br|I', s_clean))
        feat['ligand_n_amino'] = len(re.findall(r'N(?![a-z])', s_clean))
        feat['ligand_n_N_hetero'] = len(re.findall(r'n', s_clean))
        
        # RDKit parsing
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(s_clean)
            if mol is None:
                mol = Chem.MolFromSmiles(s_clean, sanitize=False)
            if mol is not None:
                try:
                    feat['ligand_MW'] = Descriptors.MolWt(mol)
                    feat['ligand_LogP'] = Descriptors.MolLogP(mol)
                    feat['ligand_n_heavy'] = mol.GetNumHeavyAtoms()
                    feat['ligand_n_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
                    feat['ligand_TPSA'] = Descriptors.TPSA(mol)
                except Exception:
                    pass
        records.append(feat)
        
    return pd.DataFrame(records)

if __name__ == '__main__':
    file_path = '252_MOF_总文件 冗余评估数据.xlsx'
    df_raw, df_y, x_features = load_mof_dataset(file_path)
    print("Y shape:", df_y.shape)
    print("X shape:", x_features.shape)
    print("Y columns:", df_y.columns.tolist()[:5])
    print("X columns sample:", x_features.columns.tolist()[:15])
