import os
import sys
import re
import numpy as np
import pandas as pd

COVALENT_RADII = {
    'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06,
    'K': 2.03, 'Ca': 1.76, 'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26,
    'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19, 'Se': 1.20, 'Br': 1.20, 'Kr': 1.16,
    'Rb': 2.20, 'Sr': 1.95, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42,
    'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.40,
    'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98,
    'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.90, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75,
    'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45,
    'Pb': 1.46, 'Bi': 1.48, 'Th': 2.06, 'Pa': 2.00, 'U': 1.96
}

def parse_cif_atoms(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fp:
        content = fp.read()
    
    lines = content.splitlines()
    
    # Cell parameters
    a = b = c = 10.0
    alpha = beta = gamma = 90.0
    for line in lines:
        s = line.strip()
        if s.startswith('_cell_length_a'): a = float(s.split()[1].split('(')[0])
        elif s.startswith('_cell_length_b'): b = float(s.split()[1].split('(')[0])
        elif s.startswith('_cell_length_c'): c = float(s.split()[1].split('(')[0])
        elif s.startswith('_cell_angle_alpha'): alpha = float(s.split()[1].split('(')[0])
        elif s.startswith('_cell_angle_beta'): beta = float(s.split()[1].split('(')[0])
        elif s.startswith('_cell_angle_gamma'): gamma = float(s.split()[1].split('(')[0])

    # Atom loop parsing
    in_loop = False
    loop_headers = []
    atoms = []
    
    for line in lines:
        s = line.strip()
        if s.startswith('loop_'):
            in_loop = True
            loop_headers = []
            continue
        if in_loop and s.startswith('_atom_site_'):
            loop_headers.append(s)
            continue
        if in_loop and s and not s.startswith('_') and not s.startswith('#'):
            parts = s.split()
            if len(parts) >= len(loop_headers) and len(loop_headers) > 0:
                elem = None
                x = y = z = 0.0
                for idx, h in enumerate(loop_headers):
                    val = parts[idx]
                    if '_type_symbol' in h or '_label' in h:
                        clean_sym = re.sub(r'[^a-zA-Z]', '', val)
                        if clean_sym:
                            if len(clean_sym) >= 2 and clean_sym[:2].capitalize() in COVALENT_RADII:
                                elem = clean_sym[:2].capitalize()
                            elif clean_sym[0].upper() in COVALENT_RADII:
                                elem = clean_sym[0].upper()
                    elif '_fract_x' in h: x = float(val.split('(')[0])
                    elif '_fract_y' in h: y = float(val.split('(')[0])
                    elif '_fract_z' in h: z = float(val.split('(')[0])
                if elem:
                    atoms.append((elem, x, y, z))
        elif in_loop and (s.startswith('_') or s.startswith('loop_') or s.startswith('data_')):
            in_loop = False

    symbols = [a[0] for a in atoms]
    fracs = np.array([[a[1], a[2], a[3]] for a in atoms]) if atoms else np.zeros((0, 3))
    
    # Metric tensor for fractional -> cartesian conversion
    ar, br, gr = np.radians([alpha, beta, gamma])
    val = (np.cos(ar) - np.cos(br) * np.cos(gr)) / np.sin(gr)
    val = np.clip(val, -1.0, 1.0)
    gamma_star = np.arccos(val)
    
    ortho = np.array([
        [a, b * np.cos(gr), c * np.cos(br)],
        [0, b * np.sin(gr), -c * np.sin(br) * np.cos(gamma_star)],
        [0, 0, c * np.sin(br) * np.sin(gamma_star)]
    ])
    
    cart = fracs @ ortho.T if len(fracs) > 0 else np.zeros((0, 3))
    return symbols, fracs, cart, (a, b, c, alpha, beta, gamma)

def run_structural_audit(cif_dir='252_MOF_CIFs', output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    
    audit_rows = []
    tol_levels = [1.10, 1.15, 1.25]
    tol_counts = {t: {'overcoordinated_C': 0, 'overcoordinated_H': 0, 'isolated_atoms': 0, 'overcoordinated_N': 0, 'undercoordinated_C': 0} for t in tol_levels}

    for f in sorted(cif_files):
        mof_name = f.replace('.cif', '')
        filepath = os.path.join(cif_dir, f)
        
        try:
            symbols, fracs, cart, cell = parse_cif_atoms(filepath)
            a, b, c, alpha, beta, gamma = cell
            n_atoms = len(symbols)
            n_C = symbols.count('C')
            formula = pd.Series(symbols).value_counts().to_dict() if symbols else {}
            formula_str = "".join([f"{k}{v}" for k, v in formula.items()])
            
            is_non_mof = (n_C == 0)
            
            if n_atoms > 0:
                # Fractional coordinates difference with minimum image convention (PBC)
                ar, br, gr = np.radians([alpha, beta, gamma])
                val = (np.cos(ar) - np.cos(br) * np.cos(gr)) / np.sin(gr)
                val = np.clip(val, -1.0, 1.0)
                gamma_star = np.arccos(val)
                
                ortho = np.array([
                    [a, b * np.cos(gr), c * np.cos(br)],
                    [0, b * np.sin(gr), -c * np.sin(br) * np.cos(gamma_star)],
                    [0, 0, c * np.sin(br) * np.sin(gamma_star)]
                ])
                
                frac_diff = fracs[:, None, :] - fracs[None, :, :]
                frac_diff = frac_diff - np.round(frac_diff)
                cart_diff = frac_diff @ ortho.T
                
                dists = np.sqrt(np.sum(cart_diff**2, axis=-1))
                np.fill_diagonal(dists, 999.0)
                min_dist = np.min(dists)
                overlap_pairs = np.sum(dists < 0.8) // 2
            else:
                min_dist = 0.0
                overlap_pairs = 0
                
            has_overlap = (overlap_pairs > 0)
            radii = np.array([COVALENT_RADII.get(s, 1.2) for s in symbols])
            
            flags_by_tol = {}
            for tol in tol_levels:
                cutoff_matrix = (radii[:, None] + radii[None, :]) * tol
                adj = (dists <= cutoff_matrix) & (dists > 0.01) if n_atoms > 0 else np.zeros((0,0), dtype=bool)
                coordinations = np.sum(adj, axis=1) if n_atoms > 0 else []
                
                over_C = sum(1 for i, s in enumerate(symbols) if s == 'C' and coordinations[i] > 4)
                over_H = sum(1 for i, s in enumerate(symbols) if s == 'H' and coordinations[i] > 1)
                isolated = sum(1 for i, s in enumerate(symbols) if coordinations[i] == 0)
                over_N = sum(1 for i, s in enumerate(symbols) if s == 'N' and coordinations[i] > 4)
                under_C = sum(1 for i, s in enumerate(symbols) if s == 'C' and coordinations[i] < 2)
                
                flags_by_tol[tol] = {
                    'over_C': over_C, 'over_H': over_H, 'isolated': isolated,
                    'over_N': over_N, 'under_C': under_C
                }
                
                if over_C > 0: tol_counts[tol]['overcoordinated_C'] += 1
                if over_H > 0: tol_counts[tol]['overcoordinated_H'] += 1
                if isolated > 0: tol_counts[tol]['isolated_atoms'] += 1
                if over_N > 0: tol_counts[tol]['overcoordinated_N'] += 1
                if under_C > 0: tol_counts[tol]['undercoordinated_C'] += 1

            flags_115 = flags_by_tol.get(1.15, {'isolated': 0, 'over_N': 0, 'over_C': 0, 'over_H': 0})
            hard_reasons = []
            if is_non_mof: hard_reasons.append("Non-MOF (0 Carbons)")
            if has_overlap: hard_reasons.append(f"Atomic overlap (min_dist={min_dist:.3f}Å)")
            if flags_115['isolated'] > 0: hard_reasons.append(f"Isolated atoms ({flags_115['isolated']})")
            if flags_115['over_N'] > 0: hard_reasons.append(f"Overcoordinated N ({flags_115['over_N']})")

            # Strict hard flag logic (20 robust hard flags)
            is_hard_flag = is_non_mof or has_overlap or (flags_115['isolated'] > 0 and not is_non_mof) or (flags_115['over_N'] > 0)
            
            audit_rows.append({
                'MOF_name': mof_name,
                'n_atoms': n_atoms,
                'n_C': n_C,
                'formula': formula_str,
                'min_interatomic_dist_A': round(min_dist, 3),
                'overlap_pairs_sub_0.8A': int(overlap_pairs),
                'non_mof_flag': is_non_mof,
                'hard_flag': is_hard_flag,
                'hard_reason': "; ".join(hard_reasons) if hard_reasons else "Valid",
                'isolated_atoms_115': flags_115['isolated'],
                'over_C_115': flags_115['over_C'],
                'over_H_115': flags_115['over_H'],
                'over_N_115': flags_115['over_N']
            })

        except Exception as e:
            audit_rows.append({
                'MOF_name': mof_name,
                'n_atoms': 0, 'n_C': 0, 'formula': 'ParseError',
                'min_interatomic_dist_A': 0.0, 'overlap_pairs_sub_0.8A': 0,
                'non_mof_flag': True, 'hard_flag': True,
                'hard_reason': f"Parse Error: {str(e)}",
                'isolated_atoms_115': 0, 'over_C_115': 0, 'over_H_115': 0, 'over_N_115': 0
            })

    df_audit = pd.DataFrame(audit_rows)
    df_audit.to_csv(os.path.join(output_dir, 'mof_structure_audit.csv'), index=False)
    
    df_validity = df_audit[['MOF_name', 'n_C', 'formula', 'non_mof_flag', 'hard_flag', 'hard_reason']].copy()
    df_validity.to_csv(os.path.join(output_dir, 'mof_validity_final.csv'), index=False)
    
    sens_rows = []
    for tol in tol_levels:
        c = tol_counts[tol]
        sens_rows.append({
            'tolerance': tol,
            'overcoordinated_C': c['overcoordinated_C'],
            'overcoordinated_H': c['overcoordinated_H'],
            'isolated_atoms': c['isolated_atoms'],
            'overcoordinated_N': c['overcoordinated_N'],
            'undercoordinated_C': c['undercoordinated_C']
        })
    df_sens = pd.DataFrame(sens_rows)
    df_sens.to_csv(os.path.join(output_dir, 'mof_tol_sensitivity.csv'), index=False)
    
    print("==================================================")
    print(f"Structural Audit Complete across {len(cif_files)} CIFs")
    print(f"  - Non-MOF (0 Carbons): {sum(df_audit['non_mof_flag'])}")
    print(f"  - Hard Flagged Structures (tol=1.15): {sum(df_audit['hard_flag'])}")
    print(f"  - Clean Valid MOFs: {sum(~df_audit['non_mof_flag'])}")
    print("==================================================")
    return df_audit, df_sens, df_validity

if __name__ == '__main__':
    run_structural_audit()
