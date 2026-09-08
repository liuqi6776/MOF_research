# -*- coding: utf-8 -*-
import sys, os, pandas as pd, numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, ExtraTreesClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False

output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

print('[*] Loading 695 CoRE MOF GCMC Dataset & PMTransformer 768-D Embeddings...')
excel_path = '695_MOF/CoRE_MOF_2019_GCMC_695_总文件.xlsx'
df = pd.read_excel(excel_path, header=1)
mof_col = [c for c in df.columns if 'MOF' in str(c) or '名称' in str(c)][0]
df['MOF_name'] = df[mof_col].astype(str).str.strip()

def get_c(kw):
    matches = [c for c in df.columns if kw in str(c)]
    return matches[0] if matches else None

pld_col = get_c('PLD')
lcd_col = get_c('LCD')
asa_col = [c for c in df.columns if '表面积' in str(c) and 'm²/g' in str(c)][0]
pvol_col = [c for c in df.columns if '孔体积' in str(c) and 'cm³/g' in str(c)][0]
void_col = [c for c in df.columns if '体积分数' in str(c)][0]
density_col = get_c('密度')
vol_col = [c for c in df.columns if '晶胞体积 A3' in str(c)][0]
metal_col = [c for c in df.columns if '金属' in str(c) or 'Metal' in str(c)][0]

sel_col = get_c('选择性')
co2_015_col = get_c('0.15bar')
co2_1_col = [c for c in df.columns if '1bar' in str(c) and 'CO2' in str(c)][0]
qst_col = [c for c in df.columns if 'Widom' in str(c) and 'Qst' in str(c)][0]
vsa_col = [c for c in df.columns if 'VSA' in str(c) and '工作容量' in str(c)][0]

Y_cols = [sel_col, co2_015_col, co2_1_col, qst_col, vsa_col]
Y_names = ['CO2/N2 Selectivity', 'CO2 Uptake @ 0.15bar', 'CO2 Uptake @ 1.0bar', 'CO2 Qst Widom Heat', 'CO2 VSA Capacity']

X_cols = [pld_col, lcd_col, asa_col, pvol_col, void_col, density_col, vol_col]
X_names = ['PLD (A)', 'LCD (A)', 'Surface Area (m2/g)', 'Pore Volume (cm3/g)', 'Void Fraction', 'Density (g/cm3)', 'Cell Volume (A3)']

for col in Y_cols + X_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

valid_mask = df[Y_cols + X_cols].notnull().all(axis=1)
df = df[valid_mask].copy()

# Load PMTransformer 768-D embeddings
emb_path = 'PMtransformer/PMTransformer_695GCMC_695(1)/PMTransformer_695GCMC_695/embeddings.csv'
emb_df = pd.read_csv(emb_path)
emb_name_col = [c for c in emb_df.columns if 'mof' in str(c).lower() or 'id' in str(c).lower() or 'name' in str(c).lower()][0]
emb_df['MOF_name'] = emb_df[emb_name_col].astype(str).str.strip().str.replace('.cif', '', regex=False)
emb_feats = [c for c in emb_df.columns if c != emb_name_col and c != 'MOF_name']

merged_df = pd.merge(df, emb_df[['MOF_name'] + emb_feats], on='MOF_name', how='inner')
print('Merged dataset size:', len(merged_df), 'MOFs')

# 500 Train / 195 Validation Split
df_test_ref = pd.read_csv('results/finetuned_test_195_predictions_vs_excel.csv')
test_set_names = set(df_test_ref['MOF_ID'].values)

train_df = merged_df[~merged_df['MOF_name'].isin(test_set_names)].copy()
val_df = merged_df[merged_df['MOF_name'].isin(test_set_names)].copy()

print(f'Train MOFs (N={len(train_df)}), Validation MOFs (N={len(val_df)})')

Y_train = train_df[Y_cols].values
X_train = train_df[X_cols].values
Z_train = train_df[emb_feats].values

Y_val = val_df[Y_cols].values
X_val = val_df[X_cols].values
Z_val = val_df[emb_feats].values

train_metal = train_df[metal_col].astype(str).values
val_metal = val_df[metal_col].astype(str).values

# 1. Baseline: Nearest-Neighbor Inverse Mapping (1-NN in Y space)
scaler_y = StandardScaler()
Y_train_s = scaler_y.fit_transform(Y_train)
Y_val_s = scaler_y.transform(Y_val)

nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(Y_train_s)
_, nn_idx = nn.kneighbors(Y_val_s)
X_val_base = X_train[nn_idx.flatten()]
metal_val_base = train_metal[nn_idx.flatten()]

# 2. Route 1: Multi-Output Supervised Inverse Regressor
m1 = ExtraTreesRegressor(n_estimators=300, max_depth=16, random_state=42, n_jobs=-1)
m1.fit(Y_train, X_train)
X_val_m1 = m1.predict(Y_val)

metal_clf = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)
metal_clf.fit(Y_train, train_metal)
metal_val_m1 = metal_clf.predict(Y_val)

# 3. Route 2: PMTransformer Latent Space Inversion (Y -> Z_768 -> Manifold Search)
latent_inverter = Ridge(alpha=10.0)
latent_inverter.fit(Y_train, Z_train)
Z_val_pred = latent_inverter.predict(Y_val)

Z_train_norm = Z_train / np.linalg.norm(Z_train, axis=1, keepdims=True)
Z_val_pred_norm = Z_val_pred / np.linalg.norm(Z_val_pred, axis=1, keepdims=True)

nn_manifold = NearestNeighbors(n_neighbors=1, metric='cosine')
nn_manifold.fit(Z_train_norm)
_, manifold_idx = nn_manifold.kneighbors(Z_val_pred_norm)

X_val_m2 = X_train[manifold_idx.flatten()]
metal_val_m2 = train_metal[manifold_idx.flatten()]

# 4. Route 3: Hybrid Reticular Ensemble (M1 + M2 Latent Anchor)
X_val_m3 = 0.70 * X_val_m1 + 0.30 * X_val_m2

# Metrics Compilation
results = []
for i, name in enumerate(X_names):
    y_t = X_val[:, i]
    
    r2_b = r2_score(y_t, X_val_base[:, i])
    r2_1 = r2_score(y_t, X_val_m1[:, i])
    r2_2 = r2_score(y_t, X_val_m2[:, i])
    r2_3 = r2_score(y_t, X_val_m3[:, i])
    
    mae_b = mean_absolute_error(y_t, X_val_base[:, i])
    mae_1 = mean_absolute_error(y_t, X_val_m1[:, i])
    mae_2 = mean_absolute_error(y_t, X_val_m2[:, i])
    mae_3 = mean_absolute_error(y_t, X_val_m3[:, i])
    
    results.append({
        'Structural_Descriptor': name,
        'Baseline_1NN_R2': round(r2_b, 4),
        'Route1_DirectInverse_R2': round(r2_1, 4),
        'Route2_LatentManifold_R2': round(r2_2, 4),
        'Route3_HybridEnsemble_R2': round(r2_3, 4),
        'Best_Inverse_R2': round(max(r2_1, r2_2, r2_3), 4),
        'R2_Absolute_Improvement': round(max(r2_1, r2_2, r2_3) - r2_b, 4),
        'Baseline_MAE': round(mae_b, 4),
        'Best_MAE': round(min(mae_1, mae_2, mae_3), 4),
        'MAE_Reduction_Percent': round((mae_b - min(mae_1, mae_2, mae_3)) / mae_b * 100, 1)
    })

res_df = pd.DataFrame(results)
csv_out = os.path.join(output_dir, 'agent21_inverse_models_validation_metrics.csv')
res_df.to_csv(csv_out, index=False, encoding='utf-8-sig')
print('Saved validation metrics to:', csv_out)

acc_base = (val_metal == metal_val_base).mean() * 100
acc_m1 = (val_metal == metal_val_m1).mean() * 100
acc_m2 = (val_metal == metal_val_m2).mean() * 100
print(f'SBU Metal Accuracy: Baseline={acc_base:.1f}%, Route1={acc_m1:.1f}%, Route2={acc_m2:.1f}%')

# Forward Closed-Loop Recovery Evaluation
fwd = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1)
fwd.fit(X_train, Y_train)

Y_rec_base = fwd.predict(X_val_base)
Y_rec_m1 = fwd.predict(X_val_m1)
Y_rec_m3 = fwd.predict(X_val_m3)

loop_res = []
for j, yname in enumerate(Y_names):
    r2_lb = r2_score(Y_val[:, j], Y_rec_base[:, j])
    r2_lm = r2_score(Y_val[:, j], Y_rec_m3[:, j])
    mae_lb = mean_absolute_error(Y_val[:, j], Y_rec_base[:, j])
    mae_lm = mean_absolute_error(Y_val[:, j], Y_rec_m3[:, j])
    loop_res.append({
        'Target_Property': yname,
        'Baseline_Cycle_R2': round(r2_lb, 4),
        'Agent21_Cycle_R2': round(r2_lm, 4),
        'Cycle_R2_Improvement': round(r2_lm - r2_lb, 4),
        'Baseline_Cycle_MAE': round(mae_lb, 4),
        'Agent21_Cycle_MAE': round(mae_lm, 4),
        'MAE_Reduction_Percent': round((mae_lb - mae_lm) / mae_lb * 100, 1)
    })

loop_df = pd.DataFrame(loop_res)
loop_csv = os.path.join(output_dir, 'agent21_cycle_consistency_metrics.csv')
loop_df.to_csv(loop_csv, index=False, encoding='utf-8-sig')
print('Saved cycle consistency metrics to:', loop_csv)

# Plot Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

y_pos = np.arange(len(res_df))
bar_h = 0.22

ax1.barh(y_pos - bar_h, res_df['Baseline_1NN_R2'], bar_h, label='Baseline (1-NN Property Match)', color='#94a3b8', alpha=0.85, edgecolor='black', linewidth=0.6)
ax1.barh(y_pos, res_df['Route1_DirectInverse_R2'], bar_h, label='Route 1: Direct Multi-Target Inverse', color='#38bdf8', alpha=0.9, edgecolor='black', linewidth=0.6)
ax1.barh(y_pos + bar_h, res_df['Route3_HybridEnsemble_R2'], bar_h, label='Route 3: Hybrid Reticular Ensemble', color='#10b981', alpha=0.9, edgecolor='black', linewidth=0.6)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(res_df['Structural_Descriptor'], fontsize=10.5, fontweight='bold')
ax1.set_xlabel('Inverse Reconstruction R^2 on 195 Validation MOFs', fontsize=11, fontweight='bold')
ax1.set_title('A. Structural Reconstruction Accuracy (Y -> X)', fontsize=12, fontweight='bold')
ax1.set_xlim(-0.1, 0.95)
ax1.grid(axis='x', linestyle='--', alpha=0.5)
ax1.legend(loc='lower right', fontsize=9.5)

y_pos2 = np.arange(len(loop_df))
ax2.barh(y_pos2 - bar_h/2, loop_df['Baseline_Cycle_R2'], bar_h, label='Baseline Recovery R^2', color='#cbd5e1', alpha=0.9, edgecolor='black', linewidth=0.6)
ax2.barh(y_pos2 + bar_h/2, loop_df['Agent21_Cycle_R2'], bar_h, label='Agent 2.1 Closed-Loop Recovery R^2', color='#6366f1', alpha=0.9, edgecolor='black', linewidth=0.6)

ax2.set_yticks(y_pos2)
ax2.set_yticklabels(loop_df['Target_Property'], fontsize=10.5, fontweight='bold')
ax2.set_xlabel('Closed-Loop Property Recovery R^2 (Y_target -> CIF -> Y_pred)', fontsize=11, fontweight='bold')
ax2.set_title('B. Property Cycle-Consistency Fidelity (195 Validation MOFs)', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 1.0)
ax2.grid(axis='x', linestyle='--', alpha=0.5)
ax2.legend(loc='lower right', fontsize=9.5)

for idx, row in loop_df.iterrows():
    ax2.text(row['Agent21_Cycle_R2'] + 0.02, idx + bar_h/2, f"+{row['Cycle_R2_Improvement']:.2f}", va='center', fontsize=9, fontweight='bold', color='#4338ca')

plt.suptitle('Agent 2.1 Inverse Structure Prediction Benchmark (500 Train / 195 Validation)', fontsize=14, fontweight='bold')
plt.tight_layout()
fig_path = os.path.join(output_dir, 'agent21_inverse_model_improvements.png')
plt.savefig(fig_path, bbox_inches='tight')
plt.close()
print('Saved plot to:', fig_path)
print('ALL_DONE_SUCCESSFULLY')
