import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import statsmodels.api as sm

# Set style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # support Chinese fonts
plt.rcParams['axes.unicode_minus'] = False  # support negative signs

# Define Directories
ARTIFACT_DIR = r"C:\Users\liuqi\.gemini\antigravity\brain\18ad86be-91e4-41dc-b713-c62e539f1151"
WORKSPACE_RESULTS_DIR = r"c:\Users\liuqi\MOF_research\results"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(WORKSPACE_RESULTS_DIR, exist_ok=True)

# Helper function to save files to both directories
def save_output_file(src_filename, content_str=None, is_csv=False, df_to_save=None, is_plot=False):
    # If it is a plot, it is saved directly using plt.savefig
    if is_plot:
        plt.savefig(os.path.join(ARTIFACT_DIR, src_filename), dpi=150)
        plt.savefig(os.path.join(WORKSPACE_RESULTS_DIR, src_filename), dpi=150)
        print(f"Saved plot to both directories: {src_filename}")
        return

    # If it's a dataframe saved to CSV
    if is_csv and df_to_save is not None:
        df_to_save.to_csv(os.path.join(ARTIFACT_DIR, src_filename))
        df_to_save.to_csv(os.path.join(WORKSPACE_RESULTS_DIR, src_filename))
        print(f"Saved CSV to both directories: {src_filename}")
        return

    # If it's a text file
    if content_str is not None:
        for d in [ARTIFACT_DIR, WORKSPACE_RESULTS_DIR]:
            with open(os.path.join(d, src_filename), "w", encoding="utf-8") as f:
                f.write(content_str)
        print(f"Saved text to both directories: {src_filename}")

# 1. Ingest Data
data_path = "AI co2_screening.xlsx"
df = pd.read_excel(data_path)

# Slice relevant columns (first 35 columns: 0 to 34)
df = df.iloc[:, 0:35]

# Clean up column names (rename the last one to avoid encoding issues)
col_names = list(df.columns)
topsis_col_name = col_names[34]
df = df.rename(columns={topsis_col_name: 'TOPSIS_score'})

# Identify column groups
feature_cols = col_names[1:29]
property_cols = col_names[29:34]
target_col = 'TOPSIS_score'

# Preprocessing: drop zero-variance features
zero_var_cols = []
for col in feature_cols:
    if df[col].dtype != 'object' and df[col].nunique() <= 1:
        zero_var_cols.append(col)
active_features = [c for c in feature_cols if c not in zero_var_cols]

# Convert has_oms to float
if 'has_oms' in active_features:
    df['has_oms'] = df['has_oms'].astype(float)

# Separate continuous and categorical features
numeric_features = []
categorical_features = []
for col in active_features:
    if df[col].dtype in ['float64', 'int64']:
        numeric_features.append(col)
    else:
        categorical_features.append(col)

# Compute and save summary statistics
summary_features = df[numeric_features].describe().T
summary_features['skewness'] = df[numeric_features].skew()
save_output_file("features_summary.csv", is_csv=True, df_to_save=summary_features)

summary_properties = df[property_cols + [target_col]].describe().T
summary_properties['skewness'] = df[property_cols + [target_col]].skew()
save_output_file("properties_summary.csv", is_csv=True, df_to_save=summary_properties)

# Plot 1: EDA Distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, col in enumerate(property_cols + [target_col]):
    axes[i].hist(df[col].dropna(), bins=10, color='teal', alpha=0.7, edgecolor='black')
    axes[i].set_title(f"{col} Distribution")
fig.delaxes(axes[5])
plt.tight_layout()
save_output_file("eda_distributions.png", is_plot=True)
plt.close()

# 2. VIF Multicollinearity Analysis
def calculate_vif(df_features):
    vifs = {}
    for col in df_features.columns:
        X = df_features.drop(columns=[col])
        y = df_features[col]
        model = LinearRegression().fit(X, y)
        r_sq = model.score(X, y)
        if r_sq == 1.0:
            vif = float('inf')
        else:
            vif = 1.0 / (1.0 - r_sq)
        vifs[col] = vif
    return pd.Series(vifs)

physical_desc = ['lcd', 'pld', 'pld_path', 'asa_m2_g', 'asa_m2_cm3', 'void_fraction', 'pore_vol_cm3_g', 'density_g_cm3', 'uc_volume_A3']
physical_desc_present = [c for c in physical_desc if c in numeric_features]

vif_initial = calculate_vif(df[physical_desc_present])
save_output_file("vif_initial.csv", is_csv=True, df_to_save=vif_initial)

# Stepwise drop
selected_physical = physical_desc_present.copy()
while True:
    vifs = calculate_vif(df[selected_physical])
    max_vif_col = vifs.idxmax()
    max_vif_val = vifs.max()
    if max_vif_val > 10.0:
        selected_physical.remove(max_vif_col)
    else:
        break
vif_final = calculate_vif(df[selected_physical])
save_output_file("vif_final.csv", is_csv=True, df_to_save=vif_final)

# 3. Correlation Matrices
spearman_corr = pd.DataFrame(index=numeric_features, columns=property_cols + [target_col])
spearman_pvals = pd.DataFrame(index=numeric_features, columns=property_cols + [target_col])
for f in numeric_features:
    for t in property_cols + [target_col]:
        r, p = stats.spearmanr(df[f], df[t])
        spearman_corr.loc[f, t] = r
        spearman_pvals.loc[f, t] = p
save_output_file("spearman_corr.csv", is_csv=True, df_to_save=spearman_corr)
save_output_file("spearman_pvals.csv", is_csv=True, df_to_save=spearman_pvals)

plot_corr = spearman_corr.copy().astype(float)
plot_pvals = spearman_pvals.copy().astype(float)
mask = plot_pvals >= 0.05
plt.figure(figsize=(12, 10))
sns.heatmap(plot_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, mask=mask, fmt=".2f", linewidths=.5)
plt.title("Spearman Correlation Heatmap (Significant relationships only, p < 0.05)")
plt.tight_layout()
save_output_file("correlation_heatmap.png", is_plot=True)
plt.close()

# 4. Generate the 5 subplots in one figure (Individual Properties Trends)
# Based on Spearman correlations, the top features for each property are:
# Henry_CO2: pld (r = -0.611)
# Qst_CO2: asa_m2_g (r = -0.668)
# Selectivity: pld (r = -0.750)
# Qreg_TSA: lcd (r = 0.624)
# Qreg_VSA: void_fraction (r = 0.815)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Henry_CO2 vs pld
sns.regplot(data=df, x='pld', y='Henry_CO2', ax=axes[0], color='blue', scatter_kws={'alpha':0.7})
axes[0].set_title("Henry_CO2 vs PLD\n(Spearman r = -0.61, p = 0.0001)")
axes[0].set_xlabel("PLD (Å)")
axes[0].set_ylabel("Henry_CO2 (mol/kg/Pa)")

# Qst_CO2 vs asa_m2_g
sns.regplot(data=df, x='asa_m2_g', y='Qst_CO2', ax=axes[1], color='orange', scatter_kws={'alpha':0.7})
axes[1].set_title("Qst_CO2 vs Accessible Surface Area\n(Spearman r = -0.67, p < 0.0001)")
axes[1].set_xlabel("asa_m2_g (m²/g)")
axes[1].set_ylabel("Qst_CO2 (kJ/mol)")

# Selectivity vs pld
sns.regplot(data=df, x='pld', y='Selectivity', ax=axes[2], color='green', scatter_kws={'alpha':0.7})
axes[2].set_title("Selectivity vs PLD\n(Spearman r = -0.75, p < 0.0001)")
axes[2].set_xlabel("PLD (Å)")
axes[2].set_ylabel("Selectivity")

# Qreg_TSA vs lcd
sns.regplot(data=df, x='lcd', y='Qreg_TSA', ax=axes[3], color='red', scatter_kws={'alpha':0.7})
axes[3].set_title("Qreg_TSA vs LCD\n(Spearman r = 0.62, p = 0.0001)")
axes[3].set_xlabel("LCD (Å)")
axes[3].set_ylabel("Qreg_TSA (kJ/mol CO₂)")

# Qreg_VSA vs void_fraction
sns.regplot(data=df, x='void_fraction', y='Qreg_VSA', ax=axes[4], color='purple', scatter_kws={'alpha':0.7})
axes[4].set_title("Qreg_VSA vs Void Fraction\n(Spearman r = 0.81, p < 0.0001)")
axes[4].set_xlabel("Void Fraction")
axes[4].set_ylabel("Qreg_VSA (kJ/mol CO₂)")

# Delete the 6th subplot
fig.delaxes(axes[5])
plt.tight_layout()
save_output_file("individual_properties_trends.png", is_plot=True)
plt.close()

# 5. Non-parametric hypothesis testing for categorical/binary variables
metal_test_results = {}
for col in property_cols + [target_col]:
    groups = [group[col].values for name, group in df.groupby('primary_metal')]
    if len(groups) > 1:
        stat, p = stats.kruskal(*groups)
        metal_test_results[col] = {'stat': stat, 'p-value': p}
metal_test_df = pd.DataFrame(metal_test_results).T
save_output_file("metal_kruskal_test.csv", is_csv=True, df_to_save=metal_test_df)

oms_test_results = {}
for col in property_cols + [target_col]:
    group_true = df[df['has_oms'] == 1.0][col].values
    group_false = df[df['has_oms'] == 0.0][col].values
    if len(group_true) > 0 and len(group_false) > 0:
        stat, p = stats.mannwhitneyu(group_true, group_false)
        oms_test_results[col] = {'stat': stat, 'p-value': p}
oms_test_df = pd.DataFrame(oms_test_results).T
save_output_file("oms_mannwhitney_test.csv", is_csv=True, df_to_save=oms_test_df)

# Boxplots of metal and OMS
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df, x='primary_metal', y=target_col, ax=axes[0], palette='Set2')
axes[0].set_title("TOPSIS score by Primary Metal")
axes[0].set_xlabel("Primary Metal")
axes[0].set_ylabel("TOPSIS score")

sns.boxplot(data=df, x='has_oms', y=target_col, ax=axes[1], palette='Set1')
axes[1].set_title("TOPSIS score by OMS status")
axes[1].set_xticklabels(['False', 'True'])
axes[1].set_xlabel("Has OMS")
axes[1].set_ylabel("TOPSIS score")
plt.tight_layout()
save_output_file("metal_comparison.png", is_plot=True)
plt.close()

# 6. TOPSIS Weight Reverse-Engineering
prop_directions = np.array([1, 1, 1, -1, -1])
X_matrix = df[property_cols].values
norm_X = X_matrix / np.sqrt(np.sum(X_matrix**2, axis=0))

def calculate_topsis(w, norm_matrix, directions):
    V = norm_matrix * w
    ideal_best = np.zeros(directions.shape)
    ideal_worst = np.zeros(directions.shape)
    for j in range(len(directions)):
        if directions[j] == 1:
            ideal_best[j] = np.max(V[:, j])
            ideal_worst[j] = np.min(V[:, j])
        else:
            ideal_best[j] = np.min(V[:, j])
            ideal_worst[j] = np.max(V[:, j])
    S_best = np.sqrt(np.sum((V - ideal_best)**2, axis=1))
    S_worst = np.sqrt(np.sum((V - ideal_worst)**2, axis=1))
    return S_worst / (S_best + S_worst)

target_scores = df[target_col].values
def loss_func(w_unnormalized):
    w = w_unnormalized / np.sum(w_unnormalized)
    calc_scores = calculate_topsis(w, norm_X, prop_directions)
    return np.mean((calc_scores - target_scores)**2)

bounds = [(0, 1) for _ in range(5)]
w0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
res = minimize(loss_func, w0, bounds=bounds)
optimized_w = res.x / np.sum(res.x)

weights_df = pd.DataFrame({'Property': property_cols, 'Optimized_Weight': optimized_w})
save_output_file("optimized_weights.csv", is_csv=True, df_to_save=weights_df)

sensitivity = np.std(norm_X, axis=0) * optimized_w
sensitivity_pct = sensitivity / np.sum(sensitivity) * 100
sensitivity_df = pd.DataFrame({
    'Property': property_cols,
    'Optimized_Weight': optimized_w,
    'Closeness_Sensitivity_Pct': sensitivity_pct
})
save_output_file("topsis_sensitivity.csv", is_csv=True, df_to_save=sensitivity_df)

# 7. Parsimonious OLS Linear Modeling for each of the 5 properties separately
regression_reports = []

# Fit regressions: we choose top 2 physical features from the VIF-cleared set for each property
# Let's see: selected_physical = ['pld', 'pld_path', 'asa_m2_cm3', 'pore_vol_cm3_g', 'density_g_cm3', 'uc_volume_A3']
# For each property, we'll sort these 6 physical features by their absolute correlation and choose the top 2.
for prop in property_cols + [target_col]:
    # Sort selected physical features by correlation with this property
    prop_corrs = spearman_corr.loc[selected_physical, prop].abs().sort_values(ascending=False)
    top_2_feats = list(prop_corrs.index[:2])
    
    X = df[top_2_feats]
    X_const = sm.add_constant(X)
    y = df[prop]
    
    model = sm.OLS(y, X_const).fit()
    
    # Check residuals normality
    residuals = model.resid
    stat_sw, p_sw = stats.shapiro(residuals)
    
    # Save OLS summary to a text file
    report_name = f"ols_summary_{prop}.txt"
    report_content = f"OLS REGRESSION FOR {prop}\n"
    report_content += f"====================================\n"
    report_content += f"Predictors used: {top_2_feats}\n"
    report_content += f"R-squared: {model.rsquared:.4f}, Adj R-squared: {model.rsquared_adj:.4f}\n"
    report_content += f"F-statistic p-value: {model.f_pvalue:.2e}\n"
    report_content += f"Shapiro-Wilk test on residuals: stat={stat_sw:.4f}, p-value={p_sw:.4f}\n\n"
    report_content += model.summary().as_text()
    
    save_output_file(report_name, content_str=report_content)
    regression_reports.append((prop, top_2_feats, model.rsquared, model.rsquared_adj, p_sw))

# Print out a summary of OLS models
print("\nOLS Regression Summary for all properties:")
for r in regression_reports:
    print(f"  {r[0]}: features={r[1]}, R2={r[2]:.3f}, AdjR2={r[3]:.3f}, Residual normality p={r[4]:.4f}")

print("\nUpdated analysis execution completed successfully!")
