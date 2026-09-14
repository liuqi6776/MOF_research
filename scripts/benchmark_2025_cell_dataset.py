# -*- coding: utf-8 -*-
"""
Benchmark Engine for 2025 Matter/Cell CoRE MOF Dataset
Evaluates:
  1. CIF -> X: Geometric and pore structural descriptors accuracy against Zeo++ ground truth
  2. CIF -> Y: Thermodynamic adsorption properties and selectivity accuracy of fine-tuned PMTransformer
"""
import os
import sys
import io
import glob
import zipfile
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.mof_property_predictor import MOFPropertyPredictor

def run_benchmark():
    zip_path = os.path.join(PROJECT_ROOT, "data_test1", "CoRE MOF 2025 Cell.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Test dataset archive not found at: {zip_path}")

    print(f"[*] Opening 2025 Matter/Cell CoRE MOF dataset: {zip_path}")
    z = zipfile.ZipFile(zip_path)

    # 1. Load Ground Truth X from features_ASR_2019.csv (6,900 MOFs, Zeo++ precision)
    print("[*] Loading Zeo++ ground truth structural features (features_ASR_2019.csv)...")
    with z.open("CoRE MOF 2025 Cell/mmc2/Diversity/features_ASR_2019.csv") as f:
        df_feat2019 = pd.read_csv(f)
    print(f"    - Loaded {len(df_feat2019)} records from features_ASR_2019.")

    # 2. Load Ground Truth metadata from ASR_data_20241125_internal.csv (8,857 MOFs)
    print("[*] Loading comprehensive metadata (ASR_data_20241125_internal.csv)...")
    with z.open("CoRE MOF 2025 Cell/mmc3/Information/ASR_data_20241125_internal.csv") as f:
        df_asr = pd.read_csv(f)
    df_asr['csd_code'] = df_asr['refcode'].astype(str).str.extract(r'^([A-Z]{6})')[0]
    print(f"    - Loaded {len(df_asr)} records from ASR_data.")

    # 3. Load 891 High-Throughput GCMC Screening table
    print("[*] Loading high-throughput GCMC screening data (data_298_423_1bar_CO2_N2_891.csv)...")
    with z.open("CoRE MOF 2025 Cell/mmc6/TSA/data_298_423_1bar_CO2_N2_891.csv") as f:
        df_891 = pd.read_csv(f)
    df_891_merged = pd.merge(df_891, df_asr[['coreid', 'csd_code', 'refcode', 'Density (g/cm3)']], on='coreid', how='inner')
    print(f"    - Loaded {len(df_891)} screening records ({len(df_891_merged)} with CSD refcodes).")

    # 4. Load 695 Full GCMC Excel table for comprehensive Y verification
    excel_695_path = os.path.join(PROJECT_ROOT, "695_MOF", "CoRE_MOF_2019_GCMC_695_总文件.xlsx")
    df_695 = pd.read_excel(excel_695_path, header=1)
    mof_col = [c for c in df_695.columns if 'MOF' in str(c) or '名称' in str(c)][0]
    df_695['MOF_ID'] = df_695[mof_col].astype(str).str.strip()

    # Create fast lookups
    lookup_2019 = {}
    for _, r in df_feat2019.iterrows():
        n = str(r['name']).strip()
        lookup_2019[n] = r
        if n.endswith('_clean'):
            lookup_2019[n.replace('_clean', '')] = r
        else:
            lookup_2019[n + '_clean'] = r

    lookup_asr_csd = {}
    for _, r in df_asr.dropna(subset=['csd_code']).iterrows():
        c = str(r['csd_code']).strip()
        lookup_asr_csd[c] = r

    lookup_891_csd = {}
    for _, r in df_891_merged.dropna(subset=['csd_code']).iterrows():
        c = str(r['csd_code']).strip()
        lookup_891_csd[c] = r

    lookup_695 = {}
    for _, r in df_695.iterrows():
        m = str(r['MOF_ID']).strip()
        lookup_695[m] = r
        if m.endswith('_clean'):
            lookup_695[m.replace('_clean', '')] = r
        else:
            lookup_695[m + '_clean'] = r

    # 5. Initialize Predictor
    print("\n[*] Initializing MOFPropertyPredictor...")
    predictor = MOFPropertyPredictor()

    cif_patterns = [
        os.path.join(PROJECT_ROOT, "PMtransformer", "PMTransformer_695GCMC_695(1)", "PMTransformer_695GCMC_695", "moftransformer_inputs", "*.cif"),
        os.path.join(PROJECT_ROOT, "252_MOF_CIFs", "*.cif")
    ]
    all_cifs = []
    for pat in cif_patterns:
        all_cifs.extend(glob.glob(pat))
    all_cifs = sorted(list(set(all_cifs)))
    print(f"[*] Found {len(all_cifs)} unique local CIF files for empirical benchmarking.")

    records = []
    print("\n[*] Parsing CIFs, extracting X features, and running fine-tuned PMTransformer model...")
    for idx, cif_path in enumerate(all_cifs):
        base_name = os.path.splitext(os.path.basename(cif_path))[0]
        csd_6 = base_name[:6]

        try:
            res = predictor.predict_properties(cif_path)
            feat = res['features']
            preds = res['predictions']
        except Exception as e:
            print(f"[!] Warning on {base_name}: {e}")
            continue

        ref_2019 = lookup_2019.get(base_name)
        ref_asr = lookup_asr_csd.get(csd_6)
        ref_695 = lookup_695.get(base_name)
        ref_891 = lookup_891_csd.get(csd_6)

        true_pld = None
        true_lcd = None
        true_lfpd = None
        true_density = None
        true_asa = None
        true_pvol = None
        true_vf = None

        if ref_2019 is not None:
            true_pld = float(ref_2019.get('Df', np.nan))
            true_lcd = float(ref_2019.get('Di', np.nan))
            true_lfpd = float(ref_2019.get('Dif', np.nan))
            true_asa = float(ref_2019.get('GSA_1_4', np.nan))
            true_pvol = float(ref_2019.get('GPOV_1_4', np.nan))
            true_vf = float(ref_2019.get('POAV_vol_frac_1_4', np.nan))
        elif ref_asr is not None:
            true_pld = float(ref_asr.get('PLD (Å)', np.nan))
            true_lcd = float(ref_asr.get('LCD (Å)', np.nan))
            true_lfpd = float(ref_asr.get('LFPD (Å)', np.nan))
            true_density = float(ref_asr.get('Density (g/cm3)', np.nan))
            true_asa = float(ref_asr.get('ASA (m2/g)', np.nan))
            true_pvol = float(ref_asr.get('PV (cm3/g)', np.nan))
            true_vf = float(ref_asr.get('VF', np.nan))

        if true_density is None and ref_asr is not None:
            true_density = float(ref_asr.get('Density (g/cm3)', np.nan))
        if true_density is None and ref_695 is not None:
            for col in ref_695.index:
                if '密度' in str(col) or 'Density' in str(col):
                    try:
                        true_density = float(ref_695[col])
                        break
                    except:
                        pass

        true_co2_1bar = None
        true_n2_1bar = None
        true_sel = None
        true_qst = None
        true_co2_015 = None

        if ref_891 is not None:
            true_co2_1bar = float(ref_891.get('298_CO2', np.nan))
            true_n2_1bar = float(ref_891.get('298_N2', np.nan))
            if true_n2_1bar and true_n2_1bar > 1e-5:
                true_sel = float(true_co2_1bar / true_n2_1bar)

        if ref_695 is not None:
            for col in ref_695.index:
                c_str = str(col)
                if '1bar' in c_str and ('CO2' in c_str or 'q_ads' in c_str):
                    try:
                        if true_co2_1bar is None:
                            true_co2_1bar = float(ref_695[col])
                    except: pass
                if 'N2' in c_str and '1bar' in c_str:
                    try:
                        if true_n2_1bar is None:
                            true_n2_1bar = float(ref_695[col])
                    except: pass
                if '实际选择性' in c_str or 'Actual Selectivity' in c_str:
                    try:
                        if true_sel is None:
                            true_sel = float(ref_695[col])
                    except: pass
                if 'Widom' in c_str or 'Qst' in c_str:
                    try:
                        true_qst = float(ref_695[col])
                    except: pass
                if '0.15bar' in c_str and 'CO2' in c_str:
                    try:
                        true_co2_015 = float(ref_695[col])
                    except: pass

        record = {
            'mof_name': base_name,
            'csd_code': csd_6,
            'parsed_pld': feat['pld_est'],
            'parsed_lcd': feat['lcd_est'],
            'parsed_density': feat['density'],
            'parsed_asa': feat['asa_est'],
            'parsed_pvol': feat['pore_vol_est'],
            'pred_pld': preds.get('pld_pred', {}).get('value'),
            'pred_lcd': preds.get('lcd_pred', {}).get('value'),
            'pred_asa': preds.get('asa_pred', {}).get('value'),
            'pred_pvol': preds.get('pvol_pred', {}).get('value'),
            'true_pld': true_pld,
            'true_lcd': true_lcd,
            'true_lfpd': true_lfpd,
            'true_density': true_density,
            'true_asa': true_asa,
            'true_pvol': true_pvol,
            'true_vf': true_vf,
            'pred_co2_1bar': preds.get('co2_1bar', {}).get('value'),
            'pred_n2_1bar': preds.get('n2_1bar', {}).get('value'),
            'pred_co2_015bar': preds.get('co2_015bar', {}).get('value'),
            'pred_sel': preds.get('selectivity_real', {}).get('value'),
            'pred_qst': preds.get('qst_widom', {}).get('value'),
            'pred_pe_vsa': preds.get('pe_vsa', {}).get('value'),
            'true_co2_1bar': true_co2_1bar,
            'true_n2_1bar': true_n2_1bar,
            'true_co2_015bar': true_co2_015,
            'true_sel': true_sel,
            'true_qst': true_qst
        }
        records.append(record)

        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_cifs):
            print(f"    - Processed {idx + 1}/{len(all_cifs)} CIFs...")

    df_res = pd.DataFrame(records)
    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    all_details_path = os.path.join(out_dir, "benchmark_2025_all_mofs_comparison_details.csv")
    df_res.to_csv(all_details_path, index=False, encoding='utf-8-sig')
    print(f"\n[✓] Detailed per-MOF comparison saved to: {all_details_path}")

    # =========================================================================
    # Compute Metrics for Dimension 1: CIF -> X
    # =========================================================================
    print("\n" + "="*95)
    print("  DIMENSION 1: CIF -> X GEOMETRIC DESCRIPTORS ACCURACY (vs. 2025 Matter/Cell Ground Truth)")
    print("="*95)

    x_benchmarks = [
        ('Pore Limiting Diameter PLD (Å)', 'parsed_pld', 'true_pld', 'pred_pld'),
        ('Largest Cavity Diameter LCD (Å)', 'parsed_lcd', 'true_lcd', 'pred_lcd'),
        ('Crystal Density (g/cm³)', 'parsed_density', 'true_density', None),
        ('Gravimetric Surface Area ASA (m²/g)', 'parsed_asa', 'true_asa', 'pred_asa'),
        ('Pore Volume (cm³/g)', 'parsed_pvol', 'true_pvol', 'pred_pvol')
    ]

    metrics_x = []
    print(f"{'Descriptor':<38} | {'N':<5} | {'R²':<7} | {'MAE':<7} | {'RMSE':<7} | {'Pearson r':<9} | {'MAPE (%)':<8}")
    print("-" * 95)

    for desc_name, parse_col, true_col, pred_col in x_benchmarks:
        col_to_use = parse_col
        valid = df_res[[col_to_use, true_col]].dropna()
        valid = valid[np.isfinite(valid[col_to_use]) & np.isfinite(valid[true_col])]
        valid = valid[valid[true_col] > 0]

        if len(valid) >= 10:
            y_t = valid[true_col].values
            y_p = valid[col_to_use].values
            r2 = r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            pr, _ = pearsonr(y_t, y_p)
            sr, _ = spearmanr(y_t, y_p)
            mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-6))) * 100.0

            metrics_x.append({
                'Descriptor': desc_name,
                'Valid_Count': len(valid),
                'R2_Score': round(r2, 4),
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'Pearson_r': round(pr, 4),
                'Spearman_rho': round(sr, 4),
                'MAPE_Percent': round(mape, 2)
            })
            print(f"{desc_name:<38} | {len(valid):<5} | {r2:7.4f} | {mae:7.3f} | {rmse:7.3f} | {pr:9.4f} | {mape:7.2f}%")

    df_mx = pd.DataFrame(metrics_x)
    metrics_x_path = os.path.join(out_dir, "benchmark_2025_cif_to_x_metrics.csv")
    df_mx.to_csv(metrics_x_path, index=False, encoding='utf-8-sig')

    # =========================================================================
    # Compute Metrics for Dimension 2: CIF -> Y
    # =========================================================================
    print("\n" + "="*95)
    print("  DIMENSION 2: CIF -> Y THERMODYNAMIC ADSORPTION PREDICTIONS (vs. 2025 Matter/Cell GCMC)")
    print("="*95)

    y_benchmarks = [
        ('CO2 Uptake @ 1.0bar 1atm (mol/kg)', 'pred_co2_1bar', 'true_co2_1bar'),
        ('N2 Uptake @ 1.0bar (mol/kg)', 'pred_n2_1bar', 'true_n2_1bar'),
        ('CO2 Uptake @ 0.15bar FlueGas (mol/kg)', 'pred_co2_015bar', 'true_co2_015bar'),
        ('CO2/N2 Actual Selectivity', 'pred_sel', 'true_sel'),
        ('CO2 Adsorption Heat Qst (kJ/mol)', 'pred_qst', 'true_qst')
    ]

    metrics_y = []
    print(f"{'Target Property':<38} | {'N':<5} | {'R²':<7} | {'MAE':<7} | {'RMSE':<7} | {'Pearson r':<9} | {'Spearman ρ':<9}")
    print("-" * 95)

    for prop_name, pred_col, true_col in y_benchmarks:
        valid = df_res[[pred_col, true_col]].dropna()
        valid = valid[np.isfinite(valid[pred_col]) & np.isfinite(valid[true_col])]
        valid = valid[valid[true_col] > 0]

        if len(valid) >= 10:
            y_t = valid[true_col].values
            y_p = valid[pred_col].values
            r2 = r2_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            rmse = np.sqrt(mean_squared_error(y_t, y_p))
            pr, _ = pearsonr(y_t, y_p)
            sr, _ = spearmanr(y_t, y_p)

            metrics_y.append({
                'Target_Property': prop_name,
                'Valid_Count': len(valid),
                'R2_Score': round(r2, 4),
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'Pearson_r': round(pr, 4),
                'Spearman_rho': round(sr, 4)
            })
            print(f"{prop_name:<38} | {len(valid):<5} | {r2:7.4f} | {mae:7.3f} | {rmse:7.3f} | {pr:9.4f} | {sr:9.4f}")

    df_my = pd.DataFrame(metrics_y)
    metrics_y_path = os.path.join(out_dir, "benchmark_2025_cif_to_y_metrics.csv")
    df_my.to_csv(metrics_y_path, index=False, encoding='utf-8-sig')

    # =========================================================================
    # Generate High-Resolution Multi-Panel Parity Plots
    # =========================================================================
    print("\n[*] Generating high-resolution Parity Scatter Plots...")
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    plot_configs = [
        (0, 0, 'PLD (Å) Pore Limiting Diameter', 'true_pld', 'parsed_pld', '#0284c7', 'True Zeo++ PLD (Å)', 'Parsed CIF PLD (Å)'),
        (0, 1, 'LCD (Å) Largest Cavity Diameter', 'true_lcd', 'parsed_lcd', '#2563eb', 'True Zeo++ LCD (Å)', 'Parsed CIF LCD (Å)'),
        (0, 2, 'Crystal Density (g/cm³)', 'true_density', 'parsed_density', '#0d9488', 'True Density (g/cm³)', 'Parsed Density (g/cm³)'),
        (1, 0, 'Gravimetric Surface Area ASA (m²/g)', 'true_asa', 'parsed_asa', '#7c3aed', 'True Zeo++ ASA (m²/g)', 'Parsed ASA (m²/g)'),
        (1, 1, 'CO2 Uptake @ 1.0 bar (mol/kg)', 'true_co2_1bar', 'pred_co2_1bar', '#e11d48', 'True GCMC CO2 Uptake (mol/kg)', 'PMTransformer Pred CO2 (mol/kg)'),
        (1, 2, 'CO2/N2 Actual Selectivity', 'true_sel', 'pred_sel', '#d97706', 'True GCMC Selectivity', 'PMTransformer Pred Selectivity')
    ]

    for row, col, title, true_k, pred_k, color, xlabel, ylabel in plot_configs:
        ax = axes[row, col]
        sub = df_res[[true_k, pred_k]].dropna()
        sub = sub[(sub[true_k] > 0) & (sub[pred_k] > 0)]

        if len(sub) > 0:
            yt = sub[true_k].values
            yp = sub[pred_k].values

            if 'Selectivity' in title:
                clip_mask = (yt < 120) & (yp < 120)
                yt, yp = yt[clip_mask], yp[clip_mask]

            r2 = r2_score(yt, yp)
            mae = mean_absolute_error(yt, yp)
            pr, _ = pearsonr(yt, yp)

            min_val = min(np.min(yt), np.min(yp)) * 0.95
            max_val = max(np.percentile(yt, 99), np.percentile(yp, 99)) * 1.05

            ax.scatter(yt, yp, alpha=0.55, s=28, color=color, edgecolors='none', label='Test MOFs')
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.6, alpha=0.8, label='Ideal y = x')

            z_fit = np.polyfit(yt, yp, 1)
            p_fit = np.poly1d(z_fit)
            x_line = np.linspace(min_val, max_val, 100)
            ax.plot(x_line, p_fit(x_line), color='#dc2626', lw=1.4, alpha=0.9, label=f'Fit: y={z_fit[0]:.2f}x+{z_fit[1]:.2f}')

            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_xlabel(xlabel, fontsize=11, fontweight='bold', color='#1e293b')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', color='#1e293b')
            ax.set_title(f"{title}\nR² = {r2:.4f} | MAE = {mae:.3f} | r = {pr:.4f} (N={len(yt)})", fontsize=12, fontweight='bold', color='#0f172a', pad=10)
            ax.legend(loc='upper left', frameon=True, fontsize=9)
            ax.tick_params(colors='#475569')

    plt.suptitle("Comprehensive Benchmark on 2025 Matter/Cell CoRE MOF Dataset: CIF -> X & Y Predictions", fontsize=15, fontweight='bold', y=0.99, color='#0f172a')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plot_path = os.path.join(out_dir, "benchmark_2025_xy_parity_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] High-resolution parity plots saved to: {plot_path}")

    artifact_dir = r"C:\Users\liuqi\.gemini\antigravity\brain\b0ea68cf-0151-4c0d-b05f-90b64cf2ef90"
    if os.path.exists(artifact_dir):
        dest_plot = os.path.join(artifact_dir, "benchmark_2025_xy_parity_plots.png")
        shutil.copyfile(plot_path, dest_plot)
        print(f"[✓] Parity plot copied to artifact directory: {dest_plot}")

    print("\n" + "="*95)
    print("  BENCHMARK EVALUATION COMPLETE SUCCESSFULLY")
    print("="*95)

if __name__ == '__main__':
    run_benchmark()
