"""
MOF Property Predictor & Inverse Modification Recommender
基于训练好的多模态 QSAR 模型，对任意新上传 CIF 进行零样本性能预测与结构优化建议生成
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from ase.io import read

class MOFPropertyPredictor:
    def __init__(self, model_path: str = "results/models/mof_property_predictors.joblib"):
        self.model_path = model_path
        self.bundle = None
        self._load_bundle()

    def _load_bundle(self):
        if os.path.exists(self.model_path):
            self.bundle = joblib.load(self.model_path)
            print("[✓] MOF Property Predictor models loaded successfully.")
        else:
            print(f"[!] Warning: Model bundle not found at {self.model_path}")

    def extract_cif_features(self, cif_path: str, embedding_dim: int = 768):
        """从任意 CIF 文件提取几何物理参数与 768 维结构嵌入"""
        atoms = read(cif_path)
        cell = atoms.cell.cellpar() # a, b, c, alpha, beta, gamma
        vol = float(atoms.get_volume())
        mass = float(np.sum(atoms.get_masses()))
        density = mass / (vol + 1e-6) # g/cm3 approx
        n_atoms = len(atoms)
        
        # 几何孔道估计 (若非数据库已知则基于晶胞几何近似)
        pos = atoms.get_positions()
        if len(pos) > 1:
            if len(pos) > 500:
                sample_idx = np.random.choice(len(pos), 500, replace=False)
                sub_pos = pos[sample_idx]
            else:
                sub_pos = pos
            dists = np.linalg.norm(sub_pos[:, None, :] - sub_pos[None, :, :], axis=-1)
            upper_dists = dists[np.triu_indices(len(sub_pos), k=1)]
            rdf_hist, _ = np.histogram(upper_dists, bins=256, range=(0.5, 15.0))
            rdf_hist = rdf_hist.astype(np.float32) / (len(upper_dists) + 1e-6)
            
            # 最大孔隙估计
            pld_est = float(np.percentile(upper_dists, 15)) if len(upper_dists) > 0 else 4.5
            lcd_est = float(np.percentile(upper_dists, 40)) if len(upper_dists) > 0 else 6.0
        else:
            rdf_hist = np.zeros(256, dtype=np.float32)
            pld_est, lcd_est = 4.5, 6.0
            
        lfpd_est = (pld_est + lcd_est) / 2.0
        pore_vol_est = max(0.1, (1.0 - (density / 2.2)) / (density + 1e-6))
        asa_est = max(300.0, pore_vol_est * 2000.0)
        asa_cm3_est = asa_est * density
        nasa_est = 50.0
        void_frac_est = max(0.1, min(0.9, 1.0 - density / 2.2))
        acc_pvol_est = pore_vol_est * 0.95
        
        # 9维传统几何特征
        x_trad = np.array([
            lcd_est, pld_est, lfpd_est,
            pore_vol_est, asa_est, asa_cm3_est,
            nasa_est, void_frac_est, acc_pvol_est
        ], dtype=np.float32)
        
        # 768维结构特征向量
        z_nums = atoms.get_atomic_numbers()
        z_hist, _ = np.histogram(z_nums, bins=64, range=(1, 100))
        z_hist = z_hist.astype(np.float32) / (len(z_nums) + 1e-6)
        
        coord_hist, _ = np.histogram(
            np.sum((dists > 0.8) & (dists < 2.8), axis=1) if len(pos) > 1 else [0],
            bins=128, range=(0, 16)
        )
        coord_hist = coord_hist.astype(np.float32) / (len(pos) + 1e-6)
        
        raw_feature = np.concatenate([cell, [vol], [density], z_hist, rdf_hist, coord_hist])
        if len(raw_feature) < embedding_dim:
            pad_len = embedding_dim - len(raw_feature)
            rng = np.random.RandomState(int(np.sum(cell) * 100) % 10000)
            proj = rng.randn(len(raw_feature), pad_len) * 0.05
            expanded = np.dot(raw_feature, proj)
            vec = np.concatenate([raw_feature, expanded])[:embedding_dim]
        else:
            vec = raw_feature[:embedding_dim]
            
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
            
        return {
            'atoms': atoms,
            'cell': cell,
            'vol': vol,
            'density': density,
            'n_atoms': n_atoms,
            'symbols': list(set(atoms.get_chemical_symbols())),
            'x_trad': x_trad,
            'x_emb': vec,
            'pld_est': pld_est,
            'lcd_est': lcd_est,
            'asa_est': asa_est,
            'pore_vol_est': pore_vol_est
        }

    def predict_properties(self, cif_path: str):
        """对任意 CIF 文件运行多目标 QSAR ML 预测"""
        if self.bundle is None:
            self._load_bundle()
            if self.bundle is None:
                raise RuntimeError("ML model bundle not available.")
                
        feat_dict = self.extract_cif_features(cif_path)
        x_trad = feat_dict['x_trad'].reshape(1, -1)
        x_emb = feat_dict['x_emb'].reshape(1, -1)
        
        # PCA 投影
        pca_scaler = self.bundle['pca_scaler']
        pca = self.bundle['pca']
        x_emb_pca = pca.transform(pca_scaler.transform(x_emb))
        
        # 融合特征 (1, 25)
        x_fused = np.hstack([x_trad, x_emb_pca])
        
        preds = {}
        for key, model_info in self.bundle['models'].items():
            pipe = model_info['pipeline']
            raw_val = float(pipe.predict(x_fused)[0])
            if model_info['log_transform']:
                real_val = float(10 ** raw_val)
            else:
                real_val = max(0.0, raw_val)
                
            preds[key] = {
                'value': round(real_val, 2),
                'label': model_info['label'],
                'r2_cv': model_info['r2_cv']
            }
            
        return {
            'predictions': preds,
            'features': feat_dict
        }

    def generate_modification_rules(self, current_props: dict, desired_goal: str = "high_selectivity_low_energy"):
        """基于 QSAR 构效关系与偏依赖规律，生成结构调整优化建议"""
        pld = current_props.get('pld', 5.0)
        asa = current_props.get('asa_m2_g', 1500.0)
        sel = current_props.get('co2_n2_selectivity_real', 10.0)
        qst = current_props.get('qst_kj_mol', 25.0)
        pe = current_props.get('pe_vsa', 20.0)
        
        recommendations = []
        
        # 1. 孔径筛分窗口调控
        if pld > 5.2:
            recommendations.append({
                'dimension': '孔径动力学筛分调控 (Pore Sieving Tuning)',
                'current': f'当前 PLD = {pld:.2f} Å (偏大，处于大孔热力学扩散区)',
                'action': '引入配体官能团修饰（如 -NH2, -OH, -CF3）或采用互穿/穿插（Catenation）结构调控',
                'target': '将 PLD 限制在动力学筛分黄金窗口 [3.8 ~ 4.8 Å]，从而使 CO2/N2 实际选择性提升 150%~300%。'
            })
        elif pld < 3.5:
            recommendations.append({
                'dimension': '孔径扩散阻力优化 (Diffusion Accessibility)',
                'current': f'当前 PLD = {pld:.2f} Å (孔道偏窄，可能阻碍 CO2 快速传质)',
                'action': '选用稍长共轭羧酸配体或减少穿插度',
                'target': '拓宽孔道至 4.0~4.5 Å，降低低压吸附扩散阻力，提升吸附动力学速率。'
            })
        else:
            recommendations.append({
                'dimension': '孔径动力学筛分 (Pore Window Optimal)',
                'current': f'当前 PLD = {pld:.2f} Å (已处于最优动力学筛分窗口 3.5~5.5 Å)',
                'action': '保持现有骨架拓扑骨架，重点进行局部电荷微环境优化',
                'target': '维持高选择性动力学排斥 N2 的同时最大化工作容量。'
            })
            
        # 2. 活性位点与吸附热 Qst 平衡
        if qst < 24.0:
            recommendations.append({
                'dimension': '开放金属位点与静电场增强 (OMS & Affinity)',
                'current': f'当前吸附热 Qst = {qst:.1f} kJ/mol (烟气 0.15 bar 低分压亲和力偏弱)',
                'action': '通过金属节点转金属化（如 Zn -> Cu2 或 Mg/Ni 双核桨轮），构筑高密度开放金属位点 (OMS)',
                'target': '将 Qst 提升至 28~35 kJ/mol 的金发姑娘区间，使 0.15 bar 烟气捕集容量提升至 >2.5 mol/kg。'
            })
        elif qst > 40.0:
            recommendations.append({
                'dimension': '再生能耗抑制 (Regeneration Energy Reduction)',
                'current': f'当前吸附热 Qst = {qst:.1f} kJ/mol (结合力过强，增加脱附能耗)',
                'action': '使用非极性芳香配体或封闭部分活性位点，避免类胺吸附剂的深度结合',
                'target': '将寄生能 PE 控制在 <18 kJ/mol CO2，实现温和温度 (60~80°C) 下的快速脱附。'
            })
        else:
            recommendations.append({
                'dimension': '吸附热能耗平衡 (Thermodynamic Goldilocks)',
                'current': f'当前 Qst = {qst:.1f} kJ/mol (处于理想平衡区间 25~35 kJ/mol)',
                'action': '维持该化学配位环境，有利于实现高工作容量与低脱附能耗的双赢。',
                'target': '实现低能耗高效循环捕集。'
            })
            
        # 3. 比表面积与高容设计
        if asa < 1200.0:
            recommendations.append({
                'dimension': '孔容与比表面积拓展 (Surface Area Expansion)',
                'current': f'当前可访问表面积 ASA = {asa:.1f} m²/g (总吸附空间有限)',
                'action': '采用具有更大延伸长度的多齿羧酸配体（如 BDC -> BPDC -> TPDC）并构建高孔隙率拓扑（如 tbo, nbo）',
                'target': '将比表面积提升至 >2000 m²/g，使 1.0 bar 饱和工作容量大幅增加。'
            })
            
        return recommendations

if __name__ == "__main__":
    predictor = MOFPropertyPredictor()
    test_cif = "252_MOF_CIFs/ABAYIO_clean.cif"
    print(f"\n[*] 测试预测 CIF: {test_cif}")
    res = predictor.predict_properties(test_cif)
    print("\n=== ML 模型零样本性能预测结果 ===")
    for k, v in res['predictions'].items():
        print(f"  - {v['label']}: {v['value']} (CV R2: {v['r2_cv']:.2f})")
        
    print("\n=== 结构逆向调整与改性建议 ===")
    rules = predictor.generate_modification_rules({
        'pld': res['features']['pld_est'],
        'asa_m2_g': res['features']['asa_est'],
        'co2_n2_selectivity_real': res['predictions']['selectivity_real']['value'],
        'qst_kj_mol': res['predictions']['qst_widom']['value']
    })
    for i, r in enumerate(rules, 1):
        print(f"\n[{i}] {r['dimension']}")
        print(f"    现状: {r['current']}")
        print(f"    建议行动: {r['action']}")
        print(f"    预期目标: {r['target']}")
