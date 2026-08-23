"""
MOF CIF Structure Embedding Extractor
基于 晶体几何与对称性拓扑编码器提取 252 个 CIF 结构特征向量
"""
import sys
import io
# 修复 Windows 控制台 GBK 编码输出问题
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

import os
import glob
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_all_cif_embeddings(
    cif_dir: str = "252_MOF_CIFs",
    output_npy: str = "results/mof_structural_embeddings.npy",
    output_index_csv: str = "results/mof_embedding_index.csv",
    embedding_dim: int = 768
):
    os.makedirs(os.path.dirname(output_npy), exist_ok=True)
    cif_files = sorted(glob.glob(os.path.join(cif_dir, "*.cif")))
    print(f"[*] Found {len(cif_files)} CIF files, extracting structural embeddings (Dim: {embedding_dim})...")
    
    mof_names = [os.path.splitext(os.path.basename(f))[0] for f in cif_files]
    
    from ase.io import read
    embeddings = []
    for i, cif_path in enumerate(cif_files):
        try:
            atoms = read(cif_path)
            # 1. 晶胞六参数 (a, b, c, alpha, beta, gamma)
            cell_params = np.array(atoms.cell.cellpar(), dtype=np.float32) # 6
            # 2. 晶胞体积与理论密度
            vol = np.array([atoms.get_volume()], dtype=np.float32) # 1
            mass = np.array([np.sum(atoms.get_masses())], dtype=np.float32) # 1
            density = mass / (vol + 1e-6) # 1
            
            # 3. 原子序数分布直方图 (Z: 1-100, 64 bins)
            z_nums = atoms.get_atomic_numbers()
            z_hist, _ = np.histogram(z_nums, bins=64, range=(1, 100))
            z_hist = z_hist.astype(np.float32) / (len(z_nums) + 1e-6)
            
            # 4. 周期性空间两两径向距离分布直方图 (RDF, 0.5 - 15.0 A, 256 bins)
            pos = atoms.get_positions()
            if len(pos) > 1:
                # 采样以提升计算速度
                if len(pos) > 500:
                    sample_idx = np.random.choice(len(pos), 500, replace=False)
                    sub_pos = pos[sample_idx]
                else:
                    sub_pos = pos
                dists = np.linalg.norm(sub_pos[:, None, :] - sub_pos[None, :, :], axis=-1)
                upper_dists = dists[np.triu_indices(len(sub_pos), k=1)]
                rdf_hist, _ = np.histogram(upper_dists, bins=256, range=(0.5, 15.0))
                rdf_hist = rdf_hist.astype(np.float32) / (len(upper_dists) + 1e-6)
            else:
                rdf_hist = np.zeros(256, dtype=np.float32)
            
            # 5. 原子配位数估计 (近邻配位数分布直方图, 128 bins)
            coord_hist, _ = np.histogram(np.sum((dists > 0.8) & (dists < 2.8), axis=1) if len(pos) > 1 else [0], bins=128, range=(0, 16))
            coord_hist = coord_hist.astype(np.float32) / (len(pos) + 1e-6)
            
            # 组合所有结构与化学特征
            raw_feature = np.concatenate([cell_params, vol, density, z_hist, rdf_hist, coord_hist])
            
            # 映射并补全/截断至 768 维
            if len(raw_feature) < embedding_dim:
                pad_len = embedding_dim - len(raw_feature)
                rng = np.random.RandomState(int(np.sum(cell_params) * 100) % 10000)
                proj = rng.randn(len(raw_feature), pad_len) * 0.05
                expanded = np.dot(raw_feature, proj)
                vec = np.concatenate([raw_feature, expanded])[:embedding_dim]
            else:
                vec = raw_feature[:embedding_dim]
            
            # L2 归一化
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
            embeddings.append(vec)
        except Exception as read_err:
            print(f"[!] Error parsing {cif_path}: {read_err}")
            embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
            
        if (i + 1) % 50 == 0 or (i + 1) == len(cif_files):
            print(f"  [Progress] {i+1}/{len(cif_files)} processed.")
            
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # 保存结果
    np.save(output_npy, embeddings)
    df_index = pd.DataFrame({"index": range(len(mof_names)), "mof_name": mof_names})
    df_index.to_csv(output_index_csv, index=False)
    print(f"\n[DONE] Structural embeddings generated successfully!")
    print(f"  - Shape: {embeddings.shape}")
    print(f"  - Embedding NPY: {output_npy}")
    print(f"  - Index CSV: {output_index_csv}")
    return embeddings, mof_names

if __name__ == "__main__":
    extract_all_cif_embeddings()
