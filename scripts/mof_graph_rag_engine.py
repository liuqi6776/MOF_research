"""
Bilingual Multi-Modal Graph RAG Engine for MOF Research (MatterChat Paradigm arXiv:2502.13107)
基于 MatterChat 范式与 DeepSeek 大模型的 MOF 多模态 Graph RAG 科学知识推理引擎 (完整字段精准映射版)
"""
import sys
import io
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass

import os
import json
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Optional

# Load from environment or local config (.env)
def get_default_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key and os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("DEEPSEEK_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass
    return key

DEFAULT_DEEPSEEK_API_KEY = get_default_api_key()
DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

class MOFMultiModalGraphRAG:
    def __init__(
        self,
        excel_path: str = "252_MOF_总文件 冗余评估数据.xlsx",
        struct_emb_path: str = "results/mof_structural_embeddings.npy",
        index_csv_path: str = "results/mof_embedding_index.csv",
        api_key: str = DEFAULT_DEEPSEEK_API_KEY,
        model_name: str = DEFAULT_MODEL
    ):
        self.excel_path = excel_path
        self.struct_emb_path = struct_emb_path
        self.index_csv_path = index_csv_path
        self.api_key = api_key
        self.model_name = model_name
        
        self.kg = nx.MultiDiGraph()
        self.mof_data_store: Dict[str, Dict[str, Any]] = {}
        self.struct_embeddings: Optional[np.ndarray] = None
        self.mof_names: List[str] = []
        
        self._load_data()
        self._build_knowledge_graph()

    def _safe_float(self, val, default=0.0):
        try:
            if pd.isna(val):
                return default
            return float(val)
        except Exception:
            return default

    def _load_data(self):
        """解析 Excel 真实字段并加载 768 维结构特征向量"""
        print("[*] 1/3 加载结构嵌入向量与 156 列真实物理化学数据...")
        
        if os.path.exists(self.struct_emb_path) and os.path.exists(self.index_csv_path):
            self.struct_embeddings = np.load(self.struct_emb_path)
            idx_df = pd.read_csv(self.index_csv_path)
            self.mof_names = idx_df['mof_name'].tolist()
            print(f"  [✓] 结构向量矩阵加载成功: {self.struct_embeddings.shape}")

        df = pd.read_excel(self.excel_path, header=1)
        for _, row in df.iterrows():
            name = str(row.get('MOF名称', '')).strip()
            if not name or name == 'nan':
                continue
            
            pld = self._safe_float(row.get('PLD (Å)\n孔道限制直径'))
            lcd = self._safe_float(row.get('LCD (Å)\n最大空腔直径'))
            asa = self._safe_float(row.get('可访问表面积\n(m²/g)'))
            if asa == 0.0:
                asa = self._safe_float(row.get('可访问表面积\n(m²/cm³)'))
            p_vol = self._safe_float(row.get('孔体积\n(cm³/g)'))
            
            # 精准气体吸附与分离性能字段
            co2_1bar = self._safe_float(row.get('q_ads_CO2_1bar_298K_mol_kg', row.get('q_CO2_298K_1bar_mol_kg')))
            co2_015bar = self._safe_float(row.get('q_CO2_298K_0.15bar_mol_kg'))
            n2_1bar = self._safe_float(row.get('q_N2_298K_1bar_mol_kg'))
            
            sel_real = self._safe_float(row.get('CO2N2实际选择性\n越高越好'))
            sel_henry = self._safe_float(row.get('CO2N2_Henry选择性\n越高越好'))
            qst = self._safe_float(row.get('CO2_Qst_Widom零覆盖(kJ/mol)\n越高越好', row.get('qst_Widom_CO2_kJ_mol')))
            
            vsa_wc = self._safe_float(row.get('CO2_VSA工作容量(mol/kg)\n越高越好'))
            tsa_wc = self._safe_float(row.get('CO2_TSA工作容量(mol/kg)\n越高越好'))
            pe_vsa = self._safe_float(row.get('PE_VSA寄生能(kJ/mol CO2)\n越低越好'))
            q_tsa_regen = self._safe_float(row.get('CO2_TSA双积分法计算再生热(kJ/mol)\n越低越好'))
            
            topo = str(row.get('拓扑代码 (Topology Code)', 'unknown')).lower().strip()
            cat = int(self._safe_float(row.get('穿插度 (Catenation)'), 0))
            metal = str(row.get('金属元素', 'unknown')).strip()
            
            mof_info = {
                "mof_name": name,
                "csd_refcode": str(row.get('MOF名称 (CSD Refcode)', name)),
                "mofid": str(row.get('完整MOFid', '')),
                "inorganic_sbu": str(row.get('无机建筑块 (Inorganic BB)', 'unknown')),
                "organic_smiles": str(row.get('有机建筑块 (Organic BB) SMILES格式', 'unknown')),
                "topology": topo,
                "catenation": cat,
                "metal": metal,
                "lcd": lcd,
                "pld": pld,
                "asa_m2_g": asa,
                "pore_vol": p_vol,
                "co2_uptake_1bar": co2_1bar,
                "co2_uptake_015bar": co2_015bar,
                "n2_uptake_1bar": n2_1bar,
                "co2_n2_selectivity_real": sel_real,
                "co2_n2_selectivity_henry": sel_henry,
                "qst_kj_mol": qst,
                "vsa_working_capacity": vsa_wc,
                "tsa_working_capacity": tsa_wc,
                "pe_vsa": pe_vsa,
                "q_tsa_regen": q_tsa_regen
            }
            
            # 生成信息密度极高的结构与性质双语卡片
            mof_info["bilingual_summary"] = (
                f"【MOF 材料】: {name} (拓扑: {topo}, 金属节点: {metal}, 穿插度: {cat})\n"
                f"  - 几何孔道: PLD = {pld:.2f} Å, LCD = {lcd:.2f} Å, 可访问表面积 ASA = {asa:.1f} m²/g, 孔体积 = {p_vol:.3f} cm³/g\n"
                f"  - 烟气捕集吸附性能: 1bar CO2容量 = {co2_1bar:.2f} mol/kg, 0.15bar烟气分压容量 = {co2_015bar:.2f} mol/kg, 1bar N2容量 = {n2_1bar:.3f} mol/kg\n"
                f"  - 分离与热力学参数: 实际CO2/N2选择性 = {sel_real:.1f}, Henry选择性 = {sel_henry:.1f}, CO2吸附热 Qst = {qst:.2f} kJ/mol\n"
                f"  - 工艺能耗指标: VSA工作容量 = {vsa_wc:.2f} mol/kg (寄生能 PE = {pe_vsa:.2f} kJ/mol CO2); TSA工作容量 = {tsa_wc:.2f} mol/kg (再生热 = {q_tsa_regen:.2f} kJ/mol)\n"
                f"  - 化学构造: SBU = {mof_info['inorganic_sbu']}, 配体 = {mof_info['organic_smiles']}"
            )
            self.mof_data_store[name] = mof_info
            
        print(f"  [✓] 成功加载 {len(self.mof_data_store)} 个 MOF 完整科学数据。")

    def _build_knowledge_graph(self):
        """构建异构科学知识图谱"""
        print("[*] 2/3 构建异构材料知识图谱 (NetworkX MultiDiGraph)...")
        for name, data in self.mof_data_store.items():
            self.kg.add_node(name, node_type="MOF", **data)
            
            metal = data['metal']
            self.kg.add_node(metal, node_type="Metal")
            self.kg.add_edge(name, metal, relation="HAS_METAL")
            
            topo = data['topology']
            self.kg.add_node(topo, node_type="Topology")
            self.kg.add_edge(name, topo, relation="HAS_TOPOLOGY")
            
            # 孔道筛分机理
            pld = data['pld']
            if pld < 3.5:
                regime = "Ultra_Microporous_Sieving (<3.5Å)"
            elif 3.5 <= pld <= 5.5:
                regime = "Optimal_CO2_Kinetic_Window (3.5-5.5Å)"
            else:
                regime = "Large_Porous_Thermodynamic (>5.5Å)"
            self.kg.add_node(regime, node_type="PoreRegime")
            self.kg.add_edge(name, regime, relation="IN_PORE_REGIME")
            
            # 工艺优选等级 (Win-Win / Top Tier)
            if data['co2_uptake_1bar'] >= 3.0 and data['co2_n2_selectivity_real'] >= 15.0:
                self.kg.add_node("High_Performance_Capture", node_type="PerformanceTier")
                self.kg.add_edge(name, "High_Performance_Capture", relation="TIER_1_PERFORMER")
                
            if data['pe_vsa'] > 0 and data['pe_vsa'] <= 25.0:
                self.kg.add_node("Low_Energy_VSA", node_type="EnergyTier")
                self.kg.add_edge(name, "Low_Energy_VSA", relation="ENERGY_EFFICIENT_VSA")

        print(f"  [✓] 知识图谱已构建: {self.kg.number_of_nodes()} 节点, {self.kg.number_of_edges()} 关系边。")

    def retrieve_multimodal_context(
        self,
        query: str,
        target_mof: Optional[str] = None,
        filter_metal: Optional[str] = None,
        filter_topology: Optional[str] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        candidate_scores = {}
        
        # 1. 结构向量余弦相似度检索
        if target_mof and target_mof in self.mof_names and self.struct_embeddings is not None:
            target_idx = self.mof_names.index(target_mof)
            target_vec = self.struct_embeddings[target_idx]
            sims = np.dot(self.struct_embeddings, target_vec) / (
                np.linalg.norm(self.struct_embeddings, axis=1) * np.linalg.norm(target_vec) + 1e-8
            )
            for idx, sim in enumerate(sims):
                candidate_scores[self.mof_names[idx]] = float(sim)
        else:
            # 基于材料科学指标综合评分
            for name, data in self.mof_data_store.items():
                score = 1.0
                q_lower = query.lower()
                # 容量加权
                score += data['co2_uptake_1bar'] * 0.8 + data['co2_uptake_015bar'] * 1.5
                # 选择性加权
                score += min(data['co2_n2_selectivity_real'], 80.0) * 0.05
                # 能耗与吸附热加权
                if data['pe_vsa'] > 0:
                    score += max(0, 40.0 - data['pe_vsa']) * 0.1
                candidate_scores[name] = score

        # 2. 知识图谱约束过滤与关联扩展
        retrieved = []
        for name, score in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True):
            if target_mof and name == target_mof:
                continue
            data = self.mof_data_store.get(name)
            if not data:
                continue
                
            if filter_metal and data['metal'].lower() != filter_metal.lower():
                continue
            if filter_topology and data['topology'].lower() != filter_topology.lower():
                continue
                
            neighbors = list(self.kg.neighbors(name))
            
            retrieved.append({
                "mof_name": name,
                "similarity_score": round(score, 4),
                "graph_neighbors": neighbors,
                "details": data,
                "summary": data['bilingual_summary']
            })
            if len(retrieved) >= top_k:
                break
                
        return retrieved

    def call_deepseek_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not system_prompt:
            system_prompt = (
                "You are an expert AI Materials Scientist and MOF specialist integrating Graph RAG with MatterChat paradigm (arXiv:2502.13107). "
                "You provide scientifically rigorous, data-grounded, and structured analysis for Metal-Organic Frameworks, "
                "focusing on post-combustion CO2 capture, structure-property relationships (QSAR), and adsorption mechanisms. "
                "Answer in clear, professional Chinese and English (bilingual friendly)."
            )
            
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
            "stream": False
        }
        
        req = urllib.request.Request(
            DEFAULT_API_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                res_body = response.read().decode('utf-8')
                res_json = json.loads(res_body)
                return res_json['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode('utf-8')
            if "model_not_found" in err_msg or "Invalid model" in err_msg or "404" in str(e.code):
                print(f"[!] 模型 {self.model_name} 暂不支持，自动降级为 deepseek-chat ...")
                payload["model"] = "deepseek-chat"
                req = urllib.request.Request(
                    DEFAULT_API_URL,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as fallback_res:
                    res_body = fallback_res.read().decode('utf-8')
                    return json.loads(res_body)['choices'][0]['message']['content']
            raise RuntimeError(f"DeepSeek API HTTP Error {e.code}: {err_msg}")
        except Exception as e:
            raise RuntimeError(f"DeepSeek API Request Failed: {e}")

    def query(self, user_question: str, target_mof: Optional[str] = None, top_k: int = 3) -> str:
        print(f"\n[?] 收到科学咨询: '{user_question}'")
        print("[*] 正在执行多模态 Graph RAG 检索 (结构相似度 + 图谱实体遍历)...")
        
        retrieved_items = self.retrieve_multimodal_context(user_question, target_mof=target_mof, top_k=top_k)
        
        context_blocks = []
        for i, item in enumerate(retrieved_items, 1):
            block = f"【候选材料 {i}】(结构相似/综合匹配分: {item['similarity_score']}):\n{item['summary']}\n知识图谱关联实体: {item['graph_neighbors']}"
            context_blocks.append(block)
        
        context_str = "\n\n".join(context_blocks)
        
        prompt = (
            f"用户提出的材料科学研究问题:\n\"{user_question}\"\n\n"
            f"=== MOF 多模态知识图谱与结构向量检索结果 (来自 252_MOF 真实数据库) ===\n"
            f"{context_str}\n\n"
            f"=== 任务指引 ===\n"
            f"请结合以上检索到的真实晶体几何参数（PLD/LCD/ASA）、金属节点配位化学、孔道尺寸筛分窗口以及吸附热Qst数据，"
            f"以材料化学家的专业视角给出严谨、条理清晰的分析与推荐：\n"
            f"1. 针对问题给出具体的优选材料推荐列表及其核心物理化学优势；\n"
            f"2. 从构效关系（孔径筛分效应、吸附热与再生能权衡）深入解释内在原因；\n"
            f"3. 总结结构拓扑与金属节点对烟气捕集分离性能的决定性规律。"
        )
        
        print("[*] 正在调用 DeepSeek LLM 进行结构-性质深度逻辑推理...")
        llm_response = self.call_deepseek_llm(prompt)
        return llm_response

if __name__ == "__main__":
    rag = MOFMultiModalGraphRAG()
    
    # 测试案例 1：针对烟气捕集的综合材料推荐
    test_query_1 = "请为燃煤电厂烟气CO2捕集（15% CO2/85% N2）推荐最优秀的 MOF 材料，要求高选择性且脱附再生能低，并深入解释其孔道构效关系与金属节点作用。"
    answer_1 = rag.query(test_query_1, top_k=3)
    print("\n" + "="*90)
    print("【DeepSeek Graph RAG 科学推理报告 - 案例 1】")
    print("="*90)
    print(answer_1)
    
    # 测试案例 2：寻找特定材料的相似结构替代物
    test_query_2 = "我想寻找与 ABAYIO_clean (Mn基 tbo拓扑) 结构类似且具备优异分离性能或更环保金属节点的替代 MOF。"
    answer_2 = rag.query(test_query_2, target_mof="ABAYIO_clean", top_k=3)
    print("\n" + "="*90)
    print("【DeepSeek Graph RAG 科学推理报告 - 案例 2】")
    print("="*90)
    print(answer_2)
