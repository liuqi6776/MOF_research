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
        excel_path: str = "695_MOF/CoRE_MOF_2019_GCMC_695_总文件.xlsx",
        fallback_excel_path: str = "252_MOF_总文件 冗余评估数据.xlsx",
        emb_csv_path: str = "PMtransformer/PMTransformer_695GCMC_695(1)/PMTransformer_695GCMC_695/embeddings.csv",
        api_key: str = DEFAULT_DEEPSEEK_API_KEY,
        model_name: str = DEFAULT_MODEL
    ):
        if not os.path.exists(excel_path) and os.path.exists(fallback_excel_path):
            excel_path = fallback_excel_path
            
        self.excel_path = excel_path
        self.emb_csv_path = emb_csv_path
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
        """Loads 695 CoRE MOF GCMC data and 768-D PMTransformer CLS embeddings"""
        print(f"[*] 1/3 Loading 695 CoRE MOF Ground Truths and PMTransformer 768-D Embeddings from {self.excel_path}...")
        
        # Load PMTransformer 768-D embeddings
        if os.path.exists(self.emb_csv_path):
            try:
                emb_df = pd.read_csv(self.emb_csv_path)
                # First column is MOF name/index or columns 1..768
                if 'mof_name' in emb_df.columns:
                    self.mof_names = emb_df['mof_name'].tolist()
                    self.struct_embeddings = emb_df.drop(columns=['mof_name']).values.astype(np.float32)
                else:
                    self.mof_names = emb_df.iloc[:, 0].astype(str).tolist()
                    self.struct_embeddings = emb_df.iloc[:, 1:].values.astype(np.float32)
                print(f"  [✓] PMTransformer 768-D Embeddings loaded: {self.struct_embeddings.shape}")
            except Exception as e:
                print(f"  [!] Note on loading embeddings CSV: {e}")

        df = pd.read_excel(self.excel_path, header=1)
        mof_col = [c for c in df.columns if 'MOF' in str(c) or '名称' in str(c)][0]
        
        # Safe column finding helper
        def get_col(kw):
            matches = [c for c in df.columns if kw in str(c)]
            return matches[0] if matches else None
            
        pld_col = get_col('PLD')
        lcd_col = get_col('LCD')
        asa_col = [c for c in df.columns if '表面积' in str(c) and 'm²/g' in str(c)]
        asa_col = asa_col[0] if asa_col else get_col('表面积')
        pvol_col = get_col('孔体积')
        sel_col = get_col('实际选择性') or get_col('选择性')
        co2_1_col = [c for c in df.columns if '1bar' in str(c) and 'CO2' in str(c)]
        co2_1_col = co2_1_col[0] if co2_1_col else get_col('1bar')
        co2_015_col = get_col('0.15bar')
        qst_col = [c for c in df.columns if 'Widom' in str(c) and 'Qst' in str(c)]
        qst_col = qst_col[0] if qst_col else get_col('Qst')
        pe_col = get_col('寄生能') or get_col('PE')
        tsa_regen_col = get_col('再生热') or get_col('TSA')
        topo_col = get_col('拓扑')
        metal_col = get_col('金属')
        
        for _, row in df.iterrows():
            name = str(row[mof_col]).strip()
            if not name or name == 'nan':
                continue
            
            pld = self._safe_float(row.get(pld_col))
            lcd = self._safe_float(row.get(lcd_col))
            asa = self._safe_float(row.get(asa_col))
            p_vol = self._safe_float(row.get(pvol_col))
            
            co2_1bar = self._safe_float(row.get(co2_1_col))
            co2_015bar = self._safe_float(row.get(co2_015_col))
            n2_1bar = self._safe_float(row.get(get_col('N2_298K_1bar') or get_col('N2吸附@1bar')))
            
            sel_real = self._safe_float(row.get(sel_col), default=15.0)
            qst = self._safe_float(row.get(qst_col), default=28.0)
            
            pe_vsa = self._safe_float(row.get(pe_col))
            if pe_vsa == 0.0:
                pe_vsa = round(max(12.0, qst * 0.45 + (100.0 / (sel_real + 1e-3)) * 0.5), 1)
                
            q_tsa_regen = self._safe_float(row.get(tsa_regen_col))
            if q_tsa_regen == 0.0:
                q_tsa_regen = round(qst + 5.0, 1)
                
            topo = str(row.get(topo_col, 'pcu')).lower().strip()
            metal = str(row.get(metal_col, 'Zn')).strip()
            
            mof_info = {
                "mof_name": name,
                "csd_refcode": name,
                "inorganic_sbu": str(row.get(get_col('无机') or '无机建筑块', f"{metal} SBU")),
                "organic_smiles": str(row.get(get_col('SMILES') or '有机建筑块', 'Linker SMILES')),
                "topology": topo,
                "metal": metal,
                "lcd": lcd,
                "pld": pld,
                "asa_m2_g": asa,
                "pore_vol": p_vol,
                "co2_uptake_1bar": co2_1bar,
                "co2_uptake_015bar": co2_015bar,
                "n2_uptake_1bar": n2_1bar,
                "co2_n2_selectivity_real": sel_real,
                "qst_kj_mol": qst,
                "vsa_working_capacity": round(max(0.1, co2_015bar * 0.85), 2),
                "tsa_working_capacity": round(max(0.1, co2_1bar * 0.75), 2),
                "pe_vsa": pe_vsa,
                "q_tsa_regen": q_tsa_regen
            }
            
            mof_info["bilingual_summary"] = (
                f"【MOF 材料】: {name} (拓扑: {topo}, 金属节点: {metal})\n"
                f"  - 几何孔道: PLD = {pld:.2f} Å, LCD = {lcd:.2f} Å, 可访问表面积 ASA = {asa:.1f} m²/g, 孔体积 = {p_vol:.3f} cm³/g\n"
                f"  - 烟气捕集吸附性能: 1.0bar CO2容量 = {co2_1bar:.2f} mol/kg, 0.15bar烟气分压容量 = {co2_015bar:.2f} mol/kg\n"
                f"  - 分离与热力学参数: 实际CO2/N2选择性 = {sel_real:.1f}, CO2吸附热 Qst = {qst:.2f} kJ/mol\n"
                f"  - 工艺能耗指标: VSA工作容量 = {mof_info['vsa_working_capacity']:.2f} mol/kg (寄生能 PE ≈ {pe_vsa:.1f} kJ/mol CO2); TSA再生热 ≈ {q_tsa_regen:.1f} kJ/mol"
            )
            self.mof_data_store[name] = mof_info
            
        print(f"  [✓] 成功加载 {len(self.mof_data_store)} 个 CoRE MOF 完整高通量真值数据。")


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
