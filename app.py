"""
MOF Chatbot - AI Materials Assistant & 3D Crystal Interactive Studio
Enhanced with Fused Multi-Modal QSAR ML Prediction & Inverse Modification Recommender
Port: 7860
"""
import os
import sys
import glob
import html
import json
import time
import numpy as np
import pandas as pd
import gradio as gr
from ase.io import read
from scripts.mof_graph_rag_engine import MOFMultiModalGraphRAG, DEFAULT_DEEPSEEK_API_KEY
from scripts.mof_property_predictor import MOFPropertyPredictor

print("[*] Initializing MOF Chatbot & ML Property Predictor...")
rag_engine = MOFMultiModalGraphRAG()
ml_predictor = MOFPropertyPredictor()

CIF_DIR = "252_MOF_CIFs"
ALL_CIF_FILES = sorted(glob.glob(os.path.join(CIF_DIR, "*.cif")))
MOF_NAME_TO_PATH = {os.path.splitext(os.path.basename(f))[0]: f for f in ALL_CIF_FILES}
MOF_CHOICES_EN = ["(Custom Upload / 自定义上传)"] + list(MOF_NAME_TO_PATH.keys())
MOF_CHOICES_ZH = ["(自定义上传 / Custom Upload)"] + list(MOF_NAME_TO_PATH.keys())

# 多语言国际化字典 (Default English)
I18N = {
    "en": {
        "title": "MOF Chatbot",
        "subtitle": "Your AI Materials Assistant for 3D Crystal Structures, QSAR Predictions & Inverse Design",
        "status_online": "● Online",
        "dataset_badge": "Knowledge Base: 252 MOFs + ML Predictor",
        "sec_structure": "📁 1. Choose or Upload MOF",
        "upload_label": "Upload CIF File (*.cif)",
        "dropdown_label": "Or Select from 252 MOF Library",
        "sec_prompt": "💬 2. Chat with Assistant",
        "prompt_label": "Your Question, Target Properties, or Modification Request",
        "prompt_placeholder": "e.g., Predict this uploaded MOF's CO2 capture performance, or suggest structural modifications to achieve higher selectivity (>30) and lower regeneration energy...",
        "default_prompt": "Can you predict this MOF's flue gas CO2 capture performance (0.15 bar uptake, selectivity, Qst), and provide structural modification suggestions to improve its separation efficiency and lower energy consumption?",
        "btn_preset_1": "⚡ Predict & Evaluate CO2 Capture",
        "btn_preset_2": "🛠️ Suggest Structural Modifications",
        "btn_preset_3": "🔍 Find Green Material Alternates",
        "preset_1_val": "Please evaluate this MOF for flue gas CO2 capture (15% CO2/85% N2). Predict its selectivity, working capacity, and parasitic energy.",
        "preset_2_val": "How can I modify this MOF structure (e.g. ligand functionalization, pore tuning, metal node replacement) to achieve higher CO2 selectivity and lower regeneration energy?",
        "preset_3_val": "Please recommend structurally similar MOF materials that use non-toxic, eco-friendly metal nodes with superior separation performance.",
        "accordion_settings": "⚙️ AI Model & Search Settings",
        "model_label": "AI Assistant Model",
        "model_choices": ["MOF Assistant (Standard)", "MOF Assistant (Deep Analysis)", "MOF Assistant (Fast)"],
        "top_k_label": "Number of Similar MOFs to Retrieve (Top-K)",
        "btn_submit": "🚀 Send / 提问",
        "tab_3d": "💬 Chat & 3D Crystal View",
        "tab_candidates": "🌐 Similar MOF Recommendations",
        "response_title": "### 🤖 MOF Chatbot Response",
        "initial_message": "*👋 Hello! Select or upload a MOF crystal on the left, type your question, and I'll analyze its structure, predict properties using our trained multi-modal ML model, and provide structural modification suggestions for you.*",
        "candidates_header": "#### 🔍 Matched Similar MOF Structures & Properties",
        "metrics_title": "Crystal & Pore Properties",
        "empty_cif": "⚛️ No MOF Crystal Selected<br><span style='font-size:12px;color:#64748b;'>Upload a CIF file or select one from the dropdown</span>",
        "viewer_hud_model": "3D Crystal View",
        "viewer_hud_note": "Ball & Stick • Unit Cell",
        "viewer_hud_ctrl": "🖱️ Left-Click: Rotate | Right-Click: Pan | Scroll: Zoom",
        "cell_params": "Lattice a, b, c:",
        "cell_angles": "Angles α, β, γ:",
        "cell_vol_den": "Volume / Density:",
        "atoms_elements": "Atoms / Elements:",
        "pld_lcd": "PLD / LCD Pore Size:",
        "asa_topo": "Surface Area / Topology:",
        "ml_pred_header": "⚡ ML Property Predictions (QSAR Model):",
        "lang_toggle_btn": "🌐 切换为中文 (Switch to Chinese)"
    },
    "zh": {
        "title": "MOF Chatbot",
        "subtitle": "您的材料智能科研助手 • 3D 晶体空间交互、性能即时预测与逆向结构优化",
        "status_online": "● 在线",
        "dataset_badge": "知识库: 252 MOF 材料 + ML 预测模型",
        "sec_structure": "📁 1. 选择或上传 MOF 晶体",
        "upload_label": "上传本地 CIF 文件 (*.cif)",
        "dropdown_label": "或从 252 数据库中选择材料",
        "sec_prompt": "💬 2. 向 MOF Chatbot 提问",
        "prompt_label": "输入您的问题、目标性能需求或结构调整要求",
        "prompt_placeholder": "例如：预测该新上传 MOF 的 CO2 捕集性能，或告诉我如何调整其配体与孔径以实现更高选择性（>30）和更低脱附能耗...",
        "default_prompt": "请预测该 MOF 在烟气 CO2 捕集（15% CO2/85% N2）下的各项性能指标（容量、选择性、Qst），并给出具体的结构调整与改性建议以优化其分离效率并降低能耗。",
        "btn_preset_1": "⚡ 预测并评估烟气 CO2 捕集",
        "btn_preset_2": "🛠️ 给出结构调整优化建议",
        "btn_preset_3": "🔍 寻找环保替代材料",
        "preset_1_val": "请评估并预测该材料在燃煤电厂烟气 CO2 捕集工况下的吸附分离表现，重点分析选择性、工作容量及寄生能耗。",
        "preset_2_val": "如何对当前 MOF 的结构（如配体官能团化、孔径微调、金属节点置换）进行针对性调整，以实现更高的 CO2 选择性与更低的再生能耗？",
        "preset_3_val": "我想寻找与当前选中材料结构类似、金属节点更绿色环保且分离性能优异的替代 MOF 材料。",
        "accordion_settings": "⚙️ 模型与检索设置",
        "model_label": "AI 助手模型",
        "model_choices": ["MOF Assistant (标准版)", "MOF Assistant (深度分析)", "MOF Assistant (极速版)"],
        "top_k_label": "推荐相似材料数量 (Top-K)",
        "btn_submit": "🚀 发送 / Send",
        "tab_3d": "💬 对话与 3D 晶体视图",
        "tab_candidates": "🌐 相似材料推荐",
        "response_title": "### 🤖 MOF Chatbot 回答",
        "initial_message": "*👋 您好！请在左侧选择或上传 MOF 晶体结构，输入您想了解的问题，我将为您深入解析其物理化学特征、运行多模态 ML 模型即时预测性质，并提供结构调整与改性方案。*",
        "candidates_header": "#### 🔍 匹配推荐的相似 MOF 材料与属性",
        "metrics_title": "晶体与孔道物理化学参数",
        "empty_cif": "⚛️ 暂未选择晶体结构<br><span style='font-size:12px;color:#64748b;'>请上传 CIF 文件或在下拉框中选择材料</span>",
        "viewer_hud_model": "3D 晶体空间模型",
        "viewer_hud_note": "周期性晶胞 • 球棍模型",
        "viewer_hud_ctrl": "🖱️ 鼠标左键旋转 | 右键平移 | 滚轮缩放",
        "cell_params": "晶胞常数 a, b, c:",
        "cell_angles": "晶胞夹角 α, β, γ:",
        "cell_vol_den": "晶胞体积 / 密度:",
        "atoms_elements": "原子数 / 元素种类:",
        "pld_lcd": "PLD / LCD 孔径:",
        "asa_topo": "比表面积 ASA / 拓扑:",
        "ml_pred_header": "⚡ 多模态 ML 模型即时预测结果:",
        "lang_toggle_btn": "🌐 Switch to English (切换为英文)"
    }
}

def generate_3d_viewer_html(cif_content: str, mof_name: str = "Structure View", lang: str = "en") -> str:
    texts = I18N.get(lang, I18N["en"])
    if not cif_content or not cif_content.strip():
        return f"""
        <div style="height: 480px; display: flex; align-items: center; justify-content: center; background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 16px; color: #64748b; font-family: sans-serif;">
            <div style="text-align: center; padding: 20px;">
                <p style="font-size: 15px; margin-bottom: 6px; color: #0284c7; font-weight: 600;">{texts['empty_cif']}</p>
            </div>
        </div>
        """
    
    js_safe_cif = cif_content.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    
    html_src = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; background: #f8fafc; font-family: 'Segoe UI', -apple-system, sans-serif; }}
        #container {{ width: 100%; height: 100%; position: relative; }}
        .hud-overlay {{
            position: absolute; top: 14px; left: 14px;
            background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(8px);
            border: 1px solid #bfdbfe; border-radius: 10px;
            padding: 8px 14px; color: #0f172a; font-size: 12px; pointer-events: none;
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.12);
        }}
        .hud-title {{ font-weight: 800; color: #0284c7; font-size: 13px; margin-bottom: 2px; display: flex; align-items: center; gap: 6px; }}
        .hud-controls {{
            position: absolute; bottom: 14px; right: 14px;
            background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(8px);
            border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 6px 12px; color: #64748b; font-size: 11px; font-family: monospace;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .glow-tag {{ background: #e0f2fe; border: 1px solid #bae6fd; color: #0369a1; padding: 2px 8px; border-radius: 6px; font-family: monospace; font-weight: 700; font-size: 11px; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div class="hud-overlay">
        <div class="hud-title"><span>⚛️</span> {texts['viewer_hud_model']}</div>
        <div style="margin-top: 3px;">Target: <span class="glow-tag">{mof_name}</span></div>
        <div style="font-size: 11px; color: #64748b; margin-top: 3px;">{texts['viewer_hud_note']}</div>
    </div>
    <div class="hud-controls">
        {texts['viewer_hud_ctrl']}
    </div>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            let element = document.getElementById("container");
            let config = {{ backgroundColor: "#f8fafc" }};
            let viewer = $3Dmol.createViewer(element, config);
            let cifData = `{js_safe_cif}`;
            
            viewer.addModel(cifData, "cif");
            viewer.setStyle({{}}, {{
                stick: {{ radius: 0.14, color: "spectrum" }},
                sphere: {{ scale: 0.24 }}
            }});
            viewer.addUnitCell({{box: {{color: "#0284c7", radius: 0.05}}}});
            viewer.zoomTo();
            viewer.render();
            
            viewer.spin(true);
            element.addEventListener('mousedown', function() {{ viewer.spin(false); }});
        }});
    </script>
</body>
</html>"""
    
    escaped_srcdoc = html.escape(html_src, quote=True)
    return f'<iframe srcdoc="{escaped_srcdoc}" style="width: 100%; height: 480px; border: 1px solid #e2e8f0; border-radius: 16px; box-shadow: 0 4px 20px -2px rgba(2, 132, 199, 0.08);" frameborder="0"></iframe>'

def get_crystal_metrics_card(cif_path: str, mof_name: str, lang: str = "en") -> str:
    texts = I18N.get(lang, I18N["en"])
    try:
        atoms = read(cif_path)
        cell = atoms.cell.cellpar()
        vol = atoms.get_volume()
        mass = np.sum(atoms.get_masses())
        density = mass / (vol + 1e-6)
        n_atoms = len(atoms)
        symbols = list(set(atoms.get_chemical_symbols()))
        
        is_in_database = mof_name in rag_engine.mof_data_store
        known_data = rag_engine.mof_data_store.get(mof_name, {})
        
        # 运行 ML 预测模型
        ml_res = ml_predictor.predict_properties(cif_path)
        preds = ml_res['predictions']
        
        if is_in_database:
            pld = f"{known_data.get('pld', 0.0):.2f}"
            lcd = f"{known_data.get('lcd', 0.0):.2f}"
            asa = f"{known_data.get('asa_m2_g', 0.0):.1f}"
            topo = known_data.get('topology', 'custom')
            badge_text = f"{mof_name} [GCMC Evaluated]"
            badge_style = "background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8;"
            
            p_co2_015 = f"{known_data.get('co2_uptake_015bar', 0.0):.2f}"
            p_co2_1 = f"{known_data.get('co2_uptake_1bar', 0.0):.2f}"
            p_sel = f"{known_data.get('co2_n2_selectivity_real', 0.0):.1f}"
            p_qst = f"{known_data.get('qst_kj_mol', 0.0):.1f}"
            p_pe = f"{known_data.get('pe_vsa', 0.0):.1f}"
        else:
            pld = f"{ml_res['features']['pld_est']:.2f}"
            lcd = f"{ml_res['features']['lcd_est']:.2f}"
            asa = f"{ml_res['features']['asa_est']:.1f}"
            topo = "Novel CIF"
            badge_text = f"{mof_name} [⚡ AI Predicted]"
            badge_style = "background: #fdf4ff; border: 1px solid #f0abfc; color: #a21caf;"
            
            p_co2_015 = f"{preds['co2_015bar']['value']:.2f}"
            p_co2_1 = f"{preds['co2_1bar']['value']:.2f}"
            p_sel = f"{preds['selectivity_real']['value']:.1f}"
            p_qst = f"{preds['qst_widom']['value']:.1f}"
            p_pe = f"{preds['pe_vsa']['value']:.1f}"
            
        return f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 18px; color: #0f172a; box-shadow: 0 4px 20px -2px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px;">
                <span style="font-size: 14px; font-weight: 800; color: #0284c7; display: flex; align-items: center; gap: 8px;">
                    <span>📊</span> {texts['metrics_title']}
                </span>
                <span style="{badge_style} padding: 3px 10px; border-radius: 6px; font-size: 11px; font-family: monospace; font-weight: 700;">
                    {badge_text}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 12px;">
                <div style="background: #f8fafc; padding: 8px 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <span style="color: #64748b; font-size: 11px;">{texts['cell_params']}</span><br>
                    <b style="color: #0f172a; font-family: monospace;">{cell[0]:.2f}, {cell[1]:.2f}, {cell[2]:.2f} Å</b>
                </div>
                <div style="background: #f8fafc; padding: 8px 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <span style="color: #64748b; font-size: 11px;">{texts['cell_angles']}</span><br>
                    <b style="color: #0f172a; font-family: monospace;">{cell[3]:.1f}°, {cell[4]:.1f}°, {cell[5]:.1f}°</b>
                </div>
                <div style="background: #f8fafc; padding: 8px 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <span style="color: #64748b; font-size: 11px;">{texts['cell_vol_den']}</span><br>
                    <b style="color: #0284c7; font-family: monospace;">{vol:.1f} Å³ / {density:.2f} g/cm³</b>
                </div>
                <div style="background: #f8fafc; padding: 8px 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <span style="color: #64748b; font-size: 11px;">{texts['atoms_elements']}</span><br>
                    <b style="color: #0f172a; font-family: monospace;">{n_atoms} ({', '.join(symbols[:4])})</b>
                </div>
                <div style="background: #f8fafc; padding: 8px 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <span style="color: #64748b; font-size: 11px;">{texts['pld_lcd']}</span><br>
                    <b style="color: #0369a1; font-family: monospace;">{pld} / {lcd} Å</b>
                </div>
                <div style="background: #f8fafc; padding: 8px 10px; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <span style="color: #64748b; font-size: 11px;">{texts['asa_topo']}</span><br>
                    <b style="color: #0284c7; font-family: monospace;">{asa} m²/g ({topo})</b>
                </div>
            </div>
            
            <!-- ML 预测/真实性能指标面板 -->
            <div style="margin-top: 12px; background: linear-gradient(135deg, #f0f9ff 0%, #f8fafc 100%); border: 1px solid #bae6fd; border-radius: 10px; padding: 10px 14px;">
                <div style="font-size: 11px; font-weight: 700; color: #0369a1; margin-bottom: 6px;">
                    {texts['ml_pred_header']}
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 11px;">
                    <div>0.15 bar CO2: <b style="color: #0284c7;">{p_co2_015} mol/kg</b></div>
                    <div>1.0 bar CO2: <b style="color: #0284c7;">{p_co2_1} mol/kg</b></div>
                    <div>CO2/N2 Selectivity: <b style="color: #0284c7;">{p_sel}</b></div>
                    <div>Adsorption Heat Qst: <b style="color: #d97706;">{p_qst} kJ/mol</b></div>
                    <div>VSA Energy PE: <b style="color: #10b981;">{p_pe} kJ/mol</b></div>
                    <div>Model: <b style="color: #64748b;">Fused QSAR (R²=0.68)</b></div>
                </div>
            </div>
        </div>
        """
    except Exception as e:
        return f"<div style='color: #ef4444; padding: 12px; background: #fef2f2; border-radius: 8px;'>Error: {e}</div>"

def handle_cif_selection(dropdown_val, file_obj, lang="en"):
    cif_path = None
    mof_name = "Custom MOF"
    
    if file_obj is not None:
        cif_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        mof_name = os.path.splitext(os.path.basename(cif_path))[0]
    elif dropdown_val and not dropdown_val.startswith("("):
        mof_name = dropdown_val
        cif_path = MOF_NAME_TO_PATH.get(dropdown_val)
        
    if not cif_path or not os.path.exists(cif_path):
        return generate_3d_viewer_html("", lang=lang), f"<div style='color: #64748b; padding: 10px;'>{I18N[lang]['empty_cif']}</div>", None
        
    with open(cif_path, "r", encoding="utf-8", errors="ignore") as f:
        cif_text = f.read()
        
    viewer_html = generate_3d_viewer_html(cif_text, mof_name, lang=lang)
    metrics_html = get_crystal_metrics_card(cif_path, mof_name, lang=lang)
    return viewer_html, metrics_html, cif_path

def execute_chat_query(
    cif_path_state,
    user_prompt,
    model_choice,
    top_k,
    lang="en"
):
    texts = I18N.get(lang, I18N["en"])
    if not user_prompt or not user_prompt.strip():
        return "⚠️ Please enter your question / 请输入您的问题。", ""
        
    internal_model = "deepseek-chat"
    if "Deep" in model_choice or "深度" in model_choice:
        internal_model = "deepseek-reasoner"
    elif "Fast" in model_choice or "极速" in model_choice:
        internal_model = "deepseek-chat"
        
    rag_engine.model_name = internal_model
    
    target_mof_name = None
    predicted_context = ""
    modification_context = ""
    
    if cif_path_state and os.path.exists(cif_path_state):
        base_name = os.path.splitext(os.path.basename(cif_path_state))[0]
        target_mof_name = base_name
        
        # 实时调用 ML 预测模型与结构调整规则引擎
        try:
            ml_res = ml_predictor.predict_properties(cif_path_state)
            preds = ml_res['predictions']
            p_dict = {
                'pld': ml_res['features']['pld_est'],
                'asa_m2_g': ml_res['features']['asa_est'],
                'co2_n2_selectivity_real': preds['selectivity_real']['value'],
                'qst_kj_mol': preds['qst_widom']['value'],
                'pe_vsa': preds['pe_vsa']['value']
            }
            predicted_context = (
                f"【ML Model Predicted Properties for this Structure】:\n"
                f"- CO2 uptake @ 0.15 bar: {preds['co2_015bar']['value']} mol/kg (CV R2: {preds['co2_015bar']['r2_cv']:.2f})\n"
                f"- CO2 uptake @ 1.0 bar: {preds['co2_1bar']['value']} mol/kg (CV R2: {preds['co2_1bar']['r2_cv']:.2f})\n"
                f"- CO2/N2 Actual Selectivity: {preds['selectivity_real']['value']} (CV R2: {preds['selectivity_real']['r2_cv']:.2f})\n"
                f"- CO2 Adsorption Heat Qst: {preds['qst_widom']['value']} kJ/mol\n"
                f"- VSA Working Capacity: {preds['vsa_working_capacity']['value']} mol/kg, Parasitic Energy: {preds['pe_vsa']['value']} kJ/mol CO2\n"
            )
            
            # 生成逆向改性规则
            rules = ml_predictor.generate_modification_rules(p_dict)
            rules_str = "\n".join([f"- {r['dimension']}: Current state={r['current']}. Recommended action={r['action']} -> Target={r['target']}" for r in rules])
            modification_context = f"【AI Structural Modification & Optimization Rules】:\n{rules_str}\n"
        except Exception as e:
            print(f"[!] ML prediction error: {e}")
            
    # 检索数据库类似物
    retrieved = rag_engine.retrieve_multimodal_context(
        query=user_prompt,
        target_mof=target_mof_name if target_mof_name in rag_engine.mof_names else None,
        top_k=int(top_k)
    )
    
    cards_html = """<div style="display: flex; flex-direction: column; gap: 14px;">"""
    for i, item in enumerate(retrieved, 1):
        d = item['details']
        cards_html += f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #0284c7; border-radius: 12px; padding: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 800; color: #0f172a; font-size: 14px; display: flex; align-items: center; gap: 6px;">
                    <span style="color: #0284c7;">#{i}</span> {item['mof_name']}
                </span>
                <span style="background: #eff6ff; border: 1px solid #bfdbfe; color: #0369a1; font-size: 11px; padding: 2px 10px; border-radius: 20px; font-family: monospace; font-weight: 700;">
                    Similarity: {item['similarity_score']}
                </span>
            </div>
            <div style="font-size: 12px; color: #475569; display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 8px;">
                <div style="background: #f8fafc; padding: 6px 8px; border-radius: 6px; border: 1px solid #f1f5f9;">Topology / Metal: <b style="color: #0f172a;">{d['topology']} ({d['metal']})</b></div>
                <div style="background: #f8fafc; padding: 6px 8px; border-radius: 6px; border: 1px solid #f1f5f9;">PLD / LCD: <b style="color: #0f172a;">{d['pld']:.2f} / {d['lcd']:.2f} Å</b></div>
                <div style="background: #f8fafc; padding: 6px 8px; border-radius: 6px; border: 1px solid #f1f5f9;">Surface Area (ASA): <b style="color: #0f172a;">{d['asa_m2_g']:.1f} m²/g</b></div>
                <div style="background: #f8fafc; padding: 6px 8px; border-radius: 6px; border: 1px solid #f1f5f9;">1bar CO2 Uptake: <b style="color: #0284c7;">{d['co2_uptake_1bar']:.2f} mol/kg</b></div>
                <div style="background: #f8fafc; padding: 6px 8px; border-radius: 6px; border: 1px solid #f1f5f9;">CO2/N2 Selectivity: <b style="color: #0284c7;">{d['co2_n2_selectivity_real']:.1f}</b></div>
                <div style="background: #f8fafc; padding: 6px 8px; border-radius: 6px; border: 1px solid #f1f5f9;">Qst / VSA Energy: <b style="color: #d97706;">{d['qst_kj_mol']:.1f} kJ / {d['pe_vsa']:.1f} kJ</b></div>
            </div>
        </div>
        """
    cards_html += "</div>"
    
    context_str = "\n\n".join([
        f"【Candidate {i}】({item['mof_name']}, Match Score: {item['similarity_score']}):\n{item['summary']}\nKG Entities: {item['graph_neighbors']}"
        for i, item in enumerate(retrieved, 1)
    ])
    
    system_prompt = (
        "You are MOF Chatbot, an elite AI Materials Assistant specialized in multi-modal crystal structure analysis, "
        "machine learning property predictions (QSAR), and target-driven inverse structural design. "
        "When the user uploads a CIF, evaluate its structure, explain its predicted properties, "
        "and provide concrete, synthetic-friendly structural modification suggestions (e.g. functionalization, pore gating, open metal sites, catenation). "
        f"Please reply in {'English' if lang == 'en' else 'Chinese (with professional scientific clarity)'}."
    )
    
    prompt = (
        f"User Inquiry:\n\"{user_prompt}\"\n\n"
        f"Target Structure: {target_mof_name}\n"
        f"{predicted_context}\n"
        f"{modification_context}\n"
        f"=== Multi-Modal Knowledge Context & Top Matched Similar MOFs (252 CoRE MOF Library) ===\n"
        f"{context_str}\n\n"
        f"=== Response Guidelines ===\n"
        f"Provide a clear, structured, and scientifically grounded response:\n"
        f"1. Property Evaluation & Predictions: Clearly summarize the performance metrics (whether from database ground truth or ML predictions);\n"
        f"2. Structure-Property Insights: Explain why this structure behaves this way based on pore limiting diameter (PLD), accessible surface area (ASA), and coordination chemistry;\n"
        f"3. Concrete Modification Recommendations: If the user asks for optimization or improvements, give specific structural adjustment steps (ligand decoration, transmetalation, pore tailoring) to achieve the target properties."
    )
    
    try:
        llm_response = rag_engine.call_deepseek_llm(prompt, system_prompt=system_prompt)
    except Exception as e:
        llm_response = f"❌ Assistant Error: {e}"
        
    return llm_response, cards_html

# ----------------- 现代纯白学术风 UI 样式 -----------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

body, .gradio-container {
    background: #f8fafc !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(2, 132, 199, 0.05) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(37, 99, 235, 0.05) 0px, transparent 50%) !important;
    color: #0f172a !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

footer {
    display: none !important;
    visibility: hidden !important;
}
.gradio-container footer {
    display: none !important;
}

.gr-panel, .gr-box, .gr-form {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.04) !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25) !important;
    transition: all 0.2s ease !important;
}

.gr-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4) !important;
}

.gr-button-secondary {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #334155 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.gr-button-secondary:hover {
    background: #f0f9ff !important;
    border-color: #0284c7 !important;
    color: #0284c7 !important;
}

.tab-nav button.selected {
    border-bottom: 2px solid #0284c7 !important;
    color: #0284c7 !important;
    font-weight: 700 !important;
}
"""

with gr.Blocks(title="MOF Chatbot - AI Materials Assistant", css=custom_css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    current_lang = gr.State(value="en")
    active_cif_path = gr.State(value=None)
    
    header_html = gr.HTML(f"""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 18px 26px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 4px 20px -2px rgba(0,0,0,0.05);">
        <div style="display: flex; align-items: center; gap: 16px;">
            <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #0284c7, #2563eb); display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);">
                ⚛️
            </div>
            <div>
                <h1 style="margin: 0; font-size: 22px; font-weight: 800; color: #0f172a; letter-spacing: -0.3px; display: flex; align-items: center; gap: 10px;">
                    <span style="color: #0284c7;">MOF Chatbot</span>
                    <span style="font-size: 11px; background: #e0f2fe; border: 1px solid #bae6fd; color: #0369a1; padding: 2px 10px; border-radius: 9999px; font-weight: 600;">AI Assistant & QSAR</span>
                </h1>
                <p id="header_sub" style="margin: 3px 0 0 0; font-size: 12px; color: #64748b;">
                    {I18N['en']['subtitle']}
                </p>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 18px;">
            <div style="text-align: right; font-family: monospace; font-size: 12px;">
                <div style="color: #64748b;">{I18N['en']['dataset_badge']}</div>
                <div style="color: #10b981; font-weight: 600;">{I18N['en']['status_online']}</div>
            </div>
        </div>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Row():
                lang_toggle_btn = gr.Button(I18N['en']['lang_toggle_btn'], size="sm", variant="secondary")

            with gr.Group():
                sec_1_title = gr.Markdown(f"### {I18N['en']['sec_structure']}")
                cif_file_input = gr.File(label=I18N['en']['upload_label'], file_types=[".cif"], type="filepath")
                cif_preset_dropdown = gr.Dropdown(
                    label=I18N['en']['dropdown_label'],
                    choices=MOF_CHOICES_EN,
                    value="ABAYIO_clean"
                )
                
            with gr.Group():
                sec_2_title = gr.Markdown(f"### {I18N['en']['sec_prompt']}")
                user_prompt_input = gr.Textbox(
                    label=I18N['en']['prompt_label'],
                    placeholder=I18N['en']['prompt_placeholder'],
                    lines=4,
                    value=I18N['en']['default_prompt']
                )
                
                with gr.Row():
                    btn_preset_1 = gr.Button(I18N['en']['btn_preset_1'], size="sm", variant="secondary")
                    btn_preset_2 = gr.Button(I18N['en']['btn_preset_2'], size="sm", variant="secondary")
                btn_preset_3 = gr.Button(I18N['en']['btn_preset_3'], size="sm", variant="secondary")
                    
            with gr.Accordion(I18N['en']['accordion_settings'], open=False) as accordion_settings:
                model_selector = gr.Dropdown(
                    label=I18N['en']['model_label'],
                    choices=I18N['en']['model_choices'],
                    value=I18N['en']['model_choices'][0]
                )
                top_k_slider = gr.Slider(label=I18N['en']['top_k_label'], minimum=1, maximum=10, value=3, step=1)

            submit_btn = gr.Button(I18N['en']['btn_submit'], variant="primary", size="lg")

        with gr.Column(scale=8):
            with gr.Tabs() as tabs_group:
                with gr.TabItem(I18N['en']['tab_3d']) as tab_3d_item:
                    with gr.Row():
                        with gr.Column(scale=6):
                            viewer_output = gr.HTML(label="3D Crystal Viewer", value=generate_3d_viewer_html("", lang="en"))
                        with gr.Column(scale=6):
                            metrics_output = gr.HTML(label="Descriptors", value="<div style='color: #64748b; padding: 12px;'>Select a material to inspect descriptors</div>")

                    response_header = gr.Markdown(I18N['en']['response_title'])
                    chat_output = gr.Markdown(value=I18N['en']['initial_message'])

                with gr.TabItem(I18N['en']['tab_candidates']) as tab_candidates_item:
                    candidates_header_md = gr.Markdown(I18N['en']['candidates_header'])
                    candidates_output = gr.HTML(value="<div style='color: #64748b; padding: 16px;'>No recommendations yet.</div>")

    def toggle_language(lang_curr):
        new_lang = "zh" if lang_curr == "en" else "en"
        t = I18N[new_lang]
        
        hdr = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 18px 26px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 4px 20px -2px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; gap: 16px;">
                <div style="width: 44px; height: 44px; border-radius: 12px; background: linear-gradient(135deg, #0284c7, #2563eb); display: flex; align-items: center; justify-content: center; font-size: 24px; color: white; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);">
                    ⚛️
                </div>
                <div>
                    <h1 style="margin: 0; font-size: 22px; font-weight: 800; color: #0f172a; letter-spacing: -0.3px; display: flex; align-items: center; gap: 10px;">
                        <span style="color: #0284c7;">MOF Chatbot</span>
                        <span style="font-size: 11px; background: #e0f2fe; border: 1px solid #bae6fd; color: #0369a1; padding: 2px 10px; border-radius: 9999px; font-weight: 600;">AI Assistant & QSAR</span>
                    </h1>
                    <p style="margin: 3px 0 0 0; font-size: 12px; color: #64748b;">
                        {t['subtitle']}
                    </p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 18px;">
                <div style="text-align: right; font-family: monospace; font-size: 12px;">
                    <div style="color: #64748b;">{t['dataset_badge']}</div>
                    <div style="color: #10b981; font-weight: 600;">{t['status_online']}</div>
                </div>
            </div>
        </div>
        """
        
        return (
            new_lang,
            hdr,
            t['lang_toggle_btn'],
            f"### {t['sec_structure']}",
            gr.update(label=t['upload_label']),
            gr.update(label=t['dropdown_label'], choices=MOF_CHOICES_ZH if new_lang == 'zh' else MOF_CHOICES_EN),
            f"### {t['sec_prompt']}",
            gr.update(label=t['prompt_label'], placeholder=t['prompt_placeholder'], value=t['default_prompt']),
            t['btn_preset_1'],
            t['btn_preset_2'],
            t['btn_preset_3'],
            gr.update(label=t['accordion_settings']),
            gr.update(label=t['model_label'], choices=t['model_choices'], value=t['model_choices'][0]),
            gr.update(label=t['top_k_label']),
            t['btn_submit'],
            t['response_title'],
            t['candidates_header']
        )

    lang_toggle_btn.click(
        fn=toggle_language,
        inputs=[current_lang],
        outputs=[
            current_lang,
            header_html,
            lang_toggle_btn,
            sec_1_title,
            cif_file_input,
            cif_preset_dropdown,
            sec_2_title,
            user_prompt_input,
            btn_preset_1,
            btn_preset_2,
            btn_preset_3,
            accordion_settings,
            model_selector,
            top_k_slider,
            submit_btn,
            response_header,
            candidates_header_md
        ]
    )

    cif_preset_dropdown.change(
        fn=handle_cif_selection,
        inputs=[cif_preset_dropdown, cif_file_input, current_lang],
        outputs=[viewer_output, metrics_output, active_cif_path]
    )
    cif_file_input.upload(
        fn=handle_cif_selection,
        inputs=[cif_preset_dropdown, cif_file_input, current_lang],
        outputs=[viewer_output, metrics_output, active_cif_path]
    )
    
    btn_preset_1.click(
        fn=lambda lang: I18N[lang]["preset_1_val"],
        inputs=[current_lang],
        outputs=[user_prompt_input]
    )
    btn_preset_2.click(
        fn=lambda lang: I18N[lang]["preset_2_val"],
        inputs=[current_lang],
        outputs=[user_prompt_input]
    )
    btn_preset_3.click(
        fn=lambda lang: I18N[lang]["preset_3_val"],
        inputs=[current_lang],
        outputs=[user_prompt_input]
    )
    
    submit_btn.click(
        fn=execute_chat_query,
        inputs=[active_cif_path, user_prompt_input, model_selector, top_k_slider, current_lang],
        outputs=[chat_output, candidates_output]
    )
    
    demo.load(
        fn=lambda: handle_cif_selection("ABAYIO_clean", None, "en"),
        inputs=[],
        outputs=[viewer_output, metrics_output, active_cif_path]
    )

if __name__ == "__main__":
    print("[*] Launching MOF Chatbot (Port: 7860)...", flush=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        show_api=False,
        prevent_thread_lock=True
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        demo.close()
