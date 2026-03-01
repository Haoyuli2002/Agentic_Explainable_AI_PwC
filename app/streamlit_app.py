import streamlit as st
import matplotlib
# CRITICAL: Set backend to non-interactive Agg before importing pyplot from ANY module
matplotlib.use('Agg')

import pandas as pd
import os
import sys
import re
import tempfile
from dotenv import load_dotenv
import uuid
from catboost import CatBoostClassifier
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import torch

# Load environment variables
load_dotenv()

# Add parent directory to path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import app as agent_app

# Import custom architectures for unpickling
from model.lstm import LoanDefaultModel
import collections

# ─────────────────────────────────────────────────────────────────────────────
# BILINGUAL TEXT DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────
TEXTS = {
    "zh": {
        # Page
        "page_title": "AI可解释性智能体",
        "app_title": "🤖 AI可解释性与公平性分析智能体",
        "lang_btn": "🌐 English",

        # Sidebar
        "sidebar_header": "配置",
        "data_model_status": "**数据与模型**",
        "dataset_label": "数据集",
        "model_label": "模型",
        "not_loaded": "未加载",
        "custom_file": "自定义文件",

        # Session Memory
        "session_memory": "会话记忆",
        "current_thread": "**当前会话ID**",
        "resume_input": "恢复历史会话（输入会话ID）:",
        "resume_btn": "恢复会话",

        # Upload
        "upload_expander": "上传自定义文件",
        "upload_csv": "数据集（CSV）",
        "upload_npz": "填充后的3D数据（.npz）",
        "upload_model": "模型文件（.cbm, .pt, .pth）",
        "upload_csv_ok": "✅ 自定义数据集加载成功！",
        "upload_csv_err": "错误",
        "upload_npz_ok": "✅ 自定义3D填充数据加载成功！维度",
        "upload_npz_err": "加载NPZ文件出错",
        "upload_cbm_ok": "✅ 自定义CatBoost模型加载成功！",
        "upload_pt_ok": "✅ 自定义PyTorch模型加载成功！",
        "upload_model_err": "加载模型出错",

        # Dataset Meta
        "dataset_meta": "数据集元数据",
        "target_var": "目标变量",
        "problem_type": "问题类型",
        "dataset_preview": "数据集预览",
        "reset_btn": "重置会话",

        # Auth warning
        "api_warning": "请确保您的 OPENAI_API_KEY 已正确配置在 .env 文件中。",

        # Tabs
        "tab_intro": "项目介绍",
        "tab_workflow": "典型工作流",
        "tab_usecase": "应用场景",
        "tab_chat": "智能对话",

        # Tab 1: Introduction
        "t1_header": "项目概述",
        "t1_welcome": "欢迎使用<strong>智能可解释人工智能</strong>平台。本平台借助多智能体系统，帮助您理解数据、解释机器学习模型，并分析模型的公平性。",
        "t1_sub1": "1. AI 智能体系统架构",
        "t1_img_caption": "AI 智能体系统架构图",
        "t1_arch_desc": """本系统基于<strong>LangGraph</strong>构建多智能体架构，协调各专业智能体处理机器学习可解释性工作流的不同阶段：
            <ul style="margin-top: 10px;">
                <li><strong>路由智能体（Router Agent）：</strong>作为入口，分析您的请求并决定将其路由至哪个专业智能体。</li>
                <li><strong>数据理解智能体（Data Understanding Agent）：</strong>分析上传的数据集，提供摘要并预览数据样本。</li>
                <li><strong>全局解释智能体（Global Explainer Agent）：</strong>提供模型在整个数据集上行为的宏观洞察（如整体特征重要性）。</li>
                <li><strong>局部解释智能体（Local Explainer Agent）：</strong>解释模型对特定单个样本的预测原因。</li>
                <li><strong>伦理与公平性分析智能体（Ethic &amp; Fairness Agent）：</strong>评估模型的潜在偏见，并计算不同人口群体的公平性指标。</li>
            </ul>""",
        "t1_sub2": "2. 各智能体及其工具说明",
        "t1_router_title": "#### 🧭 路由智能体（Router Agent）",
        "t1_router_body": """<ul style="margin-top: 0px;">
                <li><strong>职责：</strong>充当"流量控制器"。读取您的提示并设置 <code>analysis_mode</code>（如 global、local、fairness、data_understanding），将对话路由到正确的智能体。</li>
                <li><strong>工具：</strong>无。依赖大语言模型的理解能力对意图进行分类。</li>
            </ul>""",
        "t1_data_title": "#### 📊 数据理解智能体（Data Understanding Agent）",
        "t1_data_body": """<ul style="margin-top: 0px;">
                <li><strong>职责：</strong>帮助您探索上传的数据集。</li>
                <li><strong>工具：</strong>
                    <ul>
                        <li><code>get_dataset_samples</code>：获取数据集的前几行，用于快速预览。</li>
                        <li><code>update_metadata</code>：允许智能体在理解有误时以编程方式更新目标变量或问题类型。</li>
                    </ul>
                </li>
            </ul>""",
        "t1_global_title": "#### 🌍 全局解释智能体（Global Explainer Agent）",
        "t1_global_body": """<ul style="margin-top: 0px;">
                <li><strong>职责：</strong>解释模型的整体逻辑——哪些特征在全局层面最为重要？</li>
                <li><strong>工具：</strong>
                    <ul>
                        <li><code>get_global_feature_importance_shap</code>：使用 <strong>SHAP（SHapley加性解释）</strong> 生成全局特征重要性和汇总图。</li>
                    </ul>
                </li>
            </ul>""",
        "t1_local_title": "#### 🔎 局部解释智能体（Local Explainer Agent）",
        "t1_local_body": """<ul style="margin-top: 0px;">
                <li><strong>职责：</strong>解释模型为何对某个<em>特定</em>样本（行）做出该预测。</li>
                <li><strong>工具：</strong>
                    <ul>
                        <li><code>run_shap_explanation</code>：为特定预测生成SHAP瀑布图。</li>
                        <li><code>run_lime_explanation</code>：使用 <strong>LIME（局部可解释模型无关解释）</strong> 通过局部近似解释特定预测。</li>
                    </ul>
                </li>
            </ul>""",
        "t1_ethic_title": "#### ⚖️ 伦理与公平性分析智能体（Ethic & Fairness Agent）",
        "t1_ethic_body": """<ul style="margin-top: 0px;">
                <li><strong>职责：</strong>评估模型对敏感属性（如性别、种族）是否公平、无偏见。</li>
                <li><strong>工具：</strong>
                    <ul>
                        <li><code>run_ethic_analysis</code>：使用 <strong>Fairlearn</strong> 计算统计均等差异（SPD）、机会均等差异（EOD）和差异影响（DI）。</li>
                        <li><code>visualize_ethic_analysis</code>：生成条形图，可视化不同群体在公平性指标上的差异。</li>
                    </ul>
                </li>
            </ul>""",

        # Tab 2: Workflow
        "t2_header": "典型工作流",
        "t2_intro": "为充分发挥智能可解释AI平台的效能，我们建议按以下工作流程操作：",
        "t2_steps": """<ol style="margin-top: 10px;">
                <li><strong>上传数据与模型：</strong>使用侧边栏上传数据集（CSV 或 ARFF）和已训练模型（CatBoost <code>.cbm</code>，PyTorch LSTM <code>.pth</code>）。时序模型还需上传 <code>.npz</code>。</li>
                <li><strong>数据理解：</strong>前往<strong>智能对话</strong>标签页，提问「描述一下这个数据集」或「数据的格式是什么？」</li>
                <li><strong>全局解释：</strong>提问「展示全局特征重要性」。</li>
                <li><strong>局部解释：</strong>提问「客户10001的预测结果是什么？原因是什么？」</li>
                <li><strong>伦理分析：</strong>提问「按性别检查公平性」。</li>
            </ol>""",
        "t2_examples_title": "💡 示例问题推荐",
        "t2_examples": """<ul style="margin-top: 10px;">
                <li><em>「描述一下这个数据集」</em></li>
                <li><em>「展示全局特征重要性」</em></li>
                <li><em>「客户10001的预测结果是什么？原因是什么？」</em></li>
                <li><em>「按性别检查公平性问题」</em></li>
                <li><em>「哪些特征对贷款违约风险影响最大？」</em></li>
            </ul>""",

        # Tab 3: Use Case
        "t3_header": "应用场景：贷款违约预测",
        "t3_intro": "本应用场景展示了一个现实世界中AI驱动的决策直接影响人们生活的案例——也正因如此，可解释性不是锦上添花，而是不可或缺的基础要素。",
        "t3_prob_title": "🏦 问题背景",
        "t3_prob_body": """一位客户向其银行申请商业贷款。经过短暂等待后，他们收到了一封自动拒绝邮件。
            当客户联系银行寻求说明时，被告知：<br><br>
            <blockquote style="border-left: 4px solid #e74c3c; background: rgba(231,76,60,0.08); border-radius: 0 6px 6px 0; padding: 12px 16px; font-style: italic; margin: 12px 0;">
                「该决定由内部机器学习模型作出，我们无法提供进一步的详细信息。」
            </blockquote>
            客户对被拒绝的原因一无所知，既无机会纠正数据中的错误，也无申诉途径。
            这种透明度的缺失可能违反《欧盟人工智能法》和GDPR第22条规定的<strong>解释权</strong>。""",
        "t3_sol_title": "💡 我们的解决方案",
        "t3_sol_body": """本项目推出了一个<strong>智能可解释AI（XAI）</strong>平台，为自动化信贷决策引入透明度与问责机制。
            <ul style="margin-top: 10px;">
                <li><strong>对于客户：</strong>以通俗易懂的语言了解哪些财务指标导致了拒绝决策。</li>
                <li><strong>对于银行信贷员：</strong>快速审计个别决策并与整体模型行为对比，发现异常或偏见。</li>
                <li><strong>对于合规团队：</strong>生成跨人口群体的公平性证明，满足监管要求。</li>
            </ul>""",
        "t3_ex_title": "📋 具体案例",
        "t3_ex_body": """假设<strong>客户10001</strong>的贷款申请被拒绝。通过本平台，银行信贷员可以提问：
            <ul style="margin-top: 10px;">
                <li><em>「客户10001的预测结果是什么？原因是什么？」</em>——局部解释智能体识别出<strong>EBITDA负增长</strong>和<strong>高DPD-10标记</strong>是主要因素。</li>
                <li><em>「展示全局特征重要性」</em>——全局解释智能体揭示<strong>现金流量比率</strong>和<strong>契约违约次数</strong>是最高预测因子。</li>
                <li><em>「模型在不同客户群体中是否公平？」</em>——伦理与公平性智能体检查并标记任何系统性偏见。</li>
            </ul>
            这将黑盒决策转化为<strong>清晰、可审计且值得信赖</strong>的解释。""",
        "t3_why_title": "🎯 为什么这很重要",
        "t3_why_body": """<ul style="margin-top: 10px;">
                <li><strong>监管合规：</strong>《欧盟人工智能法》、GDPR第22条要求对自动化决策提供可解释的理由。</li>
                <li><strong>客户信任：</strong>透明的拒绝理由维护客户关系，减少歧视感知。</li>
                <li><strong>模型质量：</strong>可解释性促使银行批判性审视自身模型，发现错误或偏见代理。</li>
                <li><strong>运营效率：</strong>自然语言交互消除了数据科学专业知识门槛，赋能一线员工。</li>
            </ul>""",

        # Tab 4: Chat
        "t4_spinner": "思考中...",
        "t4_error": "发生错误",
        "t4_memory": "🧠 长期记忆摘要",
        "t4_empty": "👋 您好！请向我提问关于数据集、模型预测或公平性分析的问题！",
        "t4_placeholder": "请提问关于数据集、特征重要性或具体预测的问题...",
        "t4_img_caption": "生成的解释图表",
    },

    "en": {
        # Page
        "page_title": "Agentic XAI",
        "app_title": "🤖 Agentic Explainable AI",
        "lang_btn": "🌐 中文",

        # Sidebar
        "sidebar_header": "Configuration",
        "data_model_status": "**Data & Model Status**",
        "dataset_label": "Dataset",
        "model_label": "Model",
        "not_loaded": "Not loaded",
        "custom_file": "custom file",

        # Session Memory
        "session_memory": "💾 Session Memory",
        "current_thread": "**Current Thread ID**",
        "resume_input": "Resume previous session (enter Thread ID):",
        "resume_btn": "Resume Session",

        # Upload
        "upload_expander": "📂 Upload your own files",
        "upload_csv": "Dataset (CSV or ARFF)",
        "upload_npz": "Padded 3D Data (.npz)",
        "upload_model": "Model (.cbm, .pt, .pth)",
        "upload_csv_ok": "✅ Custom dataset loaded!",
        "upload_csv_err": "Error",
        "upload_npz_ok": "✅ Custom 3D Padded Data loaded! Shape",
        "upload_npz_err": "Error loading NPZ",
        "upload_cbm_ok": "✅ Custom CatBoost model loaded!",
        "upload_pt_ok": "✅ Custom PyTorch model loaded!",
        "upload_model_err": "Error loading model",

        # Dataset Meta
        "dataset_meta": "Dataset Metadata",
        "target_var": "Target Variable",
        "problem_type": "Problem Type",
        "dataset_preview": "Dataset Preview",
        "reset_btn": "🔄 Reset Session",

        # Auth warning
        "api_warning": "⚠️ Please ensure your OPENAI_API_KEY is correctly set in the .env file to proceed.",

        # Tabs
        "tab_intro": "📖 Introduction",
        "tab_workflow": "🗺️ Typical Workflow",
        "tab_usecase": "💡 Use Case",
        "tab_chat": "💬 Agent Chat",

        # Tab 1: Introduction
        "t1_header": "Project Overview",
        "t1_welcome": "Welcome to <strong>Agentic Explainable AI</strong>. This platform leverages a multi-agent system to help you understand your data, interpret machine learning models, and analyze fairness.",
        "t1_sub1": "1. Structure of our AI Agent System",
        "t1_img_caption": "AI Agent System Architecture",
        "t1_arch_desc": """Our system is powered by a graph-based multi-agent architecture (using <strong>LangGraph</strong>). It orchestrates specialized agents:
            <ul style="margin-top: 10px;">
                <li><strong>Router Agent:</strong> Analyzes your request and routes it to the correct specialized agent.</li>
                <li><strong>Data Understanding Agent:</strong> Analyzes the dataset, provides summaries, and previews data.</li>
                <li><strong>Global Explainer Agent:</strong> Provides high-level insights into model behaviour (e.g., overall feature importance).</li>
                <li><strong>Local Explainer Agent:</strong> Explains specific, individual predictions made by the model.</li>
                <li><strong>Ethic &amp; Fairness Analysis Agent:</strong> Assesses the model for potential biases and calculates fairness metrics.</li>
            </ul>""",
        "t1_sub2": "2. Explanation of each Agent and tools they use",
        "t1_router_title": "#### 🧭 Router Agent",
        "t1_router_body": """<ul style="margin-top: 0px;">
                <li><strong>Role:</strong> The 'traffic controller'. It reads your prompt and sets the <code>analysis_mode</code> to route the conversation to the correct agent.</li>
                <li><strong>Tools:</strong> None. It relies on LLM understanding to categorize the intent.</li>
            </ul>""",
        "t1_data_title": "#### 📊 Data Understanding Agent",
        "t1_data_body": """<ul style="margin-top: 0px;">
                <li><strong>Role:</strong> Helps you explore the uploaded dataset.</li>
                <li><strong>Tools:</strong>
                    <ul>
                        <li><code>get_dataset_samples</code>: Retrieves the first few rows of the dataset for a quick preview.</li>
                        <li><code>update_metadata</code>: Allows the agent to update the target variable or problem type.</li>
                    </ul>
                </li>
            </ul>""",
        "t1_global_title": "#### 🌍 Global Explainer Agent",
        "t1_global_body": """<ul style="margin-top: 0px;">
                <li><strong>Role:</strong> Explains the overall logic of the model. What features matter most generally?</li>
                <li><strong>Tools:</strong>
                    <ul>
                        <li><code>get_global_feature_importance_shap</code>: Uses <strong>SHAP</strong> to generate global feature importance and summary plots.</li>
                    </ul>
                </li>
            </ul>""",
        "t1_local_title": "#### 🔎 Local Explainer Agent",
        "t1_local_body": """<ul style="margin-top: 0px;">
                <li><strong>Role:</strong> Explains why the model made a <em>specific</em> prediction for a single instance.</li>
                <li><strong>Tools:</strong>
                    <ul>
                        <li><code>run_shap_explanation</code>: Generates a SHAP waterfall plot for a specific prediction.</li>
                        <li><code>run_lime_explanation</code>: Uses <strong>LIME</strong> to explain a specific prediction by approximating the model locally.</li>
                    </ul>
                </li>
            </ul>""",
        "t1_ethic_title": "#### ⚖️ Ethic & Fairness Analysis Agent",
        "t1_ethic_body": """<ul style="margin-top: 0px;">
                <li><strong>Role:</strong> Evaluates the model for fairness concerning sensitive attributes (e.g., gender, race).</li>
                <li><strong>Tools:</strong>
                    <ul>
                        <li><code>run_ethic_analysis</code>: Uses <strong>Fairlearn</strong> to compute SPD, EOD, and DI fairness metrics.</li>
                        <li><code>visualize_ethic_analysis</code>: Generates bar charts comparing fairness metrics across groups.</li>
                    </ul>
                </li>
            </ul>""",

        # Tab 2: Workflow
        "t2_header": "Typical Workflow",
        "t2_intro": "To get the most out of the Agentic XAI platform, we recommend following this workflow:",
        "t2_steps": """<ol style="margin-top: 10px;">
                <li><strong>Upload Data &amp; Model:</strong> Use the sidebar to upload your dataset (CSV or ARFF) and trained model (<code>.cbm</code> for CatBoost, <code>.pth</code> for PyTorch LSTM). For time-series models, also upload the padded 3D tensor (<code>.npz</code>).</li>
                <li><strong>Data Understanding:</strong> Go to <strong>Agent Chat</strong> and ask <em>"Describe the dataset"</em> or <em>"What is the data format?"</em></li>
                <li><strong>Global Explanation:</strong> Ask <em>"Show me the global feature importance"</em>.</li>
                <li><strong>Local Explanation:</strong> Ask <em>"What is the prediction for customer 10001 and why?"</em></li>
                <li><strong>Ethical Analysis:</strong> Ask <em>"Check fairness by gender"</em>.</li>
            </ol>""",
        "t2_examples_title": "💡 Example Questions to Try",
        "t2_examples": """<ul style="margin-top: 10px;">
                <li><em>"Describe the dataset"</em></li>
                <li><em>"Show me the global feature importance"</em></li>
                <li><em>"What is the prediction for customer 10001 and why?"</em></li>
                <li><em>"Check for fairness issues by gender"</em></li>
                <li><em>"Which features are most risky for loan default?"</em></li>
            </ul>""",

        # Tab 3: Use Case
        "t3_header": "Use Case: Loan Default Prediction",
        "t3_intro": "This use case illustrates a real-world scenario where AI-driven decisions directly affect people's lives — and why explainability is not optional, but essential.",
        "t3_prob_title": "🏦 The Problem",
        "t3_prob_body": """A customer applies for a business loan at their bank. After a short waiting period, they receive an automated rejection email.
            When contacting the bank for clarification, they are told:<br><br>
            <blockquote style="border-left: 4px solid #e74c3c; background: rgba(231,76,60,0.08); border-radius: 0 6px 6px 0; padding: 12px 16px; font-style: italic; margin: 12px 0;">
                "The decision was made by an internal machine learning model. We are unable to provide further details."
            </blockquote>
            The customer is left with no understanding of why they were rejected and no path to appeal.
            This lack of transparency may violate the <strong>right to explanation</strong> under the EU AI Act and GDPR Article 22.""",
        "t3_sol_title": "💡 Our Solution",
        "t3_sol_body": """This project introduces an <strong>Agentic Explainable AI (XAI)</strong> platform that brings transparency and accountability to automated credit decisions.
            <ul style="margin-top: 10px;">
                <li><strong>For the customer:</strong> Understand exactly which financial indicators led to the rejection, in plain language.</li>
                <li><strong>For the bank officer:</strong> Quickly audit individual decisions and compare against overall model behaviour to detect bias.</li>
                <li><strong>For compliance teams:</strong> Generate evidence of fairness across demographic groups to satisfy regulatory requirements.</li>
            </ul>""",
        "t3_ex_title": "📋 Concrete Example",
        "t3_ex_body": """Suppose <strong>Customer 10001</strong> is rejected for a loan. Using this platform, a bank officer can ask:
            <ul style="margin-top: 10px;">
                <li><em>"What is the prediction for customer 10001 and why?"</em> — The Local Explainer identifies <strong>negative EBITDA growth</strong> and a <strong>high DPD-10 flag</strong> as primary drivers.</li>
                <li><em>"Show me the global feature importance"</em> — The Global Explainer reveals <strong>Cash Flow Ratio</strong> and <strong>Covenant Breaches</strong> as top predictors.</li>
                <li><em>"Is the model fair across customer segments?"</em> — The Fairness Agent checks for systemic bias across groups.</li>
            </ul>
            This transforms an opaque black-box decision into a <strong>clear, auditable, and trustworthy</strong> explanation.""",
        "t3_why_title": "🎯 Why This Matters",
        "t3_why_body": """<ul style="margin-top: 10px;">
                <li><strong>Regulatory Compliance:</strong> EU AI Act, GDPR Article 22, and similar frameworks require human-interpretable explanations for automated decisions.</li>
                <li><strong>Customer Trust:</strong> Transparent rejections preserve the customer relationship and reduce perception of discriminatory treatment.</li>
                <li><strong>Model Quality:</strong> Explainability forces banks to critically review their models, catching errors or unintended proxies.</li>
                <li><strong>Operational Efficiency:</strong> Natural language interaction removes the need for data science expertise to audit decisions.</li>
            </ul>""",

        # Tab 4: Chat
        "t4_spinner": "Thinking...",
        "t4_error": "An error occurred",
        "t4_memory": "🧠 Long-term Memory Summary",
        "t4_empty": "👋 Ask me about your dataset, model predictions, or fairness analysis!",
        "t4_placeholder": "Ask about the dataset, feature importance, or specific predictions...",
        "t4_img_caption": "Generated Explanation Plot",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def extract_image_path(text):
    """
    Extracts image path from text if present and resolves it.
    Looks for pattern: `path/to/image.png` or [link](path)
    """
    potential_path = None

    match = re.search(r'`(.*?\.png)`', text)
    if match:
        potential_path = match.group(1).replace("sandbox:", "")

    if not potential_path:
        match = re.search(r'\]\((.*?\.png)\)', text)
        if match:
            potential_path = match.group(1).replace("sandbox:", "")

    if not potential_path:
        match = re.search(r'([^\s\]\(\'"]+\.png)', text)
        if match:
            potential_path = match.group(1).replace("sandbox:", "")

    if potential_path:
        potential_path = potential_path.replace("sandbox:/", "").replace("sandbox:", "")
        if potential_path.startswith("/artifacts/"):
            potential_path = potential_path.lstrip("/")
        base_name = os.path.basename(potential_path)
        if os.path.exists(potential_path):
            return potential_path
        elif os.path.exists(os.path.join("artifacts", base_name)):
            return os.path.join("artifacts", base_name)
        elif os.path.exists(base_name):
            return base_name
    return None


def load_arff_data(file_object):
    """Parses ARFF file content from a file-like object (uploaded file)."""
    data = []
    columns = []
    content = file_object.getvalue().decode("utf-8")
    lines = content.splitlines()
    data_started = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("@attribute"):
            parts = line.split()
            columns.append(parts[1])
        elif line.lower().startswith("@data"):
            data_started = True
            continue
        elif data_started:
            row = [x.strip().strip("'").strip('"') for x in line.split(',')]
            data.append(row)
    return pd.DataFrame(data, columns=columns)


# ─────────────────────────────────────────────────────────────────────────────
# PATHS TO DEFAULT FILES
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV   = os.path.join(PROJECT_ROOT, "datasets", "padded_credit_default_prediction_dataset", "credit_risk_dataset_5k.csv")
DEFAULT_NPZ   = os.path.join(PROJECT_ROOT, "datasets", "padded_credit_default_prediction_dataset", "credit_risk_dataset_5k_padded.npz")
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "model", "best_model.pth")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "default_loaded" not in st.session_state:
    st.session_state.default_loaded = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "lang" not in st.session_state:
    st.session_state.lang = "zh"          # default: Chinese

# Shortcut for current texts
t = TEXTS[st.session_state.lang]

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-LOAD DEFAULTS ON FIRST RUN
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.default_loaded:
    if os.path.exists(DEFAULT_CSV):
        try:
            df = pd.read_csv(DEFAULT_CSV, sep=';', decimal=',', on_bad_lines='warn')
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass
            st.session_state.df = df
        except Exception:
            pass

    if os.path.exists(DEFAULT_NPZ):
        try:
            import numpy as np
            npz_data = np.load(DEFAULT_NPZ)
            if 'X_padded' in npz_data:
                st.session_state.X_padded = npz_data['X_padded']
            elif len(npz_data.files) > 0:
                st.session_state.X_padded = npz_data[npz_data.files[0]]
            if 'feature_cols' in npz_data:
                st.session_state.feature_cols = list(npz_data['feature_cols'])
        except Exception:
            pass

    if os.path.exists(DEFAULT_MODEL):
        try:
            loaded = torch.load(DEFAULT_MODEL, map_location=torch.device('cpu'))
            if isinstance(loaded, collections.abc.Mapping):
                m = LoanDefaultModel(input_size=33, hidden_size=32)
                m.load_state_dict(loaded)
            else:
                m = loaded
            m.eval()
            st.session_state.model = m
        except Exception:
            pass

    st.session_state.default_loaded = True

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & TITLE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=t["page_title"], layout="wide")
st.title(t["app_title"])

# Global CSS
st.markdown("""
<style>
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] div {
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
}
[data-testid="stChatMessageContent"] h1 { font-size: 1.6rem !important; }
[data-testid="stChatMessageContent"] h2 { font-size: 1.4rem !important; }
[data-testid="stChatMessageContent"] h3 { font-size: 1.2rem !important; }
[data-testid="stChatMessageContent"] code { font-size: 0.95rem !important; }
[data-testid="stTextArea"] textarea { font-size: 1.05rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── Language toggle button (top of sidebar) ──────────────────────────────
    if st.button(t["lang_btn"], key="lang_toggle", use_container_width=True):
        st.session_state.lang = "en" if st.session_state.lang == "zh" else "zh"
        st.rerun()

    st.markdown("---")
    st.header(t["sidebar_header"])

    # Status indicators
    df_loaded    = st.session_state.df is not None
    model_loaded = st.session_state.model is not None

    st.markdown(t["data_model_status"])
    st.markdown(
        f"{'✅' if df_loaded else '❌'} {t['dataset_label']}: "
        f"{'`' + (os.path.basename(DEFAULT_CSV) if not st.session_state.get('custom_csv') else t['custom_file']) + '`' if df_loaded else t['not_loaded']}"
    )
    st.markdown(
        f"{'✅' if model_loaded else '❌'} {t['model_label']}: "
        f"{'`' + (os.path.basename(DEFAULT_MODEL) if not st.session_state.get('custom_model') else t['custom_file']) + '`' if model_loaded else t['not_loaded']}"
    )

    st.markdown("---")

    # Session Memory
    with st.expander(t["session_memory"], expanded=False):
        st.markdown(f"{t['current_thread']}: `{st.session_state.thread_id}`")
        new_thread = st.text_input(t["resume_input"])
        if st.button(t["resume_btn"]):
            if new_thread.strip():
                st.session_state.thread_id = new_thread.strip()
                st.session_state.messages = []
                st.session_state.summary = ""
                st.rerun()

    st.markdown("---")

    # File Upload
    with st.expander(t["upload_expander"]):
        uploaded_file  = st.file_uploader(t["upload_csv"],   type=["csv", "arff"], key="custom_dataset_uploader")
        uploaded_npz   = st.file_uploader(t["upload_npz"],   type=["npz"],          key="custom_npz_uploader")
        uploaded_model = st.file_uploader(t["upload_model"], type=["cbm", "pt", "pth"], key="custom_model_uploader")

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".arff"):
                    df = load_arff_data(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file, sep=';', decimal=',', on_bad_lines='warn')
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        pass
                st.session_state.df = df
                st.session_state.custom_csv = True
                st.success(t["upload_csv_ok"])
            except Exception as e:
                st.error(f"{t['upload_csv_err']}: {e}")

        if uploaded_npz is not None:
            try:
                import numpy as np
                with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp_file:
                    tmp_file.write(uploaded_npz.getvalue())
                    tmp_path = tmp_file.name
                npz_data = np.load(tmp_path)
                if 'X_padded' in npz_data:
                    X_padded = npz_data['X_padded']
                elif len(npz_data.files) > 0:
                    X_padded = npz_data[npz_data.files[0]]
                else:
                    X_padded = None
                if X_padded is not None:
                    st.session_state.X_padded = X_padded
                    if 'feature_cols' in npz_data:
                        st.session_state.feature_cols = list(npz_data['feature_cols'])
                    st.success(f"{t['upload_npz_ok']}: {X_padded.shape}")
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"{t['upload_npz_err']}: {e}")

        if uploaded_model is not None:
            try:
                if uploaded_model.name.endswith(".cbm"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".cbm") as tmp_file:
                        tmp_file.write(uploaded_model.getvalue())
                        tmp_path = tmp_file.name
                    m = CatBoostClassifier()
                    m.load_model(tmp_path)
                    os.unlink(tmp_path)
                    st.session_state.model = m
                    st.session_state.custom_model = True
                    st.success(t["upload_cbm_ok"])
                elif uploaded_model.name.endswith(".pt") or uploaded_model.name.endswith(".pth"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                        tmp_file.write(uploaded_model.getvalue())
                        tmp_path = tmp_file.name
                    loaded = torch.load(tmp_path, map_location=torch.device('cpu'))
                    if isinstance(loaded, collections.abc.Mapping):
                        m = LoanDefaultModel(input_size=33, hidden_size=32)
                        m.load_state_dict(loaded)
                    else:
                        m = loaded
                    m.eval()
                    os.unlink(tmp_path)
                    st.session_state.model = m
                    st.session_state.custom_model = True
                    st.success(t["upload_pt_ok"])
            except Exception as e:
                st.error(f"{t['upload_model_err']}: {e}")

    # Dataset Metadata
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.subheader(t["dataset_meta"])

        possible_targets = ["y", "target", "class", "label", "Default Flag"]
        default_target_ix = len(df.columns) - 1
        for pt in possible_targets:
            if pt in df.columns:
                default_target_ix = df.columns.get_loc(pt)
                break

        target_col = st.selectbox(t["target_var"], df.columns, index=default_target_ix)
        st.session_state.target_variable = target_col

        problem_type = st.selectbox(t["problem_type"], ["classification", "regression"], index=0)
        st.session_state.problem_type = problem_type

        with st.expander(t["dataset_preview"]):
            st.dataframe(df.head())

    if st.button(t["reset_btn"]):
        st.session_state.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# AUTHORIZATION CHECK
# ─────────────────────────────────────────────────────────────────────────────
if not os.environ.get("OPENAI_API_KEY"):
    st.warning(t["api_warning"])
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([t["tab_intro"], t["tab_workflow"], t["tab_usecase"], t["tab_chat"]])

# ── TAB 1: INTRODUCTION ──────────────────────────────────────────────────────
with tab1:
    st.header(t["t1_header"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t1_welcome"]}</div>', unsafe_allow_html=True)

    st.subheader(t["t1_sub1"])
    agent_sys_img_path = os.path.join(PROJECT_ROOT, "agent", "agent_system.png")
    if os.path.exists(agent_sys_img_path):
        st.image(agent_sys_img_path, caption=t["t1_img_caption"])

    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t1_arch_desc"]}</div>', unsafe_allow_html=True)

    st.subheader(t["t1_sub2"])

    st.markdown(t["t1_router_title"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t1_router_body"]}</div>', unsafe_allow_html=True)

    st.markdown(t["t1_data_title"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t1_data_body"]}</div>', unsafe_allow_html=True)

    st.markdown(t["t1_global_title"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t1_global_body"]}</div>', unsafe_allow_html=True)

    st.markdown(t["t1_local_title"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t1_local_body"]}</div>', unsafe_allow_html=True)

    st.markdown(t["t1_ethic_title"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t1_ethic_body"]}</div>', unsafe_allow_html=True)

# ── TAB 2: WORKFLOW ──────────────────────────────────────────────────────────
with tab2:
    st.header(t["t2_header"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t2_intro"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t2_steps"]}</div>', unsafe_allow_html=True)

    st.subheader(t["t2_examples_title"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t2_examples"]}</div>', unsafe_allow_html=True)

# ── TAB 3: USE CASE ──────────────────────────────────────────────────────────
with tab3:
    st.header(t["t3_header"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t3_intro"]}</div>', unsafe_allow_html=True)

    st.subheader(t["t3_prob_title"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t3_prob_body"]}</div>', unsafe_allow_html=True)

    st.subheader(t["t3_sol_title"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t3_sol_body"]}</div>', unsafe_allow_html=True)

    st.subheader(t["t3_ex_title"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t3_ex_body"]}</div>', unsafe_allow_html=True)

    st.subheader(t["t3_why_title"])
    st.markdown(f'<div style="font-size: 1.2rem; line-height: 1.6;">{t["t3_why_body"]}</div>', unsafe_allow_html=True)

# ── TAB 4: AGENT CHAT ────────────────────────────────────────────────────────
with tab4:

    # Step 1: Handle form submission FIRST (before any rendering)
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    if st.session_state.pending_prompt:
        prompt_to_run = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

        user_msg = HumanMessage(content=prompt_to_run)
        st.session_state.messages.append(user_msg)

        with st.spinner(t["t4_spinner"]):
            try:
                initial_state = {
                    "messages": st.session_state.messages,
                    "target_variable": st.session_state.get("target_variable"),
                    "problem_type": st.session_state.get("problem_type", "classification")
                }
                config = {
                    "configurable": {
                        "thread_id": st.session_state.thread_id,
                        "df": st.session_state.df,
                        "model": st.session_state.model,
                        "X_padded": st.session_state.get("X_padded"),
                        "feature_cols": st.session_state.get("feature_cols"),
                        "lang": st.session_state.lang,
                    }
                }
                result = agent_app.invoke(initial_state, config=config)
                st.session_state.messages = result.get('messages', [])
                st.session_state.summary = result.get('summary', "")
            except Exception as e:
                st.error(f"{t['t4_error']}: {e}")

    # Step 2: Display history
    if st.session_state.get("summary"):
        with st.expander(t["t4_memory"], expanded=False):
            st.info(st.session_state.summary)

    if not st.session_state.messages:
        st.markdown(
            f"<div style='text-align:center; color:#888; margin-top: 80px; font-size: 1.1rem;'>{t['t4_empty']}</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        turns = []
        current_turn = None
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                if current_turn is not None:
                    turns.append(current_turn)
                current_turn = {"human": message, "ai_messages": []}
            elif isinstance(message, AIMessage) and message.content.strip():
                if current_turn is not None:
                    current_turn["ai_messages"].append(message)
        if current_turn is not None:
            turns.append(current_turn)

        for turn in turns:
            with st.chat_message("user"):
                st.markdown(turn["human"].content)
            if turn["ai_messages"]:
                last_ai = turn["ai_messages"][-1]
                with st.chat_message("assistant"):
                    st.markdown(last_ai.content)
                    img_path = extract_image_path(last_ai.content)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption=t["t4_img_caption"])

    # Step 3: Input form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([9, 1])
        with col1:
            prompt = st.text_area(
                label="chat_input",
                placeholder=t["t4_placeholder"],
                height=80,
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("<div style='margin-top: 24px;'>", unsafe_allow_html=True)
            submitted = st.form_submit_button("➤", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    if submitted and prompt and prompt.strip():
        st.session_state.pending_prompt = prompt.strip()
        st.rerun()
