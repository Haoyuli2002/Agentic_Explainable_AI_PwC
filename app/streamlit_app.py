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

st.set_page_config(page_title="Agentic XAI", layout="wide")

st.title("🤖 Agentic Explainable AI")

# --- Global CSS: enlarge Agent Chat font size ---
st.markdown("""
<style>
/* Chat message content — main text */
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] div {
    font-size: 1.05rem !important;
    line-height: 1.7 !important;
}

/* Headings inside chat */
[data-testid="stChatMessageContent"] h1 { font-size: 1.6rem !important; }
[data-testid="stChatMessageContent"] h2 { font-size: 1.4rem !important; }
[data-testid="stChatMessageContent"] h3 { font-size: 1.2rem !important; }

/* Code blocks inside chat */
[data-testid="stChatMessageContent"] code {
    font-size: 0.95rem !important;
}

/* Chat input textarea */
[data-testid="stTextArea"] textarea {
    font-size: 1.05rem !important;
}
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def extract_image_path(text):
    """
    Extracts image path from text if present and resolves it.
    Looks for pattern: `path/to/image.png` or [link](path)
    """
    import os
    import re
    
    potential_path = None
    
    # Regex to find paths ending in .png inside backticks
    match = re.search(r'`(.*?\.png)`', text)
    if match: 
        potential_path = match.group(1).replace("sandbox:", "")
    
    # Markdown link style: [..](path.png)
    if not potential_path:
        match = re.search(r'\]\((.*?\.png)\)', text)
        if match:
            potential_path = match.group(1).replace("sandbox:", "")

    # Fallback: looks for straight paths (simplified)
    if not potential_path:
        match = re.search(r'([^\s\]\(\'"]+\.png)', text)
        if match:
            potential_path = match.group(1).replace("sandbox:", "")
            
    if potential_path:
        # Sometimes paths are prefixed with sandbox:/
        potential_path = potential_path.replace("sandbox:/", "").replace("sandbox:", "")
        
        # Resolve path
        if potential_path.startswith("/artifacts/"):
             potential_path = potential_path.lstrip("/")
        
        # If it doesn't exist directly, maybe it's in the current folder or "artifacts" folder
        base_name = os.path.basename(potential_path)
        if os.path.exists(potential_path):
            return potential_path
        elif os.path.exists(os.path.join("artifacts", base_name)):
            return os.path.join("artifacts", base_name)
        elif os.path.exists(base_name):
            return base_name
            
    return None

def load_arff_data(file_object):
    """
    Parses ARFF file content from a file-like object (uploaded file).
    """
    data = []
    columns = []
    
    # TextIOWrapper for decoding might be needed if binary
    # Streamlit UploadedFile is bytes, need to decode
    content = file_object.getvalue().decode("utf-8")
    lines = content.splitlines()
    
    data_started = False
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.lower().startswith("@attribute"):
            parts = line.split()
            columns.append(parts[1])
        elif line.lower().startswith("@data"):
            data_started = True
            continue
        elif data_started:
            # Basic CSV parsing for data lines, stripping quotes
            row = [x.strip().strip("'").strip('"') for x in line.split(',')]
            data.append(row)
            
    return pd.DataFrame(data, columns=columns)

# --- Paths to default files (relative to project root) ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV   = os.path.join(PROJECT_ROOT, "datasets", "padded_credit_default_prediction_dataset", "credit_risk_dataset_5k.csv")
DEFAULT_NPZ   = os.path.join(PROJECT_ROOT, "datasets", "padded_credit_default_prediction_dataset", "credit_risk_dataset_5k_padded.npz")
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "model", "best_model.pth")

# --- Session State Init ---
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

# --- Auto-load defaults on first run ---
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

    # Load default Padded NPZ Sequence
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

    # Load default PyTorch model
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

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # --- Status indicators ---
    df_loaded   = st.session_state.df is not None
    model_loaded = st.session_state.model is not None

    st.markdown("**Data & Model Status**")
    st.markdown(
        f"{'✅' if df_loaded else '❌'} Dataset: "
        f"{'`' + (os.path.basename(DEFAULT_CSV) if not st.session_state.get('custom_csv') else 'custom file') + '`' if df_loaded else 'Not loaded'}"
    )
    st.markdown(
        f"{'✅' if model_loaded else '❌'} Model: "
        f"{'`' + (os.path.basename(DEFAULT_MODEL) if not st.session_state.get('custom_model') else 'custom file') + '`' if model_loaded else 'Not loaded'}"
    )

    st.markdown("---")

    # --- Thread Management section ---
    with st.expander("💾 Session Memory", expanded=False):
        st.markdown(f"**Current Thread ID**: `{st.session_state.thread_id}`")
        new_thread = st.text_input("Resume previous session (enter Thread ID):")
        if st.button("Resume Session"):
            if new_thread.strip():
                st.session_state.thread_id = new_thread.strip()
                st.session_state.messages = []
                st.session_state.summary = ""
                st.rerun()
                
    st.markdown("---")

    # --- Collapsible custom upload section ---
    with st.expander("📂 Upload your own files"):
        uploaded_file    = st.file_uploader("Dataset (CSV or ARFF)", type=["csv", "arff"], key="custom_dataset_uploader")
        uploaded_npz     = st.file_uploader("Padded 3D Data (.npz)", type=["npz"],         key="custom_npz_uploader")
        uploaded_model   = st.file_uploader("Model (.cbm, .pt, .pth)", type=["cbm", "pt", "pth"], key="custom_model_uploader")

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
                st.success("✅ Custom dataset loaded!")
            except Exception as e:
                st.error(f"Error: {e}")

        if uploaded_npz is not None:
            try:
                import numpy as np
                with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp_file:
                    tmp_file.write(uploaded_npz.getvalue())
                    tmp_path = tmp_file.name
                
                # Load the padded array, usually stored as 'X_padded' or 'arr_0' inside the npz
                npz_data = np.load(tmp_path)
                # Check keys usually produced by np.savez
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
                        
                    st.success(f"✅ Custom 3D Padded Data loaded! Shape: {X_padded.shape}")
                
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Error loading NPZ: {e}")

        if uploaded_model is not None:
            try:
                # Handle CatBoost (.cbm)
                if uploaded_model.name.endswith(".cbm"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".cbm") as tmp_file:
                        tmp_file.write(uploaded_model.getvalue())
                        tmp_path = tmp_file.name
                    m = CatBoostClassifier()
                    m.load_model(tmp_path)
                    os.unlink(tmp_path)
                    st.session_state.model = m
                    st.session_state.custom_model = True
                    st.success("✅ Custom CatBoost model loaded!")
                
                # Handle PyTorch (.pt, .pth)
                elif uploaded_model.name.endswith(".pt") or uploaded_model.name.endswith(".pth"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                        tmp_file.write(uploaded_model.getvalue())
                        tmp_path = tmp_file.name
                        
                    # Safely extract from pickle
                    loaded = torch.load(tmp_path, map_location=torch.device('cpu'))
                    
                    if isinstance(loaded, collections.abc.Mapping):
                        # Construct architecture from state_dict
                        m = LoanDefaultModel(input_size=33, hidden_size=32)
                        m.load_state_dict(loaded)
                    else:
                        m = loaded
                        
                    m.eval() # Set to evaluation mode
                    os.unlink(tmp_path)
                    st.session_state.model = m
                    st.session_state.custom_model = True
                    st.success("✅ Custom PyTorch model loaded!")
                    
            except Exception as e:
                st.error(f"Error loading model: {e}")

    # --- Dataset Metadata (shown when a dataset is loaded) ---
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown("---")
        st.subheader("Dataset Metadata")

        possible_targets = ["y", "target", "class", "label", "Default Flag"]
        default_target_ix = len(df.columns) - 1
        for pt in possible_targets:
            if pt in df.columns:
                default_target_ix = df.columns.get_loc(pt)
                break

        target_col = st.selectbox("Target Variable", df.columns, index=default_target_ix)
        st.session_state.target_variable = target_col

        problem_type = st.selectbox("Problem Type", ["classification", "regression"], index=0)
        st.session_state.problem_type = problem_type

        with st.expander("Dataset Preview"):
            st.dataframe(df.head())

    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()


# --- Authorization Check ---
if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ Please ensure your OPENAI_API_KEY is correctly set in the .env file to proceed.")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📖 Introduction", "🗺️ Typical Workflow", "💡 Use Case", "💬 Agent Chat"])

with tab1:
    st.header("Project Overview")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            Welcome to <strong>Agentic Explainable AI</strong>. This platform leverages a multi-agent system to help you understand your data, interpret machine learning models, and analyze fairness.
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("1. Structure of our AI Agent System")
    
    agent_sys_img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "agent_system.png")
    if os.path.exists(agent_sys_img_path):
        st.image(agent_sys_img_path, caption="AI Agent System Architecture")
        
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            Our system is powered by a graph-based multi-agent architecture (using <strong>LangGraph</strong>). It orchestrates specialized agents to handle different stages of the machine learning explainability workflow:
            <ul style="margin-top: 10px;">
                <li><strong>Router Agent:</strong> At the entry point, it analyzes your request and determines which specialized agent should handle it.</li>
                <li><strong>Data Understanding Agent:</strong> Analyzes the uploaded dataset, provides summaries, and previews data samples.</li>
                <li><strong>Global Explainer Agent:</strong> Provides high-level insights into how the model behaves across the entire dataset (e.g., overall feature importance).</li>
                <li><strong>Local Explainer Agent:</strong> Explains specific, individual predictions made by the model.</li>
                <li><strong>Ethic & Fairness Analysis Agent:</strong> Assesses the model for potential biases and calculates fairness metrics across different demographic groups.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("2. Explanation of each Agent and tools they use")
    
    st.markdown("#### 🧭 Router Agent")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            <ul style="margin-top: 0px;">
                <li><strong>Role:</strong> The 'traffic controller'. It reads your prompt and sets the <code>analysis_mode</code> (e.g., global, local, fairness, data_understanding) to route the conversation to the correct agent.</li>
                <li><strong>Tools:</strong> None. It relies on LLM understanding to categorize the intent.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 📊 Data Understanding Agent")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            <ul style="margin-top: 0px;">
                <li><strong>Role:</strong> Helps you explore the uploaded dataset.</li>
                <li><strong>Tools:</strong>
                    <ul>
                        <li><code>get_dataset_samples</code>: Retrieves the first few rows of the dataset for a quick preview.</li>
                        <li><code>update_metadata</code>: Allows the agent to programmatically update the target variable or problem type if misunderstood.</li>
                    </ul>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🌍 Global Explainer Agent")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            <ul style="margin-top: 0px;">
                <li><strong>Role:</strong> Explains the overall logic of the model. What features matter most generally?</li>
                <li><strong>Tools:</strong>
                    <ul>
                        <li><code>get_global_feature_importance_shap</code>: Uses <strong>SHAP (SHapley Additive exPlanations)</strong> to generate global feature importance and summary plots, illustrating how each feature impacts the model's output across all data.</li>
                    </ul>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### 🔎 Local Explainer Agent")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            <ul style="margin-top: 0px;">
                <li><strong>Role:</strong> Explains why the model made a <em>specific</em> prediction for a single instance (row).</li>
                <li><strong>Tools:</strong>
                    <ul>
                        <li><code>run_shap_explanation</code>: Generates a SHAP waterfall plot for a specific prediction, showing how each feature pushed the prediction away from the base value.</li>
                        <li><code>run_lime_explanation</code>: Uses <strong>LIME (Local Interpretable Model-agnostic Explanations)</strong> to explain a specific prediction by approximating the model locally.</li>
                    </ul>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### ⚖️ Ethic & Fairness Analysis Agent")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            <ul style="margin-top: 0px;">
                <li><strong>Role:</strong> Evaluates the model to ensure it is fair and unbiased concerning sensitive attributes (e.g., gender, race).</li>
                <li><strong>Tools:</strong>
                    <ul>
                        <li><code>run_ethic_analysis</code>: Uses <strong>Fairlearn</strong> to compute fairness metrics such as Demographic Parity Difference (SPD), Equalized Odds Difference (EOD), and Disparate Impact (DI).</li>
                        <li><code>visualize_ethic_analysis</code>: Generates bar charts visualizing accuracy, selection rates, and fairness metrics across different groups based on the sensitive attribute.</li>
                    </ul>
                </li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with tab2:
    st.header("Typical Workflow")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            To get the most out of the Agentic XAI platform, we recommend following this workflow:
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            <ol style="margin-top: 10px;">
                <li><strong>Upload Data &amp; Model:</strong> Use the sidebar to upload your dataset (CSV or ARFF) and your trained model (<code>.cbm</code> for CatBoost, <code>.pth</code> for PyTorch LSTM). For time-series models, also upload the padded 3D tensor (<code>.npz</code>).</li>
                <li><strong>Data Understanding:</strong> Go to the <strong>Agent Chat</strong> tab and ask <em>"Describe the dataset"</em> or <em>"What is the data format?"</em>. The Data Understanding Agent will summarise your data and confirm the problem type and target variable.</li>
                <li><strong>Global Explanation:</strong> Ask <em>"Show me the global feature importance"</em>. The Global Explainer Agent automatically selects <strong>SHAP</strong> for tabular/tree models or <strong>Integrated Gradients</strong> for deep learning, and generates an importance plot.</li>
                <li><strong>Local Explanation:</strong> Ask <em>"What is the prediction for customer 10001 and why?"</em>. The Local Explainer Agent produces a per-instance attribution chart (SHAP waterfall or IG) showing exactly which features drove that specific decision.</li>
                <li><strong>Ethical Analysis:</strong> Ask <em>"Check fairness by gender"</em>. The Ethic &amp; Fairness Agent uses <strong>Fairlearn</strong> to compute Demographic Parity, Equalized Odds, and Disparate Impact, and generates visual comparison charts across groups.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("💡 Example Questions to Try")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            <ul style="margin-top: 10px;">
                <li><em>"Describe the dataset"</em></li>
                <li><em>"Show me the global feature importance"</em></li>
                <li><em>"What is the prediction for customer 10001 and why?"</em></li>
                <li><em>"Check for fairness issues by gender"</em></li>
                <li><em>"Which features are most risky for loan default?"</em></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("Use Case: Loan Default Prediction")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            This use case illustrates a real-world scenario where AI-driven decisions directly affect people's lives — and why explainability is not optional, but essential.
        </div>
    """, unsafe_allow_html=True)

    st.subheader("🏦 The Problem")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            A customer applies for a business loan at their bank. After a short waiting period, they receive an automated rejection email.
            When contacting the bank for clarification, they are told:<br><br>
            <blockquote style="border-left: 4px solid #e74c3c; background: rgba(231,76,60,0.08); border-radius: 0 6px 6px 0; padding: 12px 16px; color: #f000000; font-style: italic; margin: 12px 0;">
                "The decision was made by an internal machine learning model. We are unable to provide further details."
            </blockquote>
            The customer is left with no understanding of why they were rejected, no opportunity to address potential errors in the data, and no path to appeal.
            This lack of transparency is not only frustrating — it undermines trust, raises ethical concerns, and in many jurisdictions, may violate the <strong>right to explanation</strong> under regulations such as the EU AI Act and GDPR Article 22.
        </div>
    """, unsafe_allow_html=True)

    st.subheader("💡 Our Solution")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            This project introduces an <strong>Agentic Explainable AI (XAI)</strong> platform that brings transparency and accountability to automated credit decisions.
            Rather than replacing the model, we wrap it with an intelligent multi-agent system that can answer natural-language questions about its behaviour.
            <ul style="margin-top: 10px;">
                <li><strong>For the customer:</strong> Understand exactly which financial indicators led to the rejection, presented in plain language — not just raw SHAP values.</li>
                <li><strong>For the bank officer:</strong> Quickly audit individual decisions and compare them against overall model behaviour to detect anomalies or bias.</li>
                <li><strong>For compliance teams:</strong> Generate evidence of fairness across demographic groups (gender, age, region) to satisfy regulatory requirements.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("📋 Concrete Example")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            Suppose <strong>Customer 10001</strong> is rejected for a loan. Using this platform, a bank officer can ask:
            <ul style="margin-top: 10px;">
                <li><em>"What is the prediction for customer 10001 and why?"</em> — The Local Explainer Agent uses Integrated Gradients to identify that
                <strong>negative EBITDA growth</strong> and a <strong>high DPD-10 flag</strong> over the past 3 years were the primary drivers of the rejection.</li>
                <li><em>"Show me the global feature importance"</em> — The Global Explainer reveals that <strong>Cash Flow Ratio</strong> and <strong>Covenant Breaches</strong>
                are the top predictors across all customers, helping the bank understand its model's overall logic.</li>
                <li><em>"Is the model fair across customer segments?"</em> — The Ethic & Fairness Agent checks whether rejection rates differ significantly across groups, flagging any systemic bias.</li>
            </ul>
            This transforms an opaque black-box decision into a <strong>clear, auditable, and trustworthy</strong> explanation — empowering both the bank and the customer.
        </div>
    """, unsafe_allow_html=True)

    st.subheader("🎯 Why This Matters")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            <ul style="margin-top: 10px;">
                <li><strong>Regulatory Compliance:</strong> EU AI Act, GDPR Article 22, and similar frameworks increasingly require human-interpretable explanations for automated decisions affecting individuals.</li>
                <li><strong>Customer Trust:</strong> Transparent rejections preserve the customer relationship and reduce the perception of arbitrary or discriminatory treatment.</li>
                <li><strong>Model Quality:</strong> Explainability forces banks to critically review their own models, catching errors, overfitting, or unintended proxies for protected attributes.</li>
                <li><strong>Operational Efficiency:</strong> Natural language interaction removes the need for data science expertise to audit individual decisions, empowering frontline staff.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

with tab4:

    # --- Step 1: Handle form submission FIRST (before any rendering) ---
    # Use a placeholder in session state for the pending prompt
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    # Process the pending prompt if set (runs at top before any render)
    if st.session_state.pending_prompt:
        prompt_to_run = st.session_state.pending_prompt
        st.session_state.pending_prompt = None  # Clear it

        user_msg = HumanMessage(content=prompt_to_run)
        st.session_state.messages.append(user_msg)

        with st.spinner("Thinking..."):
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
                        "feature_cols": st.session_state.get("feature_cols")
                    }
                }
                result = agent_app.invoke(initial_state, config=config)
                st.session_state.messages = result.get('messages', [])
                st.session_state.summary = result.get('summary', "")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # --- Step 2: Display full history — one user msg + last AI msg per turn ---
    if st.session_state.get("summary"):
        with st.expander("🧠 Long-term Memory Summary", expanded=False):
            st.info(st.session_state.summary)

    if not st.session_state.messages:
        st.markdown(
            "<div style='text-align:center; color:#888; margin-top: 80px; font-size: 1.1rem;'>"
            "👋 Ask me about your dataset, model predictions, or fairness analysis!"
            "</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        # Group messages into turns: each turn = [HumanMessage, ...AIMessages...]
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

        # Render each turn: user msg + only the last AI msg
        for turn in turns:
            with st.chat_message("user"):
                st.markdown(turn["human"].content)
            if turn["ai_messages"]:
                last_ai = turn["ai_messages"][-1]
                with st.chat_message("assistant"):
                    st.markdown(last_ai.content)
                    img_path = extract_image_path(last_ai.content)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption="Generated Explanation Plot")

    # --- Step 3: Input form always renders BELOW messages ---
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([9, 1])
        with col1:
            prompt = st.text_area(
                label="chat_input",
                placeholder="Ask about the dataset, feature importance, or specific predictions...",
                height=80,
                label_visibility="collapsed"
            )
        with col2:
            st.markdown("<div style='margin-top: 24px;'>", unsafe_allow_html=True)
            submitted = st.form_submit_button("➤", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # On submission: store prompt and trigger rerun so agent runs BEFORE rendering
    if submitted and prompt and prompt.strip():
        st.session_state.pending_prompt = prompt.strip()
        st.rerun()
