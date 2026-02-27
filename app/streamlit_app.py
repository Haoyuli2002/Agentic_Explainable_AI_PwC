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
from catboost import CatBoostClassifier
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# Load environment variables
load_dotenv()

# Add parent directory to path to import agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.graph import app as agent_app

st.set_page_config(page_title="Agentic XAI", layout="wide")

st.title("🤖 Agentic Explainable AI")

# --- Helpers ---
def extract_image_path(text):
    """
    Extracts image path from text if present.
    Looks for pattern: `path/to/image.png`
    """
    # Regex to find paths ending in .png inside backticks or general text
    # Prioritizes explicit backticks
    match = re.search(r'`(.*?\.png)`', text)
    if match: 
        return match.group(1).replace("sandbox:", "")
    
    # Markdown link style: [..](path.png)
    match = re.search(r'\]\((.*?\.png)\)', text)
    if match:
        return match.group(1).replace("sandbox:", "")

    # Fallback: looks for straight paths (simplified)
    match = re.search(r'(\S+\.png)', text)
    if match:
        return match.group(1)
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

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    
    # Dataset Upload
    uploaded_file = st.file_uploader("Upload Dataset (CSV or ARFF)", type=["csv", "arff"])
    
    # Model Upload
    uploaded_model = st.file_uploader("Upload Model (CatBoost .cbm)", type="cbm")

    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# --- Authorization Check ---
if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ Please ensure your OPENAI_API_KEY is correctly set in the .env file to proceed.")
    st.stop()

# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None

# --- Data Loading ---
# --- Data Loading ---
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".arff"):
            df = load_arff_data(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
            
        # Basic Type Inference for ARFF loaded strings
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass

        st.session_state.df = df
        st.sidebar.success("✅ Dataset loaded!")
        
        # --- Metadata Selectors (Only show if DF is loaded) ---
        st.sidebar.markdown("---")
        st.sidebar.subheader("Dataset Metadata")
        
        # Target Variable
        # Try to guess target
        default_target_ix = 0
        possible_targets = ["y", "target", "class", "label"]
        for pt in possible_targets:
            if pt in df.columns:
                default_target_ix = df.columns.get_loc(pt)
                break
        else:
            # Default to last column if no match
            default_target_ix = len(df.columns) - 1

        target_col = st.sidebar.selectbox(
            "Target Variable", 
            df.columns, 
            index=default_target_ix
        )
        st.session_state.target_variable = target_col
        
        # Problem Type
        problem_type = st.sidebar.selectbox(
            "Problem Type",
            ["classification", "regression"],
            index=0
        )
        st.session_state.problem_type = problem_type
        
        with st.sidebar.expander("Dataset Preview"):
            st.dataframe(df.head())
            
    except Exception as e:
        st.sidebar.error(f"Error loading Dataset: {e}")

# --- Model Loading ---
if uploaded_model is not None:
    try:
        # CatBoost load_model needs a file path, so we save to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cbm") as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
            tmp_path = tmp_file.name
        
        model = CatBoostClassifier()
        model.load_model(tmp_path)
        st.session_state.model = model
        
        # Cleanup temp file
        os.unlink(tmp_path)
        
        st.sidebar.success("✅ Model loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading Model: {e}")


# --- Tabs ---
tab1, tab2 = st.tabs(["📖 Introduction", "💬 Agent Chat"])

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
    
    st.subheader("2. Typical Workflow")
    st.markdown("""
        <div style="font-size: 1.2rem; line-height: 1.6;">
            To get the most out of the Agentic XAI platform, we recommend following this typical workflow:
            <ol style="margin-top: 10px;">
                <li><strong>Upload Data & Model:</strong> Begin by uploading your dataset and your trained CatBoost model using the configuration sidebar.</li>
                <li><strong>Data Understanding:</strong> Chat with the Data Understanding Agent to learn the dataset's metadata. Check that the problem type (classification or regression) and the target variable are detected correctly.</li>
                <li><strong>Global Explanation:</strong> Ask the Global Explainer Agent for the overall feature importance to understand which features drive the model's decisions globally.</li>
                <li><strong>Local Explanation:</strong> Dive deeper by asking the Local Explainer Agent why the model made a specific prediction for a particular entry/row in your dataset.</li>
                <li><strong>Ethical Analysis:</strong> Conclude by asking the Ethic & Fairness Analysis Agent to check for potential biases against sensitive attributes (e.g., gender, age).</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    st.subheader("3. Explanation of each Agent and tools they use")
    
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
    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                # Check for images in history
                img_path = extract_image_path(message.content)
                if img_path and os.path.exists(img_path):
                    st.image(img_path)
    
    # Chat Input
    if prompt := st.chat_input("Ask about the dataset feature importance, or specific user predictions..."):
        # 1. Add user message
        user_msg = HumanMessage(content=prompt)
        st.session_state.messages.append(user_msg)
        
        with st.chat_message("user"):
            st.markdown(prompt)
    
        # 2. Invoke Agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare Initial State
                    # We pass the CURRENT session objects.
                    # Note: The agent might have internal persistence if we used a checkpointer, 
                    # but here we are passing the full history every time (in-memory).
                    initial_state = {
                        "messages": st.session_state.messages,
                        "df": st.session_state.df,
                        "model": st.session_state.model,
                        "target_variable": st.session_state.get("target_variable"),
                        "problem_type": st.session_state.get("problem_type", "classification")
                    }
    
                    # Invoke Graph
                    result = agent_app.invoke(initial_state)
                    
                    # Update History
                    full_history = result['messages']
                    
                    # Identify NEW messages
                    # We calculate the difference between the full history and what we had before (minus the user message we just appended)
                    # actually, we just appended 1 user message. So previous length was len(st.session_state.messages) - 1
                    # But safer way: just take everything after the last known message count.
                    # Since we appended 1 message (User), the "start index" for new Agent messages is len(st.session_state.messages)
                    
                    new_messages = full_history[len(st.session_state.messages):]
                    
                    # Update global state
                    st.session_state.messages = full_history
                    
                    # Render ALL new messages
                    for msg in new_messages:
                        if isinstance(msg, AIMessage):
                            st.markdown(msg.content)
                            # Check for Images to Render
                            img_path = extract_image_path(msg.content)
                            if img_path:
                                if os.path.exists(img_path):
                                    st.image(img_path, caption="Generated Explanation Plot")
                                else:
                                    st.warning(f"Plot image not found at: {img_path}")
                                    
                        elif isinstance(msg, ToolMessage):
                            # Optional: Show tool outputs if debugging, or just keep hidden.
                            # For XAI, usually the tool output is just data for the agent. 
                            # Unless it's an image path?
                            # The tools return string paths usually.
                            pass
                    
                    # If no AI message was found in new_messages (unlikely), warn.
                    if not any(isinstance(m, AIMessage) for m in new_messages):
                         st.warning("Agent finished without a text response.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
