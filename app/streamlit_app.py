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


# --- Chat Interface ---

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
