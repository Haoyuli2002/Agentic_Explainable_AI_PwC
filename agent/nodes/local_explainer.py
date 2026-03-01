from agent.state import XAIState
from agent.tools.global_tools import compute_shap_values
from agent.tools.local_tools import generate_shap_waterfall_plot, compute_lime_explanation, generate_lime_plot
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import pandas as pd
import numpy as np
import os
import uuid
import matplotlib.pyplot as plt
from typing import Annotated, Any

from Explainable_AI.xai import compute_ig_importance_local
from langchain_core.runnables.config import RunnableConfig

@tool
def run_shap_explanation(
    user_id: int,
    state: Annotated[dict, InjectedState], config: RunnableConfig
):
    """
    Generates a SHAP Waterfall plot to explain a specific user's prediction.
    Best for understanding contribution of each feature to the final score.
    """
    df = config.get("configurable", {}).get("df")
    model = config.get("configurable", {}).get("model")
    target = state.get("target_variable", None)
    
    if df is None or model is None:
        return "Error: DataFrame or Model not found in state."

    # 2. Prepare Data
    if target and target in df.columns:
        X = df.drop(columns=[target])
    else:
        X = df
        
    if user_id >= len(X):
        return f"Error: User ID {user_id} not found."

    # 3. Execution (Optimized Single Row)
    # 3. Execution (Optimized Single Row)
    X_single = X.iloc[[user_id]]
    try:
        shap_values = compute_shap_values(model, X_single)
        plot_path = generate_shap_waterfall_plot(shap_values, 0) # Index 0 of single row return

        # --- SHAP Feature List for User ---
        instance_shap = shap_values[0].values
        feature_names = X_single.columns.tolist()

        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": instance_shap,
            "abs_importance": np.abs(instance_shap)
        }).sort_values(by="abs_importance", ascending=False)

        top_9 = feat_imp.head(9)
        others_importance = feat_imp.iloc[9:]['importance'].sum() if len(feat_imp) > 9 else 0.0

        text_list = f"### Top Contributing Features for User {user_id} (Local SHAP):\\n"
        for i, row in enumerate(top_9.itertuples(), 1):
            text_list += f"{i}. **{row.feature}**: {row.importance:.4f}\\n"
        
        if len(feat_imp) > 9:
            text_list += f"10. **Others**: {others_importance:.4f}\\n"
        
        return f"{text_list}\\n\\nI have also generated the Local SHAP Waterfall Plot for User {user_id}. It is saved here: `{plot_path}`."
    except Exception as e:
        return f"CRITICAL TOOL ERROR: SHAP failed with '{e}'. This usually means you tried to run SHAP on a neural network or time-series model (like LSTM using PyTorch) without providing a valid background masker. DO NOT USE SHAP FOR THIS DATASET. You must switch to `run_ig_explanation` instead."

@tool
def run_lime_explanation(
    user_id: int,
    state: Annotated[dict, InjectedState], config: RunnableConfig
):
    """
    Generates a LIME explanation for a specific user's prediction.
    Useful for local linear approximation (Alternative to SHAP).
    """
    df = config.get("configurable", {}).get("df")
    model = config.get("configurable", {}).get("model")
    target = state.get("target_variable", None)
    
    if df is None or model is None:
        return "Error: DataFrame or Model not found in state."

    if target and target in df.columns:
        X = df.drop(columns=[target])
    else:
        X = df

    if user_id >= len(X):
        return f"Error: User ID {user_id} not found."

    lime_exp = compute_lime_explanation(model, X, X.iloc[user_id])
    plot_path = generate_lime_plot(lime_exp)
    
    # --- NEW: LIME Feature List for User ---
    # lime_exp.as_list() returns list of (feature_condition, weight) tuples
    lime_list = lime_exp.as_list()
    
    # Sort by absolute weight (LIME usually returns sorted, but just in case)
    lime_list.sort(key=lambda x: abs(x[1]), reverse=True)
    
    top_9 = lime_list[:9]
    others_list = lime_list[9:]
    others_importance = sum([x[1] for x in others_list]) if others_list else 0.0
    
    text_list = f"### Top Contributing Features for User {user_id} (LIME):\\n"
    for i, (feat, weight) in enumerate(top_9, 1):
        text_list += f"{i}. **{feat}**: {weight:.4f}\\n"

    if others_list:
         text_list += f"10. **Others**: {others_importance:.4f}\\n"

    return f"{text_list}\\n\\nI have generated the LIME explanation for User {user_id}. It is saved at `{plot_path}`."

@tool
def run_ig_explanation(
    user_id: int,
    state: Annotated[dict, InjectedState], config: RunnableConfig
):
    """
    Generates a Local Integrated Gradients explanation for a specific user's prediction.
    Best for sequential/time-series deep learning models.
    """
    df = config.get("configurable", {}).get("df")
    model = config.get("configurable", {}).get("model")
    X_padded = config.get("configurable", {}).get("X_padded")
    target = state.get("target_variable", None)
    
    if X_padded is not None:
        X = X_padded
    elif df is not None:
        if target and target in df.columns:
            X = df.drop(columns=[target])
        else:
            X = df
    else:
        return "Error: Data not found in state."

    if user_id >= len(X):
        return f"Error: User ID {user_id} not found."

    if hasattr(X, "iloc"):
        X_single = X.iloc[[user_id]]
    else:
        X_single = X[user_id:user_id+1]
        
    try:
        feature_cols_config = config.get("configurable", {}).get("feature_cols")

        if feature_cols_config is not None:
            feature_names = feature_cols_config
        elif hasattr(X, "columns"):
            feature_names = X.columns.tolist()
        elif "feature_cols" in state:
            feature_names = state["feature_cols"]
        else:
            feature_names = [f"Feature {i}" for i in range(X_single.shape[-1])]

        attr, feature_importance_signed, fig = compute_ig_importance_local(
            model=model, X_sample=X_single[0] if not hasattr(X, "iloc") else X_single.values[0], feature_cols=feature_names
        )

        save_dir = "artifacts"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        filename = f"ig_local_{user_id}_{uuid.uuid4().hex}.png"
        plot_path = os.path.join(save_dir, filename)
        
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)

        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importance_signed,
            "abs_importance": np.abs(feature_importance_signed)
        }).sort_values(by="abs_importance", ascending=False)

        top_9 = feat_imp.head(9)
        others_importance = feat_imp.iloc[9:]['importance'].sum() if len(feat_imp) > 9 else 0.0

        text_list = f"### Top Contributing Features for User {user_id} (Local IG):\\n"
        for i, row in enumerate(top_9.itertuples(), 1):
            text_list += f"{i}. **{row.feature}**: {row.importance:.4f}\\n"
        
        if len(feat_imp) > 9:
            text_list += f"10. **Others**: {others_importance:.4f}\\n"
        
        return f"{text_list}\\n\\nI have also generated the Local Integrated Gradients plots for User {user_id}. They are saved at `{plot_path}`."
    except Exception as e:
        return f"Error generating IG explanation: {e}"



def local_explainer_agent(state: XAIState, config: RunnableConfig):
    """
    Local Explainer Agent:
    - Decides which explanation method to use (SHAP vs LIME vs IG).
    - Executes the chosen tool.
    - Returns the explanation and plot path.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    lang = config.get("configurable", {}).get("lang", "en")
    lang_instruction = (
        " IMPORTANT: The user interface is set to Chinese (中文). You MUST respond entirely in Simplified Chinese (简体中文). Explain all predictions, feature contributions, and insights in Chinese."
        if lang == "zh" else
        " Respond in English."
    )

    # Bind fully functional tools
    tools = [run_shap_explanation, run_lime_explanation, run_ig_explanation]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state['messages']
    user_id = state.get('user_id')
    
    sys_msg = SystemMessage(content=(
        "You are a Local Explanation Expert. You explain specific predictions. "
        "You have access to SHAP, LIME, and Integrated Gradients (IG) tools. "
        "CRITICAL RULE: If the analysis_mode/format is 'time-series' or 'temporal', or if the model is a Deep Learning PyTorch model (like an LSTM), "
        "you ABSOLUTELY MUST USE the `run_ig_explanation` tool. SHAP and LIME will crash and burn. "
        "Only use SHAP for standard Tabular/CatBoost data."
        + lang_instruction
    ))
    
    if user_id is not None:
         # Contextual hint for the LLM
         messages = messages + [SystemMessage(content=f"The user is asking about User ID {user_id}.")]
    
    response = llm_with_tools.invoke([sys_msg] + messages)
    
    return {"messages": [response]}
