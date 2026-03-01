from agent.state import XAIState
from agent.tools.global_tools import compute_shap_values, generate_shap_summary_plot
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import pandas as pd
import numpy as np
import os
import uuid
import matplotlib.pyplot as plt
from typing import Annotated, Any

from Explainable_AI.xai import compute_ig_importance_global
from langchain_core.runnables.config import RunnableConfig

system_prompt = """
You are a data scientist who is explaining the global feature importance.
1. CRITICAL RULE: If the data/model is sequential or time-series, or if the model is a Deep Learning PyTorch model (like an LSTM), you MUST use the `get_global_feature_importance_ig` tool. SHAP will crash in this environment.
2. Only use the `get_global_feature_importance_shap` tool if the data is standard Tabular/CatBoost.
3. Do not hallucinate feature importance without calling the tool.
4. After the tool returns, you may obtain the global feature importance or some plots, then explain the results in clear, straight-forward way, for example, which features are most important, what are the driving features, etc.
"""

@tool
def get_global_feature_importance_shap(
    state: Annotated[dict, InjectedState], config: RunnableConfig
):
    """
    Calculates Global SHAP feature importance and generates a summary plot.
    TreeExplainer is best for CatBoost/XGBoost/RF
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
        
    # Sample for speed
    X_sample = X.iloc[:500]
    
    try:
        shap_values = compute_shap_values(model, X_sample)
        plot_path = generate_shap_summary_plot(shap_values, X_sample)
        
        # --- NEW: Calculate Top 9 + Others ---
        # 1. mean absolute SHAP value per feature
        global_importance = np.abs(shap_values.values).mean(axis=0)
        feature_names = X_sample.columns.tolist()
        
        # Map feature -> importance
        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": global_importance
        }).sort_values(by="importance", ascending=False)
        
        # 2. Get Top 9
        top_9 = feat_imp.head(9)
        
        # 3. Aggregate "Others"
        others_importance = feat_imp.iloc[9:]['importance'].sum() if len(feat_imp) > 9 else 0.0
        
        # 4. Format Output Text
        text_list = "### Top Driving Features (Global Importance):\\n"
        for i, row in enumerate(top_9.itertuples(), 1):
            text_list += f"{i}. **{row.feature}**: {row.importance:.4f}\\n"
            
        if others_importance > 0:
            text_list += f"10. **Others**: {others_importance:.4f}\\n"
            
        return f"{text_list}\\n\\nI have also generated the Global SHAP Summary Plot. It is saved here: `{plot_path}`."
    except Exception as e:
        return f"CRITICAL TOOL ERROR: SHAP failed with '{e}'. This usually means you tried to run SHAP on a neural network or time-series model (like LSTM using PyTorch) without providing a valid background masker. DO NOT USE SHAP FOR THIS DATASET. You must switch to `get_global_feature_importance_ig` instead."

@tool
def get_global_feature_importance_ig(
    state: Annotated[dict, InjectedState], config: RunnableConfig
):
    """
    Calculates Global Integrated Gradients (IG) feature importance and generates a summary plot.
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
        
    X_sample = X[:500] if hasattr(X, "iloc") else X[:500]
    
    try:
        feature_cols_config = config.get("configurable", {}).get("feature_cols")
        
        if feature_cols_config is not None:
            feature_names = feature_cols_config
        elif hasattr(X_sample, "columns"):
            feature_names = X_sample.columns.tolist()
        elif "feature_cols" in state:
            feature_names = state["feature_cols"]
        else:
            feature_names = None
            
        all_attr, global_feat_imp_signed, global_year_imp_abs, fig = compute_ig_importance_global(
            model=model, X_test=X_sample, feature_cols=feature_names
        )
        
        save_dir = "artifacts"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f"ig_global_{uuid.uuid4().hex}.png"
        plot_path = os.path.join(save_dir, filename)
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        
        global_importance = np.abs(global_feat_imp_signed)
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(global_importance))]
            
        feat_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": global_importance
        }).sort_values(by="importance", ascending=False)
        
        top_9 = feat_imp.head(9)
        others_importance = feat_imp.iloc[9:]['importance'].sum() if len(feat_imp) > 9 else 0.0
        
        text_list = "### Top Contributing Features (Global IG):\\n"
        for i, row in enumerate(top_9.itertuples(), 1):
            text_list += f"{i}. **{row.feature}**: {row.importance:.4f}\\n"
            
        if others_importance > 0:
            text_list += f"10. **Others**: {others_importance:.4f}\\n"
            
        return f"{text_list}\\n\\nI have also generated the Global Integrated Gradients Plot. It is saved here: `{plot_path}`."
    except Exception as e:
        return f"Error generating global explanation: {e}"

def global_explainer_agent(state: XAIState, config: RunnableConfig):
    """
    Global Explainer Node:
    - Calculates Global SHAP values (on a sample).
    - Generates a summary plot.
    - Uses LLM to explain the top driving features.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    lang = config.get("configurable", {}).get("lang", "en")
    lang_instruction = (
        "\n\nIMPORTANT: The user interface is set to Chinese (中文). You MUST respond entirely in Simplified Chinese (简体中文). Explain all feature importance results, charts, and insights in Chinese."
        if lang == "zh" else
        "\n\nRespond in English."
    )
    full_system_prompt = system_prompt + lang_instruction

    # Bind Dictionary-based tool
    tools = [get_global_feature_importance_shap, get_global_feature_importance_ig]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state.get('messages', [HumanMessage(content="Please show me the global feature importance.")])
    messages = [SystemMessage(content=full_system_prompt)] + messages
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}
