from agent.state import XAIState
from agent.tools.global_tools import compute_shap_values, generate_shap_summary_plot
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import pandas as pd
import numpy as np
from typing import Annotated, Any

system_prompt = """
You are a data scientist who is explaining the global feature importance.
1. If the user asks for global feature importance or overall model behavior, you could call tools like 'get_global_feature_importance_shap' to run SHAP analysis.
2. Do not hallucinate feature importance without calling the tool.
3. After the tool returns, you may obtain the global feature importance or some plots, then explain the results in clear, straight-forward way, for example, which features are most important, what are the driving features, etc.
"""

@tool
def get_global_feature_importance_shap(
    state: Annotated[dict, InjectedState],
):
    """
    Calculates Global SHAP feature importance and generates a summary plot.
    TreeExplainer is best for CatBoost/XGBoost/RF
    """
    df = state.get("df", None)
    model = state.get("model", None)
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
        return f"Error generating global explanation: {e}"

def global_explainer_agent(state: XAIState):
    """
    Global Explainer Node:
    - Calculates Global SHAP values (on a sample).
    - Generates a summary plot.
    - Uses LLM to explain the top driving features.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Bind Dictionary-based tool
    tools = [get_global_feature_importance_shap]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state.get('messages', [HumanMessage(content="Please show me the global feature importance.")])
    messages = [SystemMessage(content=system_prompt)] + messages
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}
