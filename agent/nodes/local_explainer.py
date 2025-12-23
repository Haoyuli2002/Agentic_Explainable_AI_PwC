from agent.state import XAIState
from agent.tools.global_tools import compute_shap_values
from agent.tools.local_tools import generate_shap_waterfall_plot, compute_lime_explanation, generate_lime_plot
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState, ToolNode

import pandas as pd
from typing import Annotated, Any

@tool
def run_shap_explanation(
    user_id: int,
    state: Annotated[dict, InjectedState],
):
    """
    Generates a SHAP Waterfall plot to explain a specific user's prediction.
    Best for understanding contribution of each feature to the final score.
    """
    df = state.get("df", None)
    model = state.get("model", None)
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
    X_single = X.iloc[[user_id]]
    shap_values = compute_shap_values(model, X_single)
    plot_path = generate_shap_waterfall_plot(shap_values, 0) # Index 0 of single row return
    
    return f"I have generated the SHAP Waterfall plot for User {user_id}. It is saved at `{plot_path}`."

@tool
def run_lime_explanation(
    user_id: int,
    state: Annotated[dict, InjectedState],
):
    """
    Generates a LIME explanation for a specific user's prediction.
    Useful for local linear approximation (Alternative to SHAP).
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

    if user_id >= len(X):
        return f"Error: User ID {user_id} not found."

    lime_exp = compute_lime_explanation(model, X, X.iloc[user_id])
    plot_path = generate_lime_plot(lime_exp)
    
    return f"I have generated the LIME explanation for User {user_id}. It is saved at `{plot_path}`."


def local_explainer_agent(state: XAIState):
    """
    Local Explainer Agent:
    - Decides which explanation method to use (SHAP vs LIME).
    - Executes the chosen tool.
    - Returns the explanation and plot path.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Bind fully functional tools
    tools = [run_shap_explanation, run_lime_explanation]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state['messages']
    user_id = state.get('user_id')
    
    sys_msg = SystemMessage(content="You are a Local Explanation Expert. You explain specific predictions. "
                                    "You have access to SHAP and LIME tools. "
                                    "Default to SHAP unless the user specifically asks for LIME.")
    
    if user_id is not None:
         # Contextual hint for the LLM
         messages = messages + [SystemMessage(content=f"The user is asking about User ID {user_id}.")]
    
    response = llm_with_tools.invoke([sys_msg] + messages)
    
    return {"messages": [response]}
