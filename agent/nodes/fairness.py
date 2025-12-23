from agent.state import XAIState
from agent import context
from agent.tools.fairness_tools import calculate_demographic_parity
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
import pandas as pd

@tool
def check_demographic_parity(sensitive_attribute: str):
    """
    Calculate Demographic Parity difference for a given sensitive attribute (e.g. 'gender', 'marital').
    Use this to check for bias in the model predictions.
    """
    try:
        model = context.get_model()
        df = context.get_dataset()
        target = context.get_target_column()
    except ValueError as e:
        return f"Error: {e}"

    if target and target in df.columns:
        X = df.drop(columns=[target])
        y = getattr(model, "predict", lambda x: [0]*len(x))(X) 
    else:
        X = df
        y = model.predict(X)
        
    # We need to construct a df with 'prediction' and 'sensitive_attribute'
    # Use a sample for speed if large
    X_sample = X.iloc[:1000].copy()
    y_sample = y[:1000]
    
    X_sample['prediction'] = y_sample
    
    if sensitive_attribute not in X_sample.columns:
         return f"Error: Column '{sensitive_attribute}' not found."

    metrics = calculate_demographic_parity(X_sample, sensitive_attribute, 'prediction', favorable_label=1)
    
    return f"**Fairness Analysis for '{sensitive_attribute}'**\n\nDemographic Parity Difference: {metrics.get('demographic_parity_difference', 'N/A'):.4f}"


def fairness_agent(state: XAIState):
    """
    Fairness Analysis Agent:
    - Identifies potentially sensitive attributes.
    - Uses tool to calculate bias metrics.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    tools = [check_demographic_parity]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state['messages']
    df = state.get('df') 
    columns = list(df.columns) if df is not None else []
    
    prompt = f"""
    You are a Fairness Expert. The dataset has columns: {columns}.
    Your goal is to identify potential fairness issues.
    
    1. Identify any likely sensitive attributes (e.g., age, gender, race, marital status).
    2. Use the `check_demographic_parity` tool for the most critical one.
    """
    
    response = llm_with_tools.invoke([SystemMessage(content=prompt)] + messages)
    
    # Standard ReAct: Return the response. Graph handles Tool execution.
    return {"messages": [response]}
