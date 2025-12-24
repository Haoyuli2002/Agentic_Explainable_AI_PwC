import json
import matplotlib.pyplot as plt
import pandas as pd
from typing import Literal, List, Optional, Annotated, Any
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agent.state import XAIState
# from agent.tools.fairness_tools import compute_fairness_metrics, plot_fairness_metrics # Not used directly anymore

from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference, true_positive_rate, false_positive_rate, demographic_parity_ratio
from sklearn.metrics import accuracy_score

system_prompt = """You are an expert AI Ethicist and Fairness Evaluator.
Your goal is to assess the machine learning model for potential biases and fairness issues.

You have access to the followings tools:
1. `run_ethic_analysis` which can calculate fairness metrics (like Demographic Parity, Equalized Odds) and save them to the state. Call this tool if there's nothing about fairness in the state.
2. `visualize_ethic_analysis` which can generate visualizations of the fairness metrics. Call this tool after the `run_ethic_analysis` tool or user wants to have visualizations.

To use this tool, you MUST identify which features in the dataset might be "sensitive" or "protected" attributes (e.g., Age, Gender, Race, Socioeconomic status) based on the dataset description or column names.

Your workflow:
1.  Analyze the available feature names (from the metadata provided in context).
2.  Decide which features are sensitive.
3.  Ask the user to confirm the sensitive features and which one they want to investigate.
Once the user has chosen a sensitive feature (if not specified, choose the first one), you could:
    *   Call `run_ethic_analysis` with the sensitive feature name.
    *   Interpret the results returned by the tool.
    *   Call `visualize_ethic_analysis` to generate plots.
    *   Provide a concise summary of your findings and any recommendations.
"""

@tool
def run_ethic_analysis(sensitive_attr: str, state: Annotated[dict, InjectedState], spd_threshold: float = 0.1, di_low: float = 0.8, di_high: float = 1.2):
    """
    Evaluate fairness using Fairlearn with visualization and textual report.
    
    Parameters
    ----------
    sensitive_attr : str
        Protected / sensitive attribute (e.g., country, gender, etc.).
    spd_threshold : float
        Acceptable absolute threshold for SPD & EOD.
    di_low, di_high : float
        Acceptable interval for DI.
        
    Returns
    -------
    JSON string containing metrics and fairness evaluation.
    """
    model = state.get("model", None)
    df = state.get("df", None)
    target = state.get("target_variable", None)
    
    if model is None or df is None:
        return "Error: Model or Data not found in state."

    if target and target in df.columns:
        X = df.drop(columns=[target])
    else:
        X = df

    if sensitive_attr not in X.columns:
        return f"Error: Sensitive attribute '{sensitive_attr}' not found in data."

    # Ensure target is 0/1 for Fairlearn
    # We use a LabelEncoder to handle "yes"/"no", "True"/"False", etc.
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # Fit on all target values to be safe
    y_true = le.fit_transform(df[target])
    
    # Check problem type and get predictions
    if state.get("problem_type") == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] > 1:
            # Assume positive class is index 1
            proba = proba[:, 1]
        y_pred = (proba >= 0.5).astype(int)
    else:
        y_pred = model.predict(X)
        # If model returns strings, encode them
        if y_pred.dtype == "object" or isinstance(y_pred[0], str):
             # Use the SAME encoder
             try:
                 y_pred = le.transform(y_pred)
             except:
                 # If unseen labels, fallback or re-fit (though unexpected for a trained model)
                 y_pred = le.fit_transform(y_pred)

    sensitive_data = X[sensitive_attr]

    # Metrics for MetricFrame (Group-wise)
    # Remove SPD and EOD from here as they are global comparison metrics
    metrics = {
        "accuracy": accuracy_score,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "selection_rate": selection_rate,
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_data
    )

    # Global fairness metrics
    spd = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_data
    )
    # demographic_parity_ratio sometimes fails if denominator is 0, handle safely
    try:
        di = demographic_parity_ratio(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_data
        )
    except Exception:
        di = 0.0

    eod = equalized_odds_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_data
    )

    result_dict = {
        "overall": mf.overall.to_dict(),
        "by_group": mf.by_group.to_dict(), # Convert DataFrame to dict
        "fairness": {"SPD": float(spd), "EOD": float(eod), "DI": float(di)},
        "sensitive_attr": sensitive_attr
    }
    
    return json.dumps(result_dict)

@tool
def visualize_ethic_analysis(state: Annotated[dict, InjectedState], spd_threshold: float = 0.1, di_low: float = 0.8, di_high: float = 1.2):
    """
    Visualize the fairness metrics extracted from the state. 
    Can be called after run_ethic_analysis.
    """
    fairness_metrics = state.get("fairness_metrics", None)
    if fairness_metrics is None:
        return "Error: Fairness metrics not found in state. Run ethic_analysis first."
    
    sensitive_attr = fairness_metrics.get("sensitive_attr", "Attribute")
    by_group = pd.DataFrame(fairness_metrics.get("by_group", {}))
    fairness = fairness_metrics.get("fairness", {})
    
    spd = fairness.get("SPD", 0)
    eod = fairness.get("EOD", 0)
    di = fairness.get("DI", 0)

    # Visualization (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # (1) Accuracy by group
    if "accuracy" in by_group.columns:
        axes[0].bar(by_group.index.astype(str), by_group["accuracy"], color="#1e88e5")
        axes[0].set_title(f"Accuracy by {sensitive_attr}")
        axes[0].set_ylabel("Accuracy")
        axes[0].tick_params(axis="x", rotation=45)

    # (2) Selection Rate by group
    if "selection_rate" in by_group.columns:
        axes[1].bar(by_group.index.astype(str), by_group["selection_rate"], color="#43a047")
        axes[1].set_title(f"Selection Rate by {sensitive_attr}")
        axes[1].set_ylabel("Selection Rate")
        axes[1].tick_params(axis="x", rotation=45)

    # (3) Fairness metrics bar
    fairness_vals = [spd, eod, di]
    fairness_labels = ["SPD", "EOD", "DI"]

    colors = []
    for val, label in zip(fairness_vals, fairness_labels):
        if label in ["SPD", "EOD"]:
            colors.append("#43a047" if abs(val) <= spd_threshold else "#e53935")
        else:  # DI
            colors.append("#43a047" if di_low <= val <= di_high else "#e53935")

    axes[2].bar(fairness_labels, fairness_vals, color=colors)
    axes[2].set_title("Fairness Metrics Overview")
    axes[2].axhline(0, color='grey', linewidth=0.8)
    
    plt.tight_layout()
    plot_path = "fairness_analysis_plot.png"
    plt.savefig(plot_path)
    plt.close()
    
    return f"Plot generated and saved to {plot_path}"
    

def ethic_analysis_agent(state: XAIState):
    """
    Node for the Ethical Analysis Agent.
    """
    # Save the fairness metrics to the state if available from previous tool call
    messages = state.get("messages", [])
    last_tool = None
    fairness_metrics = state.get("fairness_metrics")
    
    # Check if the last message was a tool output from run_ethic_analysis
    # If so, parse it and update fairness_metrics
    if messages and isinstance(messages[-1], ToolMessage) and messages[-1].name == "run_ethic_analysis":
        try:
            fairness_metrics = json.loads(messages[-1].content)
        except:
            pass
            
    model = state.get("model")
    df = state.get("df")
    
    # Check if necessary data is available
    if model is None or df is None:
        return {"messages": [SystemMessage(content="Error: Model or Data not found in state. Please run data understanding first.")]}
    
    # Bind tool to LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools([run_ethic_analysis, visualize_ethic_analysis])
        
    # Construct the full context
    full_messages = [SystemMessage(content=system_prompt)] + messages
    
    # Invoke LLM
    response = llm_with_tools.invoke(full_messages)

    # Return
    return {
        "messages": [response], 
        "fairness_metrics": fairness_metrics
    }
