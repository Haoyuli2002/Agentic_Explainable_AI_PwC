from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage
from agent.state import XAIState

# Import Nodes
from agent.nodes.data_understanding import data_understanding_agent, get_dataset_samples, update_metadata
from agent.nodes.router import router_agent
from agent.nodes.global_explainer import global_explainer_agent, get_global_feature_importance_shap
from agent.nodes.local_explainer import local_explainer_agent, run_shap_explanation, run_lime_explanation
# from agent.nodes.fairness import fairness_agent, check_demographic_parity # Disabled for now

data_tools = [get_dataset_samples, update_metadata]
data_tools_node = ToolNode(data_tools)

global_tools = [get_global_feature_importance_shap]
global_tools_node = ToolNode(global_tools)

local_tools = [run_shap_explanation, run_lime_explanation]
local_tools_node = ToolNode(local_tools)

def route_based_on_intent(state: XAIState):
    """
    Conditional Edge Logic: Reads 'analysis_mode' from Router output
    and directs to the correct agent.
    """
    mode = state.get("analysis_mode", "data_understanding")
    
    if mode == "local":
        return "local_explainer"
    elif mode == "global":
        return "global_explainer"
    # Fairness and Counterfactual disabled for now
    # elif mode == "fairness":
    #     return "fairness"
    # elif mode == "counterfactual":
    #     return "counterfactual" 
    else:
        return "data_understanding" # Default to global if unsure or fairness/counterfactual requested

# Build Graph
workflow = StateGraph(XAIState)

# Add Nodes
workflow.add_node("data_understanding", data_understanding_agent)
workflow.add_node("data_tools", data_tools_node)
workflow.add_node("router", router_agent)

workflow.add_node("global_explainer", global_explainer_agent)
workflow.add_node("global_tools", global_tools_node)

workflow.add_node("local_explainer", local_explainer_agent)
workflow.add_node("local_tools", local_tools_node)

# --- Edges ---

# 1. Start -> Data Understanding
workflow.set_entry_point("router")

# 2. Data Understanding Loop
workflow.add_conditional_edges(
    "data_understanding",
    tools_condition,
    {
        "tools": "data_tools",
        "__end__": END # Proceed to END when done (Yield to User)
    }
)
workflow.add_edge("data_tools", "data_understanding")

# 3. Router -> Specific Agent (Global or Local)
workflow.add_conditional_edges(
    "router",
    route_based_on_intent,
    {
        "local_explainer": "local_explainer",
        "global_explainer": "global_explainer",
        "data_understanding": "data_understanding"
    }
)

# 4. Global Explainer ReAct Loop
# Agent -> ToolNode (if tool call) OR End
workflow.add_conditional_edges(
    "global_explainer",
    tools_condition,
    {
        "tools": "global_tools", 
        "__end__": END
    }
)
# ToolNode -> Agent (Loop back)
workflow.add_edge("global_tools", "global_explainer")


# 5. Local Explainer ReAct Loop
workflow.add_conditional_edges(
    "local_explainer",
    tools_condition,
    {
        "tools": "local_tools", 
        "__end__": END
    }
)
workflow.add_edge("local_tools", "local_explainer")

app = workflow.compile()
