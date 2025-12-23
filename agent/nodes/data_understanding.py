from agent.state import XAIState
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
import pandas as pd
import json
import io
from typing import Annotated, Optional

@tool
def get_dataset_samples(state: Annotated[dict, InjectedState]):
    """
    Reads the dataset samples and schema. 
    Use this to understand the data structure.
    """
    df = state.get("df")
    if df is None:
        return "Error: DataFrame not found in state."
        
    try:
        buffer = io.StringIO()
        df.info(buf=buffer)
        df_info = buffer.getvalue()
        df_head = df.head(10).to_string()
        return f"--- Samples ---\n{df_head}\n\n--- Info ---\n{df_info}"
    except Exception as e:
        return f"Error: {e}"

@tool
def update_metadata(
    target_variable: str,
    problem_type: str,
    dataset_format: str,
    dataset_description: str,
    feature_description: dict
):
    """
    Updates the global state with dataset metadata.
    Call this tool when you are confident about your analysis results, or the user states the value of variables in the metadata.
    """
    # This is a placeholder function, as we cannot use tool to update the state.
    # Instead, once this function is called, we wil update the metadata within the agent node.
    return "Metadata captured. Ready to update state."

def data_understanding_agent(state: XAIState):
    """
    Data Understanding Agent (ReAct).
    1. Inspects data with `get_dataset_samples`.
    2. Updates state with `update_metadata`.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    tools = [get_dataset_samples, update_metadata] # Bind both
    llm_with_tools = llm.bind_tools(tools)
    
    messages = state.get("messages", [])
    
    # 1. System Prompt
    system_prompt = """
    You are a Data Scientist.
    1. First, call `get_dataset_samples` to inspect the dataframe.
    2. Analyze the data to determine: Target Variable, Problem Type (regression/classification), Format (tabular/temporal), and Descripton.
    3. If the values of the metadata is unclear or cannot be determined, use None instead. Also tell user which information is unclear and ask them for more information.
    4. FINALLY, call `update_metadata` to save these findings to the system state.
    """

    if not messages or not isinstance(messages[0], SystemMessage):
         messages = [SystemMessage(content=system_prompt)] + messages
         
    # 2. Invoke LLM
    response = llm_with_tools.invoke(messages)
    
    # 3. Post-Processing: Detect State Updates
    # In LangGraph, the 'Agent' node is responsible for emitting state updates.
    
    # If the LAST message was a ToolMessage from 'update_metadata', 
    # then we know the tool logic succeeded. We should now emit the state update 
    # based on the *arguments* of that tool call (which are in the AIMessage just before).
    # [ ... AIMessage(tool_calls=[update_metadata(args)]), ToolMessage(content="Success") ]
    
    # We check the last few messages to see if we need to sync state.
    
    output_state = {"messages": [response]}
    
    # Scan recent messages to see if we recently effectively called 'update_metadata'
    # We grab the latest args and apply them.
    # (We iterate backwards to find the last update)
    if isinstance(response, AIMessage) and not response.tool_calls:
        # If agent is done (no more tools), let's ensure state is synced from the last tool call
        for m in reversed(messages):
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    if tc['name'] == 'update_metadata':
                        args = tc['args']
                        # Apply to Global State Return
                        output_state.update({
                            "target_variable": args.get("target_variable"),
                            "problem_type": args.get("problem_type"),
                            "dataset_format": args.get("dataset_format"),
                            "feature_description": args.get("feature_description"),
                            "dataset_description": args.get("dataset_description"),
                            "analysis_mode": "global" # Ready to move on
                        })
                        return output_state
                        
    return output_state
