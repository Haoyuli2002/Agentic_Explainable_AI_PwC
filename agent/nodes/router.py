from agent.state import XAIState
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json

def router_agent(state: XAIState):
    """
    Router Node: Classifies the user's latest query to determine the analysis path.
    """
    messages = state.get('messages', [])
    if not messages:
        # If no messages, assume new session -> Data Understanding
        return {"analysis_mode": "data_understanding"}
        
    # Find the last HumanMessage (User Query)
    last_user_message = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_message = m.content
            break
            
    if not last_user_message:
         return {"analysis_mode": "data_understanding"}

    last_message = last_user_message
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = f"""
    You are an Intent Classifier for an Explainable AI System.
    The user said: "{last_message}"
    
    Classify the intent into one of these modes:
    - **data_understanding**: Initial analysis, "analyze this dataset", "what are the columns?", "show samples", schema/stats questions. Or if the user just uploaded data or want to modify or correct the metadata.
    - **global**: Overall model behavior, feature importance, summary plots, "how does the model work?".
    - **local**: Explanation for a specific instance/user/customer. Look for IDs (e.g., "User 5", "row 10").
    - **counterfactual**: "What if?", "Actionable changes", "How to change the outcome?".
    
    If the user mentions a specific ID (integer), extract it as `user_id`. Otherwise `0` as default.
    
    Return a strict JSON object:
    {{
        "mode": "data_understanding" | "global" | "local" | "counterfactual",
        "user_id": int or 0,
        "reason": "brief explanation"
    }}
    """
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        mode = result.get("mode", "data_understanding")
        
        # Determine fallback if LLM hallucinates invalid mode
        valid_modes = ["data_understanding", "global", "local", "counterfactual"]
        if mode not in valid_modes:
            mode = "data_understanding"

        return {
            "analysis_mode": mode,
            "user_id": result.get("user_id"),
        }
    except Exception as e:
        print(f"Router Error: {e}")
        return {
            "analysis_mode": "data_understanding",
            "user_id": 0
        }
