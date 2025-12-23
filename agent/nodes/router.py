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
        return {"analysis_mode": "global"} # Default
        
    # Find the last HumanMessage (User Query)
    last_user_message = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_user_message = m.content
            break
            
    if not last_user_message:
         # Fallback if no user message found (rare)
         return {"analysis_mode": "global"}

    last_message = last_user_message
    
    # Context from Data Understanding (to help router know valid targets/IDs if needed in future)
    # For now, we just route based on intent.
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = f"""
    You are an Intent Classifier for an Explainable AI System.
    The user said: "{last_message}"
    
    Classify the intent into one of these modes:
    - **global**: Overall model behavior, feature importance, summary plots.
    - **local**: Explanation for a specific instance/user/customer. Look for IDs (e.g., "User 5", "row 10").
    - **fairness**: Bias, demographic parity, ethical concerns, "is the model fair or ethical?".
    - **counterfactual**: "What if?", "How to change the outcome?", "Actionable changes", "What can the user do with minimal changes to receive positive predictions?", "What should the the bank do so that the user will subscribe or buy this service?".
    
    If the user mentions a specific ID (integer), extract it as `user_id`. Otherwise `0` as default.
    
    Return a strict JSON object:
    {{
        "mode": "global" | "local" | "fairness" | "counterfactual",
        "user_id": int or 0,
        "reason": "brief explanation"
    }}
    """
    
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        return {
            "analysis_mode": result.get("mode", "global"),
            "user_id": result.get("user_id"),
        }
    except Exception as e:
        print(f"Router Error: {e}")
        return {
            "analysis_mode": "global",
            "user_id": 0
        }
