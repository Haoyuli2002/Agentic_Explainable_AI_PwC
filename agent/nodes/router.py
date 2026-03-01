from agent.state import XAIState
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
import json

def router_agent(state: XAIState, config: RunnableConfig):
    """
    Router Node: Classifies the user's latest query to determine the analysis path.
    Includes the last AI message as context so follow-up answers are routed correctly.
    """
    messages = state.get('messages', [])
    if not messages:
        return {"analysis_mode": "data_understanding"}

    # Get last user message
    last_human = None
    last_ai = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and last_human is None:
            last_human = m.content
        elif isinstance(m, AIMessage) and m.content.strip() and last_ai is None:
            last_ai = m.content
        if last_human and last_ai:
            break

    if not last_human:
        return {"analysis_mode": "data_understanding"}

    # Build context string with previous AI message for follow-up awareness
    context = ""
    if last_ai:
        # Truncate long AI responses so the prompt stays manageable
        ai_preview = last_ai[:400] + "..." if len(last_ai) > 400 else last_ai
        context = f"\nThe previous AI message was:\n\"{ai_preview}\"\n"

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    lang = config.get("configurable", {}).get("lang", "en")

    lang_instruction = "\nIMPORTANT: Respond ONLY with the JSON object as instructed. Do not add any extra text." if lang == "en" else "\n重要：只需返回指定的JSON对象，不要添加任何额外文字。"

    prompt = f"""
    You are an Intent Classifier for an Explainable AI System.
    {context}
    The user's latest message: "{last_human}"

    IMPORTANT: If the previous AI message was asking a clarifying question (e.g., "Which sensitive attribute?", "Which user ID?") and the user's reply is a short answer to that question, classify it with the SAME mode as the previous question's topic.

    Classify the intent into one of these modes:
- **data_understanding**: Dataset analysis, column questions, show samples, metadata corrections.
- **global**: Overall model behavior, feature importance, summary plots.
- **local**: Explanation for a specific user/row (look for IDs like "User 5", "row 10").
- **fairness**: Bias detection, demographic parity, "is this fair?", "check discrimination", or a follow-up answer to a fairness question (e.g., user replies with a sensitive attribute name).
- **counterfactual**: "What if?", "How to change the outcome?".

Return ONLY a strict JSON object:
{{
    "mode": "data_understanding" | "global" | "local" | "fairness" | "counterfactual",
    "user_id": int or 0,
    "reason": "brief explanation"
}}{lang_instruction}"""

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)

        mode = result.get("mode", "data_understanding")
        valid_modes = ["data_understanding", "global", "local", "fairness", "counterfactual"]
        if mode not in valid_modes:
            mode = "data_understanding"

        return {
            "analysis_mode": mode,
            "user_id": result.get("user_id", 0),
        }
    except Exception as e:
        print(f"Router Error: {e}")
        return {
            "analysis_mode": "data_understanding",
            "user_id": 0
        }
