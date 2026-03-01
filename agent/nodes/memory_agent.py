from agent.state import XAIState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables.config import RunnableConfig

def summarize_memory(state: XAIState, config: RunnableConfig):
    """
    Memory Manager Node: Checks if the conversation history is getting too long (short-term window of 10).
    If it is, it summarizes the older messages into a long-term summary and removes them from the state.
    """
    messages = state.get('messages', [])
    summary = state.get('summary', "")
    
    if len(messages) <= 10:
        return {}  # No state update needed
    
    keep_idx = max(0, len(messages) - 10)
    while keep_idx < len(messages) and not isinstance(messages[keep_idx], HumanMessage):
        keep_idx += 1
        
    if keep_idx == len(messages):
        keep_idx = len(messages) - 1
        
    messages_to_summarize = messages[:keep_idx]
    
    if not messages_to_summarize:
        return {}
    
    convo_text = ""
    for m in messages_to_summarize:
        role = "User" if isinstance(m, HumanMessage) else "AI"
        convo_text += f"{role}: {m.content}\n"
    
    lang = config.get("configurable", {}).get("lang", "en")
    lang_instruction = (
        "\n\n重要：请用简体中文生成摘要。"
        if lang == "zh" else
        "\n\nReturn the summary in English."
    )

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    prompt = f"""
    You are a helpful assistant memory manager.
    Your job is to update a running summary of a conversation between a user and an AI data analysis agent.
    
    Previous Summary:
    {summary if summary else "No previous summary."}

    Recent Conversation to incorporate:
    {convo_text}

    Provide an updated, concise summary of the entire conversation so far. Focus on:
    - What dataset/model the user is analyzing.
    - What goals the user has (e.g., finding feature importance, checking fairness).
    - Any key findings or preferences established.
    Return ONLY the summary text, nothing else.{lang_instruction}
    """

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        new_summary = response.content.strip()
        
        remove_instructions = [RemoveMessage(id=m.id) for m in messages_to_summarize]
        
        return {
            "summary": new_summary,
            "messages": remove_instructions
        }
    except Exception as e:
        print(f"Memory Agent Error: {e}")
        return {}
