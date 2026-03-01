from agent.state import XAIState
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langchain_openai import ChatOpenAI

def summarize_memory(state: XAIState):
    """
    Memory Manager Node: Checks if the conversation history is getting too long (short-term window of 10).
    If it is, it summarizes the older messages into a long-term summary and removes them from the state.
    """
    messages = state.get('messages', [])
    summary = state.get('summary', "")
    
    # We want to keep the last 10 messages (5 pairs of interactions roughly)
    # Plus, the first message is often the system prompt in some setups, but here it's managed by add_messages
    
    if len(messages) <= 10:
        return {} # No state update needed
    
    # Identify messages to summarize: everything EXCEPT the last 10
    messages_to_summarize = messages
    
    # Format the messages for the summarization prompt
    convo_text = ""
    for m in messages_to_summarize:
        role = "User" if isinstance(m, HumanMessage) else "AI"
        convo_text += f"{role}: {m.content}\n"
    
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
    Return ONLY the summary text, nothing else.
    """

    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        new_summary = response.content.strip()
        
        # Remove the messages we just summarized to keep the state lightweight
        # In LangGraph, returning RemoveMessage(id=msg.id) tells the reducer to delete it
        remove_instructions = [RemoveMessage(id=m.id) for m in messages_to_summarize]
        
        return {
            "summary": new_summary,
            "messages": remove_instructions
        }
    except Exception as e:
        print(f"Memory Agent Error: {e}")
        return {}
