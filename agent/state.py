from typing import TypedDict, Annotated, List, Any, Optional
from langchain_core.messages import AnyMessage
import pandas as pd
from langgraph.graph.message import add_messages

class XAIState(TypedDict):
    """
    State for the Explainable AI Agent System.
    """
    # Conversation History (persisted)
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Context / References (can be injected or loaded)
    df: Optional[pd.DataFrame]
    model: Optional[Any]
    
    # Content understanding
    target_variable: Optional[str]
    problem_type: Optional[str]
    dataset_format: Optional[str]

    dataset_description: Optional[str]
    feature_description: Optional[dict]
    
    # Router Outputs
    analysis_mode: Optional[str]  # 'data', 'global', 'local', 'fairness', 'counterfactual'
    user_id: Optional[int]        # For local/counterfactual explanations
    
    # Results storage (optional, or just use messages)
    visualization_path: Optional[str]
