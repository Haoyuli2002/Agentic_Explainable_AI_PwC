import pandas as pd
from typing import Any, Optional, List

# Singleton storage for global context
_MODEL: Optional[Any] = None
_DF: Optional[pd.DataFrame] = None
_TARGET_COL: Optional[str] = None
_FEATURE_COLS: Optional[List[str]] = None

def set_context(model: Any, df: pd.DataFrame, target_col: str = 'y'):
    """
    Sets the global context for the XAI system.
    Should be called at application startup.
    """
    global _MODEL, _DF, _TARGET_COL, _FEATURE_COLS
    _MODEL = model
    _DF = df
    _TARGET_COL = target_col
    
    # Infer feature columns
    if target_col in df.columns:
        _FEATURE_COLS = [c for c in df.columns if c != target_col]
    else:
        _FEATURE_COLS = list(df.columns)

def get_model() -> Any:
    if _MODEL is None:
        raise ValueError("Global Model not set. Call set_context() first.")
    return _MODEL

def get_dataset() -> pd.DataFrame:
    if _DF is None:
        raise ValueError("Global DataFrame not set. Call set_context() first.")
    return _DF

def get_target_column() -> str:
    return _TARGET_COL

def get_feature_columns() -> List[str]:
    return _FEATURE_COLS
