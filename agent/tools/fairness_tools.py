import pandas as pd
from typing import Any, Dict

def calculate_demographic_parity(df: pd.DataFrame, sensitive_column: str, target_column: str, favorable_label: Any = 1) -> Dict[str, float]:
    """
    Calculates Demographic Parity Difference.
    DPD = P(y=1 | sensitive=A) - P(y=1 | sensitive=B)
    """
    # Assuming standard favorable label 1 or 'yes'
    # Simple implementation for binary sensitive attribute (or taking max diff)
    
    unique_groups = df[sensitive_column].unique()
    group_rates = {}
    
    for group in unique_groups:
        group_df = df[df[sensitive_column] == group]
        if len(group_df) > 0:
            rate = len(group_df[group_df[target_column] == favorable_label]) / len(group_df)
            group_rates[group] = rate
        else:
            group_rates[group] = 0.0
            
    # Calculate max difference
    rates = list(group_rates.values())
    if not rates:
        return {"demographic_parity_difference": 0.0}
        
    dp_diff = max(rates) - min(rates)
    
    return {
        "demographic_parity_difference": dp_diff,
        "group_rates": group_rates
    }
