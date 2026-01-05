import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Tuple
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_fairness_metrics(
    model: Any, 
    df: pd.DataFrame, 
    sensitive_features: List[str], 
    target_column: str,
    problem_type: str = "classification"
) -> Dict[str, Any]:
    """
    Computes fairness metrics using Fairlearn.
    
    Args:
        model: The trained machine learning model.
        df: The dataset (containing features and target).
        sensitive_features: List of column names to treat as sensitive.
        target_column: The name of the target variable.
        problem_type: "classification" or "regression" (currently supports classification).
        
    Returns:
        A dictionary containing the calculated metrics and a description.
    """
    # Separate features and target
    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y_true = df[target_column]
    else:
        return {"error": "Target column not found in dataset."}
    
    # Handle sensitive features
    # For simplicity, if multiple are provided, we might define intersectional groups or just pick the first one.
    # Let's support the first one primarily for now, or create a combined feature.
    if not sensitive_features or len(sensitive_features) == 0:
        return {"error": "No sensitive features provided."}
    
    sensitive_feature_name = sensitive_features[0]
    sensitive_data = df[sensitive_feature_name]
    
    # Get predictions
    try:
        y_pred = model.predict(X)
    except Exception as e:
        return {"error": f"Model prediction failed: {str(e)}"}
    
    # Define metrics of interest
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "selection_rate": selection_rate
    }
    
    # Create MetricFrame
    # We need to handle potential issues if sensitive data has many unique values -> maybe binning?
    # For now assuming categorical or low-cardinality.
    
    mf = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_data
    )
    
    # Compute aggregate metrics (differences/ratios)
    # Demographic Parity Difference (difference in selection rates)
    dp_diff = demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_data
    )
    
    # Equalized Odds Difference (requires both true positive and false positive rate differences)
    # Only applicable if model predicts 0/1 or compatible classes
    try:
        eo_diff = equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_data
        )
    except:
        eo_diff = None

    results = {
        "overall": mf.overall.to_dict(),
        "by_group": mf.by_group.to_dict(),
        "demographic_parity_difference": float(dp_diff),
        "equalized_odds_difference": float(eo_diff) if eo_diff is not None else "N/A",
        "sensitive_feature": sensitive_feature_name
    }
    
    return results

def plot_fairness_metrics(fairness_results: Dict[str, Any], output_path: str = "fairness_plot.png") -> str:
    """
    Generates a plot for the fairness metrics.
    """
    if "error" in fairness_results:
        return "Cannot plot due to errors in calculation."
        
    by_group = pd.DataFrame(fairness_results["by_group"])
    sensitive_feature = fairness_results.get("sensitive_feature", "Group")
    
    # Reset index to get the group names as a column
    by_group = by_group.reset_index().rename(columns={sensitive_feature: "Group"})
    
    # Melt for seaborn
    df_melted = by_group.melt(id_vars="Group", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Group", y="Score", hue="Metric")
    plt.title(f"Fairness Metrics by {sensitive_feature}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path
