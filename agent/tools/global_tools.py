import shap
import matplotlib.pyplot as plt
import pandas as pd
import uuid
import os

def compute_shap_values(model, X):
    """
    Computes SHAP values for the given model and dataset.
    """
    # TreeExplainer is best for CatBoost/XGBoost/RF
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap_values

def generate_shap_summary_plot(shap_values, X, save_dir="artifacts"):
    """
    Generates a SHAP summary plot and saves it to a file.
    Returns the file path.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"shap_summary_{uuid.uuid4().hex}.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return save_path
