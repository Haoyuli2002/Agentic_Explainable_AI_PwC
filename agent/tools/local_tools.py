import shap
import matplotlib.pyplot as plt
import os
import uuid
import pandas as pd
from agent.tools.global_tools import compute_shap_values

def generate_shap_waterfall_plot(shap_values, index: int, save_dir="artifacts"):
    """
    Generates a SHAP waterfall plot for a specific observation index.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"shap_waterfall_{index}_{uuid.uuid4().hex}.png"
    save_path = os.path.join(save_dir, filename)
    
    # SHAP values object contains .values (n_samples, n_features), .base_values (n_samples,), .data (n_samples, n_features)
    # We need to slice it for the specific instance
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[index], show=False)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return save_path
