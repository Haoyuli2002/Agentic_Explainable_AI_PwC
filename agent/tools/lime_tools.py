import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import os
import uuid
import matplotlib.pyplot as plt

def compute_lime_explanation(model, X_train: pd.DataFrame, instance: pd.Series, mode: str = "classification"):
    """
    Computes LIME explanation for a single instance.
    Requires the full training set (X_train) to initialize the explainer statistics.
    """
    # LIME needs categorical feature indices if data is not encoded? 
    # Assuming X_train is fully numerical logic for now (preprocessed).
    # If using CatBoost, it handles categoricals, so we might need to be careful if X_train contains strings.
    
    # Initialize Explainer
    # Note: training_data should be a numpy array for robustness with Lime
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=list(X_train.columns),
        class_names=['Class 0', 'Class 1'], # Heuristic for binary
        mode=mode
    )
    
    # Explain
    # predict_fn should be model.predict_proba for classification
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict_proba if mode == "classification" else model.predict,
        num_features=10
    )
    return exp

def generate_lime_plot(lime_exp, save_dir="artifacts"):
    """
    Saves the LIME explanation plot.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"lime_explanation_{uuid.uuid4().hex}.png"
    save_path = os.path.join(save_dir, filename)
    
    # Lime exp.as_pyplot_figure() exists
    fig = lime_exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    return save_path
