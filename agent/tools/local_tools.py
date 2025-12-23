import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid
from sklearn.preprocessing import LabelEncoder
from agent.tools.global_tools import compute_shap_values

# --- SHAP Tools ---

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

# --- LIME Tools ---

def compute_lime_explanation(model, X_train: pd.DataFrame, instance: pd.Series, problem_type: str = "classification"):
    """
    Computes LIME explanation for a single instance.
    Handles categorical features by encoding them for LIME and decoding for the model.
    """
    X_train_enc = X_train.copy()
    instance_enc = instance.copy()
    
    # identify categorical columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_indices = [X_train.columns.get_loc(c) for c in cat_cols]
    
    encoders = {}
    
    # Encode categorical columns for LIME (LIME needs numbers)
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on both train and instance to ensure all classes are known (simple approach)
        # In prod, fit on train, handle unseen in instance.
        vals = pd.concat([X_train[col], pd.Series([instance[col]])]).astype(str).unique()
        le.fit(vals)
        
        X_train_enc[col] = le.transform(X_train[col].astype(str))
        instance_enc[col] = le.transform([str(instance[col])])[0]
        encoders[col] = le

    # Wrapper for the prediction function
    # LIME passes a numpy array of perturbations (numbers)
    # We need to convert back to original format (strings) for the model
    def predict_fn_wrapper(data_numpy):
        # Convert back to DataFrame
        df_temp = pd.DataFrame(data_numpy, columns=X_train.columns)
        
        # Decode categoricals
        for col, le in encoders.items():
            # Round to nearest integer (LIME might generate floats)
            vals = df_temp[col].round().astype(int)
            # Clip to valid range of encoder
            vals = vals.clip(0, len(le.classes_) - 1)
            df_temp[col] = le.inverse_transform(vals)
            
        # Ensure numericals are correct types if needed (optional)
        
        if problem_type == "classification":
            return model.predict_proba(df_temp)
        else:
            return model.predict(df_temp)

    # Initialize Explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train_enc),
        feature_names=list(X_train.columns),
        categorical_features=cat_indices,
        class_names=['Class 0', 'Class 1'], 
        mode=problem_type,
        verbose=False
    )
    
    # Explain
    exp = explainer.explain_instance(
        data_row=np.array(instance_enc),
        predict_fn=predict_fn_wrapper,
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
    try:
        fig = lime_exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating LIME plot: {e}")
        return "Error generating plot"
    
    return save_path
