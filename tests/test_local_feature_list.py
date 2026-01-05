import sys
import os
import pandas as pd
from catboost import CatBoostClassifier

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.nodes.local_explainer import run_shap_explanation, run_lime_explanation

def load_arff_data(file_path):
    data = []
    columns = []
    with open(file_path, 'r') as f:
        data_started = False
        for line in f:
            line = line.strip()
            if not line: continue
            if line.lower().startswith("@attribute"):
                parts = line.split()
                columns.append(parts[1])
            elif line.lower().startswith("@data"):
                data_started = True
                continue
            elif data_started:
                row = [x.strip().strip("'").strip('"') for x in line.split(',')]
                data.append(row)
    return pd.DataFrame(data, columns=columns)

def test_local_feature_list():
    print("Loading data...")
    dataset_path = "datasets/banking_deposit_subscription/dataset"
    df = load_arff_data(dataset_path)
    
    # Preprocessing
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
            
    print("Loading model...")
    model = CatBoostClassifier()
    model.load_model("notebooks/models/catboost_model.cbm")
    
    # Mock User ID 0
    user_id = 0
    
    print("Mocking State...")
    state = {
        "df": df,
        "model": model,
        "target_variable": "y"
    }

    print("\n--- Testing SHAP Tool ---")
    result_shap = run_shap_explanation.invoke({"user_id": user_id, "state": state})
    print(result_shap)
    
    assert "Top Contributing Features" in result_shap, "SHAP output missing header"
    assert "(SHAP)" in result_shap, "SHAP output missing (SHAP) tag"
    assert "1. **" in result_shap, "SHAP output missing numbered list"
    
    print("\n--- Testing LIME Tool ---")
    result_lime = run_lime_explanation.invoke({"user_id": user_id, "state": state})
    print(result_lime)
    
    assert "Top Contributing Features" in result_lime, "LIME output missing header"
    assert "(LIME)" in result_lime, "LIME output missing (LIME) tag"
    assert "1. **" in result_lime, "LIME output missing numbered list"

    print("\nTest PASSED!")

if __name__ == "__main__":
    test_local_feature_list()
