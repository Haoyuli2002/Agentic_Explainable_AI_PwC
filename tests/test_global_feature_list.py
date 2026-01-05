import sys
import os
import pandas as pd
from catboost import CatBoostClassifier
from langchain_core.messages import HumanMessage

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.nodes.global_explainer import get_global_feature_importance_shap

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

def test_global_feature_list():
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
    
    print("Mocking State...")
    state = {
        "df": df,
        "model": model,
        "target_variable": "y"
    }
    
    print("Invoking Tool...")
    # Directly invoke the tool function logic (simulating tool call)
    # The tool expects a named argument 'state'
    result = get_global_feature_importance_shap.invoke({"state": state})
    
    print("\n--- Tool Result ---")
    print(result)
    
    # Verification
    assert "Top Driving Features" in result, "Output should contain the feature list header"
    assert "1. **" in result, "Output should contain numbered list items"
    # duration is usually the top feature for this banking dataset
    assert "duration" in result, "Output should likely contain 'duration' feature" 
    
    if "Others" in result:
        print("\nVerified: 'Others' category exists.")
    else:
        print("\nNote: 'Others' category not present (possibly <= 9 features).")

    print("\nTest PASSED!")

if __name__ == "__main__":
    test_global_feature_list()
