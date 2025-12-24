import sys
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent.graph import app
from agent.state import XAIState
from langchain_core.messages import HumanMessage

# Mock Data and Model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def create_mock_data():
    df = pd.DataFrame({
        'age': np.random.randint(18, 70, 100),
        'gender': np.random.choice(['Male', 'Female'], 100),
        'income': np.random.choice([0, 1], 100)
    })
    return df

def train_mock_model(df):
    le = LabelEncoder()
    df['gender_enc'] = le.fit_transform(df['gender'])
    X = df[['age', 'gender_enc']]
    y = df['income']
    model = LogisticRegression()
    model.fit(X, y)
    return model

def test_fairness_flow():
    print(">>> Setting up Mock Environment...")
    df = create_mock_data()
    model = train_mock_model(df)
    
    # Initialize State with description
    initial_state = XAIState(
        messages=[HumanMessage(content="Check if this model is fair regarding gender.")],
        df=df,
        model=model,
        target_variable="income",
        dataset_description="Mock Income Dataset with age and gender.",
        start_time=None,
        metadata={"feature_description": {"age": "Age", "gender": "Gender", "income": "Income Target"}},
        analysis_mode="data_understanding", # Initial default
        chat_history=[],
        sensitive_variables=[]
    )
    
    print(">>> invoking Graph...")
    config = {"recursion_limit": 50}
    
    output = app.invoke(initial_state, config=config)
    
    print("\n>>> Final Output Messages:")
    for m in output['messages']:
        print(f"[{m.type}]: {m.content[:500]}...")

    # usage check
    print("\n>>> Check for Plot:")
    if os.path.exists("fairness_analysis_plot.png"):
        print("Success: fairness_analysis_plot.png generated.")
        # Cleanup
        os.remove("fairness_analysis_plot.png")
    else:
        print("Failure: fairness_analysis_plot.png NOT generated.")

if __name__ == "__main__":
    test_fairness_flow()
