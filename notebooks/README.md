# CatBoost Analysis for Bank Marketing

This project utilizes the **CatBoost** gradient boosting library to predict client subscription to a term deposit based on the Bank Marketing dataset. It demonstrates a complete end-to-end workflow from data loading to advanced model interpretability using **SHAP**.

## Analysis Overview

This project focuses on three core pillars to derive actionable insights from the Bank Marketing dataset:

1.  **Model Training (CatBoost)**
    *   **Purpose**: To build a robust predictive model that accurately identifies customers likely to subscribe to a term deposit.
    *   **Goal**: Optimize marketing efficiency by targeting high-probability prospects.

2.  **Explainability (SHAP)**
    *   **Purpose**: To open the "black box" of the model and understand *why* it makes specific predictions.
    *   **Goal**: Reveal key drivers of customer behavior (e.g., call duration, account balance) and validate model logic against domain knowledge.

3.  **Counterfactual Analysis (DiCE)**
    *   **Purpose**: To explore "what-if" scenarios and generate diverse counterfactual explanations.
    *   **Goal**: Provide actionable recommendations for sales agents (e.g., *"If this customer had been contacted in May instead of October, they might have subscribed"*).

## Notebook Overview

The analysis is contained in [catboost_analysis.ipynb](catboost_analysis.ipynb) and covers the following key stages:

### 1. Data Loading & Preprocessing
- Loads the dataset, handling ARFF-like headers manually.
- Cleans categorical string columns (stripping quotes).
- Encodes the target variable (`y`) into binary format (0/1).
- Automatically identifies categorical features for CatBoost's native handling.

### 2. Model Training
- **Model**: `CatBoostClassifier`
- **Parameters**:
  - `iterations`: 500
  - `learning_rate`: 0.1
  - `depth`: 6
  - `loss_function`: Logloss
  - `eval_metric`: AUC
- **Validation**: Uses a 20% stratified test split to monitor performance and prevent overfitting (early stopping enabled).

### 3. Evaluation
The model achieves strong performance metrics:
- **Accuracy**: ~91.1%
- **ROC AUC**: ~0.936
- **Metrics**: Detailed Classification Report and Confusion Matrix.
- **Threshold Tuning**: Includes logic to find the optimal classification threshold (e.g., maximizing F1-score).

### 4. Interpretability & Explainability
A significant focus of this notebook is understanding *why* predictions are made:
- **Feature Importance**: Visualizes the default CatBoost feature importance scores.
- **SHAP (SHapley Additive exPlanations)**:
  - **Global Importance**: Uses a Beeswarm plot to show how features impact the model output across the entire dataset.
  - **Local Explanation**: Uses Waterfall plots to explain individual predictions (e.g., why a specific customer was classified as low risk).
  - **Error Analysis**: specifically analyzes False Negative cases to understand missed opportunities.
- **Counterfactual Analysis (DiCE)**:
  - Uses **DiCE (Diverse Counterfactual Explanations)** to generate "what-if" scenarios.
  - Generates actionable changes (e.g., changing contact method or month) that would flip a prediction from "No" to "Yes".
- **Similarity Analysis (KNN & LLM)**:
  - Uses **K-Nearest Neighbors (KNN)** to find real historical customers who are similar to a specific target but had a positive outcome.
  - Uses **OpenAI (LLM)** to generate a narrative comparison and actionable recommendations for sales agents based on these similar peers.

## Requirements
- `catboost`
- `shap`
- `dice-ml`
- `openai`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
